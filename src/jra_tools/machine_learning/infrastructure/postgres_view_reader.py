"""分析用のPostgresのビューを読む(usecase/iview_reader.pyの実装). SQLだけを持つ.

1つのセッションは1トランザクション。ロック(access share)はトランザクションの終わりまで続くので、
読み終わるまでapply_viewsにビューを作り直されない。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import List, Optional, Sequence

import pandas as pd
from sqlalchemy import Connection, Engine, text

from jra_tools.machine_learning.entity.input_view import LABEL_COLUMNS, LABEL_PRESENT
from jra_tools.machine_learning.entity.period import to_date

# ビューと、依存するビュー・関数を、pg_dependでたどって定義を集める。
# 表(staging)は名前だけを含める。search_pathはpg_catalogにしてから流す(名前をすべて
# スキーマ付きで出すため。接続ごとのsearch_pathでハッシュが変わらないように)。
_DIGEST_SQL = """
with recursive deps(oid, kind) as (
  select cast(:view as regclass)::oid, 'r'
  union
  select d.refobjid,
         case d.refclassid when 'pg_class'::regclass then 'r' else 'f' end
  from deps
  join pg_depend d
    on (deps.kind = 'r' and d.classid = 'pg_rewrite'::regclass
        and d.objid in (select rw.oid from pg_rewrite rw where rw.ev_class = deps.oid))
    or (deps.kind = 'f' and d.classid = 'pg_proc'::regclass and d.objid = deps.oid)
  where d.refclassid in ('pg_class'::regclass, 'pg_proc'::regclass)
    and d.refobjid <> deps.oid
)
select encode(sha256(convert_to(string_agg(def, chr(10) order by name), 'UTF8')), 'hex')
from (
  select distinct
    case kind when 'f' then oid::regprocedure::text else oid::regclass::text end as name,
    case
      when kind = 'f' then pg_get_functiondef(oid)
      when (select relkind from pg_class c where c.oid = deps.oid) in ('v', 'm')
        then 'view ' || oid::regclass::text || chr(10) || pg_get_viewdef(oid)
      else 'table ' || oid::regclass::text
    end as def
  from deps
) t
"""


def view_digest(connection: Connection, view: str) -> str:
    """ビューの定義のハッシュ.

    ビューと、依存するビュー・関数の定義から作る。空白やコメントだけの違いでは変わらない
    (Postgresが保存した形から作るため)。Postgresのメジャーバージョンを上げると、
    保存した形の出し方が変わり、ハッシュが変わることがある。

    Args:
        connection (Connection): Postgresの接続
        view (str): 「スキーマ.ビュー名」

    Raises:
        ValueError: ビューがない場合

    Returns:
        str: SHA-256の16進数

    """
    if not view_exists(connection, view):
        raise ValueError(f"ビューがありません: {view}")
    search_path = connection.execute(text("select current_setting('search_path')"))
    previous = search_path.scalar()
    connection.execute(text("select set_config('search_path', 'pg_catalog', true)"))
    try:
        return connection.execute(text(_DIGEST_SQL), {"view": view}).scalar_one()
    finally:
        connection.execute(
            text("select set_config('search_path', :p, true)"), {"p": previous}
        )


def view_exists(connection: Connection, view: str) -> bool:
    """ビュー(または表)があるか."""
    return (
        connection.execute(text("select to_regclass(:v)"), {"v": view}).scalar()
        is not None
    )


class PostgresViewSession:
    """1トランザクションの中で、ビューを読む(IViewSession)."""

    def __init__(self, connection: Connection):
        """トランザクションを始めた接続を受け取る."""
        self._connection = connection
        self._quote = connection.dialect.identifier_preparer.quote

    def exists(self, view: str) -> bool:
        """ビューがあるか."""
        return view_exists(self._connection, view)

    def lock(self, views: Sequence[str]) -> None:
        """ビューを共有ロックする(トランザクションの終わりまで)."""
        names = ", ".join(self._qualified(view) for view in views)
        self._connection.exec_driver_sql(f"lock table {names} in access share mode")

    def columns(self, view: str) -> List[str]:
        """ビューの列の名前(定義の順)."""
        return list(
            self._connection.execute(
                text(
                    "select attname from pg_attribute"
                    " where attrelid = cast(:v as regclass) and attnum > 0"
                    " and not attisdropped order by attnum"
                ),
                {"v": view},
            ).scalars()
        )

    def digest(self, view: str) -> str:
        """ビューの定義のハッシュ."""
        return view_digest(self._connection, view)

    def read(
        self,
        view: str,
        columns: Sequence[str],
        labels: Optional[str],
        start: int,
        end: int,
    ) -> pd.DataFrame:
        """期間内の出走を1頭1行で読む. ラベルのビューは左結合する."""
        quote = self._quote
        selected = [f"e.{quote(c)}" for c in columns]
        join = ""
        if labels:
            selected += [f"l.racehorsekey as {LABEL_PRESENT}"]
            selected += [f"l.{quote(c)}" for c in LABEL_COLUMNS[1:]]
            join = f" left join {self._qualified(labels)} l using (racehorsekey)"
        # 名前はすべてidentifier_preparerで引用している。値はパラメーターで渡す
        result = self._connection.execute(
            text(
                f"select {', '.join(selected)} from {self._qualified(view)} e{join}"  # noqa: S608
                " where e.race_date between :start and :end"
                " order by e.race_date, e.racekey, e.umaban"
            ),
            {"start": to_date(start), "end": to_date(end)},
        )
        return pd.DataFrame(result.all(), columns=list(result.keys()))

    def _qualified(self, name: str) -> str:
        schema, _, view = name.partition(".")
        return f"{self._quote(schema)}.{self._quote(view)}"


class PostgresViewReader:
    """分析用のPostgresのビューの読み手(IViewReader)."""

    def __init__(self, engine: Engine):
        """Postgresのエンジンを受け取る."""
        self._engine = engine

    @contextmanager
    def session(self) -> Iterator[PostgresViewSession]:
        """1トランザクションのセッション. 抜けるときにコミットする(読むだけ)."""
        with self._engine.begin() as connection:
            yield PostgresViewSession(connection)
