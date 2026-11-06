"""入力のビューを確かめる材料を、分析用のPostgresから読む(IInputViewQueryの実装).

どの問い合わせも読み取り専用のトランザクションで流す。
"""

import datetime as dt
from typing import List

from sqlalchemy import Engine, text

# 「すべての出走馬」の出どころ(model層。使う人にとって正しい出走馬の一覧)
ENTRIES_REFERENCE = (
    "select e.racehorsekey from model.entries e"
    " join model.races r on r.racekey = e.racekey"
    " where r.race_date between :start and :end"
)


class PostgresInputViewQuery:
    """1つの入力のビューと期間(開催日)について、確かめる材料を読む."""

    def __init__(self, engine: Engine, view: str, start: int, end: int):
        """「スキーマ.ビュー名」と、開始日・終了日(YYYYMMDD)を受け取る.

        Raises:
            ValueError: ビューがない場合

        """
        self._engine = engine
        self._view = view
        self._period = {"start": _to_date(start), "end": _to_date(end)}
        quote = engine.dialect.identifier_preparer.quote
        schema, _, name = view.partition(".")
        self._name = f"{quote(schema)}.{quote(name)}"
        self._quote = quote
        if self._scalar("select to_regclass(:v)", {"v": view}) is None:
            raise ValueError(f"ビューがありません: {view}")

    def columns(self) -> List[str]:
        """ビューの列の名前(定義の順)."""
        return self._keys(
            "select attname from pg_attribute"
            " where attrelid = cast(:v as regclass) and attnum > 0"
            " and not attisdropped order by attnum",
            {"v": self._view},
        )

    def duplicated_keys(self) -> List[str]:
        """期間内で2行以上ある出走馬のキー."""
        return self._keys(
            f"select racehorsekey from {self._name}"  # noqa: S608
            " where race_date between :start and :end"
            " group by racehorsekey having count(*) > 1",
            self._period,
        )

    def uncovered_keys(self) -> List[str]:
        """model層の出走馬(期間内)のうち、ビューに出ないもののキー."""
        return self._keys(
            f"select m.racehorsekey from ({ENTRIES_REFERENCE}) m"  # noqa: S608
            f" where not exists (select 1 from {self._name} v"
            " where v.racehorsekey = m.racehorsekey)",
            self._period,
        )

    def null_keys(self, column: str) -> List[str]:
        """期間内で、列が値なしの出走馬のキー."""
        return self._keys(
            f"select racehorsekey from {self._name}"  # noqa: S608
            f" where race_date between :start and :end and {self._quote(column)} is null",
            self._period,
        )

    def _keys(self, sql: str, params: dict) -> List[str]:
        with self._engine.connect() as connection:
            connection.exec_driver_sql("set transaction read only")
            return list(connection.execute(text(sql), params).scalars())

    def _scalar(self, sql: str, params: dict):
        with self._engine.connect() as connection:
            return connection.execute(text(sql), params).scalar()


def _to_date(ymd: int) -> dt.date:
    return dt.datetime.strptime(str(ymd), "%Y%m%d").date()
