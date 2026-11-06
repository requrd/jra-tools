"""入力のビュー(試作のsandboxのビューを含む)が、学習・推論に使える形かを確かめる.

入力のビューの契約(view_datasetの冒頭)のうち、データで確かめられるものを実データで流す。
どれも返したものがあれば失敗(dbtの個別のテストと同じ)。

- キーの列(ENTRY_COLUMNS)と、確かめる特徴量の列がある
- 出走馬1頭につき1行
- model層の出走馬(model.entries・model.racesのrace_dateが期間内)がすべて出る
  (内部結合で、つながらない出走馬が黙って消えていない)
- 特徴量の列に値なしがない(ビューから読む経路は値なしで止まる)

定義のハッシュ(view_digest)は照合しない。試作は学習の前に確かめるもので、まだ仕様がないため。
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Sequence, Tuple

from sqlalchemy import Connection, Engine, text

from jra_tools.machine_learning.infrastructure.mysql_loader import _to_date
from jra_tools.machine_learning.infrastructure.view_dataset import (
    ENTRY_COLUMNS,
    LABEL_COLUMNS,
)

# 「すべての出走馬」の出どころ(model層。使う人にとって正しい出走馬の一覧)
ENTRIES_REFERENCE = (
    "select e.racehorsekey from model.entries e"
    " join model.races r on r.racekey = e.racekey"
    " where r.race_date between :start and :end"
)
# 失敗のときに残す出走馬の例の数
_EXAMPLES = 5


@dataclass(frozen=True)
class InputViewProblem:
    """確かめて見つかったもの."""

    check: str  # 確かめたこと(columns・unique・covers_entries・not_null:<列>)
    rows: int  # 見つかった件数
    examples: Tuple[str, ...]  # 例(列の名前、または出走馬のキー)


def check_input_view(
    engine: Engine,
    view: str,
    start: int,
    end: int,
    features: Optional[Sequence[str]] = None,
) -> List[InputViewProblem]:
    """入力のビューを、期間内の実データで確かめる.

    Args:
        engine (Engine): Postgresのエンジン
        view (str): 「スキーマ.ビュー名」(例: sandbox.entries_try1)
        start (int): 開始日(YYYYMMDD)
        end (int): 終了日(YYYYMMDD)
        features (Optional[Sequence[str]]): 値なしを確かめる特徴量の列。
            省略時は、キーの列(ENTRY_COLUMNS)以外のすべての列

    Returns:
        List[InputViewProblem]: 見つかったもの。すべて通れば空。
            列が足りなければ、それだけを返す(ほかは確かめない)

    Raises:
        ValueError: ビューがない、またはラベルの列を特徴量に指定した場合

    """
    period = {"start": _to_date(start).date(), "end": _to_date(end).date()}
    with engine.connect() as connection:
        connection.exec_driver_sql("set transaction read only")
        quote = connection.dialect.identifier_preparer.quote
        existing = _columns_of(connection, view)
        columns = (
            list(features)
            if features is not None
            else [c for c in existing if c not in ENTRY_COLUMNS]
        )
        labels = [c for c in columns if c in LABEL_COLUMNS[1:]]
        if labels:
            raise ValueError(f"ラベルの列は入力に使えません: {labels}")
        missing = [c for c in (*ENTRY_COLUMNS, *columns) if c not in existing]
        if missing:
            return [InputViewProblem("columns", len(missing), tuple(missing))]
        name = _qualified(quote, view)
        within = "race_date between :start and :end"
        problems = [
            _problem(
                connection,
                "unique",
                f"select racehorsekey from {name} where {within}"  # noqa: S608
                " group by racehorsekey having count(*) > 1",
                period,
            ),
            _problem(
                connection,
                "covers_entries",
                f"select m.racehorsekey from ({ENTRIES_REFERENCE}) m"  # noqa: S608
                f" where not exists (select 1 from {name} v"
                " where v.racehorsekey = m.racehorsekey)",
                period,
            ),
        ]
        problems.extend(
            _problem(
                connection,
                f"not_null:{column}",
                f"select racehorsekey from {name}"  # noqa: S608
                f" where {within} and {quote(column)} is null",
                period,
            )
            for column in columns
        )
    return [p for p in problems if p is not None]


def _columns_of(connection: Connection, view: str) -> List[str]:
    if connection.execute(text("select to_regclass(:v)"), {"v": view}).scalar() is None:
        raise ValueError(f"ビューがありません: {view}")
    return list(
        connection.execute(
            text(
                "select attname from pg_attribute"
                " where attrelid = cast(:v as regclass) and attnum > 0"
                " and not attisdropped order by attnum"
            ),
            {"v": view},
        ).scalars()
    )


def _problem(
    connection: Connection, check: str, sql: str, period: dict[str, date]
) -> Optional[InputViewProblem]:
    keys = list(connection.execute(text(sql), period).scalars())
    if not keys:
        return None
    return InputViewProblem(check, len(keys), tuple(sorted(keys)[:_EXAMPLES]))


def _qualified(quote, name: str) -> str:
    schema, _, view = name.partition(".")
    return f"{quote(schema)}.{quote(view)}"
