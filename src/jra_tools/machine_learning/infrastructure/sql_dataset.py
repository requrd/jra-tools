"""必要な列だけをSQLで読み、学習用のデータセットを作る.

MysqlLoader(ORM)は開催データに紐づくすべてのテーブル・列をオブジェクトとして
読み込むため、数年分でメモリを大きく使う。こちらは仕様の特徴量・着順・単勝の払戻の
列だけを読み、列単位のndarrayにする。
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from jrdb_model import (
    BangumiData,
    CalculatedScoreData,
    KaisaiData,
    RacehorseData,
    ReturninfoData,
    SeisekiData,
    TrainAnalysisData,
    TrainOikiriData,
)
from sqlalchemy import select
from sqlalchemy.engine import Connection

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.infrastructure.mysql_loader import (
    _periods,
    _to_date,
    _to_int,
)
from jra_tools.machine_learning.usecase.table_dataset import (
    DatasetRows,
    TableDataset,
    assemble_dataset,
)

# sourceのテーブル名と、racehorseから外部結合するテーブル(feature_matrixの_ROWSと同じ範囲)
# (テーブル, 結合に使うリレーション, 行があるかを確かめる列)
_JOINED = {
    "train_analysis": (
        TrainAnalysisData,
        RacehorseData.trainanalysis,
        TrainAnalysisData.racehorsekey,
    ),
    "train_oikiri": (
        TrainOikiriData,
        RacehorseData.trainoikiri,
        TrainOikiriData.racehorsekey,
    ),
    "calculated_score": (
        CalculatedScoreData,
        RacehorseData.calculated_score,
        CalculatedScoreData.racehorsekey,
    ),
}
_ENTITIES = {
    "kaisai": KaisaiData,
    "bangumi": BangumiData,
    "racehorse": RacehorseData,
    **{table: entity for table, (entity, _, _) in _JOINED.items()},
}


def load_dataset(
    connection: Connection, spec: FeatureSpec, start: int, end: int
) -> TableDataset:
    """期間内の開催の、学習用のデータセットを作る. 1年ずつ読み込む.

    Args:
        connection (Connection): DBの接続
        spec (FeatureSpec): 特徴量の仕様
        start (int): 開始日(YYYYMMDD)
        end (int): 終了日(YYYYMMDD)

    Returns:
        TableDataset: 学習用のデータセット。レースは開催日・開催キー・レースキーの順

    """
    chunks = [
        load_rows(connection, spec, _to_int(s), _to_int(e))
        for s, e in _periods(_to_date(start), _to_date(end))
    ]
    return assemble_dataset(_concat(chunks, spec), spec)


def load_rows(
    connection: Connection, spec: FeatureSpec, start: int, end: int
) -> DatasetRows:
    """期間内の開催の出走ごとの列を読み込む.

    Args:
        connection (Connection): DBの接続
        spec (FeatureSpec): 特徴量の仕様
        start (int): 開始日(YYYYMMDD)
        end (int): 終了日(YYYYMMDD)

    Raises:
        ValueError: 仕様のテーブルや、成績・払戻のない出走がある場合

    Returns:
        DatasetRows: 出走ごとの列。開催日・開催キー・レースキー・馬番の順

    """
    sources = _sources(spec)
    joined = sorted({table for table, _ in sources} & set(_JOINED))
    stmt = (
        select(
            RacehorseData.racehorsekey,
            RacehorseData.racekey,
            RacehorseData.num,
            SeisekiData.racehorsekey.label("_seiseki"),
            SeisekiData.order_of_arrival,
            ReturninfoData.racekey.label("_returninfo"),
            ReturninfoData.win1_num,
            ReturninfoData.win1_ret,
            *[_JOINED[table][2].label(f"_{table}") for table in joined],
            *[
                getattr(_ENTITIES[table], column).label(f"{table}.{column}")
                for table, column in sources
            ],
        )
        .select_from(KaisaiData)
        .join(KaisaiData.races)
        .join(BangumiData.racehorses)
        .outerjoin(RacehorseData.result)
        .outerjoin(BangumiData.returninfo)
        .where(KaisaiData.ymd.between(start, end))
        .order_by(
            KaisaiData.ymd, KaisaiData.kaisaikey, BangumiData.racekey, RacehorseData.num
        )
    )
    for table in joined:
        stmt = stmt.outerjoin(_JOINED[table][1])
    rows = connection.execute(stmt).mappings().all()

    keys = stmt.selected_columns.keys()
    columns = {key: np.array([row[key] for row in rows], dtype=object) for key in keys}
    _check_present(columns, "_seiseki", "seiseki", "racehorsekey")
    _check_present(columns, "_returninfo", "returninfo", "racekey")
    for table in joined:
        _check_present(columns, f"_{table}", table, "racehorsekey")
    return DatasetRows(
        racehorsekeys=columns["racehorsekey"],
        racekeys=columns["racekey"],
        nums=columns["num"].astype(int),
        features={f"{t}.{c}": columns[f"{t}.{c}"] for t, c in sources},
        order_of_arrival=columns["order_of_arrival"],
        win1_num=columns["win1_num"],
        win1_ret=columns["win1_ret"],
    )


def _sources(spec: FeatureSpec) -> List[Tuple[str, str]]:
    sources = sorted({(f.table, f.column) for g in spec.groups for f in g.features})
    unknown = [f"{t}.{c}" for t, c in sources if t not in _ENTITIES]
    if unknown:
        raise ValueError(
            f"入力に使えないテーブルです: {unknown}(使えるのは{list(_ENTITIES)})"
        )
    return sources


def _check_present(
    columns: Dict[str, np.ndarray], marker: str, table: str, key: str
) -> None:
    """外部結合した行がない出走があればエラーにする(ORMの経路と同じメッセージ)."""
    missing = np.array([value is None for value in columns[marker]], dtype=bool)
    if missing.any():
        first = columns[key][missing][0]
        raise ValueError(
            f"{_MISSING_MESSAGES.get(table, table + ' がありません')}: {first}"
        )


_MISSING_MESSAGES = {
    "seiseki": "成績がありません",
    "returninfo": "払戻データがありません",
}


def _concat(chunks: Sequence[DatasetRows], spec: FeatureSpec) -> DatasetRows:
    def cat(values: Sequence[Any]) -> np.ndarray:
        return np.concatenate(values) if values else np.array([], dtype=object)

    return DatasetRows(
        racehorsekeys=cat([c.racehorsekeys for c in chunks]),
        racekeys=cat([c.racekeys for c in chunks]),
        nums=cat([c.nums for c in chunks]).astype(int),
        features={
            f"{t}.{c}": cat([chunk.features[f"{t}.{c}"] for chunk in chunks])
            for t, c in _sources(spec)
        },
        order_of_arrival=cat([c.order_of_arrival for c in chunks]),
        win1_num=cat([c.win1_num for c in chunks]),
        win1_ret=cat([c.win1_ret for c in chunks]),
    )
