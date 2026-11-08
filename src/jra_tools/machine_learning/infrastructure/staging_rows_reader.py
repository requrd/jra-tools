"""jrdb-modelの表(staging)から、出走ごとに必要な列だけをSQLで読む(IStagingRowsReaderの実装).

ORMのオブジェクトを作らず、仕様の特徴量・着順・単勝の払戻の列だけを読む。
接続先は、jrdb-modelの表があるDB(分析用のPostgresのstaging、MariaDB)。
"""

from typing import Dict, List, Tuple

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

from jra_tools.machine_learning.entity.dataset import DatasetRows
from jra_tools.machine_learning.entity.feature_spec import FeatureSpec, check_jrdb_spec

# sourceのテーブル名と、racehorseから外部結合するテーブル
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


class SqlStagingRowsReader:
    """1つの接続で、jrdb-modelの表から出走の行を読む."""

    def __init__(self, connection: Connection):
        """DBの接続を受け取る."""
        self._connection = connection

    def read_rows(
        self, spec: FeatureSpec, start: int, end: int
    ) -> Tuple[DatasetRows, Dict[str, List[str]]]:
        """期間内の開催の出走ごとの列と、表ごとの行のない出走のキーを読む.

        Args:
            spec (FeatureSpec): 特徴量の仕様
            start (int): 開始日(YYYYMMDD)
            end (int): 終了日(YYYYMMDD)

        Raises:
            ValueError: 仕様に、読めない表がある場合

        Returns:
            Tuple[DatasetRows, Dict[str, List[str]]]: 開催日・開催キー・レースキー・馬番の順の行と、
                表の名前 → 行のない出走のキー(seiseki・returninfo・仕様が使う外部結合の表)

        """
        return _read_rows(self._connection, spec, start, end)


def _read_rows(
    connection: Connection, spec: FeatureSpec, start: int, end: int
) -> Tuple[DatasetRows, Dict[str, List[str]]]:
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
    missing = {
        "seiseki": _missing(columns, "_seiseki", "racehorsekey"),
        "returninfo": _missing(columns, "_returninfo", "racekey"),
        **{table: _missing(columns, f"_{table}", "racehorsekey") for table in joined},
    }
    return DatasetRows(
        racehorsekeys=columns["racehorsekey"],
        racekeys=columns["racekey"],
        nums=columns["num"].astype(int),
        features={f"{t}.{c}": columns[f"{t}.{c}"] for t, c in sources},
        order_of_arrival=columns["order_of_arrival"],
        win1_num=columns["win1_num"],
        win1_ret=columns["win1_ret"],
    ), missing


def _sources(spec: FeatureSpec) -> List[Tuple[str, str]]:
    check_jrdb_spec(spec)
    sources = sorted({(f.table, f.column) for g in spec.groups for f in g.features})
    unknown = [f"{t}.{c}" for t, c in sources if t not in _ENTITIES]
    if unknown:
        raise ValueError(
            f"入力に使えないテーブルです: {unknown}(使えるのは{list(_ENTITIES)})"
        )
    return sources


def _missing(columns: Dict[str, np.ndarray], marker: str, key: str) -> List[str]:
    """外部結合した行がない出走(またはレース)のキー(読んだ順)."""
    absent = np.array([value is None for value in columns[marker]], dtype=bool)
    return [str(value) for value in columns[key][absent]]
