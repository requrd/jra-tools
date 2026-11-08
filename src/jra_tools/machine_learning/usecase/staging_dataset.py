"""jrdb-modelの表(staging)から、学習用のデータセットを作る手続き.

仕様の特徴量・着順・単勝の払戻の列だけを、1年ずつ読む(読み手はIStagingRowsReader)。
組み立てはtable_dataset.assemble_dataset。
"""

from typing import Any, List, Sequence

import numpy as np

from jra_tools.machine_learning.entity.dataset import DatasetRows, TableDataset
from jra_tools.machine_learning.entity.feature_spec import FeatureSpec, check_jrdb_spec
from jra_tools.machine_learning.entity.period import year_periods
from jra_tools.machine_learning.entity.staging_rows import check_joined_rows
from jra_tools.machine_learning.usecase.istaging_rows_reader import IStagingRowsReader
from jra_tools.machine_learning.usecase.table_dataset import assemble_dataset


def load_staging_dataset(
    reader: IStagingRowsReader, spec: FeatureSpec, start: int, end: int
) -> TableDataset:
    """期間内(YYYYMMDD)の開催の、学習用のデータセットを作る. 1年ずつ読み込む.

    Raises:
        ValueError: 特徴量の層から読む仕様、または成績・払戻・仕様の表の行がない出走がある場合

    Returns:
        TableDataset: レースは開催日・開催キー・レースキーの順

    """
    check_jrdb_spec(spec)
    chunks: List[DatasetRows] = []
    for s, e in year_periods(start, end):
        rows, missing = reader.read_rows(spec, s, e)
        check_joined_rows(missing)
        chunks.append(rows)
    return assemble_dataset(_concat(chunks), spec)


def _concat(chunks: Sequence[DatasetRows]) -> DatasetRows:
    def cat(values: Sequence[Any]) -> np.ndarray:
        return np.concatenate(values) if values else np.array([], dtype=object)

    keys = chunks[0].features.keys() if chunks else []
    return DatasetRows(
        racehorsekeys=cat([c.racehorsekeys for c in chunks]),
        racekeys=cat([c.racekeys for c in chunks]),
        nums=cat([c.nums for c in chunks]).astype(int),
        features={key: cat([c.features[key] for c in chunks]) for key in keys},
        order_of_arrival=cat([c.order_of_arrival for c in chunks]),
        win1_num=cat([c.win1_num for c in chunks]),
        win1_ret=cat([c.win1_ret for c in chunks]),
    )
