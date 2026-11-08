"""jrdb-modelの表(staging)から、学習用のデータセットを作る入口.

使う側はここだけを使う。仕様は最上位にsourceのないもの(各特徴量が「テーブル名.列名」)。
接続先は環境変数DB(jrdb-modelの表があるDB。分析用のPostgresのstaging、MariaDB)。

ここは組み立てだけを持つ(machine_learningの層の外)。DBからSQLの読み手
(infrastructure/staging_rows_reader.py)を作り、手続き(usecase/staging_dataset.py)に渡す。
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from sqlalchemy import Connection, create_engine

from jra_tools.machine_learning.entity.dataset import TableDataset
from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.entity.jrdbdummies import CategoryGetter
from jra_tools.machine_learning.entity.period import quarter_period, training_period
from jra_tools.machine_learning.infrastructure.dataset_files import save_table_dataset
from jra_tools.machine_learning.infrastructure.spec_files import default_feature_spec
from jra_tools.machine_learning.infrastructure.staging_rows_reader import (
    SqlStagingRowsReader,
)
from jra_tools.machine_learning.usecase.staging_dataset import load_staging_dataset

__all__ = [
    "CategoryGetter",
    "default_feature_spec",
    "load_dataset",
    "load_quarter_dataset",
    "load_training_dataset",
    "save_dataset",
]

# 接続先(jrdb-modelの表があるDB)の環境変数
DB_ENV = "DB"


def load_dataset(spec: FeatureSpec, start: int, end: int) -> TableDataset:
    """期間内(YYYYMMDD、両端を含む)の学習用のデータセットを作る. 1年ずつ読む.

    Returns:
        TableDataset: レースは開催日・開催キー・レースキーの順

    """
    with _connect() as connection:
        return load_staging_dataset(SqlStagingRowsReader(connection), spec, start, end)


def load_training_dataset(
    start_year: int, end_year: int, spec: Optional[FeatureSpec] = None
) -> TableDataset:
    """開始年の1月1日から終了年の12月31日までのデータセット. 仕様の省略時は現行の推論モデルの仕様."""
    start, end = training_period(start_year, end_year)
    return load_dataset(spec or default_feature_spec(), start, end)


def load_quarter_dataset(
    year: int, quarter: int, spec: Optional[FeatureSpec] = None
) -> TableDataset:
    """年の四半期(1〜4)のデータセット. 仕様の省略時は現行の推論モデルの仕様."""
    start, end = quarter_period(year, quarter)
    return load_dataset(spec or default_feature_spec(), start, end)


def save_dataset(dataset: TableDataset, name: str, directory: Path | str = ".") -> None:
    """データセットを書き出す(x_{name}.dump・{name}.dump・payout_{name}.dump・racekeys_{name}.npy)."""
    save_table_dataset(dataset, name, directory)


@contextmanager
def _connect() -> Iterator[Connection]:
    url = os.environ.get(DB_ENV)
    if not url:
        raise RuntimeError(
            f"環境変数{DB_ENV}に、jrdb-modelの表があるDBを設定してください"
        )
    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            yield connection
    finally:
        engine.dispose()
