"""ツールのユースケース."""

from dataclasses import dataclass
from functools import partial
from typing import List, Optional

from jrdb_model import KaisaiData, create_app, db

from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    default_feature_spec,
)
from jra_tools.machine_learning.infrastructure.mysql_loader import MysqlLoader
from jra_tools.machine_learning.infrastructure.sql_dataset import load_dataset
from jra_tools.machine_learning.interface_adapter.input_creator import InputCreator
from jra_tools.machine_learning.interface_adapter.payout_creator import PayoutCreator
from jra_tools.machine_learning.interface_adapter.quarter_kaisai_query import (
    QuarterKaisaiQuery,
)
from jra_tools.machine_learning.interface_adapter.training_kaisai_query import (
    TrainingKaisaiQuery,
)
from jra_tools.machine_learning.usecase.converter import Converter
from jra_tools.machine_learning.usecase.table_dataset import TableDataset


def _create_training(start: int, end: int, only_input: bool):
    """指定した期間のトレーニングデータを作成する.

    Args:
        start (int): YYYY
        end (int): YYYY
        only_input (bool): 入力データのみを作成するか

    Raises:
        RuntimeError: データのロードまたは保存中にエラーが発生

    """
    converter = Converter(
        TrainingKaisaiQuery(start, end, MysqlLoader),
        [InputCreator] if only_input else [InputCreator, PayoutCreator],
    )
    converter.save()


create_training_dataset = partial(_create_training, only_input=False)
create_training_input = partial(_create_training, only_input=True)


def _create_quarter(year: int, quarter: int, only_input: bool) -> None:
    """指定した四半期のデータセットをファイルに保存する.

    Args:
        year (int): YYYY
        quarter (int): 1 - 4
        only_input (bool, optional): 入力データのみを作成するか. Defaults to True.

    """
    converter = Converter(
        QuarterKaisaiQuery(year, quarter, MysqlLoader),
        [InputCreator] if only_input else [InputCreator, PayoutCreator],
    )
    converter.save()


create_quarter_dataset = partial(_create_quarter, only_input=False)
create_quarter_input = partial(_create_quarter, only_input=True)


@dataclass
class _Period:
    """TrainingKaisaiQuery・QuarterKaisaiQueryで期間(YYYYMMDD)だけを決めるためのローダー."""

    start: int
    end: int

    def load(self) -> List[KaisaiData]:
        """使わない(SQLの経路では開催データを読み込まない)."""
        return []

    @property
    def period(self) -> str:
        """データセットの期間."""
        return f"{self.start}_{self.end}"


def load_training_dataset(
    start: int, end: int, spec: Optional[FeatureSpec] = None
) -> TableDataset:
    """指定した年の範囲の学習用データセットを、必要な列だけをSQLで読んで作る.

    create_training_datasetと同じ期間・同じ値のデータを、ORMを使わずに作る。
    保存する場合は dataset.save(name) を呼ぶ(ファイルはcreate_training_datasetと同じ形式)。

    Args:
        start (int): 開始年(YYYY)
        end (int): 終了年(YYYY)
        spec (Optional[FeatureSpec]): 特徴量の仕様. 省略時は現行の推論モデルの仕様

    Returns:
        TableDataset: 入力・正解ラベル・払戻と、各レースのレースキー

    """
    return _load_by_sql(TrainingKaisaiQuery(start, end, _Period), spec)


def load_quarter_dataset(
    year: int, quarter: int, spec: Optional[FeatureSpec] = None
) -> TableDataset:
    """指定した四半期のデータセットを、必要な列だけをSQLで読んで作る.

    Args:
        year (int): YYYY
        quarter (int): 1 - 4
        spec (Optional[FeatureSpec]): 特徴量の仕様. 省略時は現行の推論モデルの仕様

    Returns:
        TableDataset: 入力・正解ラベル・払戻と、各レースのレースキー

    """
    return _load_by_sql(QuarterKaisaiQuery(year, quarter, _Period), spec)


def _load_by_sql(
    query: TrainingKaisaiQuery | QuarterKaisaiQuery, spec: Optional[FeatureSpec]
) -> TableDataset:
    app = create_app()
    with app.app_context(), db.engine.connect() as connection:
        return load_dataset(
            connection,
            spec or default_feature_spec(),
            query.loader.start,
            query.loader.end,
        )
