"""featuresのビュー(分析用のPostgres)から、学習用のデータセット・推論の入力を作る入口.

Postgresのエンジン(接続)を受け取り、PostgresViewReaderを作ってusecase/view_dataset.pyの
手続きを呼ぶだけ。決まりはentity/input_view.py、手続きはusecase/view_dataset.py、
SQLはinfrastructure/postgres_view_reader.pyにある。
batch・Jupyterはここを使う(名前・引数は0.6.0までと同じ)。
"""

from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import Connection, Engine

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.entity.input_view import (
    DEFAULT_LABELS,
    ENTRY_COLUMNS,
    LABEL_COLUMNS,
)
from jra_tools.machine_learning.infrastructure.postgres_view_reader import (
    PostgresViewReader,
    PostgresViewSession,
    view_digest,
)
from jra_tools.machine_learning.usecase import view_dataset
from jra_tools.machine_learning.usecase.frame_dataset import InputDataset
from jra_tools.machine_learning.usecase.ifinish_model import (
    PROBABILITY_OUTPUTS,
    IFinishModel,
)
from jra_tools.machine_learning.usecase.table_dataset import TableDataset

__all__ = [
    "DEFAULT_LABELS",
    "ENTRY_COLUMNS",
    "LABEL_COLUMNS",
    "PROBABILITY_OUTPUTS",
    "check_view",
    "load_view_dataset",
    "load_view_input",
    "pin_view_digest",
    "predict_finish_probabilities",
    "read_view_frame",
    "view_digest",
]


def pin_view_digest(engine: Engine, spec: FeatureSpec) -> FeatureSpec:
    """仕様のview_digestを、今のビューのハッシュにした仕様を返す(学習の前に使う)."""
    return view_dataset.pin_view_digest(PostgresViewReader(engine), spec)


def check_view(connection: Connection, spec: FeatureSpec) -> None:
    """ビューに仕様の列があり、定義が学習時と同じかを確かめる."""
    view_dataset.check_view(PostgresViewSession(connection), spec)


def load_view_input(engine: Engine, spec: FeatureSpec, ymd: int) -> InputDataset:
    """開催日(YYYYMMDD)の推論の入力を、ビューから作る."""
    return view_dataset.load_view_input(PostgresViewReader(engine), spec, ymd)


def predict_finish_probabilities(
    engine: Engine, spec: FeatureSpec, ymd: int, model: IFinishModel
) -> List[Tuple[str, float, float, float]]:
    """開催日の出走馬ごとの1着・2着・3着の確率を、ビューの入力からモデルで出す."""
    return view_dataset.predict_finish_probabilities(
        PostgresViewReader(engine), spec, ymd, model
    )


def load_view_dataset(
    engine: Engine,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: str = DEFAULT_LABELS,
) -> TableDataset:
    """期間内(YYYYMMDD)の学習用のデータセットを、ビューから作る. 1年ずつ読み込む."""
    return view_dataset.load_view_dataset(
        PostgresViewReader(engine), spec, start, end, labels
    )


def read_view_frame(
    engine: Engine,
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str] = DEFAULT_LABELS,
) -> pd.DataFrame:
    """期間内(YYYYMMDD)の出走を、1頭1行の表で読む. Noneのlabelsはラベルを読まない(推論)."""
    return view_dataset.read_view_frame(
        PostgresViewReader(engine), spec, start, end, labels
    )
