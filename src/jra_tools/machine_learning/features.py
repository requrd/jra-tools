"""特徴量の層(featuresや、その試作のsandbox)から、学習・推論の入力を作る入口.

使う側(sazanami-batch・Jupyter)はここだけを使う。読む先は仕様のsource(「スキーマ.名前」)で、
それがビューか表かは使う側には関係ない。接続先は環境変数DB(分析用のPostgres)。

ここは組み立てだけを持つ(machine_learningの層の外)。DBからPostgresの読み手
(infrastructure/postgres_view_reader.py)を、モデルのフォルダからSavedModelFolderを作り、
手続き(usecase/view_dataset.py・model_spec.py)に渡す。決まりはentityにある。
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine

from jra_tools.machine_learning.entity.dataset import InputDataset, TableDataset
from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    parse_feature_spec,
)
from jra_tools.machine_learning.entity.input_view import DEFAULT_LABELS
from jra_tools.machine_learning.infrastructure.postgres_view_reader import (
    PostgresViewReader,
)
from jra_tools.machine_learning.infrastructure.saved_model_folder import (
    SavedModelFolder,
)
from jra_tools.machine_learning.infrastructure.spec_files import (
    load_feature_spec,
    save_feature_spec,
)
from jra_tools.machine_learning.usecase import model_spec, view_dataset
from jra_tools.machine_learning.usecase.ifinish_model import IFinishModel

__all__ = [
    "DEFAULT_LABELS",
    "FeatureSpec",
    "digest",
    "load_dataset",
    "load_feature_spec",
    "load_input",
    "load_model_feature_spec",
    "parse_feature_spec",
    "pin_digest",
    "predict_finish_probabilities",
    "read_frame",
    "save_feature_spec",
    "save_model_feature_spec",
]

# 接続先(分析用のPostgres)の環境変数
DB_ENV = "DB"


def pin_digest(spec: FeatureSpec) -> FeatureSpec:
    """仕様のview_digestを、sourceの今の定義のハッシュにした仕様を返す(学習の前に使う)."""
    with _reader() as reader:
        return view_dataset.pin_view_digest(reader, spec)


def digest(source: str) -> str:
    """source(「スキーマ.名前」)の今の定義のハッシュ(依存する定義を含む)."""
    with _reader() as reader:
        return view_dataset.current_digest(reader, source)


def load_dataset(
    spec: FeatureSpec, start: int, end: int, labels: str = DEFAULT_LABELS
) -> TableDataset:
    """期間内(YYYYMMDD、両端を含む)の学習用のデータセットを作る. 1年ずつ読む."""
    with _reader() as reader:
        return view_dataset.load_view_dataset(reader, spec, start, end, labels)


def load_input(spec: FeatureSpec, ymd: int) -> InputDataset:
    """開催日(YYYYMMDD)の推論の入力を作る."""
    with _reader() as reader:
        return view_dataset.load_view_input(reader, spec, ymd)


def read_frame(
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str] = DEFAULT_LABELS,
) -> pd.DataFrame:
    """期間内(YYYYMMDD)の出走を、組み立てる前の1頭1行の表で読む. labels=Noneはラベルを読まない."""
    with _reader() as reader:
        return view_dataset.read_view_frame(reader, spec, start, end, labels)


def predict_finish_probabilities(
    spec: FeatureSpec, ymd: int, model: IFinishModel
) -> List[Tuple[str, float, float, float]]:
    """開催日の出走馬ごとの(racehorsekey, 1着, 2着, 3着の確率). 出走がなければ空."""
    with _reader() as reader:
        return view_dataset.predict_finish_probabilities(reader, spec, ymd, model)


def load_model_feature_spec(model_dir: Path | str) -> FeatureSpec:
    """モデルのフォルダ(SavedModel)の仕様を読み、モデルの入力の形と照合する."""
    return model_spec.load_model_feature_spec(SavedModelFolder(model_dir))


def save_model_feature_spec(spec: FeatureSpec, model_dir: Path | str) -> Path:
    """学習に使った仕様を、モデルの入力の形と照合してから、モデルのフォルダに書く."""
    return model_spec.save_model_feature_spec(SavedModelFolder(model_dir), spec)


@contextmanager
def _reader() -> Iterator[PostgresViewReader]:
    url = os.environ.get(DB_ENV)
    if not url:
        raise RuntimeError(
            f"環境変数{DB_ENV}に、分析用のPostgresの接続先を設定してください"
        )
    engine = create_engine(url)
    try:
        yield PostgresViewReader(engine)
    finally:
        engine.dispose()
