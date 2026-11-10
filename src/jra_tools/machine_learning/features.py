"""特徴量の層(featuresや、その試作のsandbox)から、学習・推論・評価の入力を作る入口.

使う側(sazanami-batch・Jupyter)はここだけを使う。読む先は仕様のsource(「スキーマ.名前」)で、
それがビューか表かは使う側には関係ない。接続先は環境変数DB(分析用のPostgres)。

流れ: (試作は)apply_sandbox_viewsでsandboxのビューを作る → 仕様を書く → pin_spec(ビューのハッシュとカテゴリの控えを写す)→ load_dataset で学習 →
save_model_feature_spec(出力の名前を写してモデルのフォルダへ)→ evaluate・predict_finish_probabilities

ここは組み立てだけを持つ(machine_learningの層の外)。infrastructureの接続・ファイルを作り、
interface_adapterの読み手・置き場に渡し、手続き(usecase)を呼ぶ。決まりはentityにある。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from jra_tools.machine_learning.entity.category import DEFAULT_CATEGORY_SOURCES
from jra_tools.machine_learning.entity.dataset import InputDataset, TableDataset
from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.entity.input_view import DEFAULT_LABELS
from jra_tools.machine_learning.entity.labelutil import Labelref
from jra_tools.machine_learning.infrastructure.files import (
    Directory,
    read_yaml,
    write_yaml,
)
from jra_tools.machine_learning.infrastructure.saved_model import SavedModelFolder
from jra_tools.machine_learning.infrastructure.sql_session import (
    engine_from_env,
    transaction,
)
from jra_tools.machine_learning.interface_adapter.frame_store import (
    ParquetFrameStore,
)
from jra_tools.machine_learning.interface_adapter.spec_format import (
    feature_spec_to_dict,
    parse_feature_spec,
)
from jra_tools.machine_learning.interface_adapter.view_gateway import (
    PostgresViewReader,
)
from jra_tools.machine_learning.usecase import model_spec, view_dataset
from jra_tools.machine_learning.usecase.ifinish_model import (
    IEvaluableModel,
    IFinishModel,
)
from jra_tools.machine_learning.usecase.view_dataset import PROBABILITY_COLUMNS
from jra_tools.views import load_view_set, replace_views

__all__ = [
    "DEFAULT_CATEGORY_SOURCES",
    "DEFAULT_LABELS",
    "PROBABILITY_COLUMNS",
    "FeatureSpec",
    "InputDataset",
    "Labelref",
    "TableDataset",
    "apply_sandbox_views",
    "digest",
    "evaluate",
    "load_dataset",
    "load_feature_spec",
    "load_input",
    "load_model_feature_spec",
    "parse_feature_spec",
    "pin_spec",
    "predict_finish_probabilities",
    "read_frame",
    "save_feature_spec",
    "save_model_feature_spec",
]


def pin_spec(
    spec: FeatureSpec, category_sources: Sequence[str] = DEFAULT_CATEGORY_SOURCES
) -> FeatureSpec:
    """仕様を固定する(学習の前に使う).

    sourceの今の定義のハッシュと、カテゴリの列の「コード→番号」(category_sourcesの
    code_categoriesを合わせて項目名で引く)を写した仕様を返す。
    """
    with _reader() as reader:
        return view_dataset.pin_spec(reader, spec, category_sources)


def digest(source: str) -> str:
    """source(「スキーマ.名前」)の今の定義のハッシュ(依存する定義を含む)."""
    with _reader() as reader:
        return view_dataset.current_digest(reader, source)


def read_frame(
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: Optional[str] = DEFAULT_LABELS,
    cache_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """期間内(YYYYMMDD)の出走を、組み立てる前の1頭1行の表で読む.

    labels=Noneはラベルを読まない。cache_dirを渡すと、読んだ表をParquetで残し、
    次からはそれを使う(今の仕様・期間のものでなければ止める)。
    """
    with _reader() as reader:
        return view_dataset.read_view_frame(
            reader, spec, start, end, labels, _store(cache_dir)
        )


def load_dataset(
    spec: FeatureSpec,
    start: int,
    end: int,
    labels: str = DEFAULT_LABELS,
    cache_dir: Optional[Path | str] = None,
) -> TableDataset:
    """期間内(YYYYMMDD、両端を含む)の学習用のデータセットを作る. 1年ずつ読む."""
    with _reader() as reader:
        return view_dataset.load_view_dataset(
            reader, spec, start, end, labels, _store(cache_dir)
        )


def load_input(spec: FeatureSpec, ymd: int) -> InputDataset:
    """開催日(YYYYMMDD)の推論の入力を作る."""
    with _reader() as reader:
        return view_dataset.load_view_input(reader, spec, ymd)


def predict_finish_probabilities(
    spec: FeatureSpec, ymd: int, model: IFinishModel
) -> List[Tuple[str, float, float, float]]:
    """開催日の出走馬ごとの(racehorsekey, 1着, 2着, 3着の確率). 出走がなければ空."""
    with _reader() as reader:
        return view_dataset.predict_finish_probabilities(reader, spec, ymd, model)


def evaluate(  # noqa: PLR0913 (期間・ラベル・置き場はどれも使う側が選ぶ)
    spec: FeatureSpec,
    start: int,
    end: int,
    model: IEvaluableModel,
    labels: str = DEFAULT_LABELS,
    cache_dir: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """評価期間(締めた年に限る)の、出走馬ごとの1〜5着の確率と、loss・指標の表."""
    with _reader() as reader:
        return view_dataset.evaluate_period(
            reader, spec, start, end, model, labels, _store(cache_dir)
        )


def apply_sandbox_views(directory: Path | str) -> None:
    """試作のビューの定義(views.yamlと同じ形)を、ハッシュを記録せずに反映する.

    一覧から外したビュー・関数は消える。管理するスキーマ(sandboxなど)はDBにあること。

    Args:
        directory (Path | str): views.yamlのあるフォルダ(SQLは<スキーマ>/<ビュー名>.sql)

    """
    view_set = load_view_set(directory)
    with engine_from_env() as engine:
        replace_views(engine, view_set)


def load_feature_spec(path: Path | str) -> FeatureSpec:
    """YAMLファイルから仕様を読み込む."""
    return parse_feature_spec(read_yaml(path))


def save_feature_spec(spec: FeatureSpec, path: Path | str) -> None:
    """仕様をYAMLファイルに書き出す."""
    write_yaml(feature_spec_to_dict(spec), path)


def load_model_feature_spec(model_dir: Path | str) -> FeatureSpec:
    """モデルのフォルダ(SavedModel)の仕様を読み、モデルの入力の形と照合する."""
    return model_spec.load_model_feature_spec(SavedModelFolder(model_dir))


def save_model_feature_spec(
    spec: FeatureSpec, model_dir: Path | str, output_names: Sequence[str]
) -> Path:
    """学習に使った仕様に出力の名前(1着→5着、model.output_names)を写し、モデルのフォルダに書く."""
    return model_spec.save_model_feature_spec(
        SavedModelFolder(model_dir), spec, output_names
    )


@contextmanager
def _reader() -> Iterator[PostgresViewReader]:
    with engine_from_env() as engine:
        yield PostgresViewReader(transaction(engine))


def _store(cache_dir: Optional[Path | str]) -> Optional[ParquetFrameStore]:
    return None if cache_dir is None else ParquetFrameStore(Directory(cache_dir))
