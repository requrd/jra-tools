"""推論モデルのフォルダのインターフェース(usecase/model_spec.pyが使う)."""

from pathlib import Path
from typing import Any, List, Protocol, Sequence

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec


class IModelFolder(Protocol):
    """推論モデルのフォルダ(モデルと、学習に使った仕様を置く)."""

    def input_shape(self) -> List[int]:
        """モデルの入力の形. 先頭はバッチの次元."""

    def output_names(self) -> List[str]:
        """推論で呼ぶモデルの出力の名前(predictが返す辞書のキー)."""

    def write_model(self, model: Any, output_names: Sequence[str]) -> None:
        """モデルを、出力の名前をキーにして推論できる形でフォルダに書く."""

    def read_spec(self) -> FeatureSpec:
        """フォルダの仕様を読む. なければFileNotFoundError."""

    def write_spec(self, spec: FeatureSpec) -> Path:
        """仕様をフォルダに書き、書いたファイルを返す."""
