"""着順の確率を出すモデルのインターフェース(usecase/view_dataset.pyが使う).

出力は1着〜5着の確率(各(レース, max_horses))。出力の名前の順(1着→5着)は、モデルを保存するときに
仕様のoutput_namesに写す(名前を決め打ちしない)。
"""

from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class IFinishModel(Protocol):
    """推論のモデル(Kerasのモデル・TFSMLayerなど)."""

    def predict(
        self, x: np.ndarray, verbose: Any = ...
    ) -> Mapping[str, Any] | Sequence[Any]:
        """入力(レース, max_horses, 次元)から、出力ごとの確率(名前→値、または出力の順のリスト)."""


class IEvaluableModel(IFinishModel, Protocol):
    """評価もできるモデル(Kerasのモデル)."""

    def evaluate(
        self, x: np.ndarray, y: Any, verbose: Any = ..., return_dict: bool = ...
    ) -> Mapping[str, float]:
        """入力と正解ラベル(出力の順)から、loss・指標(名前→値)."""
