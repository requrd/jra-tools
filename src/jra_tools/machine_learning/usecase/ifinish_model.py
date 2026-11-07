"""着順の確率を出すモデルのインターフェース(usecase/view_dataset.pyが使う)."""

from typing import Any, Mapping, Protocol

import numpy as np

# モデルの出力のうち、1着・2着・3着の確率(PredictFactoryと同じ)
PROBABILITY_OUTPUTS = ("activation_4", "activation_5", "activation_6")


class IFinishModel(Protocol):
    """推論のモデル(Kerasのモデル・TFSMLayerなど)."""

    def predict(self, x: np.ndarray, verbose: Any = ...) -> Mapping[str, Any]:
        """入力(レース, max_horses, 次元)から、PROBABILITY_OUTPUTSの各名前に(レース, max_horses)の確率を返す."""
