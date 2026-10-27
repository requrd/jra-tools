"""推論に必要なデータセットを作成する."""

from typing import List, Optional

import numpy as np
from jrdb_model import KaisaiData

from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    default_feature_spec,
)
from jra_tools.machine_learning.usecase.feature_matrix import create_input_matrix


class InputCreator:
    """KaisaiDataを元にndarrayを作成する."""

    def __init__(self, kaisais: List[KaisaiData], spec: Optional[FeatureSpec] = None):
        """開催データから、特徴量の仕様に従って入力データを作る.

        Args:
            kaisais (List[KaisaiData]): 開催データ一覧
            spec (Optional[FeatureSpec]): 特徴量の仕様. 省略時は現行の推論モデルの仕様

        """
        self.kaisais = kaisais
        self.spec = spec or default_feature_spec()
        self.x_data = create_input_data(kaisais, self.spec)

    def save(self, name: str) -> None:
        """入力データを永続化する.

        Args:
            name (str): ファイル名。拡張子も含めること。

        """
        save_input_data(self.x_data, name)


def save_input_data(x_data: np.ndarray, name: str) -> None:
    """入力データをx_{name}.dumpに保存する."""
    with open(f"x_{name}.dump", mode="wb") as f:
        x_data.dump(f)


def create_input_data(
    kaisais: List[KaisaiData], spec: Optional[FeatureSpec] = None
) -> np.ndarray:
    """開催一覧から推論モデル用入力データを作成する.

    Args:
        kaisais (List[KaisaiData]): 開催一覧.
        spec (Optional[FeatureSpec]): 特徴量の仕様. 省略時は現行の推論モデルの仕様

    Returns:
        np.ndarray: 推論モデルの入力データ[レース, 馬, 特徴量]

    """
    return create_input_matrix(kaisais, spec or default_feature_spec())


def save_input_data_from(kaisais: List[KaisaiData], name: str):
    """開催一覧から入力データのndarrayをローカルに保存する.

    Args:
        kaisais (List[KaisaiData]): 開催一覧
        name (str): ファイル名

    """
    save_input_data(create_input_data(kaisais), name)
