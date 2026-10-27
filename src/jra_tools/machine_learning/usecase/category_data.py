"""カテゴリ変数を取得する."""

from typing import List, Optional

import numpy as np
from jrdb_model import KaisaiData

from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    Transform,
    default_feature_spec,
)
from jra_tools.machine_learning.usecase.feature_matrix import create_group_matrix


def get_category_data(
    kaisais: List[KaisaiData], spec: Optional[FeatureSpec] = None
) -> np.ndarray:
    """開催データから、変換しないグループ(カテゴリデータ)の行列を取得する.

    Args:
        kaisais (List[KaisaiData]): 開催データ
        spec (Optional[FeatureSpec]): 特徴量の仕様. 省略時は現行の推論モデルの仕様

    Returns:
        np.ndarray: カテゴリデータ行列[レース, 馬, 特徴量]

    """
    spec = spec or default_feature_spec()
    groups = [group for group in spec.groups if group.transform == Transform.NONE]
    return np.concatenate(
        [create_group_matrix(kaisais, group, spec.max_horses) for group in groups],
        axis=2,
    )
