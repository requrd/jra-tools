"""トレーニングに利用する関数."""

from typing import List, Optional

import numpy as np
from jrdb_model import KaisaiData

from jra_tools.machine_learning.entity.feature_spec import (
    FeatureGroup,
    FeatureSpec,
    Transform,
    default_feature_spec,
)
from jra_tools.machine_learning.usecase.feature_matrix import (
    create_group_matrix,
    standardize,
)

__all__ = ["create_score_data_matrix", "standardize"]


def create_score_data_matrix(
    kaisais: List[KaisaiData], spec: Optional[FeatureSpec] = None
) -> np.ndarray:
    """開催データから、標準化する数値データの行列を作成する(標準化はしない).

    Args:
        kaisais (List[KaisaiData]): 開催データ
        spec (Optional[FeatureSpec]): 特徴量の仕様. 省略時は現行の推論モデルの仕様

    Returns:
        np.ndarray: 数値データ行列[レース, 馬, 特徴量]

    """
    spec = spec or default_feature_spec()
    return create_group_matrix(
        kaisais, _group_of(spec, Transform.STANDARDIZE_PER_RACE), spec.max_horses
    )


def _group_of(spec: FeatureSpec, transform: Transform) -> FeatureGroup:
    groups = [group for group in spec.groups if group.transform == transform]
    if len(groups) != 1:
        raise ValueError(
            f"仕様{spec.name}に変換{transform}のグループが1つではありません"
        )
    return groups[0]
