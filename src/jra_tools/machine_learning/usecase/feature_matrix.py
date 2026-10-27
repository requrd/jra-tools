"""特徴量の仕様に従って、開催データから入力データの行列を作る."""

from typing import Any, Callable, Dict, List

import numpy as np
from jrdb_model import BangumiData, KaisaiData, RacehorseData
from sklearn.preprocessing import StandardScaler

from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    FeatureGroup,
    FeatureSpec,
    Transform,
)
from jra_tools.machine_learning.entity.jrdbdummies import CategoryGetter

RowOf = Callable[[KaisaiData, BangumiData, RacehorseData], Any]

# sourceのテーブル名から、出走1頭あたりの行をたどる方法。
# 成績(seiseki・seisekirace)・払戻(returninfo)は確定後の値なので、入力には使えない。
# 馬基本(horsebase)は今の状態の値で、レースの時点の値ではないので入れていない。
_ROWS: Dict[str, RowOf] = {
    "kaisai": lambda kaisai, race, horse: kaisai,
    "bangumi": lambda kaisai, race, horse: race,
    "racehorse": lambda kaisai, race, horse: horse,
    "train_analysis": lambda kaisai, race, horse: horse.trainanalysis,
    "train_oikiri": lambda kaisai, race, horse: horse.trainoikiri,
    "calculated_score": lambda kaisai, race, horse: horse.calculated_score,
}


def create_input_matrix(kaisais: List[KaisaiData], spec: FeatureSpec) -> np.ndarray:
    """仕様のグループごとに行列を作り、変換して特徴量の次元方向につなぐ.

    Args:
        kaisais (List[KaisaiData]): 開催データ一覧
        spec (FeatureSpec): 特徴量の仕様

    Returns:
        np.ndarray: 入力データ[レース, 馬, 特徴量]

    """
    matrices = []
    for group in spec.groups:
        matrix = create_group_matrix(kaisais, group, spec.max_horses)
        if group.transform == Transform.STANDARDIZE_PER_RACE:
            matrix = standardize(matrix)
        matrices.append(matrix)
    return np.concatenate(matrices, axis=2)


def standardize(matrix: np.ndarray) -> np.ndarray:
    """レースごとに標準化する. 空き枠の0の行も含めて計算する.

    Args:
        matrix (np.ndarray): 数値データ行列[レース, 馬, 特徴量]

    Returns:
        np.ndarray: 標準化済み行列(引数の行列を書き換えて返す)

    """
    for i, element in enumerate(matrix):
        matrix[i] = StandardScaler().fit_transform(element)
    return matrix


def create_group_matrix(
    kaisais: List[KaisaiData], group: FeatureGroup, max_horses: int
) -> np.ndarray:
    """1つのグループの特徴量を、馬番-1の位置に並べた行列を作る(変換はしない).

    Args:
        kaisais (List[KaisaiData]): 開催データ一覧
        group (FeatureGroup): 特徴量のグループ
        max_horses (int): 1レースの最大頭数

    Returns:
        np.ndarray: 行列[レース, 馬, グループの次元数]。空き枠は0

    """
    _validate_sources(group)
    races = [(kaisai, race) for kaisai in kaisais for race in kaisai.races]
    matrix = np.zeros([len(races), max_horses, group.size])
    for race_num, (kaisai, race) in enumerate(races):
        for horse in race.racehorses:
            matrix[race_num, horse.num - 1] = np.concatenate(
                [
                    _encode(feature, _value(feature, kaisai, race, horse))
                    for feature in group.features
                ]
            )
    return matrix


def _validate_sources(group: FeatureGroup) -> None:
    unknown = [f.source for f in group.features if f.table not in _ROWS]
    if unknown:
        raise ValueError(
            f"入力に使えないテーブルです: {unknown}(使えるのは{list(_ROWS)})"
        )


def _value(
    feature: Feature, kaisai: KaisaiData, race: BangumiData, horse: RacehorseData
) -> Any:
    row = _ROWS[feature.table](kaisai, race, horse)
    if row is None:
        raise ValueError(f"{feature.table} がありません: {horse.racehorsekey}")
    return getattr(row, feature.column)


def _encode(feature: Feature, value: Any) -> np.ndarray:
    if feature.encode == Encode.ONEHOT:
        return CategoryGetter().categories(feature.size, value - feature.base + 1)
    return np.array([0 if isinstance(value, str) else value])
