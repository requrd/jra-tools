"""出走ごとの列(表)から、学習用のデータセット(入力・正解ラベル・払戻)を組み立てる.

列単位のndarrayから組み立てる。結果は、jra-tools 0.6.0までのORMの経路
(InputCreator・LabelCreator・PayoutCreator)と同じ(tests/machine_learning/data/の値で確かめる)。
"""

from typing import Dict, List

import numpy as np
from sklearn.preprocessing import StandardScaler

from jra_tools.machine_learning.entity.dataset import DatasetRows, TableDataset
from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    FeatureGroup,
    FeatureSpec,
    Transform,
)

# 正解ラベルを作る着順(1着〜5着)
LABEL_ORDERS = 5


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


def assemble_dataset(rows: DatasetRows, spec: FeatureSpec) -> TableDataset:
    """出走ごとの列から、仕様に従ってデータセットを組み立てる.

    Args:
        rows (DatasetRows): 出走ごとの列
        spec (FeatureSpec): 特徴量の仕様

    Returns:
        TableDataset: 学習用のデータセット

    """
    race_index, racekeys = _race_index(rows.racekeys)
    slots = _slots(rows, spec.max_horses)
    shape = (len(racekeys), spec.max_horses)
    return TableDataset(
        racekeys=racekeys,
        x=np.concatenate(
            [_group_matrix(rows, g, race_index, slots, shape) for g in spec.groups],
            axis=2,
        ),
        labels=_labels(rows, race_index, slots, shape),
        payout=_payout(rows, race_index, shape),
    )


def _race_index(racekeys: np.ndarray):
    """各行のレースの番号と、レースキーの一覧(行の並び順)."""
    if len(racekeys) == 0:
        return np.zeros(0, dtype=int), racekeys
    starts = np.r_[True, racekeys[1:] != racekeys[:-1]]
    race_index = np.cumsum(starts) - 1
    if len(np.unique(racekeys)) != race_index[-1] + 1:
        raise ValueError("行がレースごとにまとまって並んでいません")
    return race_index, racekeys[starts]


def _slots(rows: DatasetRows, max_horses: int) -> np.ndarray:
    nums = rows.nums.astype(int)
    invalid = (nums < 1) | (nums > max_horses)
    if invalid.any():
        raise ValueError(f"馬番が範囲外です: {rows.racehorsekeys[invalid][:5]}")
    return nums - 1


def _group_matrix(
    rows: DatasetRows,
    group: FeatureGroup,
    race_index: np.ndarray,
    slots: np.ndarray,
    shape,
) -> np.ndarray:
    matrix = np.zeros([*shape, group.size])
    offset = 0
    for feature in group.features:
        values = rows.features[feature.source]
        if feature.encode == Encode.ONEHOT:
            matrix[race_index, slots, offset + _onehot_index(feature, values)] = 1
        else:
            matrix[race_index, slots, offset] = _numbers(values)
        offset += feature.size
    if group.transform == Transform.STANDARDIZE_PER_RACE:
        matrix = standardize(matrix)
    return matrix


def _numbers(values: np.ndarray) -> np.ndarray:
    """数値にする. 文字列は0にする."""
    return np.array([0 if isinstance(v, str) else v for v in values], dtype=float)


def _onehot_index(feature: Feature, values: np.ndarray) -> np.ndarray:
    """one-hotで1にする位置. CategoryGetter.categoriesと同じく、負の位置は末尾から数える."""
    index = values.astype(int) - feature.base
    if ((index >= feature.size) | (index < -feature.size)).any():
        raise ValueError(f"{feature.name}の値がone-hotの範囲外です")
    return np.where(index < 0, index + feature.size, index)


def _labels(
    rows: DatasetRows, race_index: np.ndarray, slots: np.ndarray, shape
) -> List[np.ndarray]:
    labels = []
    for order in range(1, LABEL_ORDERS + 1):
        label = np.zeros(shape)
        hit = rows.order_of_arrival == order
        label[race_index[hit], slots[hit]] = 1
        labels.append(label)
    return labels


def _payout(rows: DatasetRows, race_index: np.ndarray, shape) -> np.ndarray:
    """単勝の払戻[レース, 馬]. 各レースの最初の行の値を使う."""
    payout = np.zeros(shape)
    first = np.r_[True, race_index[1:] != race_index[:-1]] if len(race_index) else []
    nums = rows.win1_num[first].astype(int)
    payout[race_index[first], nums - 1] = rows.win1_ret[first]
    return payout
