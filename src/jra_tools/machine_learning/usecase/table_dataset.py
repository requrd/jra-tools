"""出走ごとの列(表)から、学習用のデータセット(入力・正解ラベル・払戻)を組み立てる.

ORMのオブジェクトを作らずに、列単位のndarrayから組み立てる。
結果は、ORMの開催データから作る場合(InputCreator・LabelCreator・PayoutCreator)と同じ。
"""

import pickle
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    FeatureGroup,
    FeatureSpec,
    Transform,
)
from jra_tools.machine_learning.usecase.feature_matrix import standardize

# 正解ラベルを作る着順(1着〜5着)
LABEL_ORDERS = 5


@dataclass(frozen=True)
class InputRows:
    """出走1頭を1行とする入力の列. 行はレースごとにまとまって並んでいること.

    featuresは仕様のsourceごとの値。
    """

    racehorsekeys: np.ndarray
    racekeys: np.ndarray
    nums: np.ndarray
    features: Dict[str, np.ndarray]


@dataclass(frozen=True)
class DatasetRows(InputRows):
    """入力の列に、正解ラベル・払戻の列を足したもの.

    払戻(win1_num・win1_ret)はレースの値で、同じレースの行には同じ値が入る。
    """

    order_of_arrival: np.ndarray
    win1_num: np.ndarray
    win1_ret: np.ndarray


@dataclass(frozen=True)
class InputDataset:
    """推論の入力. racekeysはxの1次元目の各レース."""

    racekeys: np.ndarray
    x: np.ndarray


@dataclass(frozen=True)
class TableDataset:
    """学習用のデータセット. racekeysはx・labels・payoutの1次元目の各レース."""

    racekeys: np.ndarray
    x: np.ndarray
    labels: List[np.ndarray]
    payout: np.ndarray

    def save(self, name: str) -> None:
        """InputCreator・LabelCreator・PayoutCreatorと同じファイルに保存する.

        x_{name}.dump(入力)、{name}.dump(正解ラベル)、payout_{name}.dump(払戻)と、
        各レースのレースキーをracekeys_{name}.npyに保存する。

        Args:
            name (str): ファイル名(拡張子を除く)

        """
        with open(f"x_{name}.dump", mode="wb") as f:
            self.x.dump(f)
        with open(f"{name}.dump", mode="wb") as f:
            pickle.dump(self.labels, f)
        with open(f"payout_{name}.dump", "wb") as f:
            self.payout.dump(f)
        np.save(f"racekeys_{name}.npy", self.racekeys)


def assemble_dataset(rows: DatasetRows, spec: FeatureSpec) -> TableDataset:
    """出走ごとの列から、仕様に従ってデータセットを組み立てる.

    Args:
        rows (DatasetRows): 出走ごとの列
        spec (FeatureSpec): 特徴量の仕様

    Returns:
        TableDataset: 学習用のデータセット

    """
    inputs = assemble_input(rows, spec)
    race_index, _ = _race_index(rows.racekeys)
    slots = _slots(rows, spec.max_horses)
    shape = (len(inputs.racekeys), spec.max_horses)
    return TableDataset(
        racekeys=inputs.racekeys,
        x=inputs.x,
        labels=_labels(rows, race_index, slots, shape),
        payout=_payout(rows, race_index, shape),
    )


def assemble_input(rows: InputRows, spec: FeatureSpec) -> InputDataset:
    """出走ごとの列から、仕様に従って入力だけを組み立てる(推論用).

    Args:
        rows (InputRows): 出走ごとの入力の列
        spec (FeatureSpec): 特徴量の仕様

    Returns:
        InputDataset: 入力と、各レースのレースキー

    """
    race_index, racekeys = _race_index(rows.racekeys)
    slots = _slots(rows, spec.max_horses)
    shape = (len(racekeys), spec.max_horses)
    return InputDataset(
        racekeys=racekeys,
        x=np.concatenate(
            [_group_matrix(rows, g, race_index, slots, shape) for g in spec.groups],
            axis=2,
        ),
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


def _slots(rows: InputRows, max_horses: int) -> np.ndarray:
    nums = rows.nums.astype(int)
    invalid = (nums < 1) | (nums > max_horses)
    if invalid.any():
        raise ValueError(f"馬番が範囲外です: {rows.racehorsekeys[invalid][:5]}")
    return nums - 1


def _group_matrix(
    rows: InputRows,
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
    """数値にする. 文字列は0にする(ORMの経路の_encodeと同じ)."""
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
