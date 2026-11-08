"""学習・推論のデータセットの形."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class DatasetRows:
    """出走1頭を1行とする列. 行はレースごとにまとまって並んでいること.

    featuresは仕様のsourceごとの値。払戻(win1_num・win1_ret)はレースの値で、
    同じレースの行には同じ値が入る。
    """

    racehorsekeys: np.ndarray
    racekeys: np.ndarray
    nums: np.ndarray
    features: Dict[str, np.ndarray]
    order_of_arrival: np.ndarray
    win1_num: np.ndarray
    win1_ret: np.ndarray


@dataclass(frozen=True)
class TableDataset:
    """学習用のデータセット. racekeysはx・labels・payoutの1次元目の各レース.

    x: 入力(レース, max_horses, 次元)。labels: 1着〜5着の正解ラベル(各(レース, max_horses))。
    payout: 単勝の払戻(レース, max_horses)。
    """

    racekeys: np.ndarray
    x: np.ndarray
    labels: List[np.ndarray]
    payout: np.ndarray


@dataclass(frozen=True)
class InputDataset:
    """推論の入力. racekeysはxの1次元目の各レース."""

    racekeys: np.ndarray
    x: np.ndarray
