"""学習・推論のデータセットの形."""

from dataclasses import dataclass
from typing import List

import numpy as np

# 正解ラベルを作る着順(1着〜5着)
LABEL_ORDERS = 5


@dataclass(frozen=True)
class TableDataset:
    """学習用のデータセット. racekeysはx・labelsの1次元目の各レース.

    x: 入力(レース, max_horses, 次元)。labels: 1着〜5着の正解ラベル(各(レース, max_horses))。
    """

    racekeys: np.ndarray
    x: np.ndarray
    labels: List[np.ndarray]


@dataclass(frozen=True)
class InputDataset:
    """推論の入力. racekeysはxの1次元目の各レース."""

    racekeys: np.ndarray
    x: np.ndarray
