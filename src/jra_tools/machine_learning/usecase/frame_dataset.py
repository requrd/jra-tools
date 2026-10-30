"""1頭1行のDataFrameから、学習用のデータセット・推論の入力を組み立てる.

featuresのビューから読んだ表を、仕様に従って(レース, max_horses, 次元)のテンソルにする。
各レースを1〜max_horsesの枠に並べ(空き枠は0)、グループごとに変換してから最後にndarrayにする。
結果は、今の経路(table_dataset.assemble_dataset)と浮動小数点の誤差の範囲で同じ。

処理の場所(どの変換をするかはコードではなく仕様で決まる):
- 枠への並べ直し: to_slots
- 特徴量ごとの列への変換: _group_frame。仕様のencodeがonehotなら_onehot、それ以外は_number
  - _number: 数値にする。文字列は0
  - _onehot: one-hot(カテゴリ系。例: 枠番)。値がbaseのとき先頭が1。空き枠はすべて0
- レースごとの標準化(数値系): _group_frameの中。グループのtransformが
  standardize_per_raceのとき、列ごとに、レース内の全枠(空き枠の0を含む)の平均・標準偏差で
  標準化する
- テンソルへの変換: frame_to_inputの最後のreshape

DataFrameの列:
- racekey・umabanと、仕様の各特徴量のsourceの列。行はレースごとにまとまって並んでいること
- 学習用はさらに finish_order(着順)・win1_num・win1_ret(単勝の払戻。レースの値)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    FeatureGroup,
    FeatureSpec,
    Transform,
)
from jra_tools.machine_learning.usecase.table_dataset import LABEL_ORDERS, TableDataset

# StandardScalerと同じく、標準偏差がこれより小さい列は割らない(0にそろえるだけ)
_ZERO_SCALE = 10 * np.finfo(float).eps


@dataclass(frozen=True)
class InputDataset:
    """推論の入力. racekeysはxの1次元目の各レース."""

    racekeys: np.ndarray
    x: np.ndarray


def frame_to_input(frame: pd.DataFrame, spec: FeatureSpec) -> InputDataset:
    """表から、仕様に従って入力を組み立てる(推論用).

    Args:
        frame (pd.DataFrame): 1頭1行の表
        spec (FeatureSpec): 特徴量の仕様

    Returns:
        InputDataset: 入力と、各レースのレースキー(表に出てきた順)

    """
    slots = to_slots(frame, spec.max_horses)
    racekeys = slots.index.unique("racekey").to_numpy()
    features = pd.concat([_group_frame(slots, g) for g in spec.groups], axis=1)
    x = features.to_numpy(dtype=float).reshape(
        len(racekeys), spec.max_horses, spec.size
    )
    return InputDataset(racekeys=racekeys, x=x)


def frame_to_dataset(frame: pd.DataFrame, spec: FeatureSpec) -> TableDataset:
    """表から、仕様に従って学習用のデータセットを組み立てる.

    Args:
        frame (pd.DataFrame): 1頭1行の表(着順・払戻の列を含む)
        spec (FeatureSpec): 特徴量の仕様

    Returns:
        TableDataset: 入力・正解ラベル・払戻と、各レースのレースキー

    """
    inputs = frame_to_input(frame, spec)
    slots = to_slots(frame, spec.max_horses)
    shape = (len(inputs.racekeys), spec.max_horses)
    labels = [
        (slots["finish_order"] == order).to_numpy(dtype=float).reshape(shape)
        for order in range(1, LABEL_ORDERS + 1)
    ]
    # 払戻はレースの値。各レースの最初の行の値を、単勝の馬番の枠に置く
    win = frame.groupby("racekey", sort=False)[["win1_num", "win1_ret"]].first()
    race_of_slot = slots.index.get_level_values("racekey")
    umaban = slots.index.get_level_values("umaban").to_numpy()
    hit = umaban == win["win1_num"].reindex(race_of_slot).to_numpy()
    payout = np.where(hit, win["win1_ret"].reindex(race_of_slot).to_numpy(), 0)
    return TableDataset(
        racekeys=inputs.racekeys,
        x=inputs.x,
        labels=labels,
        payout=payout.astype(float).reshape(shape),
    )


def to_slots(frame: pd.DataFrame, max_horses: int) -> pd.DataFrame:
    """各レースを1〜max_horsesの枠に並べた表. 空き枠の行は値なし(NaN).

    Args:
        frame (pd.DataFrame): 1頭1行の表
        max_horses (int): 1レースの枠の数

    Raises:
        ValueError: 馬番が範囲外・重複している、またはレースがまとまって並んでいない場合

    Returns:
        pd.DataFrame: (racekey, umaban)をインデックスにした、(レース数 * max_horses)行の表

    """
    out_of_range = ~frame["umaban"].between(1, max_horses)
    if out_of_range.any():
        keys = frame.loc[out_of_range, "racehorsekey"].head().tolist()
        raise ValueError(f"馬番が範囲外です: {keys}")
    duplicated = frame.duplicated(["racekey", "umaban"])
    if duplicated.any():
        keys = frame.loc[duplicated, "racehorsekey"].head().tolist()
        raise ValueError(f"同じレースに同じ馬番があります: {keys}")
    racekeys = frame["racekey"]
    if (racekeys != racekeys.shift()).sum() != racekeys.nunique():
        raise ValueError("行がレースごとにまとまって並んでいません")
    index = pd.MultiIndex.from_product(
        [racekeys.unique(), range(1, max_horses + 1)], names=["racekey", "umaban"]
    )
    return frame.astype({"umaban": int}).set_index(["racekey", "umaban"]).reindex(index)


def _group_frame(slots: pd.DataFrame, group: FeatureGroup) -> pd.DataFrame:
    """グループの特徴量を次元ごとの列にし、変換する. 空き枠は0."""
    columns = [
        _onehot(slots[f.source], f)
        if f.encode == Encode.ONEHOT
        else _number(slots[f.source]).rename(f.name).to_frame()
        for f in group.features
    ]
    matrix = pd.concat(columns, axis=1).fillna(0.0)
    if group.transform == Transform.STANDARDIZE_PER_RACE:
        by_race = matrix.groupby(level="racekey", sort=False)
        std = by_race.transform("std", ddof=0)
        matrix = (matrix - by_race.transform("mean")) / std.where(std >= _ZERO_SCALE, 1)
    return matrix


def _number(values: pd.Series) -> pd.Series:
    """数値にする. 文字列は0にする(今の経路と同じ). 空き枠は値なしのまま."""
    is_text = values.map(lambda v: isinstance(v, str))
    return values.mask(is_text, 0).astype(float)


def _onehot(values: pd.Series, feature: Feature) -> pd.DataFrame:
    """one-hotの列. 値がbaseのとき先頭が1. 負の位置は末尾から数える. 空き枠はすべて0."""
    index = values - feature.base
    if ((index >= feature.size) | (index < -feature.size)).any():
        raise ValueError(f"{feature.name}の値がone-hotの範囲外です")
    position = pd.Categorical(index % feature.size, categories=range(feature.size))
    dummies = pd.get_dummies(position, dtype=float)
    dummies.index = values.index
    dummies.columns = pd.Index([f"{feature.name}_{i}" for i in range(feature.size)])
    return dummies
