"""1頭1行のDataFrameから、学習用のデータセット・推論の入力を組み立てる.

featuresのビューから読んだ表を、仕様に従って(レース, max_horses, 次元)のテンソルにする。
各レースを1〜max_horsesの枠に並べ(空き枠は0)、グループごとに変換してから最後にndarrayにする。

処理の場所(どの変換をするかはコードではなく仕様で決まる):
- 枠への並べ直し: to_slots
- 特徴量ごとの列への変換: _group_frame。仕様のencodeがcategoryなら_category、それ以外は_number
  - _number: 数値にする。文字列は0
  - _category: 仕様の控え(コード→番号)でone-hot。控えにないコード・NULLは例外の列。空き枠はすべて0
- レースごとの標準化(数値系): _group_frameの中。グループのtransformが
  standardize_per_raceのとき、列ごとに、レース内の全枠(空き枠の0を含む)の平均・標準偏差で
  標準化する
- テンソルへの変換: frame_to_inputの最後のreshape

DataFrameの列:
- racehorsekey・racekey・umabanと、仕様の各特徴量のsourceの列。行はレースごとにまとまって並んでいること
- 学習用はさらに finish_order(着順)
"""

import numpy as np
import pandas as pd

from jra_tools.machine_learning.entity.category import positions
from jra_tools.machine_learning.entity.dataset import (
    LABEL_ORDERS,
    InputDataset,
    TableDataset,
)
from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    FeatureGroup,
    FeatureSpec,
    Transform,
)

# StandardScalerと同じく、標準偏差がこれより小さい列は割らない(0にそろえるだけ)
_ZERO_SCALE = 10 * np.finfo(float).eps


def frame_to_input(frame: pd.DataFrame, spec: FeatureSpec) -> InputDataset:
    """表から、仕様に従って入力を組み立てる(推論用).

    Args:
        frame (pd.DataFrame): 1頭1行の表
        spec (FeatureSpec): 固定した仕様

    Returns:
        InputDataset: 入力と、各レースのレースキー(表に出てきた順)

    """
    spec.check_pinned()
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
        frame (pd.DataFrame): 1頭1行の表(着順の列を含む)
        spec (FeatureSpec): 固定した仕様

    Returns:
        TableDataset: 入力・正解ラベルと、各レースのレースキー

    """
    inputs = frame_to_input(frame, spec)
    slots = to_slots(frame, spec.max_horses)
    shape = (len(inputs.racekeys), spec.max_horses)
    labels = [
        (slots["finish_order"] == order).to_numpy(dtype=float).reshape(shape)
        for order in range(1, LABEL_ORDERS + 1)
    ]
    return TableDataset(racekeys=inputs.racekeys, x=inputs.x, labels=labels)


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
    occupied = slots["racehorsekey"].notna()
    columns = [
        _category(slots[f.source], occupied, f)
        if f.encode == Encode.CATEGORY
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
    """数値にする. 文字列は0にする. 空き枠は値なしのまま."""
    is_text = values.map(lambda v: isinstance(v, str))
    return values.mask(is_text, 0).astype(float)


def _category(values: pd.Series, occupied: pd.Series, feature: Feature) -> pd.DataFrame:
    """one-hotの列(最後の列が例外の列). 空き枠はすべて0."""
    categories = feature.categories or {}
    matrix = np.zeros((len(values), feature.size))
    rows = np.flatnonzero(occupied.to_numpy())
    matrix[rows, positions(values.iloc[rows], categories)] = 1.0
    return pd.DataFrame(
        matrix,
        index=values.index,
        columns=[f"{feature.name}_{i}" for i in range(feature.size)],
    )
