"""ビューから読む入力・ラベルの契約と、読んだ表を学習・推論に使ってよいかの決まり.

- 入力のビュー(仕様のsource。例: features.entries_v1): 1頭1行で、レースの前に分かる列だけ。
  ENTRY_COLUMNSと、仕様の特徴量の列
- ラベルのビュー(例: features.labels_v1): 1頭1行の確定後の値。LABEL_COLUMNS

ビューの定義のハッシュ(view_digest)は、学習の前に仕様に書き、読むたびに今のハッシュと照合する。
違えば、学習時と違う入力になるので止める。
"""

from typing import Iterable, List, Sequence

import pandas as pd

from jra_tools.machine_learning.entity.feature_spec import Encode, FeatureSpec

# 入力のビューに必要な列
ENTRY_COLUMNS = ("racehorsekey", "racekey", "race_date", "umaban")
# ラベルのビューに必要な列
LABEL_COLUMNS = ("racehorsekey", "finish_order")
# 既定のラベルのビュー
DEFAULT_LABELS = "features.labels_v1"
# 読んだ表で、ラベルのビューに行があるかを表す列(ラベルのビューのracehorsekey)
LABEL_PRESENT = "_label"


def input_columns(spec: FeatureSpec) -> List[str]:
    """仕様が読む列(重複を除き、最初に出た順).

    Raises:
        ValueError: 確定後の値(ラベルの列)を入力に使う場合

    """
    columns = list(dict.fromkeys(feature.source for feature in spec.features))
    labels = [c for c in columns if c in (LABEL_PRESENT, *LABEL_COLUMNS[1:])]
    if labels:
        raise ValueError(f"ラベルの列は入力に使えません: {labels}")
    return columns


def category_columns(spec: FeatureSpec) -> List[str]:
    """カテゴリの列(NULLを許す列)."""
    return list(
        dict.fromkeys(f.source for f in spec.features if f.encode == Encode.CATEGORY)
    )


def check_columns(view: str, existing: Iterable[str], required: Sequence[str]) -> None:
    """ビューに必要な列があるかを確かめる.

    Raises:
        ValueError: 列が足りない場合

    """
    present = set(existing)
    missing = [column for column in required if column not in present]
    if missing:
        raise ValueError(f"ビュー{view}に列がありません: {missing}")


def check_digest(spec: FeatureSpec, view: str, digest: str) -> None:
    """今のビューのハッシュが、学習時(仕様のview_digest)と同じかを確かめる.

    Raises:
        ValueError: 定義が学習時と違う場合

    """
    pinned = spec.view_digest or ""
    if digest != pinned:
        raise ValueError(
            f"ビュー{view}の定義が、仕様{spec.name}の学習時と違います"
            f"(今{digest[:12]}、学習時{pinned[:12]})。"
            "仕様を固定し直すか、ビューの版を上げてください"
        )


def check_values(frame: pd.DataFrame, spec: FeatureSpec, labels: bool) -> pd.DataFrame:
    """読んだ表に値なしがないかを確かめ、学習・推論に渡す表を返す.

    数値の列に値なしがあれば止める。カテゴリの列の値なし(記載外のコード)は例外の列にするので許す。
    ラベルを読んだときは、成績(ラベルのビューの行)がない出走も止める。

    Args:
        frame (pd.DataFrame): 読んだ表(ラベルを読んだときはLABEL_PRESENTの列を含む)
        spec (FeatureSpec): 仕様
        labels (bool): ラベルを読んだか

    Returns:
        pd.DataFrame: LABEL_PRESENTの列を除いた表

    Raises:
        ValueError: 値なしがある場合。最初の出走馬のキーを出す

    """
    nullable = set(category_columns(spec))
    for column in input_columns(spec):
        if column not in nullable:
            _check_not_null(
                frame, column, f"{spec.source}.{column}が値なしの出走があります"
            )
    if not labels:
        return frame
    _check_not_null(frame, LABEL_PRESENT, "成績がありません")
    return frame.drop(columns=LABEL_PRESENT)


def _check_not_null(frame: pd.DataFrame, column: str, message: str) -> None:
    missing = frame[column].isna()
    if missing.any():
        raise ValueError(f"{message}: {frame.loc[missing, 'racehorsekey'].iloc[0]}")
