"""ビューから読む入力・ラベルの契約と、読んだ表を学習・推論に使ってよいかの決まり.

- 入力のビュー(仕様のsource。例: features.entries_v1): 1頭1行で、レースの前に分かる列だけ。
  ENTRY_COLUMNSと、仕様の特徴量の列
- ラベルのビュー(例: features.labels_v1): 1頭1行の確定後の値。LABEL_COLUMNS

ビューの定義のハッシュ(view_digest)は、学習の前に仕様に書き、読むたびに今のハッシュと照合する。
違えば、学習時と違う入力になるので止める。
"""

from typing import Iterable, List, Sequence

import pandas as pd

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec

# 入力のビューに必要な列
ENTRY_COLUMNS = ("racehorsekey", "racekey", "race_date", "umaban")
# ラベルのビューに必要な列
LABEL_COLUMNS = ("racehorsekey", "finish_order", "win1_num", "win1_ret")
# 既定のラベルのビュー
DEFAULT_LABELS = "features.labels_v1"
# 読んだ表で、ラベルのビューに行があるかを表す列(ラベルのビューのracehorsekey)
LABEL_PRESENT = "_label"


def source_view(spec: FeatureSpec) -> str:
    """仕様が読む入力のビュー(「スキーマ.ビュー名」).

    Raises:
        ValueError: 仕様にsourceがない(jrdb-modelから読む仕様)場合

    """
    if spec.source is None:
        raise ValueError(
            f"仕様{spec.name}にsource(ビュー)がありません。jrdb-modelから読む仕様です"
        )
    return spec.source


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


def require_view_digest(spec: FeatureSpec) -> str:
    """仕様に書いた、学習時のビューのハッシュ.

    Raises:
        ValueError: 仕様にview_digestがない場合

    """
    if spec.view_digest is None:
        raise ValueError(
            f"仕様{spec.name}にview_digestがありません(pin_view_digestで書いてください)"
        )
    return spec.view_digest


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
        ValueError: view_digestがない、または定義が学習時と違う場合

    """
    pinned = require_view_digest(spec)
    if digest != pinned:
        raise ValueError(
            f"ビュー{view}の定義が、仕様{spec.name}の学習時と違います"
            f"(今{digest[:12]}、学習時{pinned[:12]})。"
            "仕様を書き直すか、ビューの版を上げてください"
        )


def check_values(
    frame: pd.DataFrame, view: str, features: Sequence[str], labels: bool
) -> pd.DataFrame:
    """読んだ表に値なしがないかを確かめ、学習・推論に渡す表を返す.

    入力の列に値なしがあれば止める(ORMの経路の「行がなければエラー」と同じ)。
    ラベルを読んだときは、成績(ラベルのビューの行)・単勝の払戻がない出走も止める。

    Args:
        frame (pd.DataFrame): 読んだ表(ラベルを読んだときはLABEL_PRESENTの列を含む)
        view (str): 入力のビュー(メッセージに出す)
        features (Sequence[str]): 入力の列
        labels (bool): ラベルを読んだか

    Returns:
        pd.DataFrame: LABEL_PRESENTの列を除いた表

    Raises:
        ValueError: 値なしがある場合。最初の出走馬のキーを出す

    """
    for column in features:
        _check_not_null(frame, column, f"{view}.{column}が値なしの出走があります")
    if not labels:
        return frame
    _check_not_null(frame, LABEL_PRESENT, "成績がありません")
    _check_not_null(frame, "win1_num", "払戻データがありません")
    _check_not_null(frame, "win1_ret", "払戻データがありません")
    return frame.drop(columns=LABEL_PRESENT)


def _check_not_null(frame: pd.DataFrame, column: str, message: str) -> None:
    missing = frame[column].isna()
    if missing.any():
        raise ValueError(f"{message}: {frame.loc[missing, 'racehorsekey'].iloc[0]}")
