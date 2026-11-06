"""入力のビュー・ラベルのビュー(分析用のPostgres)の列の契約と、入力のビューの確認の決まり.

- 入力のビュー(仕様のsource。例: features.entries_v1、sandboxの試作): 1頭1行で、レースの前に
  分かる列だけ。キーの列(ENTRY_COLUMNS)と特徴量の列
- ラベルのビュー(例: features.labels_v1): 1頭1行の確定後の値(LABEL_COLUMNS)
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# 入力のビューに必要な列
ENTRY_COLUMNS = ("racehorsekey", "racekey", "race_date", "umaban")
# ラベルのビューに必要な列
LABEL_COLUMNS = ("racehorsekey", "finish_order", "win1_num", "win1_ret")
# 失敗のときに残す出走馬の例の数
EXAMPLES = 5


@dataclass(frozen=True)
class InputViewProblem:
    """入力のビューを確かめて見つかったもの."""

    check: str  # 確かめたこと(columns・unique・covers_entries・not_null:<列>)
    rows: int  # 見つかった件数
    examples: Tuple[str, ...]  # 例(列の名前、または出走馬のキー)


def feature_columns(
    existing: Sequence[str], features: Optional[Sequence[str]]
) -> List[str]:
    """確かめる特徴量の列. 省略時はキーの列以外のすべての列.

    Raises:
        ValueError: ラベルの列(確定後の値)を特徴量にした場合

    """
    columns = (
        list(features)
        if features is not None
        else [c for c in existing if c not in ENTRY_COLUMNS]
    )
    labels = [c for c in columns if c in LABEL_COLUMNS[1:]]
    if labels:
        raise ValueError(f"ラベルの列は入力に使えません: {labels}")
    return columns


def missing_columns(
    existing: Sequence[str], features: Sequence[str]
) -> Optional[InputViewProblem]:
    """キーの列・特徴量の列のうち、ビューにないもの. すべてあればNone."""
    missing = [c for c in (*ENTRY_COLUMNS, *features) if c not in existing]
    return (
        InputViewProblem("columns", len(missing), tuple(missing)) if missing else None
    )


def keys_problem(check: str, keys: Sequence[str]) -> Optional[InputViewProblem]:
    """出走馬のキーの一覧を、見つかったものにする. 空ならNone."""
    if not keys:
        return None
    return InputViewProblem(check, len(keys), tuple(sorted(keys)[:EXAMPLES]))
