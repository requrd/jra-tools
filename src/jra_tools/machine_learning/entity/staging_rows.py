"""jrdb-modelの表(staging)から読んだ出走の行を、学習に使ってよいかの決まり.

出走には成績・払戻があること。仕様が使う表(調教分析・追切・計算スコア)の行も、
出走ごとにあること。ないものがあれば、その表と最初の出走のキーを出して止める。
"""

from typing import Mapping, Sequence

# 表ごとの、行がないときのメッセージ(なければ「<表> がありません」)
_MISSING_MESSAGES = {
    "seiseki": "成績がありません",
    "returninfo": "払戻データがありません",
}


def check_joined_rows(missing: Mapping[str, Sequence[str]]) -> None:
    """表ごとの、行のない出走(またはレース)のキーを受け取り、あれば止める.

    Args:
        missing (Mapping[str, Sequence[str]]): 表の名前 → 行のない出走・レースのキー(読んだ順)

    Raises:
        ValueError: 行のない出走がある場合(最初に見つかった表の、最初のキー)

    """
    for table, keys in missing.items():
        if keys:
            message = _MISSING_MESSAGES.get(table, f"{table} がありません")
            raise ValueError(f"{message}: {keys[0]}")
