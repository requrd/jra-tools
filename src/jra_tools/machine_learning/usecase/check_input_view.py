"""入力のビュー(試作のsandboxのビューを含む)が、学習・推論に使える形かを確かめる.

入力のビューの契約(entity/input_view.py)のうち、データで確かめられるものを流す。
どれも返したものがあれば失敗(dbtの個別のテストと同じ)。

1. キーの列と、確かめる特徴量の列がある(足りなければ、ほかは確かめない)
2. 出走馬1頭につき1行(unique)
3. model層の出走馬がすべて出る(covers_entries。内部結合で黙って消えていない)
4. 特徴量の列に値なしがない(not_null:<列>。ビューから読む経路は値なしで止まる)

定義のハッシュ(view_digest)は照合しない。試作は学習の前に確かめるもので、まだ仕様がないため。
"""

from typing import List, Optional, Sequence

from jra_tools.machine_learning.entity.input_view import (
    InputViewProblem,
    feature_columns,
    keys_problem,
    missing_columns,
)
from jra_tools.machine_learning.usecase.iinput_view_query import IInputViewQuery


def check_input_view(
    query: IInputViewQuery, features: Optional[Sequence[str]] = None
) -> List[InputViewProblem]:
    """入力のビューを確かめる.

    Args:
        query (IInputViewQuery): 確かめるビューと期間の問い合わせ
        features (Optional[Sequence[str]]): 値なしを確かめる特徴量の列。
            省略時は、キーの列以外のすべての列

    Returns:
        List[InputViewProblem]: 見つかったもの。すべて通れば空

    Raises:
        ValueError: ラベルの列を特徴量に指定した場合

    """
    existing = query.columns()
    columns = feature_columns(existing, features)
    missing = missing_columns(existing, columns)
    if missing:
        return [missing]
    problems = [
        keys_problem("unique", query.duplicated_keys()),
        keys_problem("covers_entries", query.uncovered_keys()),
        *(keys_problem(f"not_null:{c}", query.null_keys(c)) for c in columns),
    ]
    return [p for p in problems if p is not None]
