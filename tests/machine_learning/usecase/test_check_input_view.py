"""入力のビューの確認の手続きのテスト(DBなし)."""

from dataclasses import dataclass, field
from typing import Dict, List

import pytest

from jra_tools.machine_learning.entity.input_view import InputViewProblem
from jra_tools.machine_learning.usecase.check_input_view import check_input_view
from jra_tools.machine_learning.usecase.iinput_view_query import IInputViewQuery

KEYS = ["racehorsekey", "racekey", "race_date", "umaban"]


@dataclass
class FakeQuery:
    """問い合わせの結果を決め打ちにしたもの. 呼ばれた問い合わせを記録する."""

    view_columns: List[str]
    duplicated: List[str] = field(default_factory=list)
    uncovered: List[str] = field(default_factory=list)
    nulls: Dict[str, List[str]] = field(default_factory=dict)
    calls: List[str] = field(default_factory=list)

    def columns(self) -> List[str]:
        """ビューの列."""
        self.calls.append("columns")
        return self.view_columns

    def duplicated_keys(self) -> List[str]:
        """重複する出走馬."""
        self.calls.append("duplicated")
        return self.duplicated

    def uncovered_keys(self) -> List[str]:
        """出ない出走馬."""
        self.calls.append("uncovered")
        return self.uncovered

    def null_keys(self, column: str) -> List[str]:
        """値なしの出走馬."""
        self.calls.append(f"null:{column}")
        return self.nulls.get(column, [])


def test_fake_is_a_query():
    """偽の問い合わせはインターフェースを満たす."""
    assert isinstance(FakeQuery([]), IInputViewQuery)


def test_passes_when_nothing_is_found():
    """何も見つからなければ空. キーの列以外のすべての列の値なしを見る."""
    query = FakeQuery([*KEYS, "idm", "waku"])
    assert check_input_view(query) == []
    assert query.calls == [
        "columns",
        "duplicated",
        "uncovered",
        "null:idm",
        "null:waku",
    ]


def test_reports_each_problem_with_sorted_examples():
    """見つかったものごとに件数と、並べた先頭5件の例を返す."""
    query = FakeQuery(
        [*KEYS, "idm"],
        duplicated=["b", "a"],
        uncovered=[str(i) for i in range(7, 0, -1)],
        nulls={"idm": ["x"]},
    )
    assert check_input_view(query) == [
        InputViewProblem("unique", 2, ("a", "b")),
        InputViewProblem("covers_entries", 7, ("1", "2", "3", "4", "5")),
        InputViewProblem("not_null:idm", 1, ("x",)),
    ]


def test_checks_only_given_features():
    """特徴量の列を指定すれば、その列だけの値なしを見る."""
    query = FakeQuery([*KEYS, "idm", "unused"], nulls={"unused": ["x"]})
    assert check_input_view(query, features=["idm"]) == []
    assert "null:unused" not in query.calls


def test_stops_at_missing_columns():
    """列が足りなければ、それだけを返し、ほかは問い合わせない."""
    query = FakeQuery(["racehorsekey", "racekey", "umaban"])
    assert check_input_view(query, features=["idm"]) == [
        InputViewProblem("columns", 2, ("race_date", "idm"))
    ]
    assert query.calls == ["columns"]


@pytest.mark.parametrize("features", [None, ["finish_order"]])
def test_rejects_label_columns(features):
    """ラベルの列は、省略時も指定時も特徴量にできない."""
    query = FakeQuery([*KEYS, "finish_order"])
    with pytest.raises(ValueError, match="ラベルの列"):
        check_input_view(query, features=features)
