"""ビューから読む入力・ラベルの決まりのユニットテスト."""

import dataclasses

import pandas as pd
import pytest

from jra_tools.machine_learning.entity.input_view import (
    LABEL_PRESENT,
    category_columns,
    check_columns,
    check_digest,
    check_values,
    input_columns,
)
from jra_tools.machine_learning.interface_adapter.spec_format import (
    parse_feature_spec,
)

SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 4,
        "source": "sandbox.entries_try",
        "view_digest": "a" * 64,
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm"}, {"name": "idm2", "source": "idm"}],
            },
            {
                "name": "categories",
                "transform": "none",
                "features": [
                    {"name": "waku", "encode": "category", "categories": {"1": 1}}
                ],
            },
        ],
    }
)


def test_input_columns():
    """読む列は重複を除き、最初に出た順. カテゴリの列も分かる."""
    assert input_columns(SPEC) == ["idm", "waku"]
    assert category_columns(SPEC) == ["waku"]


@pytest.mark.parametrize("column", ["finish_order", "_label"])
def test_label_column_is_not_input(column):
    """確定後の値(ラベルの列)は入力に使えない."""
    feature = dataclasses.replace(SPEC.groups[0].features[0], source=column)
    group = dataclasses.replace(SPEC.groups[0], features=(feature,))
    spec = dataclasses.replace(SPEC, groups=(group, *SPEC.groups[1:]))

    with pytest.raises(ValueError, match="ラベルの列は入力に使えません"):
        input_columns(spec)


def test_check_columns():
    """足りない列を、必要な順に出す."""
    check_columns("v", ["a", "b", "c"], ["c", "a"])
    with pytest.raises(ValueError, match=r"ビューvに列がありません: \['x', 'y'\]"):
        check_columns("v", ["a"], ["x", "a", "y"])


def test_check_digest():
    """学習時と同じハッシュなら通し、違えば止める."""
    check_digest(SPEC, "v", "a" * 64)
    with pytest.raises(ValueError, match=r"学習時と違います\(今bbbbbbbbbbbb、学習時a"):
        check_digest(SPEC, "v", "b" * 64)


def _frame(**overrides) -> pd.DataFrame:
    data = {
        "racehorsekey": ["k1", "k2"],
        "idm": [1.0, 2.0],
        "waku": ["1", "2"],
        LABEL_PRESENT: ["k1", "k2"],
        "finish_order": [1, 2],
    }
    return pd.DataFrame({**data, **overrides})


def test_check_values_passes():
    """値なしがなければ、LABEL_PRESENTを除いた表を返す. 推論(ラベルなし)はそのまま."""
    assert LABEL_PRESENT not in check_values(_frame(), SPEC, labels=True)
    inputs = _frame().drop(columns=[LABEL_PRESENT, "finish_order"])
    assert check_values(inputs, SPEC, labels=False).equals(inputs)


def test_check_values_allows_null_category():
    """カテゴリの列の値なし(記載外のコード)は許す(例外の列にする)."""
    check_values(_frame(waku=["1", None]), SPEC, labels=True)


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("idm", "sandbox.entries_try.idmが値なしの出走があります: k2"),
        (LABEL_PRESENT, "成績がありません: k2"),
    ],
)
def test_check_values_null(column, message):
    """値なしの出走があれば、最初の出走馬のキーを出して止める."""
    frame = _frame(**{column: [1, None]})
    with pytest.raises(ValueError, match=message):
        check_values(frame, SPEC, labels=True)
