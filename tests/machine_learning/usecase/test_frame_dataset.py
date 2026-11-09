"""1頭1行の表からデータセットを組み立てるユニットテスト."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from jra_tools.machine_learning.interface_adapter.spec_format import (
    parse_feature_spec,
)
from jra_tools.machine_learning.usecase.frame_dataset import (
    frame_to_dataset,
    frame_to_input,
    to_slots,
)

# waku: コード"1"・"2"→1、"3"→2(まとめ方の例)。次元は2 + 例外の列1 = 3
SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 4,
        "source": "features.entries_v1",
        "view_digest": "d" * 64,
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm"}, {"name": "same"}],
            },
            {
                "name": "categories",
                "transform": "none",
                "features": [
                    {
                        "name": "waku",
                        "encode": "category",
                        "categories": {"1": 1, "2": 1, "3": 2},
                    }
                ],
            },
        ],
    }
)


def _frame(**overrides) -> pd.DataFrame:
    data = {
        "racehorsekey": ["a1", "a2", "a3", "b1", "b2"],
        "racekey": ["A", "A", "A", "B", "B"],
        "umaban": [1, 3, 4, 2, 4],
        "idm": [50.0, "x", 41.5, 40.0, 60.0],
        "same": [5.0, 5.0, 5.0, 5.0, 5.0],
        # "9"は控えにないコード、Noneは記載外(どちらも例外の列)
        "waku": ["1", "3", "9", "2", None],
        "finish_order": [2, 1, 0, 1, 2],
    }
    return pd.DataFrame({**data, **overrides})


def test_input_values():
    """空き枠は0、文字列は0として標準化、カテゴリは控えの番号の位置、控えにない・NULLは例外の列."""
    x = frame_to_input(_frame(), SPEC).x

    assert x.shape == (2, 4, 5)
    # レースAの空き枠(馬番2)は、標準化した値(0の行も含めて計算)
    idm_a = np.array([50.0, 0, 0, 41.5])
    assert np.allclose(x[0, :, 0], (idm_a - idm_a.mean()) / idm_a.std())
    assert x[0, 0, 2:].tolist() == [1, 0, 0]  # "1"→1
    assert x[0, 1, 2:].tolist() == [0, 0, 0]  # 空き枠
    assert x[0, 2, 2:].tolist() == [0, 1, 0]  # "3"→2
    assert x[0, 3, 2:].tolist() == [0, 0, 1]  # "9"→例外の列
    assert x[1, 1, 2:].tolist() == [1, 0, 0]  # "2"→1(まとめた番号)
    assert x[1, 3, 2:].tolist() == [0, 0, 1]  # NULL→例外の列


def test_dataset_labels_without_payout():
    """学習用のデータセットは入力と1〜5着のラベルだけ."""
    dataset = frame_to_dataset(_frame(), SPEC)

    assert list(dataset.racekeys) == ["A", "B"]
    assert len(dataset.labels) == 5
    assert dataset.labels[0][0].tolist() == [0, 0, 1, 0]
    assert dataset.labels[1][0].tolist() == [1, 0, 0, 0]
    assert dataset.labels[0][1].tolist() == [0, 1, 0, 0]
    assert not hasattr(dataset, "payout")


def test_needs_pinned_spec():
    """カテゴリの控えのない仕様では組み立てない."""
    group = SPEC.groups[1]
    [waku] = group.features
    unpinned = dataclasses.replace(
        SPEC,
        groups=(
            SPEC.groups[0],
            dataclasses.replace(
                group, features=(dataclasses.replace(waku, categories=None),)
            ),
        ),
    )
    with pytest.raises(ValueError, match="categoriesがありません"):
        frame_to_input(_frame(), unpinned)


def test_zero_variance():
    """全枠が同じ値の列は、StandardScalerと同じく割らずに0になる."""
    frame = _frame(
        racehorsekey=["a1", "a2", "a3", "a4", "b1"],
        racekey=["A", "A", "A", "A", "B"],
        umaban=[1, 2, 3, 4, 1],
    )

    x = frame_to_input(frame, SPEC).x

    assert x[0, :, 1].tolist() == [0, 0, 0, 0]


def test_empty_frame():
    """出走のない日は、レースが0件の入力になる."""
    empty = _frame().iloc[0:0]

    assert frame_to_input(empty, SPEC).x.shape == (0, 4, 5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"umaban": [1, 3, 5, 2, 4]}, "馬番が範囲外です"),
        ({"umaban": [1, 3, 3, 2, 4]}, "同じレースに同じ馬番"),
        ({"racekey": ["A", "B", "A", "B", "B"]}, "まとまって並んでいません"),
    ],
)
def test_invalid_frame(overrides, message):
    """組み立てられない表はエラーになる."""
    with pytest.raises(ValueError, match=message):
        frame_to_input(_frame(**overrides), SPEC)


def test_to_slots():
    """各レースを1〜max_horsesの枠に並べ、空き枠は値なし."""
    slots = to_slots(_frame(), 4)

    assert slots.index.tolist() == [(r, u) for r in "AB" for u in range(1, 5)]
    assert pd.isna(slots.loc[("A", 2), "racehorsekey"])
