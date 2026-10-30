"""1頭1行の表からデータセットを組み立てるユニットテスト."""

import numpy as np
import pandas as pd
import pytest

from jra_tools.machine_learning.entity.feature_spec import parse_feature_spec
from jra_tools.machine_learning.usecase.frame_dataset import (
    frame_to_dataset,
    frame_to_input,
    to_slots,
)
from jra_tools.machine_learning.usecase.table_dataset import (
    DatasetRows,
    assemble_dataset,
)

SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 4,
        "source": "features.entries_v1",
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm"}, {"name": "same"}],
            },
            {
                "name": "categories",
                "transform": "none",
                "features": [{"name": "waku", "encode": "onehot", "size": 3}],
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
        # 0は末尾から数えて最後の位置
        "waku": [1, 2, 0, 1, 3],
        "finish_order": [2, 1, 0, 1, 2],
        "win1_num": [3, 3, 3, 2, 2],
        "win1_ret": [150, 150, 150, 320, 320],
    }
    return pd.DataFrame({**data, **overrides})


def _rows(frame: pd.DataFrame) -> DatasetRows:
    def col(name):
        return frame[name].to_numpy(dtype=object)

    return DatasetRows(
        racehorsekeys=col("racehorsekey"),
        racekeys=col("racekey"),
        nums=frame["umaban"].to_numpy(),
        features={c: col(c) for c in ("idm", "same", "waku")},
        order_of_arrival=col("finish_order"),
        win1_num=col("win1_num"),
        win1_ret=col("win1_ret"),
    )


def test_same_as_assemble_dataset():
    """今の経路(assemble_dataset)と、誤差の範囲で同じデータセットになる."""
    frame = _frame()

    actual = frame_to_dataset(frame, SPEC)
    expected = assemble_dataset(_rows(frame), SPEC)

    assert list(actual.racekeys) == ["A", "B"]
    assert actual.x.shape == (2, 4, 5)
    assert np.allclose(actual.x, expected.x, rtol=0, atol=1e-12)
    for a, e in zip(actual.labels, expected.labels, strict=True):
        assert np.array_equal(a, e)
    assert np.array_equal(actual.payout, expected.payout)


def test_input_values():
    """空き枠は0、文字列は0として標準化、one-hotは馬番の枠."""
    x = frame_to_input(_frame(), SPEC).x

    # レースAの空き枠(馬番2)は、標準化した値(0の行も含めて計算)
    idm_a = np.array([50.0, 0, 0, 41.5])
    assert np.allclose(x[0, :, 0], (idm_a - idm_a.mean()) / idm_a.std())
    assert x[0, 1, 2:].tolist() == [0, 0, 0]
    assert x[0, 0, 2:].tolist() == [1, 0, 0]
    assert x[0, 3, 2:].tolist() == [0, 0, 1]
    assert x[1, 3, 2:].tolist() == [0, 0, 1]


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
        ({"waku": [1, 2, 4, 1, 3]}, "one-hotの範囲外"),
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
    assert slots.loc[("A", 2), "racehorsekey"] is np.nan or pd.isna(
        slots.loc[("A", 2), "racehorsekey"]
    )
