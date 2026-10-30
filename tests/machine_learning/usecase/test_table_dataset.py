"""出走ごとの列からデータセットを組み立てるユニットテスト."""

import numpy as np

from jra_tools.machine_learning.entity.feature_spec import parse_feature_spec
from jra_tools.machine_learning.usecase.table_dataset import (
    DatasetRows,
    InputRows,
    assemble_dataset,
    assemble_input,
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
                "features": [{"name": "idm"}],
            },
            {
                "name": "categories",
                "transform": "none",
                "features": [{"name": "waku", "encode": "onehot", "size": 2}],
            },
        ],
    }
)
INPUT = InputRows(
    racehorsekeys=np.array(["a1", "a2", "b1"], dtype=object),
    racekeys=np.array(["A", "A", "B"], dtype=object),
    nums=np.array([1, 3, 2]),
    features={
        "idm": np.array([50.0, "x", 40.0], dtype=object),
        "waku": np.array([1, 2, 1], dtype=object),
    },
)


def test_assemble_input_same_as_dataset():
    """入力だけを組み立てた結果が、データセットの入力と同じになる."""
    rows = DatasetRows(
        **INPUT.__dict__,
        order_of_arrival=np.array([1, 2, 1], dtype=object),
        win1_num=np.array([1, 1, 2], dtype=object),
        win1_ret=np.array([150, 150, 300], dtype=object),
    )

    inputs = assemble_input(INPUT, SPEC)
    dataset = assemble_dataset(rows, SPEC)

    assert list(inputs.racekeys) == ["A", "B"]
    assert inputs.x.shape == (2, 4, 3)
    assert np.array_equal(inputs.x, dataset.x)
    # 文字列は0、onehotは馬番の位置
    assert inputs.x[0, 2, 1:].tolist() == [0, 1]
    assert inputs.x[1, 1, 1:].tolist() == [1, 0]


def test_assemble_input_empty():
    """出走のない日は、レースが0件の入力になる."""
    empty = InputRows(
        racehorsekeys=np.array([], dtype=object),
        racekeys=np.array([], dtype=object),
        nums=np.array([], dtype=int),
        features={
            "idm": np.array([], dtype=object),
            "waku": np.array([], dtype=object),
        },
    )

    assert assemble_input(empty, SPEC).x.shape == (0, 4, 3)
