"""jrdb-modelの表から学習用のデータセットを作る手続きのユニットテスト. 読み手は偽物."""

from typing import Dict, List, Tuple

import numpy as np
import pytest

from jra_tools.machine_learning.entity.dataset import DatasetRows
from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    parse_feature_spec,
)
from jra_tools.machine_learning.usecase.staging_dataset import load_staging_dataset

SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 4,
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm", "source": "racehorse.idm"}],
            }
        ],
    }
)


def _rows(racekey: str) -> DatasetRows:
    return DatasetRows(
        racehorsekeys=np.array([f"{racekey}1", f"{racekey}2"], dtype=object),
        racekeys=np.array([racekey, racekey], dtype=object),
        nums=np.array([1, 2]),
        features={"racehorse.idm": np.array([1.0, 3.0], dtype=object)},
        order_of_arrival=np.array([2, 1], dtype=object),
        win1_num=np.array([2, 2], dtype=object),
        win1_ret=np.array([150, 150], dtype=object),
    )


class _Reader:
    def __init__(self, missing: Dict[str, List[str]] | None = None):
        self.periods: List[Tuple[int, int]] = []
        self.missing = missing or {}

    def read_rows(
        self, spec: FeatureSpec, start: int, end: int
    ) -> Tuple[DatasetRows, Dict[str, List[str]]]:
        self.periods.append((start, end))
        return _rows(f"r{start // 10000}"), {"seiseki": [], **self.missing}


def test_reads_each_year_and_assembles():
    """1年ずつ読み、つないで組み立てる."""
    reader = _Reader()

    dataset = load_staging_dataset(reader, SPEC, 20241201, 20250131)

    assert reader.periods == [(20241201, 20241231), (20250101, 20250131)]
    assert list(dataset.racekeys) == ["r2024", "r2025"]
    assert dataset.x.shape == (2, 4, 1)
    assert dataset.payout[0, 1] == 150


def test_missing_rows_stop():
    """行のない出走があれば止める."""
    with pytest.raises(ValueError, match="払戻データがありません: r1"):
        load_staging_dataset(_Reader({"returninfo": ["r1"]}), SPEC, 20250101, 20250131)


def test_source_spec_is_not_read():
    """特徴量の層から読む仕様は、読み手を呼ばずに止める."""
    spec = parse_feature_spec(
        {
            "name": "from_features",
            "max_horses": 4,
            "source": "features.entries_v1",
            "groups": [
                {"name": "s", "transform": "none", "features": [{"name": "idm"}]}
            ],
        }
    )
    reader = _Reader()

    with pytest.raises(ValueError, match="featuresを使ってください"):
        load_staging_dataset(reader, spec, 20250101, 20250131)
    assert reader.periods == []
