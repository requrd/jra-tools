"""仕様から入力データを作る処理のユニットテスト."""

import copy
from pathlib import Path
from typing import List

import numpy as np
import pytest
from jrdb_model import KaisaiData

from jra_tools.machine_learning.entity.feature_spec import (
    Feature,
    FeatureGroup,
    FeatureSpec,
    Transform,
    default_feature_spec,
)
from jra_tools.machine_learning.usecase.feature_matrix import create_input_matrix

# 仕様を読む形に変える前のjra-tools(2d8e0d0)で、sample_kaisaisから作った入力データ
EXPECTED_INPUT = Path(__file__).parents[1] / "data" / "expected_input.npy"


def test_same_as_before(sample_kaisais: List[KaisaiData]):
    """仕様を読む形に変える前と、入力データが完全に一致することを確認する."""
    x = create_input_matrix(sample_kaisais, default_feature_spec())

    assert np.array_equal(x, np.load(EXPECTED_INPUT))


def _spec(*features: Feature) -> FeatureSpec:
    return FeatureSpec("test", 18, (FeatureGroup("g", Transform.NONE, features),))


def test_unknown_table(sample_kaisais: List[KaisaiData]):
    """入力に使えないテーブル(確定後の成績など)を指定するとエラーになる."""
    spec = _spec(Feature("order", "seiseki.order_of_arrival"))

    with pytest.raises(ValueError, match="入力に使えないテーブル"):
        create_input_matrix(sample_kaisais, spec)


def test_missing_row(sample_kaisais: List[KaisaiData]):
    """sourceの行がない出走があるとエラーになる."""
    kaisais = copy.deepcopy(sample_kaisais)
    horse = kaisais[0].races[0].racehorses[0]
    horse.calculated_score = None
    spec = _spec(Feature("waku_win_rate", "calculated_score.waku_win_rate"))

    with pytest.raises(
        ValueError, match=f"calculated_score がありません: {horse.racehorsekey}"
    ):
        create_input_matrix(kaisais, spec)


def test_horse_slot(sample_kaisais: List[KaisaiData]):
    """各馬の値が馬番-1の位置に入り、空き枠は0になることを確認する."""
    spec = _spec(Feature("idm", "racehorse.idm"))

    x = create_input_matrix(sample_kaisais, spec)

    race = sample_kaisais[0].races[0]
    for horse in race.racehorses:
        assert x[0, horse.num - 1, 0] == horse.idm
    assert not x[0, len(race.racehorses) :].any()
