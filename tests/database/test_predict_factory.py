"""モデルが推論可能なことをテストする."""

from pathlib import Path
from typing import List

import keras
import pytest
from jrdb_model import KaisaiData

MODEL_DIR = Path(__file__).parent / "ResNet_4blocks_20130101_20191231"


@pytest.fixture
def v2_model() -> keras.Sequential:
    """Keras v2の既存モデル.

    Returns:
        keras.Sequential: keras v2で作成済みのモデルを取り込んだSequentailモデル

    """
    layer = keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
    return keras.Sequential([layer])


def test_predict(sample_kaisais: List[KaisaiData], v2_model: keras.Sequential):
    """モデルが推論可能なことを確認する.

    データの形がモデルと合っていないとうまく噛み合わない。
    モデルのバージョン管理はバッチで行っているため、整合性の問題あり。

    Args:
        sample_kaisais (List[KaisaiData]): 開催データ
        v2_model (keras.Sequential): 稼働中のモデル

    """
    from jra_tools.database.predict_factory import PredictFactory

    factory = PredictFactory(sample_kaisais, v2_model)
    entities = factory.create()

    _assert_predicts(entities)


def test_predict_with_model_spec(
    sample_kaisais: List[KaisaiData], v2_model: keras.Sequential
):
    """モデルのフォルダの仕様を渡して推論できることを確認する.

    Args:
        sample_kaisais (List[KaisaiData]): 開催データ
        v2_model (keras.Sequential): 稼働中のモデル

    """
    from jra_tools.database.predict_factory import PredictFactory
    from jra_tools.machine_learning.infrastructure.model_feature_spec import (
        load_model_feature_spec,
    )

    spec = load_model_feature_spec(MODEL_DIR)
    entities = PredictFactory(sample_kaisais, v2_model, spec).create()

    _assert_predicts(entities)


def _assert_predicts(entities):
    assert len(entities) == 363
    assert entities[7].pp_icchaku == pytest.approx(0.2717929184436798)
    assert entities[7].pp_nichaku == pytest.approx(0.18553103506565094)
    assert entities[7].pp_sanchaku == pytest.approx(0.1153089627623558)
