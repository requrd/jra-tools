"""モデルのフォルダの特徴量の仕様のユニットテスト."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from jra_tools.machine_learning.entity.feature_spec import (
    default_feature_spec,
    load_feature_spec,
    save_feature_spec,
)
from jra_tools.machine_learning.infrastructure.model_feature_spec import (
    MODEL_SPEC_FILE,
    load_model_feature_spec,
    save_model_feature_spec,
    saved_model_input_shape,
)

MODEL_DIR = Path(__file__).parents[2] / "database" / "ResNet_4blocks_20130101_20191231"


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """仕様のファイルがないモデルのフォルダ(入力の形の取得に要るsaved_model.pbだけ).

    Returns:
        Path: モデルのフォルダ

    """
    shutil.copy(MODEL_DIR / "saved_model.pb", tmp_path)
    return tmp_path


def _without_last_feature():
    spec = default_feature_spec()
    scores = spec.groups[0]
    return replace(
        spec,
        groups=(replace(scores, features=scores.features[:-1]), *spec.groups[1:]),
    )


def test_saved_model_input_shape():
    """saved_model.pbから入力の形を取得できることを確認する."""
    assert saved_model_input_shape(MODEL_DIR) == [-1, 18, 23]


def test_model_dir_has_spec():
    """テスト用のモデルのフォルダの仕様が、同梱の仕様と一致することを確認する."""
    assert load_model_feature_spec(MODEL_DIR) == default_feature_spec()


def test_save_and_load(model_dir: Path):
    """書き出した仕様を読み戻せることを確認する."""
    path = save_model_feature_spec(default_feature_spec(), model_dir)

    assert path == model_dir / MODEL_SPEC_FILE
    assert load_model_feature_spec(model_dir) == default_feature_spec()


def test_save_mismatch(model_dir: Path):
    """モデルの入力の形と合わない仕様は書き出さないことを確認する."""
    with pytest.raises(ValueError, match="一致しません"):
        save_model_feature_spec(_without_last_feature(), model_dir)

    assert not (model_dir / MODEL_SPEC_FILE).exists()


def test_load_without_spec(model_dir: Path):
    """仕様のファイルがないとエラーになることを確認する."""
    with pytest.raises(FileNotFoundError, match="特徴量の仕様がありません"):
        load_model_feature_spec(model_dir)


def test_load_mismatch(model_dir: Path):
    """仕様とモデルの入力の形が合わないとエラーになることを確認する."""
    save_feature_spec(_without_last_feature(), model_dir / MODEL_SPEC_FILE)

    with pytest.raises(ValueError, match="一致しません"):
        load_model_feature_spec(model_dir)


def test_round_trip(tmp_path: Path):
    """save_feature_specで書いた仕様を、load_feature_specで同じ仕様に読み戻せる."""
    path = tmp_path / "spec.yaml"
    save_feature_spec(default_feature_spec(), path)

    assert load_feature_spec(path) == default_feature_spec()
