"""モデルのフォルダ(SavedModel)の仕様の読み書きのテスト. 入口(features)とSavedModelFolderを通す."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from jra_tools.machine_learning.features import (
    load_feature_spec,
    load_model_feature_spec,
    save_feature_spec,
    save_model_feature_spec,
)
from jra_tools.machine_learning.infrastructure.saved_model import (
    MODEL_SPEC_FILE,
    SavedModelFolder,
)
from jra_tools.machine_learning.interface_adapter.spec_format import (
    parse_feature_spec,
)

MODEL_DIR = Path(__file__).parents[2] / "database" / "ResNet_4blocks_20130101_20191231"
OUTPUTS = [
    "activation_2",
    "activation_3",
    "activation_4",
    "activation_5",
    "activation_6",
]


def _spec(numbers: int = 14):
    """入力の形(None, 18, 23)のモデルに合う仕様: 数値14 + 枠のカテゴリ(8コード + 例外 = 9)."""
    return parse_feature_spec(
        {
            "name": "sample",
            "max_horses": 18,
            "source": "features.entries_v1",
            "view_digest": "d" * 64,
            "groups": [
                {
                    "name": "scores",
                    "transform": "standardize_per_race",
                    "features": [{"name": f"n{i}"} for i in range(numbers)],
                },
                {
                    "name": "categories",
                    "transform": "none",
                    "features": [
                        {
                            "name": "waku",
                            "encode": "category",
                            "categories": {str(i): i for i in range(1, 9)},
                        }
                    ],
                },
            ],
        }
    )


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """仕様のファイルがないモデルのフォルダ(入力の形の取得に要るsaved_model.pbだけ).

    Returns:
        Path: モデルのフォルダ

    """
    shutil.copy(MODEL_DIR / "saved_model.pb", tmp_path)
    return tmp_path


def test_saved_model_input_shape():
    """saved_model.pbから入力の形を取得できる."""
    assert SavedModelFolder(MODEL_DIR).input_shape() == [-1, 18, 23]


def test_save_and_load(model_dir: Path):
    """出力の名前を写して書き出し、読み戻せる."""
    path = save_model_feature_spec(_spec(), model_dir, OUTPUTS)

    assert path == model_dir / MODEL_SPEC_FILE
    assert load_model_feature_spec(model_dir) == replace(
        _spec(), output_names=tuple(OUTPUTS)
    )


def test_save_mismatch(model_dir: Path):
    """モデルの入力の形と合わない仕様は書き出さない."""
    with pytest.raises(ValueError, match="一致しません"):
        save_model_feature_spec(_spec(numbers=13), model_dir, OUTPUTS)

    assert not (model_dir / MODEL_SPEC_FILE).exists()


def test_load_without_spec(model_dir: Path):
    """仕様のファイルがないと止める."""
    with pytest.raises(FileNotFoundError, match="特徴量の仕様がありません"):
        load_model_feature_spec(model_dir)


def test_load_mismatch(model_dir: Path):
    """仕様とモデルの入力の形が合わないと止める."""
    save_feature_spec(_spec(numbers=13), model_dir / MODEL_SPEC_FILE)

    with pytest.raises(ValueError, match="一致しません"):
        load_model_feature_spec(model_dir)


def test_round_trip(tmp_path: Path):
    """save_feature_specで書いた仕様を、load_feature_specで同じ仕様に読み戻せる."""
    path = tmp_path / "spec.yaml"
    save_feature_spec(_spec(), path)

    assert load_feature_spec(path) == _spec()
    # カテゴリのコードは数字だけでも文字列のキーとして書く
    assert "'1': 1" in path.read_text(encoding="utf-8")
