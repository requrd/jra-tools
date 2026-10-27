"""特徴量の仕様のユニットテスト."""

from pathlib import Path

import pytest

from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    Transform,
    default_feature_spec,
    load_feature_spec,
)
from jra_tools.machine_learning.infrastructure.model_feature_spec import (
    saved_model_input_shape,
)

MODEL_DIR = Path(__file__).parents[2] / "database" / "ResNet_4blocks_20130101_20191231"


def test_default_spec_matches_model():
    """同梱の仕様が、稼働中のモデルの入力の形と一致することを確認する."""
    spec = default_feature_spec()

    assert spec.size == 23
    spec.check_input_shape(saved_model_input_shape(MODEL_DIR))


def test_default_spec_groups():
    """同梱の仕様のグループと次元数を確認する."""
    spec = default_feature_spec()

    assert [(g.name, g.transform, g.size) for g in spec.groups] == [
        ("scores", Transform.STANDARDIZE_PER_RACE, 15),
        ("categories", Transform.NONE, 8),
    ]


def test_check_input_shape_mismatch():
    """入力の形が仕様と一致しないとエラーになることを確認する."""
    with pytest.raises(ValueError, match="一致しません"):
        default_feature_spec().check_input_shape((None, 18, 22))


def test_load_feature_spec(tmp_path: Path):
    """YAMLファイルから仕様を読み込めることを確認する."""
    path = tmp_path / "spec.yaml"
    path.write_text(
        "name: sample\n"
        "max_horses: 18\n"
        "groups:\n"
        "  - name: g\n"
        "    transform: none\n"
        "    features:\n"
        "      - {name: idm, source: racehorse.idm}\n"
        "      - {name: waku, source: racehorse.waku, encode: onehot, size: 8}\n",
        encoding="utf-8",
    )

    spec = load_feature_spec(path)

    assert spec.size == 9
    assert spec.groups[0].features[1] == Feature(
        "waku", "racehorse.waku", Encode.ONEHOT, 8, 1
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": "idm"}, "テーブル名.列名"),
        ({"source": "racehorse.waku", "encode": Encode.ONEHOT, "size": 0}, "1以上"),
        ({"source": "racehorse.idm", "size": 2}, "numberのsizeは1"),
    ],
)
def test_invalid_feature(kwargs, message):
    """不正な特徴量の定義がエラーになることを確認する."""
    with pytest.raises(ValueError, match=message):
        Feature(name="x", **kwargs)
