"""特徴量の仕様のユニットテスト."""

import dataclasses
from datetime import date

import pytest

from jra_tools.machine_learning.entity.feature_spec import Encode, Feature
from jra_tools.machine_learning.interface_adapter.spec_format import (
    parse_feature_spec,
)


def _data(**overrides):
    data = {
        "name": "sample",
        "max_horses": 18,
        "source": "features.entries_v1",
        "view_digest": "d" * 64,
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm"}, {"name": "score", "source": "x"}],
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
    return {**data, **overrides}


def test_size_counts_exception_column():
    """数値は1次元、カテゴリは最大の番号 + 例外の列1."""
    spec = parse_feature_spec(_data())
    assert spec.size == 2 + 3
    assert spec.features[1].source == "x"
    assert [f.name for f in spec.category_features] == ["waku"]


def test_check_input_shape():
    """モデルの入力の形(先頭はバッチ)と、仕様の形を照合する."""
    spec = parse_feature_spec(_data())
    spec.check_input_shape((None, 18, 5))
    with pytest.raises(ValueError, match="一致しません"):
        spec.check_input_shape((None, 18, 4))


def test_check_pinned():
    """ハッシュ・カテゴリの控えがなければ、入力を作れない."""
    spec = parse_feature_spec(_data())
    spec.check_pinned()
    with pytest.raises(ValueError, match="view_digestがありません"):
        dataclasses.replace(spec, view_digest=None).check_pinned()
    raw = _data()
    del raw["groups"][1]["features"][0]["categories"]
    unpinned = parse_feature_spec(raw)
    with pytest.raises(ValueError, match="categoriesがありません"):
        unpinned.check_pinned()
    with pytest.raises(ValueError, match="固定してください"):
        _ = unpinned.size


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": "racehorse.idm"}, "列名だけ"),
        ({"source": ""}, "列名だけ"),
        ({"categories": {"1": 1}}, "numberの特徴量はcategoriesを持てません"),
        (
            {"encode": Encode.CATEGORY, "categories": {"1": 0}},
            "番号は1から",
        ),
    ],
)
def test_invalid_feature(kwargs, message):
    """書けない特徴量は止める."""
    with pytest.raises(ValueError, match=message):
        Feature(**{"name": "f", "source": "f", **kwargs})


def test_invalid_source():
    """仕様のsourceは「スキーマ.ビュー名」."""
    with pytest.raises(ValueError, match="スキーマ.ビュー名"):
        parse_feature_spec(_data(source="entries_v1"))


def test_training_period():
    """学習期間は両方書き、初日 <= 最終日."""
    spec = parse_feature_spec(
        _data(train_from="2013-01-01", train_to=date(2019, 12, 31))
    )
    assert spec.train_from == date(2013, 1, 1)
    with pytest.raises(ValueError, match="両方"):
        parse_feature_spec(_data(train_from="2013-01-01"))
    with pytest.raises(ValueError, match="より後"):
        parse_feature_spec(_data(train_from="2020-01-01", train_to="2019-12-31"))


def test_input_digest_follows_input_only():
    """入力の作り方(ハッシュ・控え)が変われば変わり、学習期間・出力の名前では変わらない."""
    spec = parse_feature_spec(_data())
    same = dataclasses.replace(
        spec,
        output_names=("a", "b", "c", "d", "e"),
        train_from=date(2013, 1, 1),
        train_to=date(2019, 12, 31),
    )
    assert spec.input_digest() == same.input_digest()
    raw = _data()
    raw["groups"][1]["features"][0]["categories"] = {"1": 1, "2": 2, "3": 2}
    assert parse_feature_spec(raw).input_digest() != spec.input_digest()
    assert (
        dataclasses.replace(spec, view_digest="e" * 64).input_digest()
        != spec.input_digest()
    )
