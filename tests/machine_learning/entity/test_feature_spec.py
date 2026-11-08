"""特徴量の仕様のユニットテスト."""

from datetime import date

import pytest

from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    check_jrdb_spec,
    feature_spec_to_dict,
    parse_feature_spec,
)

# jrdb-modelの表から読む仕様(最上位にsourceがない)
JRDB_SPEC = parse_feature_spec(
    {
        "name": "jrdb",
        "max_horses": 18,
        "groups": [
            {
                "name": "g",
                "transform": "none",
                "features": [{"name": "idm", "source": "racehorse.idm"}],
            }
        ],
    }
)


def test_check_input_shape():
    """入力の形が仕様と一致すれば通し、一致しないとエラーになる."""
    JRDB_SPEC.check_input_shape((None, 18, 1))
    with pytest.raises(ValueError, match="一致しません"):
        JRDB_SPEC.check_input_shape((None, 18, 2))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": "racehorse.waku", "encode": Encode.ONEHOT, "size": 0}, "1以上"),
        ({"source": "racehorse.idm", "size": 2}, "numberのsizeは1"),
    ],
)
def test_invalid_feature(kwargs, message):
    """不正な特徴量の定義がエラーになることを確認する."""
    with pytest.raises(ValueError, match=message):
        Feature(name="x", **kwargs)


def _data(**head):
    return {
        "name": "sample",
        "max_horses": 18,
        **head,
        "groups": [
            {
                "name": "g",
                "transform": "none",
                "features": [
                    {"name": "idm"},
                    {
                        "name": "waku",
                        "source": "waku_no",
                        "encode": "onehot",
                        "size": 8,
                    },
                ],
            }
        ],
    }


def test_view_spec():
    """特徴量の層から読む仕様: 特徴量のsourceは列名で、省略するとnameになる. 辞書にして読み戻せる."""
    spec = parse_feature_spec(_data(source="features.entries_v1", view_digest="abc"))

    assert spec.source == "features.entries_v1"
    assert spec.view_digest == "abc"
    assert [f.source for f in spec.features] == ["idm", "waku_no"]
    assert parse_feature_spec(feature_spec_to_dict(spec)) == spec


def test_jrdb_spec_has_no_view_fields():
    """jrdb-modelの表から読む仕様は、sourceの項目を持たず、辞書にしても増えない."""
    data = feature_spec_to_dict(JRDB_SPEC)

    assert JRDB_SPEC.source is None
    assert JRDB_SPEC.view_digest is None
    assert not {"source", "view_digest", "train_from", "train_to"} & set(data)
    assert parse_feature_spec(data) == JRDB_SPEC


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {
                **_data(),
                "groups": [
                    {
                        "name": "g",
                        "transform": "none",
                        "features": [{"name": "idm", "source": "idm"}],
                    }
                ],
            },
            "テーブル名.列名",
        ),
        (_data(source="entries_v1"), "スキーマ.ビュー名"),
        (
            {
                **_data(source="features.entries_v1"),
                "groups": [
                    {
                        "name": "g",
                        "transform": "none",
                        "features": [{"name": "idm", "source": "racehorse.idm"}],
                    }
                ],
            },
            "列名だけ",
        ),
    ],
)
def test_invalid_source(data, message):
    """sourceの書式が仕様の読み方と合わないとエラーになる."""
    with pytest.raises(ValueError, match=message):
        parse_feature_spec(data)


def test_digest_without_source():
    """view_digestはビューから読む仕様にだけ書ける."""
    data = {**_data(), "view_digest": "abc"}
    data["groups"][0]["features"] = [{"name": "idm", "source": "racehorse.idm"}]

    with pytest.raises(ValueError, match="view_digest"):
        parse_feature_spec(data)


def test_training_period():
    """学習期間を書いた仕様は、日付として読み、辞書にして読み戻せる."""
    spec = parse_feature_spec(
        _data(
            source="features.entries_v1",
            train_from=date(2013, 1, 1),
            train_to="2019-12-31",
        )
    )
    assert (spec.train_from, spec.train_to) == (date(2013, 1, 1), date(2019, 12, 31))
    assert feature_spec_to_dict(spec)["train_from"] == date(2013, 1, 1)
    assert parse_feature_spec(feature_spec_to_dict(spec)) == spec


@pytest.mark.parametrize(
    ("period", "message"),
    [
        ({"train_from": date(2013, 1, 1)}, "両方"),
        ({"train_from": date(2020, 1, 1), "train_to": date(2019, 12, 31)}, "より後"),
    ],
)
def test_invalid_training_period(period, message):
    """学習期間は両方書き、初日は最終日より後にしない."""
    with pytest.raises(ValueError, match=message):
        parse_feature_spec(_data(source="features.entries_v1", **period))


def test_check_jrdb_spec():
    """jrdb-modelの表から読む仕様は通し、sourceのある仕様は止める."""
    check_jrdb_spec(JRDB_SPEC)
    spec = parse_feature_spec(
        {
            "name": "from_features",
            "max_horses": 18,
            "source": "features.entries_v1",
            "groups": [
                {"name": "s", "transform": "none", "features": [{"name": "idm"}]}
            ],
        }
    )
    with pytest.raises(ValueError, match="featuresを使ってください"):
        check_jrdb_spec(spec)
