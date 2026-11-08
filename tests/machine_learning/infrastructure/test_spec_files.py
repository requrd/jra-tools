"""仕様のYAMLファイルと、同梱の既定の仕様のテスト."""

from datetime import date
from pathlib import Path

from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    Transform,
    parse_feature_spec,
)
from jra_tools.machine_learning.infrastructure.saved_model_folder import (
    SavedModelFolder,
)
from jra_tools.machine_learning.infrastructure.spec_files import (
    default_feature_spec,
    load_feature_spec,
    save_feature_spec,
)

MODEL_DIR = Path(__file__).parents[2] / "database" / "ResNet_4blocks_20130101_20191231"


def test_default_spec_matches_model():
    """同梱の仕様が、稼働中のモデルの入力の形と一致する."""
    spec = default_feature_spec()

    assert spec.size == 23
    spec.check_input_shape(SavedModelFolder(MODEL_DIR).input_shape())


def test_default_spec_groups():
    """同梱の仕様のグループと次元数."""
    spec = default_feature_spec()

    assert [(g.name, g.transform, g.size) for g in spec.groups] == [
        ("scores", Transform.STANDARDIZE_PER_RACE, 15),
        ("categories", Transform.NONE, 8),
    ]


def test_load_feature_spec(tmp_path: Path):
    """YAMLファイルから仕様を読み込める."""
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


def test_round_trip(tmp_path: Path):
    """書き出した仕様を、同じ仕様に読み戻せる(sourceのない仕様・ある仕様・学習期間)."""
    view_spec = parse_feature_spec(
        {
            "name": "sample",
            "max_horses": 18,
            "source": "features.entries_v1",
            "view_digest": "abc",
            "train_from": date(2013, 1, 1),
            "train_to": date(2019, 12, 31),
            "groups": [
                {"name": "g", "transform": "none", "features": [{"name": "idm"}]}
            ],
        }
    )
    for spec in (default_feature_spec(), view_spec):
        path = tmp_path / f"{spec.name}.yaml"
        save_feature_spec(spec, path)
        assert load_feature_spec(path) == spec
    assert "train_from: 2013-01-01" in (tmp_path / "sample.yaml").read_text("utf-8")


def test_jrdb_spec_file_has_no_view_fields(tmp_path: Path):
    """sourceのない仕様は、書き出してもsource・view_digest・学習期間が増えない."""
    path = tmp_path / "spec.yaml"
    save_feature_spec(default_feature_spec(), path)

    text = path.read_text(encoding="utf-8")
    for key in ("source: features", "view_digest", "train_from", "train_to"):
        assert key not in text
