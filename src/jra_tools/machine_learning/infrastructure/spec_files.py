"""特徴量の仕様のYAMLファイルを読み書きする. 同梱の既定の仕様もここで読む."""

from importlib import resources
from pathlib import Path

import yaml

from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    feature_spec_to_dict,
    parse_feature_spec,
)

# パッケージに同梱している仕様のファイルを置くパッケージ
SPECS_PACKAGE = "jra_tools.machine_learning.specs"
# 現行の推論モデル(ResNet_4blocks_20130101_20191231)の仕様
DEFAULT_SPEC = "resnet4_2013_2019.yaml"


def load_feature_spec(path: Path | str) -> FeatureSpec:
    """YAMLファイルから仕様を読み込む."""
    with open(path, encoding="utf-8") as f:
        return parse_feature_spec(yaml.safe_load(f))


def save_feature_spec(spec: FeatureSpec, path: Path | str) -> None:
    """仕様をYAMLファイルに書き出す."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            feature_spec_to_dict(spec), f, allow_unicode=True, sort_keys=False
        )


def default_feature_spec() -> FeatureSpec:
    """現行の推論モデルの仕様を、パッケージに同梱したファイルから読み込む."""
    text = resources.files(SPECS_PACKAGE).joinpath(DEFAULT_SPEC).read_text("utf-8")
    return parse_feature_spec(yaml.safe_load(text))
