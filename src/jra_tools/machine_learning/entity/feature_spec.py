"""特徴量の仕様(推論モデルの入力の形)を定義する.

仕様はYAMLで書き、学習に使った仕様をモデルと一緒に保存する。
入力の次元は、groupsの並び順・featuresの並び順のとおりに並ぶ。
"""

from dataclasses import dataclass
from enum import StrEnum
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import yaml

# パッケージに同梱している仕様のファイルを置くパッケージ
SPECS_PACKAGE = "jra_tools.machine_learning.specs"
# 現行の推論モデル(ResNet_4blocks_20130101_20191231)の仕様
DEFAULT_SPEC = "resnet4_2013_2019.yaml"


class Transform(StrEnum):
    """グループ単位の変換."""

    NONE = "none"
    # レースごとにStandardScalerで標準化する(空き枠の0の行も含めた18行で計算する)
    STANDARDIZE_PER_RACE = "standardize_per_race"


class Encode(StrEnum):
    """特徴量ごとの値の表し方."""

    NUMBER = "number"
    ONEHOT = "onehot"


@dataclass(frozen=True)
class Feature:
    """1つの特徴量.

    sourceは「テーブル名.列名」(例: racehorse.idm)。
    onehotの場合、値がbaseのときに先頭の次元が1になる。
    """

    name: str
    source: str
    encode: Encode = Encode.NUMBER
    size: int = 1
    base: int = 1

    def __post_init__(self):
        """sourceの書式とonehotの次元数を検証する."""
        table, _, column = self.source.partition(".")
        if not table or not column:
            raise ValueError(f"sourceは「テーブル名.列名」で指定してください: {self}")
        if self.encode == Encode.ONEHOT and self.size < 1:
            raise ValueError(f"onehotのsizeは1以上を指定してください: {self}")
        if self.encode == Encode.NUMBER and self.size != 1:
            raise ValueError(f"numberのsizeは1です: {self}")

    @property
    def table(self) -> str:
        """sourceのテーブル名."""
        return self.source.partition(".")[0]

    @property
    def column(self) -> str:
        """sourceの列名."""
        return self.source.partition(".")[2]


@dataclass(frozen=True)
class FeatureGroup:
    """変換をまとめて適用する特徴量の組."""

    name: str
    transform: Transform
    features: Tuple[Feature, ...]

    @property
    def size(self) -> int:
        """グループの次元数."""
        return sum(feature.size for feature in self.features)


@dataclass(frozen=True)
class FeatureSpec:
    """推論モデルの入力の仕様. 入力の形は(レース数, max_horses, size)."""

    name: str
    max_horses: int
    groups: Tuple[FeatureGroup, ...]

    @property
    def size(self) -> int:
        """1頭あたりの特徴量の次元数."""
        return sum(group.size for group in self.groups)

    def check_input_shape(self, shape: Sequence[Any]) -> None:
        """推論モデルの入力の形(先頭はバッチの次元)が仕様と一致するか確かめる.

        Args:
            shape (Sequence[Any]): モデルの入力の形(例: (None, 18, 23))

        Raises:
            ValueError: 形が仕様と一致しない場合

        """
        if tuple(shape[1:]) != (self.max_horses, self.size):
            raise ValueError(
                f"モデルの入力の形{tuple(shape)}が、仕様{self.name}"
                f"(None, {self.max_horses}, {self.size})と一致しません"
            )


def parse_feature_spec(data: Mapping[str, Any]) -> FeatureSpec:
    """辞書(YAMLを読み込んだもの)から仕様を作る."""
    return FeatureSpec(
        name=data["name"],
        max_horses=data["max_horses"],
        groups=tuple(
            FeatureGroup(
                name=group["name"],
                transform=Transform(group["transform"]),
                features=tuple(
                    Feature(
                        name=feature["name"],
                        source=feature["source"],
                        encode=Encode(feature.get("encode", Encode.NUMBER)),
                        size=feature.get("size", 1),
                        base=feature.get("base", 1),
                    )
                    for feature in group["features"]
                ),
            )
            for group in data["groups"]
        ),
    )


def feature_spec_to_dict(spec: FeatureSpec) -> Dict[str, Any]:
    """仕様を、parse_feature_specで読み戻せる辞書にする."""
    return {
        "name": spec.name,
        "max_horses": spec.max_horses,
        "groups": [
            {
                "name": group.name,
                "transform": str(group.transform),
                "features": [
                    {"name": feature.name, "source": feature.source}
                    if feature.encode == Encode.NUMBER
                    else {
                        "name": feature.name,
                        "source": feature.source,
                        "encode": str(feature.encode),
                        "size": feature.size,
                        "base": feature.base,
                    }
                    for feature in group.features
                ],
            }
            for group in spec.groups
        ],
    }


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
