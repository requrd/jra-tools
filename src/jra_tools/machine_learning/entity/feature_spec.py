"""特徴量の仕様(推論モデルの入力の形)を定義する.

仕様はYAMLで書き、学習に使った仕様をモデルと一緒に保存する(辞書との変換は
interface_adapter/spec_format.py、ファイルはinfrastructure)。
入力の次元は、groupsの並び順・featuresの並び順のとおりに並ぶ。

- source: 読む入力のビュー(「スキーマ.名前」。featuresや試作のsandbox)。各特徴量のsourceはその列名
- view_digest: 学習時のsourceの定義のハッシュ。読むたびにDBと照合する
- categories: カテゴリの列の「コード→番号」。学習の前にcode_categoriesから写す控え
  (変換の決まりの正はDBのcode_categories。推論はこの控えだけを使う)
- output_names: モデルの出力の名前(1着→5着の順)。モデルを保存するときに写す
- train_from・train_to: 学習期間(初日・最終日を含む。省略可)
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Mapping, Optional, Sequence, Tuple


class Transform(StrEnum):
    """グループ単位の変換."""

    NONE = "none"
    # レースごとに標準化する(空き枠の0の行も含めたmax_horses行で計算する)
    STANDARDIZE_PER_RACE = "standardize_per_race"


class Encode(StrEnum):
    """特徴量ごとの値の表し方."""

    NUMBER = "number"
    # コードをcategoriesで番号にし、one-hotにする。控えにないコード・NULLは例外の列(最後の次元)
    CATEGORY = "category"


@dataclass(frozen=True)
class Feature:
    """1つの特徴量. sourceは入力のビューの列名(項目の名前)."""

    name: str
    source: str
    encode: Encode = Encode.NUMBER
    categories: Optional[Mapping[str, int]] = None

    def __post_init__(self):
        """sourceの書式と、categoriesを持てるかを検証する."""
        if not self.source or "." in self.source:
            raise ValueError(f"特徴量のsourceは列名だけです: {self}")
        if self.encode == Encode.NUMBER and self.categories is not None:
            raise ValueError(f"numberの特徴量はcategoriesを持てません: {self}")
        if self.categories is not None:
            numbers = set(self.categories.values())
            if not numbers or min(numbers) < 1:
                raise ValueError(f"categoriesの番号は1からです: {self.name}")

    @property
    def pinned(self) -> bool:
        """次元が決まっているか(numberか、categoriesを写したcategory)."""
        return self.encode == Encode.NUMBER or self.categories is not None

    @property
    def size(self) -> int:
        """次元数. categoryは最大の番号 + 例外の列1.

        Raises:
            ValueError: categoriesを写していないcategoryの場合

        """
        if self.encode == Encode.NUMBER:
            return 1
        if self.categories is None:
            raise ValueError(
                f"{self.name}のcategoriesがありません(仕様を固定してください)"
            )
        return max(self.categories.values()) + 1


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
    source: str
    groups: Tuple[FeatureGroup, ...]
    view_digest: Optional[str] = None
    output_names: Optional[Tuple[str, ...]] = None
    train_from: Optional[date] = None
    train_to: Optional[date] = None

    def __post_init__(self):
        """学習期間・sourceの書式を検証する."""
        if (self.train_from is None) != (self.train_to is None):
            raise ValueError("train_fromとtrain_toは両方書いてください")
        if (
            self.train_from is not None
            and self.train_to is not None
            and self.train_from > self.train_to
        ):
            raise ValueError(
                f"学習期間の初日{self.train_from}が最終日{self.train_to}より後です"
            )
        schema, _, view = self.source.partition(".")
        if not schema or not view:
            raise ValueError(
                f"仕様のsourceは「スキーマ.ビュー名」で指定してください: {self.source}"
            )

    @property
    def features(self) -> Tuple[Feature, ...]:
        """すべての特徴量(groupsの並び順)."""
        return tuple(feature for group in self.groups for feature in group.features)

    @property
    def category_features(self) -> Tuple[Feature, ...]:
        """カテゴリの特徴量."""
        return tuple(f for f in self.features if f.encode == Encode.CATEGORY)

    @property
    def size(self) -> int:
        """1頭あたりの特徴量の次元数."""
        return sum(group.size for group in self.groups)

    def check_pinned(self) -> None:
        """入力を作れるか(ビューのハッシュとカテゴリの控えがあるか)を確かめる.

        Raises:
            ValueError: view_digest・categoriesがない場合

        """
        if self.view_digest is None:
            raise ValueError(
                f"仕様{self.name}にview_digestがありません(仕様を固定してください)"
            )
        missing = [f.name for f in self.features if not f.pinned]
        if missing:
            raise ValueError(
                f"仕様{self.name}のcategoriesがありません(仕様を固定してください): {missing}"
            )

    def check_input_shape(self, shape: Sequence[Any]) -> None:
        """推論モデルの入力の形(先頭はバッチの次元)が仕様と一致するか確かめる.

        Raises:
            ValueError: 形が仕様と一致しない場合

        """
        if tuple(shape[1:]) != (self.max_horses, self.size):
            raise ValueError(
                f"モデルの入力の形{tuple(shape)}が、仕様{self.name}"
                f"(None, {self.max_horses}, {self.size})と一致しません"
            )

    def input_digest(self) -> str:
        """入力の作り方のハッシュ(読む先・ハッシュ・特徴量・カテゴリの控え).

        学習期間・出力の名前は入力の作り方に関わらないので含めない。
        ビューから読んで残した表が、今の仕様のものかを確かめるのに使う。
        """
        data = {
            "max_horses": self.max_horses,
            "source": self.source,
            "view_digest": self.view_digest,
            "groups": [
                {
                    "name": g.name,
                    "transform": str(g.transform),
                    "features": [
                        {
                            "name": f.name,
                            "source": f.source,
                            "encode": str(f.encode),
                            "categories": dict(sorted(f.categories.items()))
                            if f.categories is not None
                            else None,
                        }
                        for f in g.features
                    ],
                }
                for g in self.groups
            ],
        }
        text = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
