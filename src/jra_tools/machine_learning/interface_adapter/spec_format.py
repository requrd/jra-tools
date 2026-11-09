"""特徴量の仕様と、YAMLの辞書との変換."""

from datetime import date
from typing import Any, Dict, Mapping, Optional

from jra_tools.machine_learning.entity.feature_spec import (
    Encode,
    Feature,
    FeatureGroup,
    FeatureSpec,
    Transform,
)


def parse_feature_spec(data: Mapping[str, Any]) -> FeatureSpec:
    """辞書(YAMLを読み込んだもの)から仕様を作る. 特徴量のsourceを省略するとname."""
    output_names = data.get("output_names")
    return FeatureSpec(
        name=data["name"],
        max_horses=data["max_horses"],
        source=data["source"],
        groups=tuple(
            FeatureGroup(
                name=group["name"],
                transform=Transform(group["transform"]),
                features=tuple(
                    Feature(
                        name=feature["name"],
                        source=feature.get("source", feature["name"]),
                        encode=Encode(feature.get("encode", Encode.NUMBER)),
                        categories=_categories(feature.get("categories")),
                    )
                    for feature in group["features"]
                ),
            )
            for group in data["groups"]
        ),
        view_digest=data.get("view_digest"),
        output_names=tuple(output_names) if output_names is not None else None,
        train_from=_date(data.get("train_from")),
        train_to=_date(data.get("train_to")),
    )


def feature_spec_to_dict(spec: FeatureSpec) -> Dict[str, Any]:
    """仕様を、parse_feature_specで読み戻せる辞書にする."""
    head: Dict[str, Any] = {
        "name": spec.name,
        "max_horses": spec.max_horses,
        "source": spec.source,
    }
    if spec.view_digest is not None:
        head["view_digest"] = spec.view_digest
    if spec.output_names is not None:
        head["output_names"] = list(spec.output_names)
    if spec.train_from is not None:
        head["train_from"] = spec.train_from
        head["train_to"] = spec.train_to
    return {
        **head,
        "groups": [
            {
                "name": group.name,
                "transform": str(group.transform),
                "features": [_feature_to_dict(f) for f in group.features],
            }
            for group in spec.groups
        ],
    }


def _feature_to_dict(feature: Feature) -> Dict[str, Any]:
    data: Dict[str, Any] = {"name": feature.name, "source": feature.source}
    if feature.encode == Encode.CATEGORY:
        data["encode"] = str(feature.encode)
        if feature.categories is not None:
            data["categories"] = dict(feature.categories)
    return data


def _categories(value: Optional[Mapping[Any, Any]]) -> Optional[Dict[str, int]]:
    """コードは文字列にそろえる(YAMLでは数字だけのキーが整数に読まれるため)."""
    if value is None:
        return None
    return {str(code): int(number) for code, number in value.items()}


def _date(value: Any) -> Optional[date]:
    """YAMLの日付(dateに読まれる)か、YYYY-MM-DDの文字列を日付にする."""
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))
