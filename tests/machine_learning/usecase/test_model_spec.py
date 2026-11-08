"""モデルと一緒に持つ仕様を読み書きする手続きのユニットテスト. フォルダは偽物."""

from pathlib import Path
from typing import List, Optional

import pytest

from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    parse_feature_spec,
)
from jra_tools.machine_learning.usecase.model_spec import (
    load_model_feature_spec,
    save_model_feature_spec,
)

SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 18,
        "groups": [
            {
                "name": "s",
                "transform": "none",
                "features": [{"name": "idm", "source": "racehorse.idm"}],
            }
        ],
    }
)


class _Folder:
    def __init__(self, shape: List[int], spec: Optional[FeatureSpec] = None):
        self.shape = shape
        self.spec = spec

    def input_shape(self) -> List[int]:
        return self.shape

    def read_spec(self) -> FeatureSpec:
        if self.spec is None:
            raise FileNotFoundError("なし")
        return self.spec

    def write_spec(self, spec: FeatureSpec) -> Path:
        self.spec = spec
        return Path("feature_spec.yaml")


def test_load_checks_shape():
    """読んだ仕様がモデルの入力の形と合えば返し、合わなければ止める."""
    assert load_model_feature_spec(_Folder([-1, 18, 1], SPEC)) == SPEC
    with pytest.raises(ValueError, match="一致しません"):
        load_model_feature_spec(_Folder([-1, 18, 2], SPEC))


def test_save_checks_shape_before_writing():
    """モデルの入力の形と合わない仕様は書かない."""
    folder = _Folder([-1, 18, 2])

    with pytest.raises(ValueError, match="一致しません"):
        save_model_feature_spec(folder, SPEC)
    assert folder.spec is None

    folder = _Folder([-1, 18, 1])
    assert save_model_feature_spec(folder, SPEC) == Path("feature_spec.yaml")
    assert folder.spec == SPEC
