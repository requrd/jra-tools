"""モデルと一緒に持つ仕様を読み書きする手続きのユニットテスト. フォルダは偽物."""

import dataclasses
from pathlib import Path
from typing import List, Optional

import pytest

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.interface_adapter.spec_format import (
    parse_feature_spec,
)
from jra_tools.machine_learning.usecase.model_spec import (
    load_model_feature_spec,
    save_model_feature_spec,
)

OUTPUTS = ["o1", "o2", "o3", "o4", "o5"]
SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 18,
        "source": "features.entries_v1",
        "view_digest": "d" * 64,
        "groups": [{"name": "s", "transform": "none", "features": [{"name": "idm"}]}],
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
    """モデルの入力の形と合わない仕様は書かない. 書くときは出力の名前を写す."""
    folder = _Folder([-1, 18, 2])
    with pytest.raises(ValueError, match="一致しません"):
        save_model_feature_spec(folder, SPEC, OUTPUTS)
    assert folder.spec is None

    folder = _Folder([-1, 18, 1])
    assert save_model_feature_spec(folder, SPEC, OUTPUTS) == Path("feature_spec.yaml")
    assert folder.spec == dataclasses.replace(SPEC, output_names=tuple(OUTPUTS))


def test_save_needs_five_outputs_and_pinned_spec():
    """出力は1着→5着の5つ。固定していない仕様は書かない."""
    with pytest.raises(ValueError, match="5つ"):
        save_model_feature_spec(_Folder([-1, 18, 1]), SPEC, OUTPUTS[:3])
    unpinned = dataclasses.replace(SPEC, view_digest=None)
    with pytest.raises(ValueError, match="固定"):
        save_model_feature_spec(_Folder([-1, 18, 1]), unpinned, OUTPUTS)
