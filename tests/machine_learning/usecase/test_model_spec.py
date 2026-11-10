"""モデルと一緒に持つ仕様を読み書きする手続きのユニットテスト. フォルダは偽物."""

import dataclasses
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pytest

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.interface_adapter.spec_format import (
    parse_feature_spec,
)
from jra_tools.machine_learning.usecase.model_spec import (
    load_model_feature_spec,
    save_model,
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
    def __init__(
        self,
        shape: List[int],
        spec: Optional[FeatureSpec] = None,
        outputs: Sequence[str] = tuple(OUTPUTS),
    ):
        self.shape = shape
        self.spec = spec
        self.outputs = list(outputs)
        self.model: Any = None

    def input_shape(self) -> List[int]:
        return self.shape

    def output_names(self) -> List[str]:
        return self.outputs

    def write_model(self, model: Any, output_names: Sequence[str]) -> None:
        self.model = model
        self.shape = [-1, *model.input_shape[1:]]
        self.outputs = list(output_names)

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


def test_load_checks_output_names():
    """仕様の出力の名前で、フォルダのモデルの出力を引けなければ止める(順は問わない)."""
    named = dataclasses.replace(SPEC, output_names=tuple(OUTPUTS))
    assert load_model_feature_spec(_Folder([-1, 18, 1], named, OUTPUTS[::-1])) == named
    with pytest.raises(ValueError, match="出力の名前"):
        load_model_feature_spec(
            _Folder([-1, 18, 1], named, [f"output_{i}" for i in range(5)])
        )


def test_save_spec_checks_output_names():
    """フォルダのモデルの出力の名前と合わない名前は書かない."""
    folder = _Folder([-1, 18, 1], outputs=[f"output_{i}" for i in range(5)])
    with pytest.raises(ValueError, match="出力の名前"):
        save_model_feature_spec(folder, SPEC, OUTPUTS)
    assert folder.spec is None


class _Model:
    def __init__(self, size: int):
        self.input_shape = (None, 18, size)
        self.output_names = OUTPUTS


def test_save_model_writes_model_and_spec():
    """モデルを出力の名前で書き、その名前を写した仕様を書く."""
    folder = _Folder([], outputs=[])
    model = _Model(1)
    assert save_model(folder, model, SPEC) == Path("feature_spec.yaml")
    assert folder.model is model
    assert folder.outputs == OUTPUTS
    assert folder.spec == dataclasses.replace(SPEC, output_names=tuple(OUTPUTS))


def test_save_model_checks_shape_before_writing():
    """仕様と合わない入力の形のモデルは、モデルも仕様も書かない."""
    folder = _Folder([], outputs=[])
    with pytest.raises(ValueError, match="一致しません"):
        save_model(folder, _Model(2), SPEC)
    assert folder.model is None
    assert folder.spec is None
