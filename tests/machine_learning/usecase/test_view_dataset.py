"""ビューから読む手続きのユニットテスト. 読み手・置き場・モデルは偽物(DBを使わない).

Postgresを通す確認は、入口のtests/machine_learning/test_integration_features.pyと、
tests/machine_learning/interface_adapter/test_integration_view_gateway.py。
"""

import dataclasses
from contextlib import contextmanager
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

from jra_tools.machine_learning.entity.category import CategoryRow
from jra_tools.machine_learning.entity.input_view import ENTRY_COLUMNS, LABEL_PRESENT
from jra_tools.machine_learning.interface_adapter.spec_format import (
    parse_feature_spec,
)
from jra_tools.machine_learning.usecase.view_dataset import (
    PROBABILITY_COLUMNS,
    evaluate_period,
    load_view_dataset,
    pin_spec,
    predict_finish_probabilities,
    read_view_frame,
)

VIEW = "sandbox.entries_try"
LABELS = "features.labels_v1"
DIGEST = "d" * 64
OUTPUTS = ("o1", "o2", "o3", "o4", "o5")
RAW_SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 4,
        "source": VIEW,
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm"}],
            },
            {
                "name": "categories",
                "transform": "none",
                "features": [{"name": "waku", "encode": "category"}],
            },
        ],
    }
)
WAKU = {"1": 1, "2": 2}
SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 4,
        "source": VIEW,
        "view_digest": DIGEST,
        "output_names": list(OUTPUTS),
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm"}],
            },
            {
                "name": "categories",
                "transform": "none",
                "features": [
                    {"name": "waku", "encode": "category", "categories": WAKU}
                ],
            },
        ],
    }
)


class _Session:
    def __init__(self, reader: "_Reader"):
        self.reader = reader

    def exists(self, view: str) -> bool:
        return view in self.reader.views

    def lock(self, views: Sequence[str]) -> None:
        self.reader.calls.append(("lock", tuple(views)))

    def columns(self, view: str) -> List[str]:
        return self.reader.views[view]

    def digest(self, view: str) -> str:
        self.reader.calls.append(("digest", view))
        return self.reader.digest

    def read(
        self,
        view: str,
        columns: Sequence[str],
        labels: Optional[str],
        start: int,
        end: int,
    ) -> pd.DataFrame:
        self.reader.calls.append(("read", start, end, labels))
        rows = self.reader.rows
        frame = rows[(rows["ymd"] >= start) & (rows["ymd"] <= end)]
        keep = list(columns)
        if labels:
            keep += [LABEL_PRESENT, "finish_order"]
        return frame[keep].reset_index(drop=True)

    def code_categories(
        self, sources: Sequence[str], items: Sequence[str]
    ) -> List[CategoryRow]:
        self.reader.calls.append(("code_categories", tuple(sources), tuple(items)))
        return [r for r in self.reader.categories if r[0] in sources and r[1] in items]

    def incomplete_years(self, start: int, end: int) -> List[int]:
        return self.reader.incomplete


class _Reader:
    """呼ばれた順を残す読み手. sessionsは始めたセッションの数."""

    def __init__(self, rows: pd.DataFrame, digest: str = DIGEST):
        self.views: Dict[str, List[str]] = {
            VIEW: [*ENTRY_COLUMNS, "idm", "waku"],
            LABELS: ["racehorsekey", "finish_order"],
        }
        self.rows = rows
        self.digest = digest
        self.categories: List[CategoryRow] = [
            ("model.code_categories", "waku", "1", 1),
            ("model.code_categories", "waku", "2", 2),
        ]
        self.incomplete: List[int] = []
        self.calls: list = []
        self.sessions = 0

    @contextmanager
    def session(self):
        self.sessions += 1
        yield _Session(self)


class _Store:
    def __init__(self):
        self.saved: Dict[str, Tuple[pd.DataFrame, Mapping[str, str]]] = {}

    def read(self, name: str):
        return self.saved.get(name)

    def write(self, name: str, frame: pd.DataFrame, metadata: Mapping[str, str]):
        self.saved[name] = (frame.copy(), dict(metadata))


class _Model:
    """出力ごとに、枠の番号(0から)を値にした確率を返すモデル."""

    def __init__(self, as_dict: bool = True):
        self.as_dict = as_dict
        self.evaluated: list = []

    def predict(self, x: np.ndarray, verbose=0):
        outputs = [
            np.tile(np.arange(x.shape[1], dtype=float) + 10 * i, (x.shape[0], 1))
            for i in range(len(OUTPUTS))
        ]
        return dict(zip(OUTPUTS, outputs, strict=True)) if self.as_dict else outputs

    def evaluate(self, x, y, verbose=0, return_dict=True):
        self.evaluated.append((x.shape, len(y)))
        return {"loss": 1.5, "o1_loss": 0.3, "o1_accuracy": 0.25}


def _rows() -> pd.DataFrame:
    """2年にまたがる3レース(2頭ずつ)."""
    races = [(20241228, "r1"), (20250105, "r2"), (20250105, "r3")]
    records = []
    for ymd, racekey in races:
        for umaban in (1, 2):
            key = f"{racekey}{umaban}"
            records.append(
                {
                    "ymd": ymd,
                    "racehorsekey": key,
                    "racekey": racekey,
                    "umaban": umaban,
                    "idm": float(umaban),
                    "waku": str(umaban),
                    LABEL_PRESENT: key,
                    "finish_order": umaban,
                }
            )
    return pd.DataFrame(records)


def test_pin_spec_writes_digest_and_categories():
    """仕様の固定で、ビューのハッシュとカテゴリの控え(コード→番号)を写す."""
    reader = _Reader(_rows())
    pinned = pin_spec(reader, RAW_SPEC)
    assert pinned.view_digest == DIGEST
    [waku] = pinned.category_features
    assert dict(waku.categories) == WAKU
    assert waku.size == 3
    # 既定の探す先(model・features・sandbox)を渡している
    [call] = [c for c in reader.calls if c[0] == "code_categories"]
    assert call[1] == (
        "model.code_categories",
        "features.code_categories",
        "sandbox.code_categories",
    )


def test_pin_spec_stops_when_item_is_missing_or_duplicated():
    """項目がどこにもない・2か所以上にあれば止める(s1)."""
    reader = _Reader(_rows())
    reader.categories = []
    with pytest.raises(ValueError, match="項目がありません"):
        pin_spec(reader, RAW_SPEC)
    reader.categories = [
        ("model.code_categories", "waku", "1", 1),
        ("sandbox.code_categories", "waku", "1", 1),
    ]
    with pytest.raises(ValueError, match="2か所以上"):
        pin_spec(reader, RAW_SPEC)


def test_pin_spec_uses_given_sources():
    """探す先は引数で替えられる."""
    reader = _Reader(_rows())
    reader.categories = [("test.codes", "waku", "1", 1)]
    pinned = pin_spec(reader, RAW_SPEC, category_sources=["test.codes"])
    assert dict(pinned.category_features[0].categories) == {"1": 1}


def test_read_needs_pinned_spec():
    """固定していない仕様では読まない."""
    with pytest.raises(ValueError, match="固定"):
        read_view_frame(_Reader(_rows()), RAW_SPEC, 20241201, 20250131)


def test_read_by_year_locks_before_digest():
    """1年ずつ1セッションで読み、ロックしてからハッシュを照合する."""
    reader = _Reader(_rows())
    frame = read_view_frame(reader, SPEC, 20241201, 20250131)
    assert reader.sessions == 2
    assert len(frame) == 6
    assert "finish_order" in frame.columns
    assert LABEL_PRESENT not in frame.columns
    first_lock = reader.calls.index(("lock", (VIEW, LABELS)))
    first_digest = reader.calls.index(("digest", VIEW))
    assert first_lock < first_digest


def test_read_stops_on_digest_mismatch():
    """ビューの定義が学習時と違えば止める."""
    with pytest.raises(ValueError, match="学習時と違います"):
        read_view_frame(_Reader(_rows(), digest="e" * 64), SPEC, 20250101, 20250131)


def test_null_category_is_allowed_but_null_number_stops():
    """カテゴリの列の値なしは許し(例外の列)、数値の列の値なしは止める."""
    rows = _rows()
    rows.loc[0, "waku"] = None
    frame = read_view_frame(_Reader(rows), SPEC, 20241201, 20250131)
    assert frame["waku"].isna().sum() == 1
    rows.loc[0, "idm"] = None
    with pytest.raises(ValueError, match="idmが値なし"):
        read_view_frame(_Reader(rows), SPEC, 20241201, 20250131)


def test_read_stops_without_result():
    """ラベルを読むとき、成績(ラベルのビューの行)がない出走で止める."""
    rows = _rows()
    rows.loc[1, LABEL_PRESENT] = None
    with pytest.raises(ValueError, match="成績がありません: r12"):
        read_view_frame(_Reader(rows), SPEC, 20241201, 20250131)


def test_store_keeps_and_reuses_frame():
    """置き場があれば読んだ表を残し、次はビューを読まずにそれを使う."""
    reader, store = _Reader(_rows()), _Store()
    first = read_view_frame(reader, SPEC, 20241201, 20250131, store=store)
    [(name, (_, metadata))] = store.saved.items()
    assert name == "sample_20241201_20250131"
    assert metadata["input_digest"] == SPEC.input_digest()
    sessions = reader.sessions
    second = read_view_frame(reader, SPEC, 20241201, 20250131, store=store)
    assert reader.sessions == sessions
    pd.testing.assert_frame_equal(first, second)


def test_store_stops_when_saved_frame_is_for_another_spec():
    """残した表が今の仕様のものでなければ止める."""
    store = _Store()
    read_view_frame(_Reader(_rows()), SPEC, 20241201, 20250131, store=store)
    other = dataclasses.replace(SPEC, view_digest="e" * 64)
    with pytest.raises(ValueError, match="今の仕様・期間のものではありません"):
        read_view_frame(_Reader(_rows()), other, 20241201, 20250131, store=store)


def test_load_view_dataset_has_no_payout():
    """学習用のデータセットは入力とラベルだけ(払戻は持たない)."""
    dataset = load_view_dataset(_Reader(_rows()), SPEC, 20241201, 20250131)
    assert dataset.x.shape == (3, 4, 4)
    assert len(dataset.labels) == 5
    assert not hasattr(dataset, "payout")


@pytest.mark.parametrize("as_dict", [True, False])
def test_predict_finish_probabilities_uses_output_names(as_dict: bool):
    """出力の名前の順(仕様のoutput_names)で、出走馬ごとの1〜3着の確率にする."""
    result = predict_finish_probabilities(
        _Reader(_rows()), SPEC, 20250105, _Model(as_dict)
    )
    assert result == [
        ("r21", 0.0, 10.0, 20.0),
        ("r22", 1.0, 11.0, 21.0),
        ("r31", 0.0, 10.0, 20.0),
        ("r32", 1.0, 11.0, 21.0),
    ]


def test_predict_without_entries_is_empty():
    """出走がない日は空(モデルを呼ばない)."""
    assert predict_finish_probabilities(_Reader(_rows()), SPEC, 20250106, None) == []


def test_predict_needs_output_names():
    """仕様に出力の名前がなければ止める."""
    spec = dataclasses.replace(SPEC, output_names=None)
    with pytest.raises(ValueError, match="出力の名前"):
        predict_finish_probabilities(_Reader(_rows()), spec, 20250105, _Model())


def test_evaluate_period_returns_predictions_and_metrics():
    """評価期間の1〜5着の確率と、出力ごとのloss・指標を表で返す."""
    model = _Model()
    predictions, metrics = evaluate_period(
        _Reader(_rows()), SPEC, 20250101, 20250131, model
    )
    assert list(predictions.columns) == ["racehorsekey", *PROBABILITY_COLUMNS]
    assert predictions["racehorsekey"].tolist() == ["r21", "r22", "r31", "r32"]
    assert predictions["fifth_prob"].tolist() == [40.0, 41.0, 40.0, 41.0]
    assert model.evaluated == [((2, 4, 4), 5)]
    assert metrics.to_dict("records") == [
        {"output": "total", "metric": "loss", "value": 1.5},
        {"output": "o1", "metric": "loss", "value": 0.3},
        {"output": "o1", "metric": "accuracy", "value": 0.25},
    ]


def test_evaluate_period_stops_on_incomplete_year():
    """評価期間に締めていない年があれば止める."""
    reader = _Reader(_rows())
    reader.incomplete = [2025]
    with pytest.raises(ValueError, match="締めていない年"):
        evaluate_period(reader, SPEC, 20250101, 20250131, _Model())
