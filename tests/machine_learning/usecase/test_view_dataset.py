"""ビューから読む手続きのユニットテスト. 読み手は偽物(DBを使わない).

Postgresを通す確認は、入口のtests/machine_learning/test_integration_features.pyと、
tests/machine_learning/infrastructure/test_integration_postgres_view_reader.py。
"""

import dataclasses
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from jra_tools.machine_learning.entity.feature_spec import parse_feature_spec
from jra_tools.machine_learning.entity.input_view import ENTRY_COLUMNS, LABEL_PRESENT
from jra_tools.machine_learning.usecase.view_dataset import (
    load_view_dataset,
    pin_view_digest,
    predict_finish_probabilities,
    read_view_frame,
)

VIEW = "sandbox.entries_try"
LABELS = "features.labels_v1"
DIGEST = "d" * 64
SPEC = parse_feature_spec(
    {
        "name": "sample",
        "max_horses": 4,
        "source": VIEW,
        "view_digest": DIGEST,
        "groups": [
            {
                "name": "scores",
                "transform": "standardize_per_race",
                "features": [{"name": "idm"}],
            }
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
        self.reader.calls.append(("columns", view))
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
        self.reader.calls.append(("read", view, tuple(columns), labels, start, end))
        rows = self.reader.rows
        frame = rows[(rows["ymd"] >= start) & (rows["ymd"] <= end)]
        keep = list(columns)
        if labels:
            keep += [LABEL_PRESENT, "finish_order", "win1_num", "win1_ret"]
        return frame[keep].reset_index(drop=True)


class _Reader:
    """呼ばれた順を残す読み手. sessionsは始めたセッションの数."""

    def __init__(self, rows: pd.DataFrame, digest: str = DIGEST):
        self.views: Dict[str, List[str]] = {
            VIEW: [*ENTRY_COLUMNS, "idm"],
            LABELS: ["racehorsekey", "finish_order", "win1_num", "win1_ret"],
        }
        self.rows = rows
        self.digest = digest
        self.calls: list = []
        self.sessions = 0

    @contextmanager
    def session(self):
        self.sessions += 1
        self.calls.append(("session",))
        yield _Session(self)


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
                    LABEL_PRESENT: key,
                    "finish_order": umaban,
                    "win1_num": 1,
                    "win1_ret": 200,
                }
            )
    return pd.DataFrame(records)


def test_reads_each_year_in_its_own_session():
    """1年ずつ、1年ごとに1つのセッションで、ロック→照合→読むの順に呼ぶ."""
    reader = _Reader(_rows())

    frame = read_view_frame(reader, SPEC, 20241201, 20250131, labels=LABELS)

    assert reader.sessions == 2
    first = reader.calls[: reader.calls.index(("session",), 1)]
    assert first == [
        ("session",),
        ("lock", (VIEW, LABELS)),
        ("columns", VIEW),
        ("digest", VIEW),
        ("columns", LABELS),
        (
            "read",
            VIEW,
            ("racehorsekey", "racekey", "umaban", "idm"),
            LABELS,
            20241201,
            20241231,
        ),
    ]
    assert reader.calls[-1][4:] == (20250101, 20250131)
    assert list(frame["racehorsekey"]) == ["r11", "r12", "r21", "r22", "r31", "r32"]
    assert LABEL_PRESENT not in frame.columns


def test_inference_reads_no_labels():
    """推論(labels=None)はラベルのビューをロックも読みもしない."""
    reader = _Reader(_rows())

    frame = read_view_frame(reader, SPEC, 20250105, 20250105, labels=None)

    assert ("lock", (VIEW,)) in reader.calls
    assert reader.calls[-1][3] is None
    assert "finish_order" not in frame.columns


def test_digest_mismatch_stops_before_reading():
    """定義が学習時と違えば、読まずに止める."""
    reader = _Reader(_rows(), digest="e" * 64)

    with pytest.raises(ValueError, match="学習時と違います"):
        read_view_frame(reader, SPEC, 20250105, 20250105)
    assert not [c for c in reader.calls if c[0] == "read"]


def test_missing_view_stops_before_locking():
    """ビューがなければ、ロックの前に止める."""
    reader = _Reader(_rows())
    del reader.views[LABELS]

    with pytest.raises(ValueError, match=f"ビューがありません: {LABELS}"):
        read_view_frame(reader, SPEC, 20250105, 20250105)
    assert not [c for c in reader.calls if c[0] == "lock"]


def test_pin_view_digest():
    """今のハッシュを仕様に書く."""
    reader = _Reader(_rows(), digest="f" * 64)

    spec = pin_view_digest(reader, dataclasses.replace(SPEC, view_digest=None))

    assert spec.view_digest == "f" * 64


def test_load_view_dataset():
    """読んだ表を、レースごとのデータセットにする."""
    dataset = load_view_dataset(_Reader(_rows()), SPEC, 20241201, 20250131)

    assert list(dataset.racekeys) == ["r1", "r2", "r3"]
    assert dataset.x.shape[:2] == (3, 4)


class _Model:
    """出力の値がレースの位置・枠で決まるモデル."""

    def __init__(self):
        self.called = False

    def predict(self, x, verbose="auto"):
        self.called = True
        base = np.arange(x.shape[0])[:, None] * 100 + np.arange(x.shape[1])[None, :]
        return {
            "activation_4": base + 0.1,
            "activation_5": base + 0.2,
            "activation_6": base + 0.3,
        }


def test_predict_finish_probabilities():
    """出走馬ごとに、その馬のレース・馬番の枠の確率を、読んだ順に返す."""
    actual = predict_finish_probabilities(_Reader(_rows()), SPEC, 20250105, _Model())

    assert actual == [
        ("r21", 0.1, 0.2, 0.3),
        ("r22", 1.1, 1.2, 1.3),
        ("r31", 100.1, 100.2, 100.3),
        ("r32", 101.1, 101.2, 101.3),
    ]


def test_predict_no_races():
    """出走のない日は、モデルを呼ばずに空を返す."""
    model = _Model()

    assert predict_finish_probabilities(_Reader(_rows()), SPEC, 20250106, model) == []
    assert not model.called
