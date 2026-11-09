"""ビューから読んだ表をParquetで残す置き場のテスト(ファイルはinfrastructureのDirectory)."""

from pathlib import Path

import pandas as pd

from jra_tools.machine_learning.infrastructure.files import Directory
from jra_tools.machine_learning.interface_adapter.frame_store import (
    ParquetFrameStore,
)


def test_round_trip_keeps_order_and_metadata(tmp_path: Path):
    """表の並び・値なし・メタデータを保って読み戻せる. なければNone."""
    store = ParquetFrameStore(Directory(tmp_path / "cache"))
    assert store.read("x") is None

    frame = pd.DataFrame(
        {
            "racehorsekey": ["b1", "a1", "a2"],
            "racekey": ["B", "A", "A"],
            "umaban": [1, 1, 2],
            "idm": [1.5, None, 3.0],
            "waku": ["1", None, "3"],
        }
    )
    store.write("x", frame, {"input_digest": "abc", "start": "20250101"})

    saved = store.read("x")
    assert saved is not None
    loaded, metadata = saved
    pd.testing.assert_frame_equal(loaded, frame)
    assert metadata == {"input_digest": "abc", "start": "20250101"}
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["x.parquet"]
