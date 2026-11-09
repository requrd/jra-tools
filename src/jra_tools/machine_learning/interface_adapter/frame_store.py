"""ビューから読んだ表を、Parquetのファイルとして残す置き場(usecase/iframe_store.pyの実装).

ファイルの読み書きはinfrastructureのものを受け取る。ファイル名は「名前.parquet」。
"""

from typing import Mapping, Optional, Protocol, Tuple

import pandas as pd

from jra_tools.machine_learning.interface_adapter.frame_format import (
    frame_to_parquet,
    parquet_to_frame,
)


class IBinaryFiles(Protocol):
    """1つの場所のファイル(バイト列)."""

    def read_bytes(self, name: str) -> Optional[bytes]:
        """ファイルの中身. なければNone."""

    def write_bytes(self, name: str, data: bytes) -> None:
        """ファイルに書く."""


class ParquetFrameStore:
    """Parquetのファイルの置き場(IFrameStore)."""

    def __init__(self, files: IBinaryFiles):
        """ファイルの読み書きを受け取る."""
        self._files = files

    def read(self, name: str) -> Optional[Tuple[pd.DataFrame, Mapping[str, str]]]:
        """名前の表とメタデータ. なければNone."""
        data = self._files.read_bytes(f"{name}.parquet")
        return None if data is None else parquet_to_frame(data)

    def write(
        self, name: str, frame: pd.DataFrame, metadata: Mapping[str, str]
    ) -> None:
        """表をメタデータと一緒に残す."""
        self._files.write_bytes(f"{name}.parquet", frame_to_parquet(frame, metadata))
