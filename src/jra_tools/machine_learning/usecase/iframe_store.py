"""ビューから読んだ表を残す置き場のインターフェース(usecase/view_dataset.pyが使う).

読み直しの時間を減らすため、ビューから読んだ1頭1行の表をそのまま残す(テンソルは毎回表から作る)。
"""

from typing import Mapping, Optional, Protocol, Tuple

import pandas as pd


class IFrameStore(Protocol):
    """表の置き場."""

    def read(self, name: str) -> Optional[Tuple[pd.DataFrame, Mapping[str, str]]]:
        """名前の表とメタデータ. なければNone."""

    def write(
        self, name: str, frame: pd.DataFrame, metadata: Mapping[str, str]
    ) -> None:
        """表をメタデータと一緒に残す(行の並びを保つ)."""
