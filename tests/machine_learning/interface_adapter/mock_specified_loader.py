"""SpecifiedLoaderのモック."""

from dataclasses import dataclass
from typing import List

from jrdb_model import KaisaiData


@dataclass
class MockSpecifiedLoader:
    """DBに接続せずに期間だけを保持するローダー."""

    start: int
    end: int

    @property
    def period(self) -> str:
        """固定の期間名を返す."""
        return "sample_period"

    def load(self) -> List[KaisaiData]:
        """何もしない."""
