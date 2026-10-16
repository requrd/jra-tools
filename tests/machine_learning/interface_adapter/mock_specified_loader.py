from dataclasses import dataclass
from typing import List

from jrdb_model import KaisaiData


@dataclass
class MockSpecifiedLoader:
    start: int
    end: int

    @property
    def period(self) -> str:
        return "sample_period"

    def load(self) -> List[KaisaiData]:
        pass
