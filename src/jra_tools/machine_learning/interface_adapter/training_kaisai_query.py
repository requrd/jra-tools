"""トレーニングデータ取得用クエリ."""

from typing import Callable, List

from jrdb_model import KaisaiData

from jra_tools.machine_learning.interface_adapter.ispecified_loader import (
    SpecifiedLoader,
)

MIN_YEAR = 2000
MAX_YEAR = 2100
# 期間名を"data"とする従来のデータセットの期間
LEGACY_PERIOD = (2012, 2018)


def _validate_year(year: int):
    if not isinstance(year, int):
        raise TypeError("年は整数で指定してください")
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError("2000年以降を指定してください")


class TrainingKaisaiQuery:
    """トレーニングデータ用クエリ."""

    def __init__(
        self,
        start: int,
        end: int,
        loader_factory: Callable[[int, int], SpecifiedLoader],
    ):
        """開始年・終了年を検証し、期間に対応するローダーを作る."""
        _validate_year(start)
        _validate_year(end)
        if start > end:
            raise ValueError("終了年は開始年以降にしてください")
        self.start = start
        self.end = end
        self.loader = loader_factory(self._start, self._end)

    def load(self) -> List[KaisaiData]:
        """期間内の開催データを取得する."""
        return self.loader.load()

    @property
    def _start(self):
        return int(f"{self.start}0101")

    @property
    def _end(self):
        return int(f"{self.end}1231")

    @property
    def period(self):
        """データセットの期間."""
        return (
            "data"
            if (self.start, self.end) == LEGACY_PERIOD
            else f"{self.start}_{self.end}"
        )
