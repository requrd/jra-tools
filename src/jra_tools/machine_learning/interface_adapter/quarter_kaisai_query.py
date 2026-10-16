"""四半期データ用クエリ."""

from typing import Callable, List

from jrdb_model import KaisaiData

from jra_tools.machine_learning.interface_adapter.ispecified_loader import (
    SpecifiedLoader,
)

MIN_YEAR = 2000
MAX_YEAR = 2100
QUARTERS = 4


class QuarterKaisaiQuery:
    """四半期データ用クエリ."""

    def __init__(
        self,
        year: int,
        quarter: int,
        loader_factory: Callable[[int, int], SpecifiedLoader],
    ):
        """年・四半期を検証し、期間に対応するローダーを作る."""
        if not isinstance(year, int):
            raise TypeError("年は整数で指定してください")
        if not MIN_YEAR <= year <= MAX_YEAR:
            raise ValueError("2000年以降を指定してください")
        self.year = year
        if not 1 <= quarter <= QUARTERS:
            raise ValueError("1 - 4を指定してください")
        self.quarter = quarter
        self.loader = loader_factory(self._start, self._end)

    def load(self) -> List[KaisaiData]:
        """四半期内の開催データを取得する."""
        return self.loader.load()

    @property
    def _start(self) -> int:
        return int(f"{self.year}{self._start_month:02}01")

    @property
    def _end(self) -> int:
        return int(
            f"{self.year}{self._end_month:02}{30 if self._end_month in [2, 4, 6, 9, 11] else 31}"
        )

    @property
    def period(self) -> str:
        """データセットの期間."""
        return f"{self.year}_{self._start_month:02}_{self._end_month:02}"

    @property
    def _start_month(self) -> int:
        return int(12 / 4 * (self.quarter - 1) + 1)

    @property
    def _end_month(self) -> int:
        return self._start_month + 2
