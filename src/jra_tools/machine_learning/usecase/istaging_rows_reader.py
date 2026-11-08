"""jrdb-modelの表(staging)から出走の行を読むインターフェース(usecase/staging_dataset.pyが使う)."""

from typing import Dict, List, Protocol, Tuple

from jra_tools.machine_learning.entity.dataset import DatasetRows
from jra_tools.machine_learning.entity.feature_spec import FeatureSpec


class IStagingRowsReader(Protocol):
    """出走の行の読み手."""

    def read_rows(
        self, spec: FeatureSpec, start: int, end: int
    ) -> Tuple[DatasetRows, Dict[str, List[str]]]:
        """期間内(開催日、YYYYMMDD、両端を含む)の出走の行と、表ごとの行のない出走のキー.

        行のない出走のキーは、成績(seiseki)・払戻(returninfo。レースのキー)と、
        仕様が使う外部結合の表について返す(なければ空のリスト)。

        Returns:
            Tuple[DatasetRows, Dict[str, List[str]]]: 開催日・開催キー・レースキー・馬番の順の行と、
                表の名前 → 行のない出走のキー

        """
