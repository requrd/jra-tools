"""開催データに紐づくデータを結合した状態でロードする."""

from dataclasses import dataclass
from typing import List

from jrdb_model import KaisaiData, create_app
from sqlalchemy.orm import joinedload

from jra_tools.machine_learning.entity.period import year_periods


def get_kaisais(start: int, end: int) -> List[KaisaiData]:
    """指定した期間の開催データを取得する.

    Args:
        start (int): YYYYMMDD
        end (int): YYYYMMDD

    Returns:
        list[KaisaiData]: 開催データ一覧

    """
    app = create_app()
    with app.app_context():
        return (
            KaisaiData.query.filter(KaisaiData.ymd >= start, KaisaiData.ymd <= end)
            .options(joinedload("*"))
            .all()
        )


@dataclass
class MysqlLoader:
    """MySQLからエンティティを取得する."""

    start: int
    end: int

    def load(self) -> List[KaisaiData]:
        """期間内の開催データを取得する."""
        return load(self.start, self.end)

    @property
    def period(self) -> str:
        """データセットの期間."""
        return f"{self.start}_{self.end}"


def load(start: int, end: int) -> List[KaisaiData]:
    """データを1年毎にまとめてロードして、マージ結果を取得する.

    Args:
        start (int): YYYYMMDD
        end (int): YYYYMMDD

    Returns:
        List[KaisaiData]: 開催データ一覧

    """
    return [
        kaisai
        for start_date, end_date in year_periods(start, end)
        for kaisai in get_kaisais(start_date, end_date)
    ]
