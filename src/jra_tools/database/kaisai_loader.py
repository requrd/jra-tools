"""jrdb-modelの開催データを、紐づく表ごと読み込む."""

from typing import List

from jrdb_model import KaisaiData, create_app
from sqlalchemy.orm import joinedload

__all__ = ["get_kaisais"]


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
