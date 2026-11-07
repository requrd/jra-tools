"""読み込む期間を1年ずつに分ける.

数年分を一度に読むとメモリを大きく使うため、どの経路(ORM・SQL・ビュー)も1年ずつ読む。
日付はYYYYMMDDの整数で受け渡す。
"""

import datetime as dt
from typing import List, Tuple


def year_periods(start: int, end: int) -> List[Tuple[int, int]]:
    """期間を、年ごとの(開始日, 終了日)に分ける. 両端を含む.

    Args:
        start (int): 開始日(YYYYMMDD)
        end (int): 終了日(YYYYMMDD)

    Returns:
        List[Tuple[int, int]]: 年ごとの(開始日, 終了日)(YYYYMMDD)。年の順

    Raises:
        RuntimeError: 開始日が終了日より後の場合

    """
    s, e = to_date(start), to_date(end)
    if s > e:
        raise RuntimeError("開始日 <= 終了日となるように入力してください")
    return [
        (
            to_ymd(max(s, dt.date(year, 1, 1))),
            to_ymd(min(e, dt.date(year, 12, 31))),
        )
        for year in range(s.year, e.year + 1)
    ]


def to_date(ymd: int) -> dt.date:
    """YYYYMMDDの整数を日付にする."""
    return dt.datetime.strptime(str(ymd), "%Y%m%d").date()


def to_ymd(date: dt.date) -> int:
    """日付をYYYYMMDDの整数にする."""
    return int(date.strftime("%Y%m%d"))
