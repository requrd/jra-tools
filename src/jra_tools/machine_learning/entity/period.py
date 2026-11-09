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


# 年で指定できる範囲
MIN_YEAR = 2000
MAX_YEAR = 2100
QUARTERS = 4


def training_period(start_year: int, end_year: int) -> Tuple[int, int]:
    """開始年の1月1日から終了年の12月31日まで(YYYYMMDD).

    Raises:
        TypeError: 年が整数でない場合
        ValueError: 年が範囲外、または終了年が開始年より前の場合

    """
    _check_year(start_year)
    _check_year(end_year)
    if start_year > end_year:
        raise ValueError("終了年は開始年以降にしてください")
    return int(f"{start_year}0101"), int(f"{end_year}1231")


def quarter_period(year: int, quarter: int) -> Tuple[int, int]:
    """年の四半期(1〜4)の初日から末日まで(YYYYMMDD).

    Raises:
        TypeError: 年が整数でない場合
        ValueError: 年・四半期が範囲外の場合

    """
    _check_year(year)
    if not 1 <= quarter <= QUARTERS:
        raise ValueError("1 - 4を指定してください")
    first_month = 3 * (quarter - 1) + 1
    last_month = first_month + 2
    last_day = 30 if last_month in (6, 9) else 31
    return int(f"{year}{first_month:02}01"), int(f"{year}{last_month:02}{last_day}")


def _check_year(year: int) -> None:
    if not isinstance(year, int):
        raise TypeError("年は整数で指定してください")
    if not MIN_YEAR <= year <= MAX_YEAR:
        raise ValueError("2000年以降を指定してください")
