"""期間を1年ずつに分けるユニットテスト."""

import pytest

from jra_tools.machine_learning.entity.period import (
    quarter_period,
    training_period,
    year_periods,
)


def test_same_year():
    """同じ年なら、そのまま1つ."""
    assert year_periods(20250301, 20250310) == [(20250301, 20250310)]


def test_one_day():
    """1日だけでもよい(推論)."""
    assert year_periods(20261108, 20261108) == [(20261108, 20261108)]


def test_years():
    """年をまたぐと、最初と最後の年は端まで、間の年はまるごと."""
    assert year_periods(20131005, 20160220) == [
        (20131005, 20131231),
        (20140101, 20141231),
        (20150101, 20151231),
        (20160101, 20160220),
    ]


@pytest.mark.parametrize(("start", "end"), [(20250310, 20250301), (20260101, 20251231)])
def test_reversed(start, end):
    """開始日が終了日より後なら止める."""
    with pytest.raises(RuntimeError, match="開始日 <= 終了日"):
        year_periods(start, end)


def test_training_period():
    """開始年の1月1日から終了年の12月31日まで."""
    assert training_period(2013, 2019) == (20130101, 20191231)


@pytest.mark.parametrize(
    ("quarter", "expected"),
    [
        (1, (20250101, 20250331)),
        (2, (20250401, 20250630)),
        (3, (20250701, 20250930)),
        (4, (20251001, 20251231)),
    ],
)
def test_quarter_period(quarter, expected):
    """四半期の初日から末日まで."""
    assert quarter_period(2025, quarter) == expected


@pytest.mark.parametrize(
    ("call", "error", "message"),
    [
        (lambda: training_period(2019, 2013), ValueError, "終了年は開始年以降"),
        (lambda: training_period(1999, 2013), ValueError, "2000年以降"),
        (lambda: training_period("2013", 2019), TypeError, "整数"),
        (lambda: quarter_period(2025, 5), ValueError, "1 - 4"),
    ],
)
def test_invalid_period(call, error, message):
    """年・四半期が範囲外なら止める."""
    with pytest.raises(error, match=message):
        call()
