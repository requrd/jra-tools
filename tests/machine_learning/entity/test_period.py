"""期間を1年ずつに分けるユニットテスト."""

import pytest

from jra_tools.machine_learning.entity.period import year_periods


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
