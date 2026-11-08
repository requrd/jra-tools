"""jrdb-modelの表から読んだ行の決まりのユニットテスト."""

import pytest

from jra_tools.machine_learning.entity.staging_rows import check_joined_rows


def test_passes_without_missing():
    """行のない出走がなければ通す."""
    check_joined_rows({"seiseki": [], "returninfo": [], "calculated_score": []})


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ({"seiseki": ["k1", "k2"], "returninfo": []}, "成績がありません: k1"),
        ({"seiseki": [], "returninfo": ["r1"]}, "払戻データがありません: r1"),
        ({"seiseki": [], "train_oikiri": ["k3"]}, "train_oikiri がありません: k3"),
    ],
)
def test_missing(missing, message):
    """最初に見つかった表の、最初のキーを出して止める."""
    with pytest.raises(ValueError, match=message):
        check_joined_rows(missing)
