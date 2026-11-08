"""CategoryGetterのユニットテスト."""

import numpy as np
import pytest

from jra_tools.machine_learning.entity.jrdbdummies import CategoryGetter


@pytest.mark.parametrize(
    ("code", "position"),
    [(0, 1), (1, 2), (7, 8), (11, 9), (16, 14), (21, 15)],
)
def test_restreasoncode(code: int, position: int):
    """休養理由分類コードが15種類のone-hotベクトルに変換されることを確認する."""
    expected = np.zeros(15)
    expected[position - 1] = 1
    np.testing.assert_array_equal(CategoryGetter().getRestreasoncode(code), expected)


@pytest.mark.parametrize("code", [8, 10, 17, 20, 22])
def test_restreasoncode_unknown(code: int):
    """コード表にない値はKeyErrorになることを確認する."""
    with pytest.raises(KeyError):
        CategoryGetter().getRestreasoncode(code)
