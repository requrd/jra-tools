"""カテゴリの列の決まりのユニットテスト."""

import pandas as pd
import pytest

from jra_tools.machine_learning.entity.category import pick_categories, positions


def test_pick_categories_from_one_source():
    """項目ごとに、行がある1か所の「コード→番号」を返す. コードは文字列にそろえる."""
    rows = [
        ("model.code_categories", "waku", 1, 1),
        ("model.code_categories", "waku", "2", 1),
        ("features.code_categories", "waku_group", "1", 1),
    ]
    assert pick_categories(["waku", "waku_group"], rows) == {
        "waku": {"1": 1, "2": 1},
        "waku_group": {"1": 1},
    }


def test_pick_categories_stops_when_missing():
    """項目がどこにもなければ止める."""
    with pytest.raises(ValueError, match=r"項目がありません: \['course'\]"):
        pick_categories(["course"], [("model.code_categories", "waku", "1", 1)])


def test_pick_categories_stops_when_duplicated():
    """同じ項目が2か所以上にあれば止める(s1)."""
    rows = [
        ("model.code_categories", "waku", "1", 1),
        ("sandbox.code_categories", "waku", "1", 1),
    ]
    with pytest.raises(ValueError, match="2か所以上"):
        pick_categories(["waku"], rows)


def test_positions():
    """番号nは位置n - 1。控えにないコード・NULLは例外の列(最大の番号の位置)."""
    values = pd.Series(["1", "2", "3", "9", None, float("nan")], dtype=object)
    assert positions(values, {"1": 1, "2": 1, "3": 2}).tolist() == [0, 0, 1, 2, 2, 2]
