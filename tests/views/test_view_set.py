"""ビューの定義の読み込みのテスト."""

from pathlib import Path

import pytest

from jra_tools.views import load_view_set, parse_view_set

FILES = {
    "functions.sql": "create function model.twice(x int) returns int"
    " language sql immutable return x * 2;",
    "model/races.sql": "select 1 as racekey",
    "model/entries.sql": "select racekey from model.races",
}


def _parse(data, files=FILES):
    return parse_view_set(data, files.__getitem__)


DATA = {
    "schemas": ["model"],
    "functions": ["functions.sql"],
    "views": ["model.races", "model.entries"],
}


def test_parse_keeps_order():
    """ビューはviews.yamlの順に並び、SQLは<スキーマ>/<ビュー名>.sqlから読む."""
    view_set = _parse(DATA)
    assert view_set.schemas == ("model",)
    assert [v.qualified_name for v in view_set.views] == [
        "model.races",
        "model.entries",
    ]
    assert view_set.views[1].sql == "select racekey from model.races"
    assert view_set.functions == (FILES["functions.sql"],)


def test_functions_can_be_omitted():
    """functionsは省略できる."""
    view_set = _parse({"schemas": ["model"], "views": ["model.races"]})
    assert view_set.functions == ()


def test_digest_changes_only_with_definitions():
    """ハッシュは定義が変わったときだけ変わる."""
    base = _parse(DATA).digest
    assert _parse(dict(DATA)).digest == base
    changed = {**FILES, "model/races.sql": "select 2 as racekey"}
    assert _parse(DATA, changed).digest != base
    reordered = {**DATA, "views": ["model.entries", "model.races"]}
    assert _parse(reordered).digest != base


@pytest.mark.parametrize(
    ("views", "message"),
    [
        (["races"], "スキーマ.ビュー名"),
        (["features.races"], "schemasにありません"),
        (["model.races", "model.races"], "同じビュー"),
    ],
)
def test_invalid_views(views, message):
    """ビューの指定の誤りを読み込むときに止める."""
    with pytest.raises(ValueError, match=message):
        _parse({"schemas": ["model"], "views": views})


def test_load_from_directory(tmp_path: Path):
    """フォルダから定義とテストを読む. テストはハッシュに含めない."""
    (tmp_path / "model").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "views.yaml").write_text(
        "schemas: [model]\nviews:\n  - model.races\n", encoding="utf-8"
    )
    (tmp_path / "model" / "races.sql").write_text(
        "select 1 as racekey\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "b_second.sql").write_text("select 1", encoding="utf-8")
    (tmp_path / "tests" / "a_first.sql").write_text("select 2", encoding="utf-8")
    view_set = load_view_set(tmp_path)
    assert [t.name for t in view_set.tests] == ["a_first", "b_second"]
    digest = view_set.digest
    (tmp_path / "tests" / "a_first.sql").write_text("select 3", encoding="utf-8")
    assert load_view_set(tmp_path).digest == digest
