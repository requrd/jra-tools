"""ビューの反映の結合テスト. 環境変数TEST_POSTGRESのPostgresを使う(なければ飛ばす)."""

import os
import threading
import uuid
from typing import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from jra_tools.views import (
    ApplyOptions,
    ViewTest,
    applied_digest,
    apply_views,
    parse_view_set,
    run_view_tests,
)

pytestmark = pytest.mark.skipif(
    "TEST_POSTGRES" not in os.environ, reason="TEST_POSTGRESがない"
)

FILES = {
    "functions.sql": "create function model.twice(x int) returns int"
    " language sql immutable return x * 2;",
    "model/races.sql": "select racekey, model.twice(distance) as distance2"
    " from staging.races where name like '%S';",
    "features/entries.sql": "select racekey from model.races",
}
DATA = {
    "schemas": ["model", "features"],
    "functions": ["functions.sql"],
    "views": ["model.races", "features.entries"],
}


def _view_set(files=FILES, tests=()):
    return parse_view_set(DATA, files.__getitem__, tests)


@pytest.fixture
def engine() -> Iterator[Engine]:
    """stagingに小さな表を置いたPostgres. 終わったらスキーマを消す."""
    engine = create_engine(os.environ["TEST_POSTGRES"])
    with engine.begin() as conn:
        for schema in ("model", "features", "meta", "staging"):
            conn.exec_driver_sql(f"drop schema if exists {schema} cascade")
        conn.exec_driver_sql("create schema staging")
        conn.exec_driver_sql(
            "create table staging.races (racekey text primary key, distance int, name text)"
        )
        conn.exec_driver_sql(
            "insert into staging.races values ('01', 1200, 'A S'), ('02', 1600, 'B')"
        )
    yield engine
    with engine.begin() as conn:
        for schema in ("model", "features", "meta", "staging"):
            conn.exec_driver_sql(f"drop schema if exists {schema} cascade")
    engine.dispose()


def test_apply_creates_views_once(engine: Engine):
    """初回は作り、定義が同じなら何もしない."""
    view_set = _view_set()
    assert apply_views(engine, view_set, ApplyOptions(label="sha-1"))
    with engine.connect() as conn:
        rows = conn.execute(text("select * from model.races")).all()
        assert rows == [("01", 2400)]
        assert conn.execute(text("select count(*) from features.entries")).scalar() == 1
        assert applied_digest(conn) == view_set.digest
    assert not apply_views(engine, view_set, ApplyOptions(label="sha-2"))
    with engine.connect() as conn:
        labels = (
            conn.execute(text("select label from meta.view_version")).scalars().all()
        )
        assert labels == ["sha-1"]


def test_apply_replaces_changed_columns(engine: Engine):
    """定義が変われば、列の削除・改名を含めて作り直す."""
    apply_views(engine, _view_set())
    changed = {
        **FILES,
        "model/races.sql": "select racekey as key, name from staging.races",
        "features/entries.sql": "select key from model.races",
    }
    assert apply_views(engine, _view_set(changed))
    with engine.connect() as conn:
        columns = conn.execute(text("select * from model.races order by 1")).keys()
        assert list(columns) == ["key", "name"]


def test_failed_definition_keeps_current_views(engine: Engine):
    """stagingにない列を参照する定義は失敗し、今のビューと記録が残る."""
    view_set = _view_set()
    apply_views(engine, view_set)
    broken = {**FILES, "features/entries.sql": "select no_such_column from model.races"}
    with pytest.raises(ProgrammingError):
        apply_views(engine, _view_set(broken))
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from features.entries")).scalar() == 1
        assert applied_digest(conn) == view_set.digest


def test_readers_can_select(engine: Engine):
    """readersのロールは管理するスキーマのビューを読める."""
    role = f"reader_{uuid.uuid4().hex[:8]}"
    with engine.begin() as conn:
        conn.exec_driver_sql(f"create role {role}")
    try:
        apply_views(engine, _view_set(), ApplyOptions(readers=(role,)))
        with engine.connect() as conn:
            for schema, view in (("model", "races"), ("features", "entries")):
                assert conn.execute(
                    text("select has_table_privilege(:role, :view, 'select')"),
                    {"role": role, "view": f"{schema}.{view}"},
                ).scalar()
    finally:
        with engine.begin() as conn:
            for schema in ("model", "features"):
                conn.exec_driver_sql(f"drop schema if exists {schema} cascade")
            conn.exec_driver_sql(f"drop role {role}")


def test_run_view_tests(engine: Engine):
    """行を返したテストを失敗として返す."""
    tests = (
        ViewTest("passes", "select * from model.races where racekey is null;"),
        ViewTest("fails", "select * from staging.races where name like '%'"),
    )
    view_set = _view_set(tests=tests)
    apply_views(engine, view_set)
    failures = run_view_tests(engine, view_set)
    assert [(f.name, f.rows) for f in failures] == [("fails", 2)]


def test_gives_up_when_readers_hold_the_lock(engine: Engine):
    """読んでいる最中のトランザクションがあれば、lock_timeoutで諦めてやり直し、最後は失敗する."""
    apply_views(engine, _view_set())
    changed = {
        **FILES,
        "features/entries.sql": "select racekey, 1 as x from model.races",
    }
    with engine.connect() as reader:
        reader.execute(text("select * from features.entries")).all()
        with pytest.raises(OperationalError):
            apply_views(
                engine,
                _view_set(changed),
                ApplyOptions(lock_timeout="100ms", retries=1, retry_wait=0),
            )
        reader.rollback()
    assert apply_views(engine, _view_set(changed), ApplyOptions(lock_timeout="100ms"))


def test_concurrent_applies_run_one_at_a_time(engine: Engine):
    """同時に流しても、片方が作り直し、もう片方は何もしない."""
    view_set = _view_set()
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(apply_views(engine, view_set)))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False, True]
