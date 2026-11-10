"""ビューの反映の結合テスト. 環境変数TEST_POSTGRESのPostgresを使う(なければ飛ばす)."""

import logging
import os
import threading
import uuid
from typing import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DBAPIError, OperationalError, ProgrammingError

from jra_tools.machine_learning import features
from jra_tools.views import (
    ApplyOptions,
    ViewTest,
    ViewVersionTableMissing,
    applied_digest,
    apply_views,
    parse_view_set,
    replace_views,
    run_view_tests,
)

pytestmark = pytest.mark.skipif(
    "TEST_POSTGRES" not in os.environ, reason="TEST_POSTGRESがない"
)

FILES = {
    "functions.sql": "create or replace function model.twice(x int) returns int"
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
        for schema in ("model", "features", "meta", "staging", "sandbox"):
            conn.exec_driver_sql(f"drop schema if exists {schema} cascade")
        conn.exec_driver_sql("create schema staging")
        conn.exec_driver_sql(
            "create table staging.races (racekey text primary key, distance int, name text)"
        )
        conn.exec_driver_sql(
            "insert into staging.races values ('01', 1200, 'A S'), ('02', 1600, 'B')"
        )
        # 表を作るコマンド(sazanami-batchのapply-model-and-meta-tables)の代わり
        conn.exec_driver_sql("create schema meta")
        conn.exec_driver_sql(
            "create table meta.view_version ("
            " digest text not null,"
            " applied_at timestamptz not null default now(),"
            " label text not null default '')"
        )
    yield engine
    with engine.begin() as conn:
        for schema in ("model", "features", "meta", "staging", "sandbox"):
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


def test_stops_without_view_version_table(engine: Engine):
    """meta.view_versionがなければ、何も作らずに止まる."""
    with engine.begin() as conn:
        conn.exec_driver_sql("drop table meta.view_version")
    with pytest.raises(ViewVersionTableMissing, match="apply-model-and-meta-tables"):
        apply_views(engine, _view_set())
    with engine.connect() as conn:
        assert conn.execute(text("select to_regclass('model.races')")).scalar() is None


def test_keeps_tables_in_managed_schemas(engine: Engine):
    """作り直しても、管理するスキーマに置いた表と行は残り、表の上のビューも作り直せる."""
    apply_views(engine, _view_set())
    with engine.begin() as conn:
        conn.exec_driver_sql("create table model.predict (racekey text, p1 real)")
        conn.exec_driver_sql("insert into model.predict values ('01', 0.25)")
    on_table = {
        **FILES,
        "model/predict_odds.sql": "select racekey, 1 / p1 as odds from model.predict",
    }
    data = {**DATA, "views": [*DATA["views"], "model.predict_odds"]}
    assert apply_views(engine, parse_view_set(data, on_table.__getitem__))
    assert apply_views(engine, _view_set())
    with engine.connect() as conn:
        assert conn.execute(text("select * from model.predict")).all() == [("01", 0.25)]
        assert (
            conn.execute(text("select to_regclass('model.predict_odds')")).scalar()
            is None
        )


def test_drops_views_and_functions_removed_from_definition(engine: Engine):
    """定義から外したビュー・関数は消え、依存するほかのスキーマのビューも消える."""
    apply_views(engine, _view_set())
    with engine.begin() as conn:
        conn.exec_driver_sql("create schema sandbox")
        conn.exec_driver_sql(
            "create view sandbox.mine as select * from features.entries"
        )
    without = {
        "model/races.sql": "select racekey from staging.races",
        "features/entries.sql": "select racekey from model.races",
    }
    data = {"schemas": ["model", "features"], "views": DATA["views"]}
    assert apply_views(engine, parse_view_set(data, without.__getitem__))
    with engine.connect() as conn:
        assert (
            conn.execute(text("select to_regprocedure('model.twice(int)')")).scalar()
            is None
        )
        assert conn.execute(text("select to_regclass('sandbox.mine')")).scalar() is None


def test_stops_when_a_table_uses_a_function(engine: Engine):
    """表の列の既定値が使う関数を定義から外すと、黙って消さずに止まり、全体が戻る."""
    view_set = _view_set()
    apply_views(engine, view_set)
    with engine.begin() as conn:
        conn.exec_driver_sql("create table model.uses (x int default model.twice(1))")
    without = {
        "model/races.sql": "select racekey from staging.races",
        "features/entries.sql": "select racekey from model.races",
    }
    data = {"schemas": ["model", "features"], "views": DATA["views"]}
    with pytest.raises(DBAPIError):
        apply_views(engine, parse_view_set(data, without.__getitem__))
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from features.entries")).scalar() == 1
        assert applied_digest(conn) == view_set.digest


def _sandbox(engine: Engine, *views: str) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql("create schema if not exists sandbox")
        for sql in views:
            conn.exec_driver_sql(sql)


def _exists(engine: Engine, name: str) -> bool:
    with engine.connect() as conn:
        return (
            conn.execute(text("select to_regclass(:n)"), {"n": name}).scalar()
            is not None
        )


def test_keeps_sandbox_views_when_columns_stay(engine: Engine, caplog):
    """列が同じか末尾に足すだけなら、その場で置き換え、依存するsandboxのビューが残る."""
    apply_views(engine, _view_set())
    _sandbox(
        engine,
        "create view sandbox.mine as select * from features.entries",
        "create view sandbox.on_mine as select count(*) as n from sandbox.mine",
        "create view sandbox.uses_function as select model.twice(3) as six",
    )
    changed = {
        **FILES,
        "functions.sql": "create or replace function model.twice(x int) returns int"
        " language sql immutable return x + x;",
        "model/races.sql": "select racekey, model.twice(distance) as distance2,"
        " name from staging.races",
        "features/entries.sql": "select racekey, distance2 from model.races",
    }
    with caplog.at_level(logging.INFO, logger="jra_tools.views.apply"):
        assert apply_views(engine, _view_set(changed))
    assert "作り直します" not in caplog.text
    assert "消えます" not in caplog.text
    with engine.connect() as conn:
        assert conn.execute(text("select n from sandbox.on_mine")).scalar() == 2
        assert conn.execute(text("select six from sandbox.uses_function")).scalar() == 6
        columns = conn.execute(text("select * from features.entries")).keys()
        assert list(columns) == ["racekey", "distance2"]


def test_rebuilds_and_names_sandbox_views_when_columns_change(engine: Engine, caplog):
    """列を消すと作り直し、消えるsandboxのビュー(関数だけを使うものも)の名前をログに残す."""
    apply_views(engine, _view_set())
    _sandbox(
        engine,
        "create view sandbox.mine as select * from features.entries",
        "create view sandbox.on_mine as select count(*) as n from sandbox.mine",
        "create materialized view sandbox.snapshot as select * from model.races",
        "create view sandbox.uses_function as select model.twice(3) as six",
        "create view sandbox.unrelated as select 1 as one",
    )
    changed = {
        **FILES,
        "model/races.sql": "select racekey as key from staging.races",
        "features/entries.sql": "select key from model.races",
    }
    with caplog.at_level(logging.INFO, logger="jra_tools.views.apply"):
        assert apply_views(engine, _view_set(changed))
    assert "作り直します" in caplog.text
    dropped = [r.getMessage() for r in caplog.records if "消えます" in r.getMessage()]
    assert len(dropped) == 1
    for name in ("mine", "on_mine", "snapshot", "uses_function"):
        assert f"sandbox.{name}" in dropped[0]
        assert not _exists(engine, f"sandbox.{name}")
    assert "unrelated" not in dropped[0]
    assert _exists(engine, "sandbox.unrelated")


def test_rebuilds_when_functions_are_not_replaceable(engine: Engine, caplog):
    """CREATE OR REPLACEで書いていない関数は置き換えられないので、作り直す."""
    plain = {
        **FILES,
        "functions.sql": "create function model.twice(x int) returns int"
        " language sql immutable return x * 2;",
    }
    apply_views(engine, _view_set(plain))
    _sandbox(engine, "create view sandbox.mine as select * from features.entries")
    changed = {
        **plain,
        "features/entries.sql": "select racekey, 1 as x from model.races",
    }
    with caplog.at_level(logging.INFO, logger="jra_tools.views.apply"):
        assert apply_views(engine, _view_set(changed))
    assert "作り直します" in caplog.text
    assert "sandbox.mine" in caplog.text
    assert not _exists(engine, "sandbox.mine")


def test_names_sandbox_views_on_views_removed_from_definition(engine: Engine, caplog):
    """定義から外したビューに依存するsandboxのビューは消え、名前をログに残す. ほかは残る."""
    on_extra = {**FILES, "model/extra.sql": "select racekey from staging.races"}
    data = {**DATA, "views": [*DATA["views"], "model.extra"]}
    apply_views(engine, parse_view_set(data, on_extra.__getitem__))
    _sandbox(
        engine,
        "create view sandbox.on_extra as select * from model.extra",
        "create view sandbox.mine as select * from features.entries",
    )
    with caplog.at_level(logging.INFO, logger="jra_tools.views.apply"):
        assert apply_views(engine, _view_set())
    assert "作り直します" not in caplog.text
    assert "sandbox.on_extra" in caplog.text
    assert not _exists(engine, "sandbox.on_extra")
    assert not _exists(engine, "model.extra")
    assert _exists(engine, "sandbox.mine")


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


def test_drops_only_functions_removed_from_definition_in_place(engine: Engine, caplog):
    """その場で置き換えるときも、定義から外した関数だけを消し、sandboxのビューは残る."""
    two = {
        **FILES,
        "functions.sql": FILES["functions.sql"]
        + "\ncreate or replace function model.thrice(x int) returns int"
        " language sql immutable return x * 3;",
    }
    apply_views(engine, _view_set(two))
    _sandbox(engine, "create view sandbox.mine as select * from features.entries")
    without_thrice = {
        **FILES,
        "model/races.sql": "select racekey, distance * 2 as distance2"
        " from staging.races where name like '%S';",
    }
    with caplog.at_level(logging.INFO, logger="jra_tools.views.apply"):
        assert apply_views(engine, _view_set(without_thrice))
    assert "作り直します" not in caplog.text
    with engine.connect() as conn:
        assert (
            conn.execute(text("select to_regprocedure('model.thrice(int)')")).scalar()
            is None
        )
        assert (
            conn.execute(text("select to_regprocedure('model.twice(int)')")).scalar()
            is not None
        )
    assert _exists(engine, "sandbox.mine")


SANDBOX_DATA = {"schemas": ["sandbox"], "views": ["sandbox.a", "sandbox.b"]}
SANDBOX_FILES = {
    "sandbox/a.sql": "select racekey, distance from staging.races where name like '%S'",
    "sandbox/b.sql": "select count(*) as n from sandbox.a",
}


@pytest.fixture
def lab_engine(engine: Engine) -> Iterator[Engine]:
    """sazanami_labと同じ権限のロール(stagingを読み、sandboxに作れる。metaは読めず、スキーマは作れない)."""
    role = f"lab_{uuid.uuid4().hex[:8]}"
    secret = uuid.uuid4().hex
    with engine.begin() as conn:
        conn.exec_driver_sql(f"create role {role} login password '{secret}'")
        conn.exec_driver_sql("create schema sandbox")
        conn.exec_driver_sql(f"grant usage, create on schema sandbox to {role}")
        conn.exec_driver_sql(f"grant usage on schema staging to {role}")
        conn.exec_driver_sql(f"grant select on all tables in schema staging to {role}")
    lab = create_engine(engine.url.set(username=role, password=secret))
    yield lab
    lab.dispose()
    with engine.begin() as conn:
        conn.exec_driver_sql("drop schema if exists sandbox cascade")
        conn.exec_driver_sql(f"revoke all on schema staging from {role}")
        conn.exec_driver_sql(f"revoke all on all tables in schema staging from {role}")
        conn.exec_driver_sql(f"drop owned by {role}")
        conn.exec_driver_sql(f"drop role {role}")


def _sandbox_set(files=SANDBOX_FILES, data=SANDBOX_DATA):
    return parse_view_set(data, files.__getitem__)


def test_replace_views_without_view_version(lab_engine: Engine):
    """meta.view_versionを読めないロールでも反映でき、同じ定義でも毎回流す(記録しない)."""
    replace_views(lab_engine, _sandbox_set())
    with lab_engine.begin() as conn:
        assert conn.execute(text("select n from sandbox.b")).scalar() == 1
        conn.exec_driver_sql("drop view sandbox.b")
    replace_views(lab_engine, _sandbox_set())
    with lab_engine.connect() as conn:
        assert conn.execute(text("select n from sandbox.b")).scalar() == 1


def test_replace_views_drops_views_removed_from_list(lab_engine: Engine):
    """一覧から外したビュー(手で作ったものも)は消える."""
    replace_views(lab_engine, _sandbox_set())
    with lab_engine.begin() as conn:
        conn.exec_driver_sql("create view sandbox.by_hand as select 1 as one")
    replace_views(
        lab_engine, _sandbox_set(data={"schemas": ["sandbox"], "views": ["sandbox.a"]})
    )
    assert _exists(lab_engine, "sandbox.a")
    assert not _exists(lab_engine, "sandbox.b")
    assert not _exists(lab_engine, "sandbox.by_hand")


def test_replace_views_rebuilds_when_columns_change(lab_engine: Engine, caplog):
    """列を消すと、sandboxのビューを作り直す(表は残る)."""
    replace_views(lab_engine, _sandbox_set())
    with lab_engine.begin() as conn:
        conn.exec_driver_sql("create table sandbox.kept (x int)")
    changed = {
        **SANDBOX_FILES,
        "sandbox/a.sql": "select racekey from staging.races where name like '%S'",
    }
    with caplog.at_level(logging.INFO, logger="jra_tools.views.apply"):
        replace_views(lab_engine, _sandbox_set(changed))
    assert "作り直します" in caplog.text
    with lab_engine.connect() as conn:
        assert list(conn.execute(text("select * from sandbox.a")).keys()) == ["racekey"]
        assert conn.execute(text("select n from sandbox.b")).scalar() == 1
    assert _exists(lab_engine, "sandbox.kept")


def test_replace_views_keeps_current_views_on_error(lab_engine: Engine):
    """定義が誤っていれば全体が戻り、今のビューが残る."""
    replace_views(lab_engine, _sandbox_set())
    broken = {**SANDBOX_FILES, "sandbox/b.sql": "select no_such_column from sandbox.a"}
    with pytest.raises(ProgrammingError):
        replace_views(lab_engine, _sandbox_set(broken))
    with lab_engine.connect() as conn:
        assert conn.execute(text("select n from sandbox.b")).scalar() == 1


def test_apply_sandbox_views_reads_directory_and_env(
    lab_engine: Engine, tmp_path, monkeypatch
):
    """入口のapply_sandbox_viewsは、フォルダの定義を環境変数DBの接続先に反映する."""
    (tmp_path / "sandbox").mkdir()
    (tmp_path / "views.yaml").write_text(
        "schemas: [sandbox]\nviews: [sandbox.a, sandbox.b]\n", encoding="utf-8"
    )
    for path, sql in SANDBOX_FILES.items():
        (tmp_path / path).write_text(sql, encoding="utf-8")
    monkeypatch.setenv("DB", lab_engine.url.render_as_string(hide_password=False))
    features.apply_sandbox_views(tmp_path)
    with lab_engine.connect() as conn:
        assert conn.execute(text("select n from sandbox.b")).scalar() == 1
