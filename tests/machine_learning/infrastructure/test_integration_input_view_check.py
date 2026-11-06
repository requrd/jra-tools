"""入力のビューの確認の結合テスト. 環境変数TEST_POSTGRESのPostgresを使う(なければ飛ばす)."""

import os
from typing import Iterator, List

import pytest
from jrdb_model import KaisaiData, db
from sqlalchemy import Engine, create_engine

from jra_tools.machine_learning.infrastructure.input_view_check import (
    InputViewProblem,
    check_input_view,
)
from tests.machine_learning.infrastructure.conftest import sample_rows
from tests.machine_learning.infrastructure.test_integration_view_dataset import (
    ENTRIES_SQL,
    FUNCTION_SQL,
)

pytestmark = pytest.mark.skipif(
    "TEST_POSTGRES" not in os.environ, reason="TEST_POSTGRESがない"
)

SCHEMAS = ("staging", "model", "features", "sandbox")

# model層の出走馬(確認の比べる相手)
MODEL_SQL = (
    "create view model.races as select b.racekey, model.jrdb_date(b.ymd) as race_date"
    " from staging.bangumi b",
    "create view model.entries as select r.racehorsekey, r.racekey"
    " from staging.racehorse r",
)


def _drop(engine: Engine) -> None:
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.exec_driver_sql(f"drop schema if exists {schema} cascade")


@pytest.fixture
def engine(sample_kaisais: List[KaisaiData]) -> Iterator[Engine]:
    """sample_kaisaisをstagingに登録し、model層の出走馬とfeatures.entries_v1を作ったPostgres."""
    engine = create_engine(
        os.environ["TEST_POSTGRES"],
        connect_args={"options": "-csearch_path=staging"},
    )
    _drop(engine)
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.exec_driver_sql(f"create schema {schema}")
        db.metadata.create_all(conn)
        conn.exec_driver_sql("set local session_replication_role = replica")
        for name, rows in sample_rows(sample_kaisais).items():
            conn.execute(db.metadata.tables[name].insert(), rows)
        conn.exec_driver_sql(FUNCTION_SQL)
        for sql in MODEL_SQL:
            conn.exec_driver_sql(sql)
        conn.exec_driver_sql(ENTRIES_SQL)
    yield engine
    _drop(engine)
    engine.dispose()


@pytest.fixture
def period(sample_kaisais: List[KaisaiData]):
    """サンプルの開催の初日・最終日."""
    days = [kaisai.ymd for kaisai in sample_kaisais]
    return min(days), max(days)


def _sandbox(engine: Engine, sql: str) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(f"create view sandbox.try as {sql}")


def _checks(problems: List[InputViewProblem]) -> List[str]:
    return [p.check for p in problems]


def test_passes_for_entries_v1(engine: Engine, period):
    """本番の入力のビューは、すべて通る(サンプルの期間)."""
    assert check_input_view(engine, "features.entries_v1", *period) == []


def test_passes_for_a_sandbox_copy_with_a_new_column(engine: Engine, period):
    """sandboxで列を足した試作も、同じ確認で通る."""
    _sandbox(engine, "select e.*, e.idm * 2 as idm2 from features.entries_v1 e")
    assert check_input_view(engine, "sandbox.try", *period) == []


def test_finds_duplicates(engine: Engine, period):
    """出走馬が2行になる結合を見つける."""
    _sandbox(
        engine,
        "select * from features.entries_v1"
        " union all select * from features.entries_v1 where umaban = 1",
    )
    problems = check_input_view(engine, "sandbox.try", *period)
    assert _checks(problems) == ["unique"]
    assert problems[0].rows > 0
    assert len(problems[0].examples) <= 5


def test_finds_entries_dropped_by_an_inner_join(engine: Engine, period):
    """内部結合で黙って消えた出走馬を、model層の出走馬と比べて見つける."""
    _sandbox(engine, "select * from features.entries_v1 where umaban <> 1")
    assert _checks(check_input_view(engine, "sandbox.try", *period)) == [
        "covers_entries"
    ]


def test_finds_nulls_in_feature_columns(engine: Engine, period):
    """特徴量の列の値なしを、列ごとに見つける. キーの列は対象にしない."""
    _sandbox(
        engine,
        "select e.*, case when e.umaban = 1 then null else 1.0 end as maybe"
        " from features.entries_v1 e",
    )
    assert _checks(check_input_view(engine, "sandbox.try", *period)) == [
        "not_null:maybe"
    ]


def test_checks_only_given_features(engine: Engine, period):
    """特徴量の列を指定すれば、その列だけの値なしを見る."""
    _sandbox(
        engine,
        "select e.*, null::float as unused from features.entries_v1 e",
    )
    assert check_input_view(engine, "sandbox.try", *period, features=["idm"]) == []


def test_reports_missing_columns_only(engine: Engine, period):
    """キーの列・指定した列が足りなければ、それだけを返す."""
    _sandbox(engine, "select racehorsekey, racekey, umaban from features.entries_v1")
    problems = check_input_view(engine, "sandbox.try", *period, features=["idm"])
    assert problems == [InputViewProblem("columns", 2, ("race_date", "idm"))]


def test_rejects_label_columns(engine: Engine, period):
    """確定後の値(ラベルの列)は特徴量にできない."""
    _sandbox(engine, "select e.*, 1 as finish_order from features.entries_v1 e")
    with pytest.raises(ValueError, match="ラベルの列"):
        check_input_view(engine, "sandbox.try", *period)


def test_rejects_missing_view(engine: Engine, period):
    """ビューがなければ止まる."""
    with pytest.raises(ValueError, match="ビューがありません"):
        check_input_view(engine, "sandbox.nothing", *period)
