"""Postgresの定義のハッシュ(SQL)の結合テスト. 環境変数TEST_POSTGRESのPostgresを使う(なければ飛ばす)."""

import os

import pytest
from sqlalchemy import Engine, create_engine, text

from jra_tools.machine_learning.infrastructure.postgres_view_reader import view_digest
from tests.machine_learning.test_integration_features import (
    engine,
)

pytestmark = pytest.mark.skipif(
    "TEST_POSTGRES" not in os.environ, reason="TEST_POSTGRESがない"
)


def test_digest_is_per_view(engine: Engine):
    """関係のないビューを足してもハッシュは変わらず、依存する定義を変えると変わる."""
    with engine.connect() as conn:
        before = view_digest(conn, "features.entries_v1")
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "create view features.entries_v2 as select racehorsekey from features.entries_v1"
        )
    with engine.connect() as conn:
        assert view_digest(conn, "features.entries_v1") == before
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "create or replace function model.jrdb_date(v integer) returns date"
            " language sql immutable return to_date(v::text, 'YYYYMMDD') + 0"
        )
    with engine.connect() as conn:
        assert view_digest(conn, "features.entries_v1") != before


def test_digest_ignores_search_path(engine: Engine):
    """接続のsearch_pathが違っても、同じハッシュになる."""
    other = create_engine(os.environ["TEST_POSTGRES"])
    try:
        with engine.connect() as a, other.connect() as b:
            assert view_digest(a, "features.entries_v1") == view_digest(
                b, "features.entries_v1"
            )
            assert a.execute(text("show search_path")).scalar() == "staging"
    finally:
        other.dispose()
