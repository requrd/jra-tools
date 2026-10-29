"""model・featuresのビューをPostgresに反映する.

ビューはデータを持たないので、マイグレーションはしない。定義(ハッシュ)が変わったときだけ、
1トランザクションで管理するスキーマを作り直してビューを作り、ハッシュを記録する。
定義とstagingの列が合わなければCREATE VIEWでPostgresがエラーにし、全体が戻る(今のビューが残る)。
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from sqlalchemy import Connection, Engine, text
from sqlalchemy.exc import DBAPIError, OperationalError

from jra_tools.views.view_set import ViewSet

# apply_viewsどうしの排他に使うadvisory lockのキー(任意の固定値)
ADVISORY_LOCK_KEY = 7_316_042_901
# lock_timeoutにかかったときのSQLSTATE
_LOCK_NOT_AVAILABLE = "55P03"

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApplyOptions:
    """apply_viewsの設定."""

    # 管理するスキーマを読めるようにするロール
    readers: Tuple[str, ...] = ()
    # 記録に残す文字列(batchのイメージのタグなど)
    label: str = ""
    # ロックを待つ上限(Postgresの書式。例: 10s)
    lock_timeout: str = "10s"
    # lock_timeoutにかかったときにやり直す回数
    retries: int = 5
    # やり直すまでの秒数
    retry_wait: float = 5.0


DEFAULT_OPTIONS = ApplyOptions()


@dataclass(frozen=True)
class TestFailure:
    """失敗したテスト."""

    name: str
    rows: int


def apply_views(
    engine: Engine, view_set: ViewSet, options: ApplyOptions = DEFAULT_OPTIONS
) -> bool:
    """定義が変わっていれば、ビューを作り直す.

    スキーマを作り直すときの排他ロックは、読んでいる最中のクエリ(Metabaseなど)が終わるまで待つ。
    待っている間は新しいクエリも止まるため、lock_timeoutで諦めて、間を置いてやり直す。

    Args:
        engine (Engine): Postgresのエンジン
        view_set (ViewSet): 定義一式
        options (ApplyOptions): 設定

    Returns:
        bool: 作り直したらTrue。定義が変わっていなければFalse

    """
    for attempt in range(options.retries + 1):
        try:
            with engine.begin() as conn:
                return _apply(conn, view_set, options)
        except OperationalError as e:
            if _sqlstate(e) != _LOCK_NOT_AVAILABLE or attempt == options.retries:
                raise
            _logger.warning(
                "ロックを取れませんでした。%s秒後にやり直します(%d/%d)",
                options.retry_wait,
                attempt + 1,
                options.retries,
            )
            time.sleep(options.retry_wait)
    raise AssertionError("unreachable")


def applied_digest(conn: Connection) -> Optional[str]:
    """最後に反映した定義のハッシュ. 一度も反映していなければNone."""
    exists = conn.execute(text("select to_regclass('meta.view_version')")).scalar()
    if exists is None:
        return None
    return conn.execute(
        text("select digest from meta.view_version order by applied_at desc limit 1")
    ).scalar()


def run_view_tests(engine: Engine, view_set: ViewSet) -> List[TestFailure]:
    """テストを流し、行を返したテストを返す. ビューは反映済みであること.

    Args:
        engine (Engine): Postgresのエンジン
        view_set (ViewSet): 定義一式

    Returns:
        List[TestFailure]: 失敗したテスト。すべて通れば空

    """
    failures = []
    with engine.connect() as conn:
        for test in view_set.tests:
            cursor = conn.connection.cursor()
            _execute_definition(conn, test.sql, cursor)
            rows = len(cursor.fetchall())
            if rows:
                failures.append(TestFailure(test.name, rows))
    return failures


def _apply(conn: Connection, view_set: ViewSet, options: ApplyOptions) -> bool:
    conn.execute(
        text("select set_config('lock_timeout', :timeout, true)"),
        {"timeout": options.lock_timeout},
    )
    conn.execute(text("select pg_advisory_xact_lock(:key)"), {"key": ADVISORY_LOCK_KEY})
    if applied_digest(conn) == view_set.digest:
        _logger.info("ビューの定義は変わっていません(%s)", view_set.digest[:12])
        return False
    _logger.info("ビューを作り直します(%s)", view_set.digest[:12])
    quote = conn.dialect.identifier_preparer.quote
    for schema in view_set.schemas:
        # ほかのスキーマに置いた、このスキーマに依存するビュー(sandboxなど)も消える
        conn.exec_driver_sql(f"drop schema if exists {quote(schema)} cascade")
        conn.exec_driver_sql(f"create schema {quote(schema)}")
    for sql in view_set.functions:
        _execute_definition(conn, sql)
    for view in view_set.views:
        _execute_definition(
            conn,
            f"create view {quote(view.schema)}.{quote(view.name)} as\n{_body(view.sql)}",
        )
    for schema in view_set.schemas:
        for reader in options.readers:
            conn.exec_driver_sql(
                f"grant usage on schema {quote(schema)} to {quote(reader)}"
            )
            conn.exec_driver_sql(
                f"grant select on all tables in schema {quote(schema)} to {quote(reader)}"
            )
    conn.exec_driver_sql("create schema if not exists meta")
    conn.exec_driver_sql(
        "create table if not exists meta.view_version ("
        " digest text not null,"
        " applied_at timestamptz not null default now(),"
        " label text not null default '')"
    )
    conn.execute(
        text("insert into meta.view_version (digest, label) values (:digest, :label)"),
        {"digest": view_set.digest, "label": options.label},
    )
    _logger.info("ビューを%d個作りました", len(view_set.views))
    return True


def _execute_definition(conn: Connection, sql: str, cursor: Any = None) -> None:
    """定義のSQLを、パラメーターなしでドライバに渡す.

    SQLAlchemyを通すと、SQLの「%」(LIKEの'%S'など)がプレースホルダーとして解釈されるため。
    """
    dbapi_error = conn.dialect.loaded_dbapi.Error
    try:
        (cursor or conn.connection.cursor()).execute(sql)
    except dbapi_error as e:
        # SQLAlchemyを通したときと同じ例外(ProgrammingError・OperationalErrorなど)にする
        raise DBAPIError.instance(sql, None, e, dbapi_error) from e


def _body(sql: str) -> str:
    """末尾の空白とセミコロンを取り除く(サブクエリやCREATE VIEWに埋め込むため)."""
    return sql.rstrip().rstrip(";").rstrip()


def _sqlstate(error: OperationalError) -> Optional[str]:
    orig = error.orig
    return getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
