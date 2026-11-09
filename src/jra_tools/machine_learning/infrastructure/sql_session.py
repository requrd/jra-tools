"""分析用のPostgresへの接続とトランザクション. 接続先は環境変数DB."""

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Callable

from sqlalchemy import Connection, Engine, create_engine

# 接続先(分析用のPostgres)の環境変数
DB_ENV = "DB"


@contextmanager
def engine_from_env() -> Iterator[Engine]:
    """環境変数DBのエンジン. 抜けるときに接続を閉じる.

    Raises:
        RuntimeError: 環境変数DBがない場合

    """
    url = os.environ.get(DB_ENV)
    if not url:
        raise RuntimeError(
            f"環境変数{DB_ENV}に、分析用のPostgresの接続先を設定してください"
        )
    engine = create_engine(url)
    try:
        yield engine
    finally:
        engine.dispose()


def transaction(engine: Engine) -> Callable[[], AbstractContextManager[Connection]]:
    """トランザクションを始めた接続を返す関数(抜けるときにコミット)."""
    return engine.begin
