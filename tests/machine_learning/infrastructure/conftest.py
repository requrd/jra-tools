"""DBを使うテストのフィクスチャー."""

from pathlib import Path
from typing import Dict, List

import pytest
from jrdb_model import KaisaiData, db
from sqlalchemy import create_engine

# 出走に紐づけて登録する、読み込み済みのリレーション。
# ORMの経路(joinedload("*"))はhorse_base・trainanalysis・trainoikiriを内部結合するため、
# これらの行がないと開催ごと読み込まれない。
_HORSE_RELATIONS = [
    "result",
    "calculated_score",
    "trainanalysis",
    "trainoikiri",
    "horse_base",
]


@pytest.fixture
def sample_db(
    sample_kaisais: List[KaisaiData], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """sample_kaisaisを登録したSQLiteのDB. 環境変数DBに設定する(create_appが読む).

    Returns:
        str: DBのURL

    """
    url = f"sqlite:///{tmp_path / 'sample.db'}"
    engine = create_engine(url)
    db.metadata.create_all(engine)
    with engine.begin() as connection:
        for name, table_rows in sample_rows(sample_kaisais).items():
            connection.execute(db.metadata.tables[name].insert(), table_rows)
    engine.dispose()
    monkeypatch.setenv("DB", url)
    return url


def sample_rows(sample_kaisais: List[KaisaiData]) -> Dict[str, List[dict]]:
    """sample_kaisaisの行を、テーブルごとに(外部キーの順に)まとめる."""
    rows: Dict[str, Dict[tuple, dict]] = {}

    def add(entity):
        if entity is None:
            return
        table = entity.__table__
        key = tuple(getattr(entity, c.name) for c in table.primary_key.columns)
        rows.setdefault(table.name, {})[key] = {
            c.name: getattr(entity, c.name) for c in table.columns
        }

    for kaisai in sample_kaisais:
        add(kaisai)
        for race in kaisai.races:
            add(race)
            add(race.returninfo)
            for horse in race.racehorses:
                add(horse)
                for relation in _HORSE_RELATIONS:
                    add(getattr(horse, relation))
    return {
        table.name: list(rows[table.name].values())
        for table in db.metadata.sorted_tables
        if table.name in rows
    }
