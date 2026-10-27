"""必要な列だけをSQLで読むデータセットのユニットテスト."""

from pathlib import Path
from typing import List

import numpy as np
import pytest
from jrdb_model import KaisaiData, create_app, db
from sqlalchemy import create_engine, text

from jra_tools.machine_learning.entity.feature_spec import default_feature_spec
from jra_tools.machine_learning.infrastructure.mysql_loader import get_kaisais
from jra_tools.machine_learning.infrastructure.sql_dataset import load_dataset
from jra_tools.machine_learning.interface_adapter.input_creator import InputCreator
from jra_tools.machine_learning.interface_adapter.label_creator import LabelCreator
from jra_tools.machine_learning.interface_adapter.payout_creator import PayoutCreator
from jra_tools.machine_learning.usecase.table_dataset import TableDataset
from jra_tools.machine_learning.util import load_training_dataset

# 仕様を読む形に変える前のjra-tools(2d8e0d0)で、sample_kaisaisから作った入力データ
EXPECTED_INPUT = Path(__file__).parents[1] / "data" / "expected_input.npy"


def _period(kaisais: List[KaisaiData]):
    days = [kaisai.ymd for kaisai in kaisais]
    return min(days), max(days)


def _load(sample_kaisais: List[KaisaiData]) -> TableDataset:
    start, end = _period(sample_kaisais)
    app = create_app()
    with app.app_context(), db.engine.connect() as connection:
        return load_dataset(connection, default_feature_spec(), start, end)


def _reorder(dataset: TableDataset, racekeys: List[str]) -> TableDataset:
    """ORMの経路のレースの並びに合わせる."""
    order = [int(np.where(dataset.racekeys == key)[0][0]) for key in racekeys]
    return TableDataset(
        racekeys=dataset.racekeys[order],
        x=dataset.x[order],
        labels=[label[order] for label in dataset.labels],
        payout=dataset.payout[order],
    )


def test_same_as_orm(sample_db: str, sample_kaisais: List[KaisaiData]):
    """同じDBから、ORMの経路と同じ入力・正解ラベル・払戻ができることを確認する."""
    start, end = _period(sample_kaisais)
    app = create_app()
    with app.app_context():
        kaisais = get_kaisais(start, end)
        racekeys = [race.racekey for kaisai in kaisais for race in kaisai.races]
        x = InputCreator(kaisais).x_data
        labels = LabelCreator(kaisais).labels
        payout = PayoutCreator(kaisais).payout

    dataset = _reorder(_load(sample_kaisais), racekeys)

    assert len(dataset.racekeys) == 24
    assert np.array_equal(dataset.x, x)
    for actual, expected in zip(dataset.labels, labels, strict=True):
        assert np.array_equal(actual, expected)
    assert np.array_equal(dataset.payout, payout)


def test_same_as_before(sample_db: str, sample_kaisais: List[KaisaiData]):
    """仕様を読む形に変える前(2d8e0d0)の入力データと完全に一致することを確認する."""
    racekeys = [race.racekey for kaisai in sample_kaisais for race in kaisai.races]

    dataset = _reorder(_load(sample_kaisais), racekeys)

    assert np.array_equal(dataset.x, np.load(EXPECTED_INPUT))


def test_race_order(sample_db: str, sample_kaisais: List[KaisaiData]):
    """レースが開催日・開催キー・レースキーの順に並ぶことを確認する."""
    dataset = _load(sample_kaisais)
    expected = sorted(
        (kaisai.ymd, kaisai.kaisaikey, race.racekey)
        for kaisai in sample_kaisais
        for race in kaisai.races
    )

    assert list(dataset.racekeys) == [racekey for _, _, racekey in expected]


def test_load_training_dataset(
    sample_db: str, sample_kaisais: List[KaisaiData], tmp_path: Path, monkeypatch
):
    """年の範囲を指定して読み込み、create_training_datasetと同じ形式で保存できる."""
    year = _period(sample_kaisais)[0] // 10000
    monkeypatch.chdir(tmp_path)

    dataset = load_training_dataset(year, year)
    dataset.save("sample")

    assert dataset.x.shape == (24, 18, 23)
    assert np.array_equal(np.load("x_sample.dump", allow_pickle=True), dataset.x)
    assert np.array_equal(
        np.load("payout_sample.dump", allow_pickle=True), dataset.payout
    )
    assert list(np.load("racekeys_sample.npy", allow_pickle=True)) == list(
        dataset.racekeys
    )
    assert (tmp_path / "sample.dump").exists()


@pytest.mark.parametrize(
    ("table", "key", "message"),
    [
        ("calculated_score", "racehorsekey", "calculated_score がありません"),
        ("seiseki", "racehorsekey", "成績がありません"),
        ("returninfo", "racekey", "払戻データがありません"),
    ],
)
def test_missing_row(
    sample_db: str, sample_kaisais: List[KaisaiData], table, key, message
):
    """紐づく行がない出走があると、ORMの経路と同じメッセージでエラーになる."""
    horse = sample_kaisais[0].races[0].racehorses[0]
    value = getattr(horse, key)
    engine = create_engine(sample_db)
    with engine.begin() as connection:
        connection.execute(
            text(f"DELETE FROM {table} WHERE {key} = :value"),  # noqa: S608 テストの固定値
            {"value": value},
        )
    engine.dispose()

    with pytest.raises(ValueError, match=f"{message}: {value}"):
        _load(sample_kaisais)
