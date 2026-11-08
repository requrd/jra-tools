"""jrdb-modelの表(staging)から読む入口(staging)のテスト. sample_kaisaisを登録したSQLiteを使う.

入口は接続先を環境変数DBから読む(sample_dbがSQLiteのURLを設定する)。
"""

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pytest
from jrdb_model import KaisaiData
from sqlalchemy import create_engine, text

from jra_tools.machine_learning import staging
from jra_tools.machine_learning.entity.dataset import TableDataset
from jra_tools.machine_learning.entity.feature_spec import parse_feature_spec

DATA = Path(__file__).parent / "data"
# jra-tools 0.6.0までのORMの経路(InputCreator・LabelCreator・PayoutCreator)で、sample_kaisaisから
# 作った入力・正解ラベル(1〜5着)・払戻。入力は仕様を読む形に変える前(2d8e0d0)の値
EXPECTED_INPUT = DATA / "expected_input.npy"
EXPECTED_LABELS = DATA / "expected_labels.npy"
EXPECTED_PAYOUT = DATA / "expected_payout.npy"


def _period(kaisais: List[KaisaiData]):
    days = [kaisai.ymd for kaisai in kaisais]
    return min(days), max(days)


def _load(sample_kaisais: List[KaisaiData]) -> TableDataset:
    start, end = _period(sample_kaisais)
    return staging.load_dataset(staging.default_feature_spec(), start, end)


def _reorder(dataset: TableDataset, racekeys: List[str]) -> TableDataset:
    """sample_kaisaisのレースの並び(期待値を作ったときの並び)に合わせる."""
    order = [int(np.where(dataset.racekeys == key)[0][0]) for key in racekeys]
    return TableDataset(
        racekeys=dataset.racekeys[order],
        x=dataset.x[order],
        labels=[label[order] for label in dataset.labels],
        payout=dataset.payout[order],
    )


def test_same_as_orm_path(sample_db: str, sample_kaisais: List[KaisaiData]):
    """入力・正解ラベル・払戻が、ORMの経路で作った値と完全に一致する."""
    racekeys = [race.racekey for kaisai in sample_kaisais for race in kaisai.races]

    dataset = _reorder(_load(sample_kaisais), racekeys)

    assert len(dataset.racekeys) == 24
    assert np.array_equal(dataset.x, np.load(EXPECTED_INPUT))
    assert np.array_equal(np.stack(dataset.labels), np.load(EXPECTED_LABELS))
    assert np.array_equal(dataset.payout, np.load(EXPECTED_PAYOUT))


def test_race_order(sample_db: str, sample_kaisais: List[KaisaiData]):
    """レースが開催日・開催キー・レースキーの順に並ぶ."""
    dataset = _load(sample_kaisais)
    expected = sorted(
        (kaisai.ymd, kaisai.kaisaikey, race.racekey)
        for kaisai in sample_kaisais
        for race in kaisai.races
    )

    assert list(dataset.racekeys) == [racekey for _, _, racekey in expected]


def test_load_training_dataset_and_save(
    sample_db: str, sample_kaisais: List[KaisaiData], tmp_path: Path
):
    """年の範囲で読み、0.6.0までと同じ形式のファイルに書き出せる."""
    year = _period(sample_kaisais)[0] // 10000

    dataset = staging.load_training_dataset(year, year)
    staging.save_dataset(dataset, "sample", tmp_path)

    assert dataset.x.shape == (24, 18, 23)
    assert np.array_equal(
        np.load(tmp_path / "x_sample.dump", allow_pickle=True), dataset.x
    )
    with open(tmp_path / "sample.dump", "rb") as f:
        labels = pickle.load(f)  # noqa: S301 このテストで書いたファイル
    assert all(
        np.array_equal(a, b) for a, b in zip(labels, dataset.labels, strict=True)
    )
    assert np.array_equal(
        np.load(tmp_path / "payout_sample.dump", allow_pickle=True), dataset.payout
    )
    assert list(np.load(tmp_path / "racekeys_sample.npy", allow_pickle=True)) == list(
        dataset.racekeys
    )


def test_load_quarter_dataset(sample_db: str, sample_kaisais: List[KaisaiData]):
    """四半期で読める(サンプルの開催がある四半期は、年で読んだときと同じ)."""
    start, _ = _period(sample_kaisais)
    year, month = start // 10000, start // 100 % 100

    dataset = staging.load_quarter_dataset(year, (month - 1) // 3 + 1)

    assert len(dataset.racekeys) > 0


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
    """紐づく行がない出走があると、表と最初のキーを出して止める."""
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


def test_source_spec_rejected(sample_db: str, sample_kaisais: List[KaisaiData]):
    """特徴量の層から読む仕様(sourceがある)は、読まずに止める."""
    spec = parse_feature_spec(
        {
            "name": "from_features",
            "max_horses": 18,
            "source": "features.entries_v1",
            "groups": [
                {
                    "name": "scores",
                    "transform": "standardize_per_race",
                    "features": [{"name": "idm"}],
                }
            ],
        }
    )
    start, end = _period(sample_kaisais)
    with pytest.raises(ValueError, match="featuresを使ってください"):
        staging.load_dataset(spec, start, end)


def test_db_required(monkeypatch: pytest.MonkeyPatch, sample_kaisais):
    """接続先(環境変数DB)がなければ止める."""
    monkeypatch.delenv("DB", raising=False)
    start, end = _period(sample_kaisais)
    with pytest.raises(RuntimeError, match="環境変数DB"):
        staging.load_dataset(staging.default_feature_spec(), start, end)
