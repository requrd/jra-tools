"""特徴量の層から読む入口(features)の結合テスト. 環境変数TEST_POSTGRESのPostgresを使う(なければ飛ばす).

入口は接続先を環境変数DBから読むので、DBにTEST_POSTGRESを設定して呼ぶ。
stagingにsample_kaisaisを入れ、model(code_categories・races)とfeatures(entries_v1・labels_v1)を
テスト用に作る。
"""

import dataclasses
import os
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pytest
from jrdb_model import KaisaiData, db
from sqlalchemy import Engine, create_engine, text

from jra_tools.machine_learning import features
from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from tests.machine_learning.conftest import sample_rows

pytestmark = pytest.mark.skipif(
    "TEST_POSTGRES" not in os.environ, reason="TEST_POSTGRESがない"
)

SCHEMAS = ("staging", "model", "features", "sandbox")
NUMBERS = (
    "turf_baba_in",
    "idm",
    "jockey_score",
    "info_score",
    "train_score",
    "trainer_score",
    "train_code",
    "jockey_rate_rentai",
    "kakutoku_money",
    "ls_score_order",
    "up_score_order",
    "expect_jokey_win_rate",
    "expect_jokey_rentai_rate",
    "waku_win_rate",
    "waku_rentai_rate",
)
OUTPUTS = ("o1", "o2", "o3", "o4", "o5")

# 数値15列とカテゴリの枠(コードは文字列。記載外の0はmodelと同じくNULLにする)
ENTRIES_SQL = """
create view features.entries_v1 as
select r.racehorsekey, r.racekey, model.jrdb_date(k.ymd) as race_date, r.num as umaban,
       k.turf_baba_in, r.idm, r.jockey_score, r.info_score, r.train_score,
       r.trainer_score, r.train_code, r.jockey_rate_rentai, r.kakutoku_money,
       r.ls_score_order, r.up_score_order, r.expect_jokey_win_rate,
       r.expect_jokey_rentai_rate, c.waku_win_rate, c.waku_rentai_rate,
       nullif(r.waku, 0)::text as waku
from staging.kaisai k
join staging.bangumi b on b.kaisaikey = k.kaisaikey and b.ymd = k.ymd
join staging.racehorse r on r.racekey = b.racekey
left join staging.calculated_score c on c.racehorsekey = r.racehorsekey
"""
LABELS_SQL = """
create view features.labels_v1 as
select r.racehorsekey, s.order_of_arrival as finish_order
from staging.racehorse r
join staging.seiseki s on s.racehorsekey = r.racehorsekey
"""
FUNCTION_SQL = """
create function model.jrdb_date(v integer) returns date
language sql immutable return to_date(v::text, 'YYYYMMDD')
"""
# 枠番1〜8を1つずつ(model)、まとめ方(features)
CATEGORIES_SQL = """
create view model.code_categories as
select 'waku'::text as item, n::text as code, n as category_no, n || '枠' as label
from generate_series(1, 8) n;
create view features.code_categories as
select 'waku_group'::text as item, n::text as code,
       case when n <= 3 then 1 when n <= 5 then 2 else 3 end as category_no,
       '' as label
from generate_series(1, 8) n;
"""
RACES_SQL = """
create table model.year_completeness (year integer primary key, year_complete boolean);
create view model.races as
select b.racekey, model.jrdb_date(b.ymd) as race_date,
       coalesce(y.year_complete, false) as year_complete
from staging.bangumi b
left join model.year_completeness y on y.year = b.ymd / 10000
"""


def _raw_spec(category: str = "waku") -> FeatureSpec:
    """数値15列(レースごとに標準化)と、枠のカテゴリの仕様(固定する前)."""
    return features.parse_feature_spec(
        {
            "name": "resnet4_view",
            "max_horses": 18,
            "source": "features.entries_v1",
            "groups": [
                {
                    "name": "scores",
                    "transform": "standardize_per_race",
                    "features": [{"name": n} for n in NUMBERS],
                },
                {
                    "name": "categories",
                    "transform": "none",
                    "features": [
                        {"name": category, "source": "waku", "encode": "category"}
                    ]
                    if category == "waku"
                    else [{"name": category, "encode": "category"}],
                },
            ],
        }
    )


def _pinned() -> FeatureSpec:
    return dataclasses.replace(features.pin_spec(_raw_spec()), output_names=OUTPUTS)


def _drop(engine: Engine) -> None:
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.exec_driver_sql(f"drop schema if exists {schema} cascade")


@pytest.fixture
def engine(
    sample_kaisais: List[KaisaiData], monkeypatch: pytest.MonkeyPatch
) -> Iterator[Engine]:
    """sample_kaisaisをstagingに登録し、model・featuresのビューを作ったPostgres. DBにも設定する."""
    monkeypatch.setenv("DB", os.environ["TEST_POSTGRES"])
    engine = create_engine(
        os.environ["TEST_POSTGRES"],
        connect_args={"options": "-csearch_path=staging"},
    )
    _drop(engine)
    with engine.begin() as conn:
        for schema in SCHEMAS:
            conn.exec_driver_sql(f"create schema {schema}")
        db.metadata.create_all(conn)
        # seisekiの外部キー(seisekirace)の行はサンプルにないため、制約を効かせずに入れる
        conn.exec_driver_sql("set local session_replication_role = replica")
        for name, rows in sample_rows(sample_kaisais).items():
            conn.execute(db.metadata.tables[name].insert(), rows)
        conn.exec_driver_sql(FUNCTION_SQL)
        conn.exec_driver_sql(ENTRIES_SQL)
        conn.exec_driver_sql(LABELS_SQL)
        conn.exec_driver_sql(CATEGORIES_SQL)
        conn.exec_driver_sql(RACES_SQL)
    yield engine
    _drop(engine)
    engine.dispose()


def _period(kaisais: List[KaisaiData]):
    days = [kaisai.ymd for kaisai in kaisais]
    return min(days), max(days)


def test_load_input(engine: Engine, sample_kaisais: List[KaisaiData]):
    """開催日の入力が、学習用のデータセットのその日の分と同じになる."""
    spec = _pinned()
    start, end = _period(sample_kaisais)
    dataset = features.load_dataset(spec, start, end)
    ymd = sample_kaisais[0].ymd
    day_racekeys = {
        race.racekey for k in sample_kaisais if k.ymd == ymd for race in k.races
    }

    inputs = features.load_input(spec, ymd)

    assert set(inputs.racekeys) == day_racekeys
    index = [list(dataset.racekeys).index(key) for key in inputs.racekeys]
    assert np.array_equal(inputs.x, dataset.x[index])


def test_digest(engine: Engine):
    """sourceの今のハッシュ. pin_specが仕様に書く値と同じ. なければ止める."""
    assert features.digest("features.entries_v1") == (_pinned().view_digest)
    with pytest.raises(ValueError, match="ビューがありません: features.no_such"):
        features.digest("features.no_such")


def test_db_required(engine: Engine, monkeypatch: pytest.MonkeyPatch):
    """接続先(環境変数DB)がなければ止める."""
    monkeypatch.delenv("DB")
    with pytest.raises(RuntimeError, match="環境変数DB"):
        features.digest("features.entries_v1")


def test_digest_mismatch(engine: Engine, sample_kaisais: List[KaisaiData]):
    """学習時とビューの定義が違うと、読み込みを止める."""
    spec = _pinned()
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "create or replace function model.jrdb_date(v integer) returns date"
            " language sql immutable return to_date(v::text, 'YYYYMMDD') + 0"
        )

    with pytest.raises(ValueError, match="学習時と違います"):
        features.load_input(spec, sample_kaisais[0].ymd)


def test_pinned_spec_required(engine: Engine, sample_kaisais: List[KaisaiData]):
    """固定していない仕様では読まない."""
    with pytest.raises(ValueError, match="固定してください"):
        features.load_input(_raw_spec(), sample_kaisais[0].ymd)


def test_missing_column(engine: Engine, sample_kaisais: List[KaisaiData]):
    """ビューにない列を仕様が使うと止める."""
    spec = _pinned()
    group = dataclasses.replace(
        spec.groups[0],
        features=(
            *spec.groups[0].features,
            dataclasses.replace(spec.groups[0].features[0], source="no_such"),
        ),
    )
    spec = dataclasses.replace(spec, groups=(group, *spec.groups[1:]))

    with pytest.raises(ValueError, match=r"列がありません: \['no_such'\]"):
        features.load_input(spec, sample_kaisais[0].ymd)


@pytest.mark.parametrize(
    ("table", "key", "message"),
    [
        (
            "calculated_score",
            "racehorsekey",
            "features.entries_v1.waku_win_rateが値なし",
        ),
        ("seiseki", "racehorsekey", "成績がありません"),
    ],
)
def test_missing_row(
    engine: Engine, sample_kaisais: List[KaisaiData], table, key, message
):
    """値のない出走があると止める(ORMの経路の「行がなければエラー」と同じ)."""
    horse = sample_kaisais[0].races[0].racehorses[0]
    spec = _pinned()
    with engine.begin() as conn:
        conn.execute(
            text(f"delete from staging.{table} where {key} = :v"),  # noqa: S608 テストの固定値
            {"v": getattr(horse, key)},
        )
    start, end = _period(sample_kaisais)

    with pytest.raises(ValueError, match=message):
        features.load_dataset(spec, start, end)


def test_read_frame(engine: Engine, sample_kaisais: List[KaisaiData]):
    """1頭1行の表を、開催日・レースキー・馬番の順で読める. 推論ではラベルを読まない."""
    spec = _pinned()
    start, end = _period(sample_kaisais)

    frame = features.read_frame(spec, start, end)
    inputs = features.read_frame(spec, start, end, labels=None)

    horses = sum(len(r.racehorses) for k in sample_kaisais for r in k.races)
    assert len(frame) == horses
    assert "finish_order" in frame.columns
    assert not {"win1_num", "win1_ret"} & set(frame.columns)
    assert "_label" not in frame.columns
    assert "finish_order" not in inputs.columns
    assert frame[["racekey", "umaban"]].equals(inputs[["racekey", "umaban"]])
    assert frame["racekey"].is_monotonic_increasing or len(sample_kaisais) > 1


def test_label_column_as_feature(engine: Engine, sample_kaisais: List[KaisaiData]):
    """確定後の値(ラベルの列)を入力に使う仕様は止める."""
    spec = _pinned()
    feature = dataclasses.replace(spec.groups[0].features[0], source="finish_order")
    group = dataclasses.replace(spec.groups[0], features=(feature,))
    spec = dataclasses.replace(spec, groups=(group, *spec.groups[1:]))

    with pytest.raises(ValueError, match="ラベルの列は入力に使えません"):
        features.load_input(spec, sample_kaisais[0].ymd)


class _FakeModel:
    """出力の値がレースの位置・枠で決まるモデル. 受け取った入力を残す."""

    def __init__(self):
        self.x = None

    def predict(self, x, verbose="auto"):
        self.x = x
        races, slots = x.shape[0], x.shape[1]
        base = np.arange(races)[:, None] * 100 + np.arange(slots)[None, :]
        offsets = (0.1, 0.2, 0.3, 0.4, 0.5)
        return {name: base + d for name, d in zip(OUTPUTS, offsets, strict=True)}

    def evaluate(self, x, y, verbose="auto", return_dict=False):
        self.x = x
        return {"loss": 1.0, "o1_loss": 0.5}


def test_predict_finish_probabilities(engine: Engine, sample_kaisais: List[KaisaiData]):
    """出走馬ごとに、その馬のレース・馬番の枠の確率を返す. 入力はload_view_inputと同じ."""
    spec = _pinned()
    ymd = sample_kaisais[0].ymd
    inputs = features.load_input(spec, ymd)
    model = _FakeModel()

    actual = features.predict_finish_probabilities(spec, ymd, model)

    assert np.array_equal(model.x, inputs.x)
    horses = [
        (h.racehorsekey, race.racekey, h.num)
        for k in sample_kaisais
        if k.ymd == ymd
        for race in k.races
        for h in race.racehorses
    ]
    assert len(actual) == len(horses)
    racekeys = list(inputs.racekeys)
    expected = {
        key: tuple(racekeys.index(racekey) * 100 + num - 1 + d for d in (0.1, 0.2, 0.3))
        for key, racekey, num in horses
    }
    assert {key: probs for key, *probs in actual} == {
        k: list(v) for k, v in expected.items()
    }
    assert all(isinstance(v, float) for row in actual for v in row[1:])


def test_predict_finish_probabilities_no_races(engine: Engine):
    """出走のない日は、モデルを呼ばずに空を返す."""
    spec = _pinned()
    model = _FakeModel()

    assert features.predict_finish_probabilities(spec, 19000101, model) == []
    assert model.x is None


def test_pin_spec_copies_categories(engine: Engine):
    """カテゴリの列は、code_categoriesの「コード→番号」を仕様に写す(modelの枠、featuresのまとめ方)."""
    [waku] = _pinned().category_features
    assert dict(waku.categories) == {str(n): n for n in range(1, 9)}
    assert waku.size == 9
    [group] = features.pin_spec(_raw_spec("waku_group")).category_features
    assert dict(group.categories) == {
        "1": 1,
        "2": 1,
        "3": 1,
        "4": 2,
        "5": 2,
        "6": 3,
        "7": 3,
        "8": 3,
    }


def test_pin_spec_stops_on_duplicated_item(engine: Engine):
    """同じ項目がsandboxにもあれば止める(s1)."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "create view sandbox.code_categories as"
            " select 'waku'::text as item, '1'::text as code, 1 as category_no, '' as label"
        )
    with pytest.raises(ValueError, match="2か所以上"):
        features.pin_spec(_raw_spec())


def test_null_category_goes_to_exception_column(
    engine: Engine, sample_kaisais: List[KaisaiData]
):
    """記載外(modelがNULLにした)コードは止めずに例外の列にする."""
    spec = _pinned()
    horse = sample_kaisais[0].races[0].racehorses[0]
    with engine.begin() as conn:
        conn.execute(
            text("update staging.racehorse set waku = 0 where racehorsekey = :k"),
            {"k": horse.racehorsekey},
        )
    inputs = features.load_input(spec, sample_kaisais[0].ymd)
    race = list(inputs.racekeys).index(horse.racekey)
    assert inputs.x[race, horse.num - 1, -9:].tolist() == [0] * 8 + [1]


def test_cache_dir_keeps_frame(
    engine: Engine, sample_kaisais: List[KaisaiData], tmp_path: Path
):
    """cache_dirを渡すと、読んだ表をParquetで残し、次はビューを読まずに同じデータセットを作る."""
    spec = _pinned()
    start, end = _period(sample_kaisais)
    first = features.load_dataset(spec, start, end, cache_dir=tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == [
        f"resnet4_view_{start}_{end}.parquet"
    ]
    with engine.begin() as conn:
        conn.exec_driver_sql("drop view features.labels_v1")
    second = features.load_dataset(spec, start, end, cache_dir=tmp_path)
    assert np.array_equal(first.x, second.x)
    assert list(first.racekeys) == list(second.racekeys)


def test_evaluate(engine: Engine, sample_kaisais: List[KaisaiData]):
    """締めた年だけ評価する. 出走馬ごとの1〜5着の確率と、loss・指標を返す."""
    spec = _pinned()
    start, end = _period(sample_kaisais)
    with pytest.raises(ValueError, match="締めていない年"):
        features.evaluate(spec, start, end, _FakeModel())
    years = sorted({k.ymd // 10000 for k in sample_kaisais})
    with engine.begin() as conn:
        for year in years:
            conn.execute(
                text("insert into model.year_completeness values (:y, true)"),
                {"y": year},
            )
    predictions, metrics = features.evaluate(spec, start, end, _FakeModel())
    horses = sum(len(r.racehorses) for k in sample_kaisais for r in k.races)
    assert len(predictions) == horses
    assert list(predictions.columns) == ["racehorsekey", *features.PROBABILITY_COLUMNS]
    assert metrics.to_dict("records") == [
        {"output": "total", "metric": "loss", "value": 1.0},
        {"output": "o1", "metric": "loss", "value": 0.5},
    ]
