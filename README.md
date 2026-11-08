# jra-tools
Sazanami-AIのデータ・学習のための道具。JRDBの取り込みの補助・開催日程(Yahoo競馬)の照合(`jra_tools`)、分析用のPostgresのビューの反映(`jra_tools.views`)、学習・推論の入力を作ること(`jra_tools.machine_learning`の入口`features`・`staging`)を持つ。

## Usage
### Install
```
$pip install jra-tools
```
分析用のPostgres(特徴量の層)から読む場合は、ドライバ(psycopg)も入れる。
```
$pip install "jra-tools[postgres]"
```
## Test
```
$uv run pytest
```
Postgresを使う結合テスト(`tests/views`・`tests/machine_learning/test_integration_features.py`・`tests/machine_learning/infrastructure/test_integration_postgres_view_reader.py`)は、環境変数`TEST_POSTGRES`(例: `postgresql+psycopg://test_user:test_password@127.0.0.1:5432/test_database`)がなければ飛ばす。

## ビューの仕組み(`jra_tools.views`)
分析用のPostgresのmodel・featuresのビューを、定義から作る。具体的な定義は持たず、使う側(sazanami-batchの`schemas/`)から渡す。ビューはデータを持たないので、マイグレーションはしない。

```
schemas/
  views.yaml        管理するスキーマ・関数のSQL・作る順のビューの一覧
  model/races.sql   ビュー1つにつき1ファイル。SELECT文だけを書く
  tests/*.sql       テスト。返した行があれば失敗
```

```yaml
schemas: [model, features]
functions: [functions.sql]   # ビューより先に流すSQL。省略できる
views:                       # 作る順に並べる
  - model.races
  - model.entries
```

```python
from jra_tools.views import ApplyOptions, apply_views, load_view_set, run_view_tests

view_set = load_view_set("schemas")
apply_views(engine, view_set, ApplyOptions(readers=("sazanami_ro",), label="xxxxxxx"))
failures = run_view_tests(engine, view_set)  # 反映済みのビューにテストを流す(CI・分析用のPostgresの実データ)
```

- `apply_views`は、定義(views.yaml・関数・ビューのSQL)のハッシュが`meta.view_version`の最後の記録と同じなら何もしない。違えば1トランザクションでビューと関数を反映し、`readers`に読む権限を付け、ハッシュを記録する
- スキーマは消さない(なければ作る)。同じスキーマに置いた表・マテリアライズドビュー(推論の結果など)と、拡張機能が作ったものには触らない
- `meta.view_version`は作らない。なければ`ViewVersionTableMissing`で止まる。表を作るのは使う側(sazanami-batchの`apply-model-and-meta-tables`)
- 反映は、まず今のビュー・関数をその場で置き換える(`CREATE OR REPLACE`)。ほかのスキーマに置いた、これらに依存するビュー(sandboxの試作など)は残る。定義から外したビュー・関数だけを消す
  - 関数は`create or replace function`で書く(`create function`だと、あるものを作れず、下の作り直しになる)
  - ビューは、列が同じか末尾に足すだけなら置き換えられる
- 置き換えられないとき(ビューの列を消す・改名する・型を変える、関数の戻り値の型を変えるなど)は、`schemas`のスキーマのビューと関数をすべて消して作り直す。依存するほかのスキーマのビュー(関数だけを使うものを含む)も消え、その名前を警告のログに残す
- 関数はCASCADEで消さない。表の列の既定値などが関数を使っていれば、エラーで止まる(全体が戻る)
- 定義とstagingの列が合わなければ`CREATE VIEW`でPostgresがエラーにし、全体が戻る(今のビューと記録が残る)
- 読んでいる最中のクエリがあると、ビューを消す排他ロックを待つ。`lock_timeout`(既定10秒)で諦め、間を置いて`retries`回(既定5回)やり直す
- 同時に流しても、advisory lockで1つずつ動く
- テストはハッシュに含めない(テストだけを変えても作り直さない)
## 特徴量の層から読む(`jra_tools.machine_learning.features`)
特徴量の仕様の最上位に`source`(「スキーマ.名前」。`features.entries_v1`や、試作の`sandbox.entries_try1`など)を書くと、学習・推論の入力をそこから作る。sourceがビューか表かは使う側には関係ない。`source`のない仕様(今の本番モデルの`feature_spec.yaml`)は、jrdb-modelの表から読む(下の`staging`。sourceのある仕様を`staging`に渡すとエラー)。

```yaml
name: resnet4_view
max_horses: 18
source: features.entries_v1   # 入力を読む先
view_digest: 1c445b...         # 学習時のsourceの定義のハッシュ(pin_digestが書く)
train_from: 2013-01-01         # 学習期間(初日・最終日を含む。省略可)
train_to: 2019-12-31
groups:
  - name: scores
    transform: standardize_per_race
    features:
      - {name: idm}                     # sourceを省略すると列名=name
      - {name: waku_win_rate, source: waku_win_rate}
```

使う側は`features`だけを使う。接続先は環境変数`DB`(分析用のPostgres)から読む。

```python
import dataclasses
from datetime import date

from jra_tools.machine_learning import features

spec = features.load_feature_spec("spec.yaml")
spec = features.pin_digest(spec)                          # 学習の前に、今のsourceのハッシュを書く
dataset = features.load_dataset(spec, 20130101, 20191231)  # labels="features.labels_v1"
spec = dataclasses.replace(spec, train_from=date(2013, 1, 1), train_to=date(2019, 12, 31))
features.save_model_feature_spec(spec, model_dir)        # モデルと一緒に保存する(学習期間も書く)

spec = features.load_model_feature_spec(model_dir)       # 推論
inputs = features.load_input(spec, 20261101)             # その日の入力だけ
probs = features.predict_finish_probabilities(spec, 20261101, model)  # [(racehorsekey, 1着, 2着, 3着の確率)]
frame = features.read_frame(spec, 20250101, 20251231)    # 組み立てる前の1頭1行のDataFrame
features.digest("features.entries_v1")                    # 今の定義のハッシュ
```

- 読んだ表(1頭1行のpandasのDataFrame)を、仕様に従ってテンソルにする。各レースを1〜18の枠に並べ(空き枠は0)、グループごとにone-hot・レースごとの標準化をしてから、最後に`(レース, 18, 次元)`のndarrayにする
- jrdb-modelの経路(`assemble_dataset`)とは、浮動小数点の誤差(1e-15程度)の範囲で一致する。正解ラベル・払戻・並び順は完全に一致する
- 学習期間(`train_from`・`train_to`)は、学習に使った期間をモデルの仕様に残すためのもの。使う側(batch)は、推論する日が学習期間に入っていないかをこれで確かめる
- 入力の列: `racehorsekey`・`racekey`・`race_date`(date)・`umaban`と仕様の列。ラベルの列: `racehorsekey`・`finish_order`・`win1_num`・`win1_ret`
- 読むたびに、仕様の列がsourceにあるか、sourceの定義のハッシュが`view_digest`と同じかを確かめ、違えば止める。定義を変えたら(バグ修正でも)、仕様を書き直すか、版を上げる(`entries_v2`を並べる)
- ハッシュは、sourceと、依存するビュー・関数の定義(Postgresが保存した形)から作る。関係のないものを足したり変えたりしても変わらない。空白・コメントだけの違いでも変わらない。stagingの表は名前だけを含める。Postgresのメジャーバージョンを上げると変わることがある。名前も含むので、sandboxで学習したモデルは、本採用(`features.`へ移す)のあと学習し直す
- 関数の中から呼ぶ関数をたどれるのは、`RETURN`で書いたSQLの関数だけ(`AS $$...$$`の本体はPostgresが依存を記録しない)
- 1回の読み込みは1トランザクションで、先にsourceを共有ロックする(読み終わるまで`apply_views`に作り直されない)。学習は1年ずつ読む
- `predict_finish_probabilities`は確率だけを素の値で返す(連対率・オッズは使う側のビューで出す)。`model.predict(x)`は`activation_4`・`activation_5`・`activation_6`(1・2・3着)の`(レース, 18)`を返すこと(`PredictFactory`と同じ出力)。出走のない日は空
- 値のない出走(NULL)があれば止める(jrdb-modelの経路の「行がなければエラー」と同じ)。文字列は0にする

## jrdb-modelの表から読む(`jra_tools.machine_learning.staging`)
最上位に`source`のない仕様(各特徴量が「テーブル名.列名」)で、jrdb-modelの表(分析用のPostgresのstaging、MariaDB)から学習用のデータセットを作る。仕様の特徴量・着順・単勝の払戻の列だけをSQLで読む。接続先は環境変数`DB`。

```python
from jra_tools.machine_learning import staging

dataset = staging.load_training_dataset(2013, 2019)    # 仕様の省略時は現行の推論モデルの仕様
dataset = staging.load_quarter_dataset(2025, 1)
dataset = staging.load_dataset(spec, 20130101, 20191231)
staging.save_dataset(dataset, "2013_2019")             # x_….dump・….dump・payout_….dump・racekeys_….npy
```

- 0.6.0まであったORMのオブジェクトから作る経路(`InputCreator`・`LabelCreator`・`PayoutCreator`・`PredictFactory`など)は0.7.1でなくした。結果は同じで(`tests/machine_learning/data/`の値で確かめる)、2019年の1年分でORMの経路は17.0秒・1,865MB、この経路は4.0秒・263MBだった(2026-10-27、分析用のMySQL)
- 成績・払戻・仕様の表(調教分析・追切・計算スコア)の行がない出走があれば、表と最初のキーを出して止める
- 書き出すファイルは0.6.0までと同じ形式

## 置き場所(`machine_learning`の層)
使う側は入口(`features`・`staging`)と、仕様の型(入口から取れる`FeatureSpec`など)だけを使う。

| 置き場所 | 中身 |
|---|---|
| `features.py`・`staging.py`(層の外) | 入口。環境変数`DB`から読み手を作り、手続きに渡す(組み立てだけ) |
| `entity/` | 仕様(`feature_spec.py`。ファイルは扱わない)、データセットの形(`dataset.py`)、特徴量の層の入力の列の契約と決まり(`input_view.py`)、jrdb-modelの表の行の決まり(`staging_rows.py`)、期間(`period.py`) |
| `usecase/` | 手続き:特徴量の層から読む(`view_dataset.py`)、jrdb-modelの表から読む(`staging_dataset.py`)、テンソルへの組み立て(`frame_dataset.py`・`table_dataset.py`)、モデルの仕様の読み書き(`model_spec.py`)。外に触るところはProtocol(`iview_reader.py`・`istaging_rows_reader.py`・`imodel_folder.py`・`ifinish_model.py`) |
| `infrastructure/` | Postgresへの問い合わせ(`postgres_view_reader.py`)、jrdb-modelの表を読むSQL(`staging_rows_reader.py`)、SavedModelのフォルダ(`saved_model_folder.py`)、仕様のYAML・同梱の既定の仕様(`spec_files.py`)、データセットのファイル(`dataset_files.py`) |

開催データを紐づく表ごと読む`get_kaisais`は、機械学習ではなくDBの読み込みなので`jra_tools`(`database/kaisai_loader.py`)にある。
