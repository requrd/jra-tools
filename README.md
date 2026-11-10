# jra-tools
Sazanami-AIのデータ・学習のための道具。JRDBの取り込みの補助・開催日程(Yahoo競馬)の照合(`jra_tools`)、分析用のPostgresのビューの反映(`jra_tools.views`)、学習・推論・評価の入力を作ること(`jra_tools.machine_learning`の入口`features`)を持つ。

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
Postgresを使う結合テスト(`tests/views`・`tests/machine_learning/test_integration_features.py`・`tests/machine_learning/interface_adapter/test_integration_view_gateway.py`)は、環境変数`TEST_POSTGRES`(例: `postgresql+psycopg://test_user:test_password@127.0.0.1:5432/test_database`)がなければ飛ばす。

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
- `replace_views(engine, view_set)`は、ハッシュを読まず・記録せずに毎回同じ手順で反映する(`meta.view_version`を読めない試作のsandbox向け。使う側は入口の`features.apply_sandbox_views`から使う)。スキーマは作らない(あること)。`readers`・`label`は使わない
## 特徴量の層から読む(`jra_tools.machine_learning.features`)
特徴量の仕様の`source`(「スキーマ.名前」。`features.entries_v1`や、試作の`sandbox.entries_try1`など)から、学習・推論・評価の入力を作る。sourceがビューか表かは使う側には関係ない。jra-toolsはstagingを直接読まない。

```yaml
name: resnet4_view
max_horses: 18
source: features.entries_v1   # 入力を読む先
groups:
  - name: scores
    transform: standardize_per_race
    features:
      - {name: idm}                     # sourceを省略すると列名=name
      - {name: waku_win_rate, source: waku_win_rate}
  - name: categories
    transform: none
    features:
      - {name: waku, encode: category}  # 列の値はコード。番号はcode_categoriesから(pin_specが写す)
```

固定した仕様(`pin_spec`・`save_model_feature_spec`が書き足す):

```yaml
view_digest: 1c445b...                          # 学習時のsourceの定義のハッシュ
output_names: [activation_2, ..., activation_6] # モデルの出力の名前(1着→5着)
train_from: 2013-01-01                          # 学習期間(初日・最終日を含む。省略可)
train_to: 2019-12-31
groups:
  - name: categories
    transform: none
    features:
      - name: waku
        encode: category
        categories: {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}  # 学習時の控え
```

使う側は`features`だけを使う。接続先は環境変数`DB`(分析用のPostgres)から読む。

```python
import dataclasses
from datetime import date

from jra_tools.machine_learning import features

spec = features.load_feature_spec("spec.yaml")
spec = features.pin_spec(spec)          # 学習の前に、ハッシュとカテゴリの控えを写す
dataset = features.load_dataset(spec, 20130101, 20241231, cache_dir="cache")  # 入力と1〜5着のラベル
spec = dataclasses.replace(spec, train_from=date(2013, 1, 1), train_to=date(2024, 12, 31))
features.save_model_feature_spec(spec, model_dir, model.output_names)  # モデルと一緒に保存する

predictions, metrics = features.evaluate(spec, 20250101, 20250331, model)  # 評価期間(締めた年だけ)
spec = features.load_model_feature_spec(model_dir)                       # 推論
probs = features.predict_finish_probabilities(spec, 20261101, model)    # [(racehorsekey, 1着, 2着, 3着の確率)]
frame = features.read_frame(spec, 20250101, 20251231)  # 組み立てる前の1頭1行のDataFrame
features.digest("features.entries_v1")                  # 今の定義のハッシュ
```

試作のビュー(sandbox)は、`views.yaml`と同じ形のフォルダ(例: `schemas/views.yaml`に`schemas: [sandbox]`と作る順のビュー、SQLは`schemas/sandbox/<ビュー名>.sql`)を反映する。接続先は同じ環境変数`DB`。

```python
features.apply_sandbox_views("schemas")  # 一覧から外したsandboxのビューは消える
```

- 毎回反映する(ハッシュを記録しない)。手順と排他は`apply_views`と同じ(その場で置き換え、できなければsandboxのビューをすべて作り直す。表は残る)
- 一覧にないsandboxのビュー(手で作ったものも)は消える。sandboxのスキーマと、ビューが読む表は先にあること

- 読んだ表(1頭1行のpandasのDataFrame)を、仕様に従ってテンソルにする。各レースを1〜18の枠に並べ(空き枠は0)、グループごとにカテゴリのone-hot・レースごとの標準化をしてから、最後に`(レース, 18, 次元)`のndarrayにする
- 学習のデータセットは入力と1〜5着の正解ラベルだけで、払戻は持たない。回収率は、`evaluate`の確率(`racehorsekey`ごと)を`model.entries`・`model.payouts`と結合して出す(使う側のsandboxのビュー・pandas)
- 入力の列: `racehorsekey`・`racekey`・`race_date`(date)・`umaban`と仕様の列。ラベルの列: `racehorsekey`・`finish_order`
- 数値の列に値なし(NULL)があれば止める。文字列は0にする。成績(ラベルのビューの行)のない出走があれば止める
- 学習期間(`train_from`・`train_to`)は、学習に使った期間をモデルの仕様に残すためのもの。使う側(batch)は、推論する日が学習期間に入っていないかをこれで確かめる

### カテゴリの列
- 変換の決まりの正はDBの`code_categories`(1コード1行 `item, code, category_no, label`)。列の名前=項目の名前、列の値=コード(文字列)。JRDBの項目名(`waku`など)はmodelが使い、まとめ方・試作には別の名前(例: `waku_group`)を付ける
- `pin_spec`は、`model.code_categories`・`features.code_categories`・`sandbox.code_categories`(DBにあるものだけ。引数`category_sources`で替えられる)を合わせて項目名で引き、「コード→番号」を仕様に写す。項目がどこにもない・2か所以上にあれば止める
- 次元は最大の番号 + 例外の列1。控えにないコード(学習のあとに増えたコードなど)・NULL(modelで記載外にしたコード)は例外の列。推論はDBの`code_categories`を読まず、モデルのフォルダの控えだけを使う(seedsの更新でビューのハッシュは変わらず、番号がずれうるため)

### ビューのハッシュ
- 読むたびに、仕様の列がsourceにあるか、sourceの定義のハッシュが`view_digest`と同じかを確かめ、違えば止める。定義を変えたら(バグ修正でも)、仕様を固定し直すか、版を上げる(`entries_v2`を並べる)
- ハッシュは、sourceと、依存するビュー・関数の定義(Postgresが保存した形)から作る。関係のないものを足したり変えたりしても変わらない。空白・コメントだけの違いでも変わらない。stagingの表は名前だけを含める。Postgresのメジャーバージョンを上げると変わることがある。名前も含むので、sandboxで学習したモデルは、本採用(`features.`へ移す)のあと学習し直す
- 関数の中から呼ぶ関数をたどれるのは、`RETURN`で書いたSQLの関数だけ(`AS $$...$$`の本体はPostgresが依存を記録しない)
- 1回の読み込みは1トランザクションで、先にsourceを共有ロックする(読み終わるまで`apply_views`に作り直されない)。1年ずつ読む

### 読み直しの節約(`cache_dir`)
- `cache_dir`を渡すと、ビューから読んだ表を`<仕様の名前>_<開始日>_<終了日>.parquet`で残し、次からはビューを読まずにそれを使う。テンソルは毎回表から作る
- メタデータに入力の作り方のハッシュ(source・view_digest・特徴量・カテゴリの控え)・期間・ラベルのビューを入れ、今の仕様と違えば止める(ファイルを消すか、置き場所を替える)
- 行の並び(開催日・レースキー・馬番)を保つ(Kerasの`validation_split`は並びの末尾を検証に使う)

### 評価・推論
- `evaluate`は評価期間を1回読み、Kerasの`evaluate`(出力ごとのloss・指標)と`predict`(出走馬ごとの1〜5着の確率)を表で返す。評価期間は締めた年(`model.races`の`year_complete`)に限る。DBに書くのは使う側
  - 確率の列: `racehorsekey`・`first_prob`〜`fifth_prob`(sandboxの`trial_predictions`と同じ名前)。loss・指標の列: `output`(出力の名前、合計は`total`)・`metric`・`value`
- 出力の名前は決め打ちせず、`save_model_feature_spec`でモデルの`output_names`(1着→5着の順)を仕様に写し、その順で読む。`model.predict`の結果は名前→値でも、出力の順のリストでもよい
- `predict_finish_probabilities`は1〜3着の確率だけを素の値で返す(連対率・オッズは使う側のビューで出す)。出走のない日は空

## 置き場所(`machine_learning`の層)
使う側は入口(`features`)と、仕様の型(入口から取れる`FeatureSpec`など)だけを使う。engine・ファイルの扱いは見せない。

| 置き場所 | 中身 |
|---|---|
| `features.py`(層の外) | 入口。infrastructureの接続・ファイルを作り、interface_adapterの読み手・置き場に渡し、手続きを呼ぶ(組み立てだけ) |
| `entity/` | 仕様(`feature_spec.py`)、カテゴリの決まり(`category.py`。探す先の既定・項目を1つに決める・コード→位置)、データセットの形(`dataset.py`)、入力・ラベルの列の契約と決まり(`input_view.py`)、期間(`period.py`)、馬券の馬番の組み合わせとカテゴリ番号の変換`Labelref`(`labelutil.py`) |
| `usecase/` | 手続き:仕様の固定・読む・推論・評価(`view_dataset.py`)、テンソルへの組み立て(`frame_dataset.py`)、モデルの仕様の読み書き(`model_spec.py`)。外に触るところはProtocol(`iview_reader.py`・`iframe_store.py`・`imodel_folder.py`・`ifinish_model.py`) |
| `interface_adapter/` | PostgresのSQL(`view_gateway.py`。照合・ハッシュ・読む・code_categories・締めた年)、表⇔Parquet(`frame_format.py`)と置き場(`frame_store.py`)、仕様⇔YAMLの辞書(`spec_format.py`) |
| `infrastructure/` | 接続・トランザクション(`sql_session.py`)、ファイル(`files.py`)、SavedModelのフォルダ(`saved_model.py`) |

JRDBのコードの値と解釈(0.7.1まであった`CategoryGetter`の知識)は、分析用のPostgresのseeds(JRDBのコード表の写し)とmodel・featuresの`code_categories`にある(sazanami-batch)。0.7.2で`CategoryGetter`をなくした。

0.6.0まであったORMのオブジェクトから作る経路(`InputCreator`・`LabelCreator`・`PayoutCreator`・`PredictFactory`など)、jrdb-modelの表から読む経路、onehot(`size`・`base`)、`.dump`の書き出しはなくした。2019年の1年分でORMの経路は17.0秒・1,865MB、SQLの経路は4.0秒・263MBだった(2026-10-27、分析用のMySQL)。

開催データを紐づく表ごと読む`get_kaisais`は、機械学習ではなくDBの読み込みなので`jra_tools`(`database/kaisai_loader.py`)にある。
