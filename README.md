# jra-tools
For jrdb view.
https://github.com/requrd/horseview

## Usage
### Install
```
$pip install jra-tools
```
## Test
```
$pip install jra_tools/
$pytest jra_tools/tests
```
`tests/views`の結合テストはPostgresを使う。環境変数`TEST_POSTGRES`(例: `postgresql+psycopg://test_user:test_password@127.0.0.1:5432/test_database`)がなければ飛ばす。

## ビューの仕組み(`jra_tools.views`)
分析用のPostgresのmodel・featuresのビューを、定義から作る。具体的な定義は持たず、使う側(sazanami-batchの`views/`)から渡す。ビューはデータを持たないので、マイグレーションはしない。

```
views/
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

view_set = load_view_set("views")
apply_views(engine, view_set, ApplyOptions(readers=("sazanami_ro",), label="sha-xxxxxxx"))
failures = run_view_tests(engine, view_set)  # CIで使う
```

- `apply_views`は、定義(views.yaml・関数・ビューのSQL)のハッシュが`meta.view_version`の最後の記録と同じなら何もしない。違えば1トランザクションで、`schemas`のスキーマを消して作り直し、関数とビューを作り、`readers`に読む権限を付け、ハッシュを記録する
- 定義とstagingの列が合わなければ`CREATE VIEW`でPostgresがエラーにし、全体が戻る(今のビューと記録が残る)
- スキーマを消すと、ほかのスキーマに置いた、これらのビューに依存するビュー(sandboxなど)も消える
- 読んでいる最中のクエリがあると、スキーマを消す排他ロックを待つ。`lock_timeout`(既定10秒)で諦め、間を置いて`retries`回(既定5回)やり直す
- 同時に流しても、advisory lockで1つずつ動く
- テストはハッシュに含めない(テストだけを変えても作り直さない)
## featuresのビューから読む(`view_dataset`)
特徴量の仕様の最上位に`source`(featuresのビュー)を書くと、学習・推論の入力をそのビューから作る。`source`のない仕様(今の本番モデルの`feature_spec.yaml`)は、今までどおりjrdb-modelから読む(ビューから読む仕様を今の経路に渡すとエラー)。

```yaml
name: resnet4_view
max_horses: 18
source: features.entries_v1   # 入力のビュー
view_digest: 1c445b...         # 学習時のビューの定義のハッシュ(pin_view_digestが書く)
groups:
  - name: scores
    transform: standardize_per_race
    features:
      - {name: idm}                     # sourceを省略すると列名=name
      - {name: waku_win_rate, source: waku_win_rate}
```

```python
from jra_tools.machine_learning.infrastructure.view_dataset import (
    load_view_dataset, load_view_input, pin_view_digest, read_view_frame,
)

spec = pin_view_digest(engine, spec)                 # 学習の前に、今のビューのハッシュを書く
dataset = load_view_dataset(engine, spec, 20130101, 20191231)  # labels="features.labels_v1"
save_model_feature_spec(spec, model_dir)             # モデルと一緒に保存する
inputs = load_view_input(engine, spec, 20261101)    # 推論: その日の入力だけ
frame = read_view_frame(engine, spec, 20250101, 20251231)  # 組み立てる前の1頭1行のDataFrame
```

- 読んだ表(1頭1行のpandasのDataFrame)を、`frame_dataset`(`frame_to_dataset`・`frame_to_input`)が仕様に従ってテンソルにする。各レースを1〜18の枠に並べ(空き枠は0)、グループごとにone-hot・レースごとの標準化をしてから、最後に`(レース, 18, 次元)`のndarrayにする
- 今の経路(`assemble_dataset`)とは、浮動小数点の誤差(1e-15程度)の範囲で一致する。正解ラベル・払戻・並び順は完全に一致する

- 入力のビューの列: `racehorsekey`・`racekey`・`race_date`(date)・`umaban`と仕様の列。ラベルのビューの列: `racehorsekey`・`finish_order`・`win1_num`・`win1_ret`
- 読むたびに、仕様の列がビューにあるか、ビューの定義のハッシュが`view_digest`と同じかを確かめ、違えば止める。ビューの定義を変えたら(バグ修正でも)、仕様を書き直すか、ビューの版を上げる(`entries_v2`を並べる)
- ハッシュは、そのビューと、依存するビュー・関数の定義(Postgresが保存した形)から作る。関係のないビューを足したり変えたりしても変わらない。空白・コメントだけの違いでも変わらない。stagingの表は名前だけを含める。Postgresのメジャーバージョンを上げると変わることがある
- 関数の中から呼ぶ関数をたどれるのは、`RETURN`で書いたSQLの関数だけ(`AS $$...$$`の本体はPostgresが依存を記録しない)
- 1回の読み込みは1トランザクションで、先にビューを共有ロックする(読み終わるまで`apply_views`に作り直されない)。学習は1年ずつ読む
- 値のない出走(NULL)があれば止める(今の経路の「行がなければエラー」と同じ)。文字列は0にする
