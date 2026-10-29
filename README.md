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