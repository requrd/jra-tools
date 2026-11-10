# ビューを反映する(`jra_tools.views`)
分析用のPostgresのmodel・featuresのビューを、定義のファイルから作る。具体的な定義は持たず、使う側(sazanami-batchの`schemas/`)から渡す。ビューはデータを持たないので、マイグレーションはしない。

## 定義のフォルダ
```
schemas/
  views.yaml        管理するスキーマ・関数のSQL・作る順のビューの一覧
  model/races.sql   ビュー1つにつき1ファイル(<スキーマ>/<ビュー名>.sql)。SELECT文だけを書く
  tests/*.sql       テスト。返した行があれば失敗
```

```yaml
schemas: [model, features]
functions: [functions.sql]   # ビューより先に流すSQL。省略できる
views:                       # 作る順に並べる
  - model.races
  - model.entries
```

## 使い方
```python
from jra_tools.views import ApplyOptions, apply_views, load_view_set, run_view_tests

view_set = load_view_set("schemas")
apply_views(engine, view_set, ApplyOptions(readers=("sazanami_ro",), label="xxxxxxx"))
failures = run_view_tests(engine, view_set)  # 反映済みのビューにテストを流す(CI・分析用のPostgresの実データ)
```

| 名前 | 中身 |
|---|---|
| `load_view_set(directory)`・`parse_view_set(data, read, tests)` | 定義を読む(`ViewSet`) |
| `apply_views(engine, view_set, options)` | 定義が変わっていれば反映し、ハッシュを記録する。反映したら`True` |
| `replace_views(engine, view_set, options)` | ハッシュを読まず・記録せずに毎回反映する(試作のsandbox向け。Jupyterは`features.apply_sandbox_views`から使う) |
| `run_view_tests(engine, view_set)` | テストを流し、行を返したテスト(`TestFailure`)を返す |
| `applied_digest(conn)` | 最後に反映したハッシュ |
| `ApplyOptions` | `readers`(読む権限を付けるロール)・`label`(記録に残す文字列)・`lock_timeout`(既定10秒)・`retries`(既定5回)・`retry_wait`(既定5秒) |

## `apply_views`の動き
- 定義(views.yaml・関数・ビューのSQL)のハッシュが`meta.view_version`の最後の記録と同じなら何もしない。違えば1トランザクションでビューと関数を反映し、`readers`に読む権限を付け、ハッシュを記録する
- テストはハッシュに含めない(テストだけを変えても作り直さない)
- `meta.view_version`は作らない。なければ`ViewVersionTableMissing`で止まる。表を作るのは使う側(sazanami-batchの`apply-model-and-meta-tables`)
- スキーマは消さない(なければ作る)。同じスキーマに置いた表・マテリアライズドビュー(推論の結果など)と、拡張機能が作ったものには触らない

### 置き換えと作り直し
- まず今のビュー・関数をその場で置き換える(`CREATE OR REPLACE`)。ほかのスキーマに置いた、これらに依存するビュー(sandboxの試作など)は残る。定義から外したビュー・関数だけを消す
  - 関数は`create or replace function`で書く(`create function`だと、あるものを作れず、作り直しになる)
  - ビューは、列が同じか末尾に足すだけなら置き換えられる
- 置き換えられないとき(ビューの列を消す・改名する・型を変える、関数の戻り値の型を変えるなど)は、`schemas`のスキーマのビューと関数をすべて消して作り直す。依存するほかのスキーマのビュー(関数だけを使うものを含む)も消え、その名前を警告のログに残す
- 関数はCASCADEで消さない。表の列の既定値などが関数を使っていれば、エラーで止まる(全体が戻る)
- 定義とstagingの列が合わなければ`CREATE VIEW`でPostgresがエラーにし、全体が戻る(今のビューと記録が残る)

### ロック
- 読んでいる最中のクエリがあると、ビューを消す排他ロックを待つ。`lock_timeout`で諦め、間を置いて`retries`回やり直す
- 同時に流しても、advisory lockで1つずつ動く(`replace_views`も同じロックを取る)

## `replace_views`の動き
- `apply_views`と同じ手順(置き換え・作り直し・一覧から外したものを消す・ロック)で、毎回反映する
- `meta.view_version`を読まない・書かない。スキーマは作らない(あること)。`readers`・`label`は使わない
