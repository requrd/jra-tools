# jra-tools
Sazanami-AIのデータ・学習のための道具。次の3つを持つ。

| 使い道 | モジュール | 主に使うところ |
|---|---|---|
| 学習・推論・評価の入力を作り、モデルを保存・読み込みする | `jra_tools.machine_learning.features` | Jupyter・sazanami-batch |
| 分析用のPostgresのビューを反映する | `jra_tools.views` | sazanami-batch・Jupyter |
| JRDBの取り込みの補助・開催日程の照合 | `jra_tools`(`database/`) | sazanami-batch |

## ドキュメント
| 何を知りたいか | どこ |
|---|---|
| 全体のどこにいて、何を持ち・持たないか、中の層 | [docs/architecture.md](https://github.com/requrd/jra-tools/blob/master/docs/architecture.md) |
| 仕様のYAML・ビューの列・テンソル・カテゴリ・モデルとモデルのフォルダの約束・ビューの反映 | [docs/specification.md](https://github.com/requrd/jra-tools/blob/master/docs/specification.md) |
| 関数ごとの引数・戻り値 | APIリファレンス(下の「APIリファレンス」で生成) |
| 学習・評価・本採用の手順、モデルの置き場所 | 使う側のREADME(Jupyter: One-Stone/jupyter、推論: sazanami-batch) |

## 入れる
```
pip install "jra-tools[postgres]"   # 分析用のPostgresから読むとき(ドライバのpsycopgも入る)
pip install jra-tools               # JRDBの補助だけ
```

## 学習の入力を作る(例)
接続先は環境変数`DB`(分析用のPostgres。例: `postgresql+psycopg://user:pass@host:5432/sazanami`)。engineを作る必要はない。

```python
from jra_tools.machine_learning import features

spec = features.pin_spec(features.load_feature_spec("spec.yaml"))   # ビューのハッシュとカテゴリの番号を写す
dataset = features.load_dataset(spec, 20130101, 20241231)           # dataset.x・dataset.labels
model.fit(dataset.x, dataset.labels)                                 # Keras
features.save_model(model, spec, model_dir)                          # 推論できる形のモデルのフォルダ
```

`model`は使う側が作るKerasのモデル(満たす約束は[モデルの約束](https://github.com/requrd/jra-tools/blob/master/docs/specification.md#5-モデルの約束))。

## ビューを反映する(例)
```python
from jra_tools.views import apply_views, load_view_set

apply_views(engine, load_view_set("schemas"))   # 定義が変わっていればビューを作り直す
```

定義のフォルダの形・反映の動きは[ビューの定義のフォルダと反映](https://github.com/requrd/jra-tools/blob/master/docs/specification.md#8-ビューの定義のフォルダと反映)。

## JRDB・開催日程
`from jra_tools import ...`で使う(sazanami-batchの取り込み・照合)。

| 名前 | 中身 |
|---|---|
| `annual_schedule`・`open_days`・`annual_kaisais`・`open_kaisais`(async) | Yahoo競馬の開催日・開催の一覧 |
| `missing_days`・`db_days` | DBに登録済みの開催日と、足りない日 |
| `get_kaisais(start, end)` | jrdb-modelの開催データを、紐づく表ごと読む |
| `BaoImporter` | BAOのオッズのCSVをDBに取り込む |
| `PredictRaceFactory` | 推論するレースのデータを作る |

## APIリファレンス
docstringから生成する(公開していない。手元で見る)。
```
uv run pdoc jra_tools.machine_learning.features jra_tools.views          # ブラウザで開く
uv run pdoc jra_tools.machine_learning.features jra_tools.views -o site  # HTMLに書き出す
```

## テスト
```
uv run pytest
```
Postgresを使う結合テスト(`tests/views`・`tests/machine_learning/test_integration_features.py`・`tests/machine_learning/interface_adapter/test_integration_view_gateway.py`)は、環境変数`TEST_POSTGRES`(例: `postgresql+psycopg://test_user:test_password@127.0.0.1:5432/test_database`)がなければ飛ばす。テストはスキーマ(model・features・meta・staging・sandbox)を消して作り直すので、使い捨てのPostgresを指す。
