# jra-tools
Sazanami-AIのデータ・学習のための道具。次の3つを持つ。

| 使い道 | モジュール | 主に使うところ | 説明 |
|---|---|---|---|
| 学習・推論・評価の入力を作る | `jra_tools.machine_learning.features` | Jupyter・sazanami-batch | [docs/features.md](https://github.com/requrd/jra-tools/blob/master/docs/features.md) |
| 分析用のPostgresのビューを反映する | `jra_tools.views` | sazanami-batch | [docs/views.md](https://github.com/requrd/jra-tools/blob/master/docs/views.md) |
| JRDBの取り込みの補助・開催日程の照合 | `jra_tools`(`database/`) | sazanami-batch | 下の「JRDB・開催日程」 |

内部の層の置き場所と、なくしたものの経緯は[docs/architecture.md](https://github.com/requrd/jra-tools/blob/master/docs/architecture.md)。

## 入れる
```
pip install "jra-tools[postgres]"   # 分析用のPostgresから読むとき(ドライバのpsycopgも入る)
pip install jra-tools               # JRDBの補助だけ
```

## 学習の入力を作る(最短の例)
接続先は環境変数`DB`(分析用のPostgres。例: `postgresql+psycopg://user:pass@host:5432/sazanami`)。engineを作る必要はない。

```python
from jra_tools.machine_learning import features

spec = features.load_feature_spec("spec.yaml")   # どのビューのどの列を使うか
spec = features.pin_spec(spec)                     # ビューのハッシュとカテゴリの番号を仕様に写す
dataset = features.load_dataset(spec, 20130101, 20241231)
# dataset.x: (レース数, 18, 次元)、dataset.labels: 1〜5着の正解(5つ、各(レース数, 18))

model.fit(dataset.x, dataset.labels)               # Keras
features.save_model(model, spec, "models/try1")    # 推論(sazanami-batch)が読める形で保存
```

仕様の書き方・評価・推論・試作のビュー(sandbox)は[docs/features.md](https://github.com/requrd/jra-tools/blob/master/docs/features.md)。

## ビューを反映する(最短の例)
```python
from jra_tools.views import apply_views, load_view_set

apply_views(engine, load_view_set("schemas"))   # 定義が変わっていればビューを作り直す
```

定義のフォルダの形・反映の動きは[docs/views.md](https://github.com/requrd/jra-tools/blob/master/docs/views.md)。

## JRDB・開催日程
`from jra_tools import ...`で使う(sazanami-batchの取り込み・照合)。

| 名前 | 中身 |
|---|---|
| `annual_schedule`・`open_days`・`annual_kaisais`・`open_kaisais`(async) | Yahoo競馬の開催日・開催の一覧 |
| `missing_days`・`db_days` | DBに登録済みの開催日と、足りない日 |
| `get_kaisais(start, end)` | jrdb-modelの開催データを、紐づく表ごと読む |
| `BaoImporter` | BAOのオッズのCSVをDBに取り込む |
| `PredictRaceFactory` | 推論するレースのデータを作る |

## テスト
```
uv run pytest
```
Postgresを使う結合テスト(`tests/views`・`tests/machine_learning/test_integration_features.py`・`tests/machine_learning/interface_adapter/test_integration_view_gateway.py`)は、環境変数`TEST_POSTGRES`(例: `postgresql+psycopg://test_user:test_password@127.0.0.1:5432/test_database`)がなければ飛ばす。テストはスキーマ(model・features・meta・staging・sandbox)を消して作り直すので、使い捨てのPostgresを指す。
