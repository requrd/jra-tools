# 学習・推論・評価の入力を作る(`jra_tools.machine_learning.features`)
分析用のPostgresのビュー(`features.entries_v1`や、試作の`sandbox.entries_try1`)から、Kerasに渡す入力を作る。使う側はこの`features`だけを使う。

- 接続先は環境変数`DB`(分析用のPostgres)。engine・ファイルの扱いは`features`の中にあり、使う側は作らない
- jra-toolsはstagingを直接読まない。読むのは仕様の`source`のビュー(表でもよい)、ラベルのビュー(`features.labels_v1`)、`code_categories`、締めた年(`model.races.year_complete`)だけ

## 目次
1. [仕様を書く](#1-仕様を書く)
2. [仕様を固定する](#2-仕様を固定する)
3. [学習データを読む](#3-学習データを読む)
4. [モデルを保存する](#4-モデルを保存する)
5. [評価する](#5-評価する)
6. [推論する](#6-推論する)
7. [試作のビュー(sandbox)を作る](#7-試作のビューsandboxを作る)
8. [関数の一覧](#関数の一覧)
9. [仕様の項目](#仕様の項目)
10. [詳しい決まり](#詳しい決まり)

## 1. 仕様を書く
仕様(spec.yaml)は、どのビューのどの列を、どう変換して入力にするかを書いたもの。

```yaml
name: entries_try1
max_horses: 18
source: sandbox.entries_try1          # 入力を読むビュー(「スキーマ.名前」)
groups:
  - name: scores
    transform: standardize_per_race   # レースの中で標準化する
    features:
      - {name: idm}                   # name = ビューの列の名前
      - {name: jockey_score}
  - name: categories
    transform: none
    features:
      - {name: wakuban, encode: category}   # 列の値はコード。番号はcode_categoriesから
```

- 数値の列はそのまま、`encode: category`の列はone-hotにする
- カテゴリの列の名前は`code_categories`の項目名にする(`wakuban`・`waku_group`など。[カテゴリの列](#カテゴリの列))
- 入力の形は`(レース数, max_horses, 次元)`。次元は数値の列の数と、カテゴリの列ごとの「番号の数+1」の合計

## 2. 仕様を固定する
```python
spec = features.load_feature_spec("spec.yaml")
spec = features.pin_spec(spec)
features.save_feature_spec(spec, "spec.pinned.yaml")   # 残すなら
```

`pin_spec`は、学習の前に次の2つを仕様に写す。

- `view_digest`: `source`のビューの今の定義のハッシュ。あとで読むとき、定義が変わっていれば止める
- カテゴリの列の`categories`: 「コード→番号」の控え。推論はこの控えだけで番号にする

## 3. 学習データを読む
```python
dataset = features.load_dataset(spec, 20130101, 20241231, cache_dir="cache")
model.fit(dataset.x, dataset.labels, validation_split=0.1)
```

- 期間はYYYYMMDDの整数で、両端を含む。1年ずつ読む
- `dataset.x`: `(レース数, 18, 次元)`。空き枠は0
- `dataset.labels`: 1着〜5着の正解のリスト(5つ、各`(レース数, 18)`のone-hot)
- `dataset.racekeys`: 各レースのキー
- `cache_dir`を渡すと、ビューから読んだ表をParquetで残し、次からはそれを使う([読み直しの節約](#読み直しの節約cache_dir))
- 組み立てる前の1頭1行の表は`features.read_frame(spec, start, end)`で読める

## 4. モデルを保存する
```python
import dataclasses
from datetime import date

spec = dataclasses.replace(spec, train_from=date(2013, 1, 1), train_to=date(2024, 12, 31))
features.save_model(model, spec, "models/try1")
```

- モデルをSavedModelで書き、同じフォルダに仕様(`feature_spec.yaml`)を書く。仕様には学習期間と、モデルの出力の名前(`output_names`、1着→5着の順)が入る
- Keras 3の`model.export`は使わない。出力の名前が`output_0`…になり、推論(sazanami-batch)が出力を名前で引けないため。`save_model`は`serving_default`が「出力の名前→値」を返すように書く
- 仕様とモデルの入力の形が合わなければ止める(何も書かない)
- `train_from`・`train_to`は、推論する日が学習期間に入っていないかを使う側(batch)が確かめるためのもの

## 5. 評価する
```python
spec = features.load_model_feature_spec("models/try1")
predictions, metrics = features.evaluate(spec, 20250101, 20250331, model)
```

- `model`はcompileしたKerasのモデル(`evaluate`を呼ぶため)
- 評価期間は締めた年(`model.races`の`year_complete`)に限る。締めていない年があれば止める
- `predictions`: 出走馬ごとの1〜5着の確率。列は`racehorsekey`・`first_prob`〜`fifth_prob`
- `metrics`: Kerasの`evaluate`の結果。列は`output`(出力の名前、合計は`total`)・`metric`(`loss`・`accuracy`など)・`value`
- DBに書くのは使う側(Jupyterの`lab.evaluate_quarter`)。回収率は、確率を`model.entries`・`model.payouts`と`racehorsekey`で結合して出す(jra-toolsは買い方・集計を持たない)

## 6. 推論する
```python
spec = features.load_model_feature_spec("models/try1")
probs = features.predict_finish_probabilities(spec, 20261101, model)
# [(racehorsekey, 1着の確率, 2着の確率, 3着の確率), ...]。出走がなければ空
```

- `load_model_feature_spec`は、仕様とモデルの入力の形・出力の名前を照合し、合わなければ止める
- `model.predict`の結果は「名前→値」(`TFSMLayer`)でも、出力の順のリスト(Kerasのモデル)でもよい。名前のときは仕様の`output_names`で引く
- 確率は素の値。連対率・オッズは使う側のビューで出す
- 入力だけが要るときは`features.load_input(spec, ymd)`

## 7. 試作のビュー(sandbox)を作る
試作の特徴量は、sandboxのビューにして仕様の`source`に書く。定義は`views.yaml`と同じ形のフォルダに置く。

```
schemas/
  views.yaml               schemas: [sandbox] と、作る順のビューの一覧
  sandbox/entries_try1.sql ビュー1つにつき1ファイル。SELECT文だけ
```

```python
features.apply_sandbox_views("schemas")
```

- 毎回、一覧のビューをその場で置き換える(できなければsandboxのビューをすべて作り直す。表は残る)。ハッシュは記録しない
- 一覧にないsandboxのビュー(手で作ったものも)は消える
- sandboxのスキーマと、ビューが読む表は先にあること
- sandboxのビューで学習したモデルは、本採用(`features.`のビューへ移す)のあとに学習し直す(ビューの名前もハッシュに入るため)

## 関数の一覧
| 関数 | 中身 |
|---|---|
| `load_feature_spec(path)`・`save_feature_spec(spec, path)` | 仕様のYAMLを読む・書く |
| `parse_feature_spec(data)` | 辞書から仕様を作る |
| `pin_spec(spec, category_sources=DEFAULT_CATEGORY_SOURCES)` | ビューのハッシュとカテゴリの控えを写す |
| `digest(source)` | ビューの今の定義のハッシュ |
| `read_frame(spec, start, end, labels=DEFAULT_LABELS, cache_dir=None)` | 組み立てる前の1頭1行の表(DataFrame) |
| `load_dataset(spec, start, end, labels=DEFAULT_LABELS, cache_dir=None)` | 学習のデータセット(`TableDataset`: `racekeys`・`x`・`labels`) |
| `save_model(model, spec, model_dir)` | モデルと、出力の名前・学習期間を写した仕様を保存する |
| `save_model_feature_spec(spec, model_dir, output_names)` | 仕様だけを保存する(モデルは保存済みのとき) |
| `load_model_feature_spec(model_dir)` | モデルのフォルダの仕様を読む(入力の形・出力の名前を照合) |
| `evaluate(spec, start, end, model, labels=DEFAULT_LABELS, cache_dir=None)` | 評価期間の確率とloss・指標(DataFrame 2つ) |
| `load_input(spec, ymd)` | 開催日の推論の入力(`InputDataset`) |
| `predict_finish_probabilities(spec, ymd, model)` | 開催日の出走馬ごとの1〜3着の確率 |
| `apply_sandbox_views(directory)` | sandboxのビューを反映する |

| 定数・型 | 中身 |
|---|---|
| `DEFAULT_LABELS` | ラベルのビュー(`features.labels_v1`) |
| `DEFAULT_CATEGORY_SOURCES` | `pin_spec`がカテゴリを探す先(`model.code_categories`・`features.code_categories`・`sandbox.code_categories`) |
| `PROBABILITY_COLUMNS` | 確率の列の名前(`first_prob`〜`fifth_prob`) |
| `FeatureSpec`・`TableDataset`・`InputDataset` | 仕様・学習のデータセット・推論の入力 |
| `Labelref` | 馬券の馬番の組み合わせとカテゴリ番号の変換 |

## 仕様の項目
| 項目 | 書く人 | 中身 |
|---|---|---|
| `name` | 使う人 | 仕様の名前(`cache_dir`のファイル名にも使う) |
| `max_horses` | 使う人 | 1レースの枠の数(18) |
| `source` | 使う人 | 入力を読むビュー(「スキーマ.名前」) |
| `groups[].name` | 使う人 | グループの名前 |
| `groups[].transform` | 使う人 | `none`、または`standardize_per_race`(レースごとに標準化。空き枠の0の行も含めた`max_horses`行で計算) |
| `groups[].features[].name` | 使う人 | 特徴量の名前 |
| `groups[].features[].source` | 使う人 | ビューの列の名前(省略すると`name`) |
| `groups[].features[].encode` | 使う人 | `number`(既定)、または`category` |
| `groups[].features[].categories` | `pin_spec` | コード→番号の控え |
| `view_digest` | `pin_spec` | 学習したときの`source`の定義のハッシュ |
| `train_from`・`train_to` | 使う人(保存の前) | 学習期間(初日・最終日を含む) |
| `output_names` | `save_model` | モデルの出力の名前(1着→5着) |

固定・保存したあとの仕様の例:

```yaml
name: entries_try1
max_horses: 18
source: sandbox.entries_try1
view_digest: 1c445b...
output_names: [activation_4, activation_5, activation_6, activation_7, activation_8]
train_from: 2013-01-01
train_to: 2024-12-31
groups:
  - name: categories
    transform: none
    features:
      - name: wakuban
        encode: category
        categories: {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}
```

## 詳しい決まり
### 入力とラベルの列
- 入力のビューに要る列: `racehorsekey`・`racekey`・`race_date`(date)・`umaban`と、仕様の列
- ラベルのビューの列: `racehorsekey`・`finish_order`
- 数値の列に値なし(NULL)があれば止める。文字列は0にする
- 成績(ラベルのビューの行)のない出走があれば止める
- 各レースを1〜18の枠に並べ(空き枠は0)、グループごとにカテゴリのone-hot・レースごとの標準化をしてから、`(レース, 18, 次元)`にする
- 学習のデータセットは入力と1〜5着の正解だけで、払戻は持たない

### カテゴリの列
- 変換の決まりの正はDBの`code_categories`(1コード1行 `item, code, category_no, label`)。列の名前=項目の名前、列の値=コード(文字列)
- JRDBの項目名(`wakuban`など)はmodelが使う。まとめ方・試作には別の名前(例: `waku_group`)を付ける
- `pin_spec`は、`category_sources`の`code_categories`(DBにあるものだけ)を合わせて項目名で引く。項目がどこにもない・2か所以上にあれば止める
- 次元は最大の番号+1(例外の列)。控えにないコード(学習のあとに増えたコードなど)・NULL(modelで記載外にしたコード)は例外の列
- 推論はDBの`code_categories`を読まず、仕様の控えだけを使う(seedsを更新してもビューのハッシュは変わらず、番号がずれうるため)

### ビューのハッシュ
- 読むたびに、仕様の列が`source`にあるか、`source`の定義のハッシュが`view_digest`と同じかを確かめ、違えば止める。定義を変えたら(バグ修正でも)、仕様を固定し直すか、版を上げる(`entries_v2`を並べる)
- ハッシュは、`source`と、依存するビュー・関数の定義(Postgresが保存した形)から作る。関係のないものを変えても変わらない。空白・コメントだけの違いでも変わらない。stagingの表は名前だけを含める。Postgresのメジャーバージョンを上げると変わることがある
- 関数の中から呼ぶ関数をたどれるのは、`RETURN`で書いたSQLの関数だけ(`AS $$...$$`の本体はPostgresが依存を記録しない)
- 1回の読み込みは1トランザクションで、先に`source`を共有ロックする(読み終わるまで`apply_views`に作り直されない)

### 読み直しの節約(`cache_dir`)
- ビューから読んだ表を`<仕様の名前>_<開始日>_<終了日>.parquet`で残し、次からはビューを読まずにそれを使う。テンソルは毎回表から作る
- メタデータに入力の作り方のハッシュ(`source`・`view_digest`・特徴量・カテゴリの控え)・期間・ラベルのビューを入れ、今の仕様と違えば止める(ファイルを消すか、置き場所を替える)
- 行の並び(開催日・レースキー・馬番)を保つ(Kerasの`validation_split`は並びの末尾を検証に使う)

### 出力の名前
- 出力の名前は決め打ちせず、`save_model`でモデルの`output_names`(1着→5着の順)を仕様に写し、その順で読む
- `load_model_feature_spec`・`save_model_feature_spec`は、仕様の`output_names`とフォルダのモデルの出力の名前が一致しなければ止める(推論で引けないため)
