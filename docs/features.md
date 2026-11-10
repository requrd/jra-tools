# 入口`jra_tools.machine_learning.features`
分析用のPostgresのビューから学習・推論・評価の入力を作り、学習したモデルを推論できる形で保存・読み込みする道具。ここは各関数が何をするか・何を約束するかを書く。作業の手順(どこで学習し、どこに置き、どう本採用するか)は使う側のREADMEにある(Jupyter: One-Stone/jupyter、推論: sazanami-batch)。

- 使う側は`features`だけを使う。接続先は環境変数`DB`(分析用のPostgres)。engine・ファイルの扱いは`features`の中にある
- 読むのは、仕様の`source`のビュー(表でもよい)、ラベルのビュー(`features.labels_v1`)、`code_categories`、締めた年(`model.races.year_complete`)だけ。stagingは直接読まない

## 目次
1. [仕様](#1-仕様)
2. [データを読む](#2-データを読む)
3. [モデルのフォルダ](#3-モデルのフォルダ)
4. [評価・推論](#4-評価推論)
5. [sandboxのビュー](#5-sandboxのビュー)
6. [定数・型](#6-定数型)

## 1. 仕様
仕様は、どのビューのどの列を、どう変換して入力にするかを書いたもの。YAMLのファイルで持つ。

| 関数 | 中身 |
|---|---|
| `load_feature_spec(path)`・`save_feature_spec(spec, path)` | YAMLのファイルを読む・書く |
| `parse_feature_spec(data)` | 辞書(YAMLを読み込んだもの)から作る |
| `pin_spec(spec, category_sources=DEFAULT_CATEGORY_SOURCES)` | 学習の前に、`view_digest`とカテゴリの`categories`を写した仕様を返す |
| `digest(source)` | ビューの今の定義のハッシュ |

### 項目
| 項目 | 書く人 | 中身 |
|---|---|---|
| `name` | 使う人 | 仕様の名前(`cache_dir`のファイル名にも使う) |
| `max_horses` | 使う人 | 1レースの枠の数(18) |
| `source` | 使う人 | 入力を読むビュー(「スキーマ.名前」) |
| `groups[].name` | 使う人 | グループの名前 |
| `groups[].transform` | 使う人 | `none`、または`standardize_per_race`(レースごとに標準化。空き枠の0の行も含めた`max_horses`行で計算) |
| `groups[].features[].name` | 使う人 | 特徴量の名前 |
| `groups[].features[].source` | 使う人 | ビューの列の名前(省略すると`name`) |
| `groups[].features[].encode` | 使う人 | `number`(既定)、または`category`(one-hot) |
| `groups[].features[].categories` | `pin_spec` | コード→番号の控え |
| `view_digest` | `pin_spec` | 学習したときの`source`の定義のハッシュ |
| `train_from`・`train_to` | 使う人(保存の前) | 学習期間(初日・最終日を含む) |
| `output_names` | `save_model` | モデルの出力の名前(1着→5着) |

```yaml
name: sample
max_horses: 18
source: features.entries_v1
view_digest: 1c445b...                    # pin_specが写す
output_names: [activation_4, activation_5, activation_6, activation_7, activation_8]  # save_modelが写す
train_from: 2013-01-01
train_to: 2024-12-31
groups:
  - name: scores
    transform: standardize_per_race
    features:
      - {name: idm}
  - name: categories
    transform: none
    features:
      - name: wakuban
        encode: category
        categories: {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}  # pin_specが写す
```

- 入力の形は`(レース数, max_horses, 次元)`。次元は、数値の列の数と、カテゴリの列ごとの「最大の番号+1」の合計

### カテゴリの列
- 変換の決まりの正はDBの`code_categories`(1コード1行 `item, code, category_no, label`)。列の名前=項目の名前、列の値=コード(文字列)
- JRDBの項目名(`wakuban`など)はmodelが使う。まとめ方・試作には別の名前(例: `waku_group`)を付ける
- `pin_spec`は、`category_sources`の`code_categories`(DBにあるものだけ)を合わせて、特徴量の`source`(列の名前)で項目を引く。項目がどこにもない・2か所以上にあれば止める
- 番号の数+1列目は例外の列。控えにないコード(学習のあとに増えたコードなど)・NULL(modelで記載外にしたコード)はここに入る
- 推論はDBの`code_categories`を読まず、仕様の控えだけを使う(seedsを更新してもビューのハッシュは変わらず、番号がずれうるため)

### ビューのハッシュ
- 読むたびに、仕様の列が`source`にあるか、`source`の定義のハッシュが`view_digest`と同じかを確かめ、違えば止める。定義を変えたら(バグ修正でも)、仕様を固定し直すか、版を上げる(`entries_v2`を並べる)
- ハッシュは、`source`と、依存するビュー・関数の定義(Postgresが保存した形)から作る。関係のないものを変えても変わらない。空白・コメントだけの違いでも変わらない。stagingの表は名前だけを含める。Postgresのメジャーバージョンを上げると変わることがある
- 名前も含むので、sandboxのビューで固定した仕様は、同じ定義を`features.`へ移すと合わなくなる
- 関数の中から呼ぶ関数をたどれるのは、`RETURN`で書いたSQLの関数だけ(`AS $$...$$`の本体はPostgresが依存を記録しない)

## 2. データを読む
| 関数 | 中身 |
|---|---|
| `load_dataset(spec, start, end, labels=DEFAULT_LABELS, cache_dir=None)` | 学習のデータセット(`TableDataset`) |
| `read_frame(spec, start, end, labels=DEFAULT_LABELS, cache_dir=None)` | 組み立てる前の1頭1行の表(DataFrame)。`labels=None`ならラベルを付けない |
| `load_input(spec, ymd)` | 開催日の推論の入力(`InputDataset`) |

```python
dataset = features.load_dataset(spec, 20130101, 20241231, cache_dir="cache")
```

- 期間はYYYYMMDDの整数で、両端を含む。1年ずつ読む。仕様は`pin_spec`で固定してあること
- `TableDataset`: `x`(`(レース数, max_horses, 次元)`。空き枠は0)、`labels`(1着〜5着の正解のリスト。各`(レース数, max_horses)`のone-hot)、`racekeys`(各レースのキー)。払戻は持たない
- 1回の読み込みは1トランザクションで、先に`source`を共有ロックする(読み終わるまで`apply_views`に作り直されない)

### 列の約束
- 入力のビューに要る列: `racehorsekey`・`racekey`・`race_date`(date)・`umaban`と、仕様の列
- ラベルのビューの列: `racehorsekey`・`finish_order`
- 数値の列に値なし(NULL)があれば止める。文字列は0にする
- 成績(ラベルのビューの行)のない出走があれば止める
- 各レースを1〜`max_horses`の枠に並べ(空き枠は0)、グループごとにカテゴリのone-hot・レースごとの標準化をしてから組み立てる

### 読み直しの節約(`cache_dir`)
- ビューから読んだ表を`<仕様の名前>_<開始日>_<終了日>.parquet`で残し、次からはビューを読まずにそれを使う。テンソルは毎回表から作る
- メタデータに入力の作り方のハッシュ(`source`・`view_digest`・特徴量・カテゴリの控え)・期間・ラベルのビューを入れ、今の仕様と違えば止める(ファイルを消すか、置き場所を替える)
- 行の並び(開催日・レースキー・馬番)を保つ(Kerasの`validation_split`は並びの末尾を検証に使う)

## 3. モデルのフォルダ
学習したモデルと、その入力の作り方(仕様)を1つのフォルダに置く。推論する側(`predict_finish_probabilities`、sazanami-batch)は、このフォルダだけで入力を作り、モデルを呼べる。

| 関数 | 中身 |
|---|---|
| `save_model(model, spec, model_dir)` | Kerasのモデルと、出力の名前を写した仕様を書く |
| `save_model_feature_spec(spec, model_dir, output_names)` | 仕様だけを書く(モデルはフォルダに保存済みのとき) |
| `load_model_feature_spec(model_dir)` | 仕様を読み、フォルダのモデルと照合する |

### フォルダの中身
```
<model_dir>/
  saved_model.pb       TensorFlowのSavedModel
  variables/           重み
  assets/
  fingerprint.pb
  feature_spec.yaml    仕様(view_digest・categories・output_names・学習期間を含む)
```

- `serving_default`のシグネチャが、入力1つ(`(None, max_horses, 次元)`)を受け取り、「出力の名前→1着〜5着それぞれの確率(`(None, max_horses)`)」の辞書を返す
- `feature_spec.yaml`の`output_names`は、その辞書のキーを1着→5着の順に並べたもの

### 書く(`save_model`)
- `model`はKerasのモデル(`input_shape`・`output_names`を持ち、出力は1着→5着の順の5つ)。`spec`は`pin_spec`で固定したもの。学習期間を残すなら`train_from`・`train_to`を入れておく
- 書く前に、仕様とモデルの入力の形を照合する。合わなければ何も書かない
- Keras 3の`model.export`は使わない(出力の名前が`output_0`…になり、名前で引けないため)。`save_model`は`serving_default`が出力の名前で返すように書く

### 読む(`load_model_feature_spec`)
- 仕様が固定されていること、仕様とモデルの入力の形、仕様の`output_names`とモデルの出力の名前を照合し、合わなければ止める
- モデル本体は読まない。推論に使うモデルは使う側が読む(例: `keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")`。sazanami-batchはこの読み方)
- `train_from`・`train_to`は、推論する日が学習期間に入っていないかを使う側が確かめるためのもの

## 4. 評価・推論
| 関数 | 中身 |
|---|---|
| `evaluate(spec, start, end, model, labels=DEFAULT_LABELS, cache_dir=None)` | 評価期間の出走馬ごとの1〜5着の確率と、Kerasの`evaluate`の結果(DataFrame 2つ) |
| `predict_finish_probabilities(spec, ymd, model)` | 開催日の出走馬ごとの`(racehorsekey, 1着, 2着, 3着の確率)`のリスト。出走がなければ空 |

- `spec`は`load_model_feature_spec`で読んだもの(`output_names`が要る)
- `evaluate`の`model`はcompileしたKerasのモデル(`evaluate`・`predict`を呼ぶ)。評価期間は締めた年(`model.races`の`year_complete`)に限り、締めていない年があれば止める
  - 確率の表: `racehorsekey`・`first_prob`〜`fifth_prob`
  - `evaluate`の表: `output`(出力の名前、合計は`total`)・`metric`(`loss`・`accuracy`など)・`value`
- `predict_finish_probabilities`の`model`は`predict`を持つもの(Kerasのモデル、`TFSMLayer`)。`predict`の結果は「名前→値」でも出力の順のリストでもよく、名前のときは`output_names`で引く。確率は素の値(連対率・オッズは使う側のビューで出す)
- DBには書かない。買い方・回収率の集計も持たない(確率を`model.entries`・`model.payouts`と`racehorsekey`で結合して、使う側が出す)

## 5. sandboxのビュー
| 関数 | 中身 |
|---|---|
| `apply_sandbox_views(directory)` | `views.yaml`と同じ形のフォルダの定義で、sandboxのビューを反映する |

```
<directory>/
  views.yaml                 schemas: [sandbox] と、作る順のビューの一覧
  sandbox/<ビュー名>.sql     ビュー1つにつき1ファイル。SELECT文だけ
```

- 毎回、一覧のビューをその場で置き換える(できなければsandboxのビューをすべて作り直す。表は残る)。ハッシュは記録しない(`meta`を読めないロールでも流せる)
- 一覧にないsandboxのビュー(手で作ったものも)は消える
- sandboxのスキーマと、ビューが読む表は先にあること
- 反映の手順・ロックは`jra_tools.views`の`replace_views`と同じ([views.md](views.md))

## 6. 定数・型
| 名前 | 中身 |
|---|---|
| `DEFAULT_LABELS` | ラベルのビュー(`features.labels_v1`) |
| `DEFAULT_CATEGORY_SOURCES` | `pin_spec`がカテゴリを探す先(`model.code_categories`・`features.code_categories`・`sandbox.code_categories`) |
| `PROBABILITY_COLUMNS` | 確率の列の名前(`first_prob`〜`fifth_prob`) |
| `FeatureSpec` | 仕様 |
| `TableDataset` | 学習のデータセット(`racekeys`・`x`・`labels`) |
| `InputDataset` | 推論の入力(`racekeys`・`x`) |
| `Labelref` | 馬券の馬番の組み合わせとカテゴリ番号の変換 |
