# 仕様
jra-toolsが受け取るもの・作るものが満たす約束。全体の流れは[architecture.md](architecture.md)、関数ごとの引数・戻り値はAPIリファレンス(READMEの「APIリファレンス」)にある。手順(どこで学習し、どこに置くか)は使う側のREADME(Jupyter: One-Stone/jupyter、推論: sazanami-batch)にある。

## 目次
1. [特徴量の仕様](#1-特徴量の仕様)
2. [ビューの列](#2-ビューの列)
3. [テンソル](#3-テンソル)
4. [カテゴリ](#4-カテゴリ)
5. [モデルの約束](#5-モデルの約束)
6. [モデルのフォルダ](#6-モデルのフォルダ)
7. [ビューのハッシュと読み直しの節約](#7-ビューのハッシュと読み直しの節約)
8. [ビューの定義のフォルダと反映](#8-ビューの定義のフォルダと反映)

## 1. 特徴量の仕様
どのビューのどの列を、どう変換して入力にするかを書いたYAMLのファイル。学習・評価・推論のどれも、入力はこの仕様だけから作る。

```yaml
name: sample                           # 仕様の名前
max_horses: 18                         # 1レースの枠の数
source: features.entries_v1            # 入力を読むビュー(表でもよい)
view_digest: 1c445b...                 # 固定するときに写る
train_from: 2013-01-01                 # 学習期間(両端を含む。保存する前に書く)
train_to: 2024-12-31
output_names: [activation_4, activation_5, activation_6, activation_7, activation_8]  # 保存するときに写る
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
        categories: {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}  # 固定するときに写る
```

| 項目 | 書く人 | 中身 |
|---|---|---|
| `name` | 使う人 | 仕様の名前(読み直しの節約のファイル名にも使う) |
| `max_horses` | 使う人 | 1レースの枠の数(18) |
| `source` | 使う人 | 入力を読むビュー(「スキーマ.名前」) |
| `groups[].name` | 使う人 | グループの名前 |
| `groups[].transform` | 使う人 | `none`、または`standardize_per_race`(レースごとに標準化。空き枠の0の行も含めた`max_horses`行で計算) |
| `groups[].features[].name` | 使う人 | 特徴量の名前 |
| `groups[].features[].source` | 使う人 | ビューの列の名前(省略すると`name`) |
| `groups[].features[].encode` | 使う人 | `number`(既定)、または`category`(one-hot) |
| `groups[].features[].categories` | 固定(`pin_spec`) | コード→番号の控え([4. カテゴリ](#4-カテゴリ)) |
| `view_digest` | 固定(`pin_spec`) | `source`の定義のハッシュ([7.](#7-ビューのハッシュと読み直しの節約)) |
| `train_from`・`train_to` | 使う人 | 学習期間。推論する日が学習期間に入っていないかを、使う側が確かめるためのもの |
| `output_names` | 保存(`save_model`) | モデルの出力の名前(1着→5着の順。[5.](#5-モデルの約束)) |

### 固定
- 入力を作る(読む・保存する・推論する)には、仕様が固定されていること(`view_digest`とすべての`categories`がある)。なければ止まる
- 固定は学習の前に1回だけ。固定した仕様をモデルと一緒に保存し、推論はそれを使う(推論のときにDBからカテゴリの番号を引き直さない)

## 2. ビューの列
| ビュー | 要る列 |
|---|---|
| 入力(仕様の`source`) | `racehorsekey`・`racekey`・`race_date`(date)・`umaban`と、仕様の特徴量の列 |
| ラベル(既定は`features.labels_v1`) | `racehorsekey`・`finish_order` |

- 数値の列に値なし(NULL)があれば止まる(NULLの扱いはビューで決める)。文字列は0にする
- 成績(ラベルのビューの行)のない出走があれば止まる
- 期間はYYYYMMDDの整数で、両端を含む。1年ずつ読む
- 1回の読み込みは1トランザクションで、先に`source`を共有ロックする(読み終わるまでビューは作り直されない)
- 評価の期間は締めた年(`model.races.year_complete`)だけ。締めていない年があれば止まる

## 3. テンソル
| 名前 | 形 | 中身 |
|---|---|---|
| `x` | `(レース数, max_horses, 次元)` | 入力。馬番 − 1 の位置に各馬を置き、空き枠は0 |
| `labels` | 5つの`(レース数, max_horses)`のリスト | 1着〜5着の正解のone-hot(1着→5着の順) |
| `racekeys` | `(レース数,)` | 各レースのキー |

- 次元 = 数値の列の数 + カテゴリの列ごとの(最大の番号 + 1)の合計。仕様の`size`で取れる
- グループごとに、カテゴリのone-hot・レースごとの標準化をしてから、グループの順に並べる
- レースの並びは開催日・レースキーの順(Kerasの`validation_split`は末尾を検証に使うので、検証は新しいレースになる)
- 推論の入力は`x`と`racekeys`だけ(ラベルを読まない)

## 4. カテゴリ
- 変換の決まりの正は、分析用のPostgresの`code_categories`(1コード1行 `item, code, category_no, label`)。番号は1から
- 特徴量の`source`(列の名前)を項目の名前(`item`)として引く。列の値はコード(文字列)
- 探す先は`model.code_categories`・`features.code_categories`・`sandbox.code_categories`のうちDBにあるもの。項目がどこにもない・2か所以上にあれば止まる
- JRDBの項目名(`wakuban`など)はmodelが使う。まとめ方を替えるときは別の項目名(例: `waku_group`)を付ける。ビューの列の名前が項目名と違うときは、ビューで別名を付ける(例: `waku as wakuban`)
- one-hotは(番号の数 + 1)列。最後の列は例外の列で、控えにないコード(学習のあとに増えたものなど)とNULLが入る。例: 枠は8 + 1 = 9次元

## 5. モデルの約束
jra-toolsはモデルを作らない。使う側が作ったモデルを、保存・評価・推論のときに受け取る。モデルは次を満たすこと。

| 項目 | 約束 |
|---|---|
| 入力 | 1つ。形は`(None, max_horses, 次元)`(仕様の`max_horses`・`size`と同じ) |
| 出力 | 5つ。順に1着・2着・3着・4着・5着の確率で、それぞれ`(None, max_horses)`。各レースの枠で合計1(softmax) |
| 出力の名前 | 5つとも違う名前(Kerasの`model.output_names`)。保存するときに仕様の`output_names`に写り、推論はこの名前で出力を引く |
| 学習 | `labels`(1着→5着の順の5つ)を正解にする。損失は出力ごとの`categorical_crossentropy`が素直 |
| 評価に渡すとき | compile済み(`evaluate`・`predict`を呼ぶ)。指標の名前は`<出力の名前>_<指標>`で、出力ごとに分ける |
| 推論に渡すとき | `predict(x)`が、出力の名前→値の辞書か、出力の順のリストを返す(Kerasのモデル、`TFSMLayer`のどちらでもよい) |

最小の形(Keras):
```python
import keras

x_in = keras.Input((spec.max_horses, spec.size))
h = keras.layers.Conv1D(64, 1, activation="relu")(x_in)          # 枠ごとに同じ重みをかける
outputs = [
    keras.layers.Activation("softmax")(keras.layers.Flatten()(keras.layers.Conv1D(1, 1)(h)))
    for _ in range(5)                                             # 1着→5着
]
model = keras.Model(x_in, outputs)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"] * 5)
model.fit(dataset.x, dataset.labels)
```
実際に使っているモデル(ResNet)は、Jupyterの`model_creator.py`にある。

## 6. モデルのフォルダ
学習したモデルと、その入力の作り方(仕様)を置くフォルダ。推論する側は、このフォルダだけで入力を作り、モデルを呼べる。

```
<model_dir>/
  saved_model.pb       TensorFlowのSavedModel
  variables/           重み
  assets/
  fingerprint.pb
  feature_spec.yaml    固定した仕様(output_names・学習期間を含む)
```

- `serving_default`のシグネチャが、入力1つ(`(None, max_horses, 次元)`)を受け取り、「出力の名前→確率(`(None, max_horses)`)」の辞書を返す
- `feature_spec.yaml`の`output_names`は、その辞書のキーを1着→5着の順に並べたもの
- 書くとき(`save_model`)・読むとき(`load_model_feature_spec`)に、仕様が固定されていること、仕様とモデルの入力の形、`output_names`とモデルの出力の名前を照合し、合わなければ止まる(書くときは何も書かない)
- Keras 3の`model.export`で書いたフォルダは、出力の名前が`output_0`…になり約束を満たさない。`save_model`は出力を名前で返すように書く
- jra-toolsはモデル本体を読み込まない。推論に使うモデルは使う側が読む(sazanami-batchは`keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")`)。評価には、フォルダではなくcompileしたKerasのモデルを渡す

## 7. ビューのハッシュと読み直しの節約
### ハッシュ
- 読むたびに、仕様の列が`source`にあるか、`source`の定義のハッシュが`view_digest`と同じかを確かめ、違えば止まる。定義を変えたら(バグ修正でも)、仕様を固定し直すか、版を上げたビューを並べる(`entries_v2`)
- ハッシュは、`source`と、依存するビュー・関数の定義(Postgresが保存した形)から作る。関係のないものを変えても変わらない。空白・コメントだけの違いでは変わらない。stagingの表は名前だけを含める。Postgresのメジャーバージョンを上げると変わることがある
- 名前も含むので、sandboxのビューで固定した仕様は、同じ定義を`features.`へ移すと合わない
- 関数の中から呼ぶ関数をたどれるのは、`RETURN`で書いたSQLの関数だけ(`AS $$...$$`の本体はPostgresが依存を記録しない)

### 読み直しの節約(`cache_dir`)
- ビューから読んだ表を`<仕様の名前>_<開始日>_<終了日>.parquet`で残し、次からはビューを読まずにそれを使う。テンソルは毎回表から作る
- ファイルには入力の作り方のハッシュ(`source`・`view_digest`・特徴量・カテゴリの控え)・期間・ラベルのビューを入れ、今の仕様と違えば止まる(ファイルを消すか、置き場所を替える)

## 8. ビューの定義のフォルダと反映
### 定義のフォルダ
```
schemas/
  views.yaml               管理するスキーマ・関数のSQL・作る順のビューの一覧
  <スキーマ>/<ビュー名>.sql  ビュー1つにつき1ファイル。SELECT文だけ
  tests/*.sql              テスト。行を返したら失敗(省略できる)
```
```yaml
schemas: [model, features]
functions: [functions.sql]   # ビューより先に流すSQL。省略できる
views:                       # 作る順
  - model.races
  - model.entries
```
sandboxの試作は`schemas: [sandbox]`で、ビューだけを書く。

### 反映(model・features: `apply_views`)
- 定義(views.yaml・関数・ビューのSQL)のハッシュが`meta.view_version`の最後の記録と同じなら何もしない。違えば1トランザクションでビューと関数を反映し、読む権限を付け、ハッシュを記録する。テストはハッシュに含めない
- `meta.view_version`は作らない(なければ止まる。作るのはsazanami-batch)。スキーマは消さない(なければ作る)。同じスキーマの表・マテリアライズドビュー・拡張機能のものには触らない
- まず今のビュー・関数をその場で置き換える(`CREATE OR REPLACE`)。ほかのスキーマの、これらに依存するビュー(sandboxの試作など)は残る。定義から外したビュー・関数だけを消す
  - ビューは、列が同じか末尾に足すだけなら置き換えられる。関数は`create or replace function`で書く
- 置き換えられないとき(列を消す・改名する・型を変える、関数の戻り値の型を変えるなど)は、`schemas`のスキーマのビューと関数をすべて消して作り直す。依存するほかのスキーマのビューも消え、その名前を警告のログに残す
- 関数はCASCADEで消さない。表の列の既定値などが関数を使っていれば止まる(全体が戻る)
- 定義とstagingの列が合わなければPostgresがエラーにし、全体が戻る(今のビューと記録が残る)

### 反映(sandbox: `replace_views`)
- 上と同じ手順で、ハッシュを読まず・記録せずに毎回反映する(`meta`を読めないロールでも流せる)。スキーマは作らない。権限は付けない
- 一覧にないsandboxのビュー(手で作ったものも)は消える。表は残る

### ロック
- 読んでいる最中のクエリがあると、ビューを消す排他ロックを待つ。待つのは既定10秒で、間を置いて既定5回やり直す
- 同時に流しても、advisory lockで1つずつ動く(`apply_views`・`replace_views`で共通)
