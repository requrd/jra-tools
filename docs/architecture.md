# アーキテクチャ
jra-toolsが全体のどこにいて、何を持ち、何を持たないか。データ・ファイル・モデルが満たす約束は[specification.md](specification.md)、関数ごとの引数・戻り値はAPIリファレンス(READMEの「APIリファレンス」)にある。

## 1. 全体の中での位置
```
[ビューを作る]
  sazanami-batchのschemas/ ──apply_views──────────▶ model・featuresのビュー
  Jupyterのschemas/        ──apply_sandbox_views──▶ sandboxのビュー(試作)
        (どちらもjra_tools.views。Postgresの層: staging ▶ model ▶ features、sandbox)

[学習する](Jupyter)
  仕様(YAML) ─pin_spec─▶ 固定した仕様
  features・sandboxのビュー ─load_dataset─▶ テンソル x(レース, 18, 次元)・labels(1〜5着)
  モデル(Keras) ◀── 使う側が作る(Jupyterのmodel_creator.py)。jra-toolsは作らない
  model.fit(x, labels)
  学習したモデル + 固定した仕様 ─save_model─▶ モデルのフォルダ(SavedModel + feature_spec.yaml)

[推論する](sazanami-batch)
  モデルのフォルダ ─load_model_feature_spec─▶ 仕様    ─TFSMLayer─▶ モデル(batchが読む)
  featuresのビュー ─predict_finish_probabilities(仕様, 日, モデル)─▶ 確率 ─batchが書く─▶ model.predict
```

- ビューの定義(SQL)は使う側にある。model・featuresはsazanami-batchの`schemas/`、試作のsandboxはJupyterの`schemas/`。jra-toolsは定義を受け取って反映するだけ
- jra-toolsが読むのは、仕様の`source`のビュー、ラベルのビュー、`code_categories`、締めた年だけ。stagingを直接は読まない
- 学習・評価・推論のどれも、入力は同じ仕様から同じ手順で作る(学習と推論で入力の作り方がずれない)

## 2. 持つもの・持たないもの
| 持つ | 持たない(持つ場所) |
|---|---|
| ビューの定義のフォルダを読み、Postgresに反映する | ビューの定義(sazanami-batch・Jupyterの`schemas/`) |
| 仕様を読み書きし、学習の前に固定する(ビューのハッシュ・カテゴリの番号を写す) | JRDBのコードの値と意味(Postgresのseeds・`code_categories`) |
| ビューを読み、テンソルにする(学習・評価・推論の入力) | モデルの構造・学習の手順(Jupyterの`model_creator.py`・ノートブック) |
| モデルのフォルダを書く・読む(モデルと仕様の照合) | モデルを推論に読み込むこと(使う側。batchは`TFSMLayer`) |
| モデルから確率・評価の指標を出す | DBに書くこと(推論の結果はbatch、試作の記録はJupyterの`lab.py`) |
| | 買い方・回収率の集計(使う側のビュー) |

モデルは使う側が作って渡す。jra-toolsはモデルの中身を知らず、[モデルの約束](specification.md#5-モデルの約束)(入力の形・出力の数と順・出力の名前)だけを求める。

## 3. 層
`jra_tools.machine_learning`はクリーンアーキテクチャで分ける。依存は外から内へ(入口 → infrastructure・interface_adapter → usecase → entity)。

| 置き場所 | 役割 | 中身 |
|---|---|---|
| `features.py`(層の外) | 入口。使う側はここだけを使う | 接続・ファイル(infrastructure)を作り、読み手・置き場(interface_adapter)に渡し、手続き(usecase)を呼ぶ。組み立てだけ |
| `entity/` | 決まり | 仕様(`feature_spec.py`)、カテゴリの決まり(`category.py`)、データセットの形(`dataset.py`)、入力・ラベルの列の約束(`input_view.py`)、期間(`period.py`)、馬券の組み合わせの番号`Labelref`(`labelutil.py`) |
| `usecase/` | 手続きと、外とのつなぎ目 | 仕様の固定・読む・推論・評価(`view_dataset.py`)、テンソルへの組み立て(`frame_dataset.py`)、モデルのフォルダの読み書き(`model_spec.py`) |
| `interface_adapter/` | 外の形との変換 | PostgresのSQL(`view_gateway.py`)、表⇔Parquet(`frame_format.py`・`frame_store.py`)、仕様⇔YAMLの辞書(`spec_format.py`) |
| `infrastructure/` | 外そのもの | 接続・トランザクション(`sql_session.py`)、ファイル(`files.py`)、SavedModelのフォルダ(`saved_model.py`) |

### 外とのつなぎ目(usecaseのProtocol)
usecaseは外のものを、次のProtocolとしてだけ扱う。テストではこれを偽物に替える。

| Protocol | 何を表すか | 実装 |
|---|---|---|
| `IViewReader`(`iview_reader.py`) | ビュー・`code_categories`・締めた年を読むもの | `PostgresViewReader` |
| `IFrameStore`(`iframe_store.py`) | 読んだ表の置き場(`cache_dir`) | `ParquetFrameStore` |
| `IModelFolder`(`imodel_folder.py`) | モデルのフォルダ | `SavedModelFolder` |
| `IFinishModel`・`IEvaluableModel`・`ITrainedModel`(`ifinish_model.py`) | 推論・評価・保存に渡すモデル | 使う側のKerasのモデル、`TFSMLayer` |

### `jra_tools.views`
定義を読むこと(`view_set.py`)と反映すること(`apply.py`)だけの小さなモジュールで、層に分けていない。sazanami-batchは`apply_views`を、Jupyterは`features.apply_sandbox_views`(中で`replace_views`)を使う。

### それ以外
JRDBの取り込みの補助・開催日程(`jra_tools`直下・`database/`)は機械学習と関係なく、sazanami-batchの取り込みが使う。開催データを紐づく表ごと読む`get_kaisais`もここ(DBの読み込みのため)。

## 4. なくしたもの
- JRDBのコードの値と解釈(0.7.1まであった`CategoryGetter`)は、分析用のPostgresのseeds(JRDBのコード表の写し)とmodel・featuresの`code_categories`へ移した(sazanami-batch)。0.7.2でなくした
- 0.6.0まであったORMのオブジェクトから作る経路(`InputCreator`・`LabelCreator`・`PayoutCreator`・`PredictFactory`など)、jrdb-modelの表から読む経路、onehot(`size`・`base`)、`.dump`の書き出しはなくした。2019年の1年分でORMの経路は17.0秒・1,865MB、SQLの経路は4.0秒・263MBだった(2026-10-27、分析用のMySQL)
