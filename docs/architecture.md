# 置き場所と経緯
使う側は入口(`features`)と、そこから取れる型だけを使う。ここは中を直す人のためのもの。

## `machine_learning`の層
クリーンアーキテクチャ(entity・usecase・interface_adapter・infrastructure)と、層の外の入口。置き場所は、隣のファイルではなく層の役割で決める。

| 置き場所 | 中身 |
|---|---|
| `features.py`(層の外) | 入口。infrastructureの接続・ファイルを作り、interface_adapterの読み手・置き場に渡し、手続きを呼ぶ(組み立てだけ) |
| `entity/` | 仕様(`feature_spec.py`)、カテゴリの決まり(`category.py`。探す先の既定・項目を1つに決める・コード→位置)、データセットの形(`dataset.py`)、入力・ラベルの列の契約と決まり(`input_view.py`)、期間(`period.py`)、馬券の馬番の組み合わせとカテゴリ番号の変換`Labelref`(`labelutil.py`) |
| `usecase/` | 手続き:仕様の固定・読む・推論・評価(`view_dataset.py`)、テンソルへの組み立て(`frame_dataset.py`)、モデルとその仕様の読み書き(`model_spec.py`)。外に触るところはProtocol(`iview_reader.py`・`iframe_store.py`・`imodel_folder.py`・`ifinish_model.py`) |
| `interface_adapter/` | PostgresのSQL(`view_gateway.py`。照合・ハッシュ・読む・code_categories・締めた年)、表⇔Parquet(`frame_format.py`)と置き場(`frame_store.py`)、仕様⇔YAMLの辞書(`spec_format.py`) |
| `infrastructure/` | 接続・トランザクション(`sql_session.py`)、ファイル(`files.py`)、SavedModelのフォルダ(`saved_model.py`。入力の形・出力の名前を読む、モデルを書く) |

`jra_tools.views`(`view_set.py`・`apply.py`)は、定義を読むことと反映することだけの小さなモジュールで、層に分けていない。

開催データを紐づく表ごと読む`get_kaisais`は、機械学習ではなくDBの読み込みなので`jra_tools`(`database/kaisai_loader.py`)にある。

## なくしたもの
- JRDBのコードの値と解釈(0.7.1まであった`CategoryGetter`の知識)は、分析用のPostgresのseeds(JRDBのコード表の写し)とmodel・featuresの`code_categories`にある(sazanami-batch)。0.7.2で`CategoryGetter`をなくした
- 0.6.0まであったORMのオブジェクトから作る経路(`InputCreator`・`LabelCreator`・`PayoutCreator`・`PredictFactory`など)、jrdb-modelの表から読む経路、onehot(`size`・`base`)、`.dump`の書き出しはなくした。2019年の1年分でORMの経路は17.0秒・1,865MB、SQLの経路は4.0秒・263MBだった(2026-10-27、分析用のMySQL)
