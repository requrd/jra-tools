"""TensorFlowのSavedModelのフォルダ(usecase/imodel_folder.pyのIModelFolderの実装).

仕様はフォルダのfeature_spec.yaml。入力の形・出力の名前は、変数を読み込まずにsaved_model.pbから取る。
モデルは、serving_defaultが出力の名前→値の辞書を返すように書く(Keras 3のmodel.exportは
出力の名前をoutput_0…にするため、名前を渡して書く)。
"""

from pathlib import Path
from typing import Any, List, Sequence

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.infrastructure.files import read_yaml, write_yaml
from jra_tools.machine_learning.interface_adapter.spec_format import (
    feature_spec_to_dict,
    parse_feature_spec,
)

# モデルのフォルダ(SavedModel)に置く仕様のファイル名
MODEL_SPEC_FILE = "feature_spec.yaml"
# 推論で呼び出すシグネチャ
SIGNATURE = "serving_default"


class SavedModelFolder:
    """SavedModelのフォルダ."""

    def __init__(self, model_dir: Path | str):
        """フォルダを受け取る."""
        self.model_dir = Path(model_dir)

    def input_shape(self) -> List[int]:
        """モデルの入力の形. 先頭はバッチの次元(-1).

        Raises:
            ValueError: 入力が1つではない場合

        """
        inputs = self._signature().inputs
        if len(inputs) != 1:
            raise ValueError(f"モデルの入力が1つではありません: {list(inputs)}")
        [tensor] = inputs.values()
        return [dim.size for dim in tensor.tensor_shape.dim]

    def output_names(self) -> List[str]:
        """推論で呼ぶシグネチャの出力の名前."""
        return list(self._signature().outputs)

    def write_model(self, model: Any, output_names: Sequence[str]) -> None:
        """Kerasのモデルを、出力の名前→値の辞書を返すSavedModelとして書く.

        Args:
            model (Any): Kerasのモデル(入力は1つ、出力はoutput_namesの順)
            output_names (Sequence[str]): 出力の名前

        """
        # TensorFlow・Kerasの読み込みは重いため、使うときにimportする
        import keras
        import tensorflow as tf

        names = list(output_names)

        def serve(x: Any) -> dict:
            return dict(zip(names, model(x, training=False), strict=True))

        archive = keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint(
            name=SIGNATURE,
            fn=serve,
            input_signature=[tf.TensorSpec(shape=model.input_shape, dtype=tf.float32)],
        )
        archive.write_out(str(self.model_dir), verbose=False)

    def read_spec(self) -> FeatureSpec:
        """フォルダの仕様を読む.

        Raises:
            FileNotFoundError: フォルダに仕様がない場合

        """
        path = self.model_dir / MODEL_SPEC_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"モデルのフォルダに特徴量の仕様がありません: {path}"
            )
        return parse_feature_spec(read_yaml(path))

    def write_spec(self, spec: FeatureSpec) -> Path:
        """仕様をフォルダに書く."""
        path = self.model_dir / MODEL_SPEC_FILE
        write_yaml(feature_spec_to_dict(spec), path)
        return path

    def _signature(self) -> Any:
        # TensorFlowの読み込みは重いため、使うときにimportする
        from tensorflow.core.protobuf import saved_model_pb2

        saved_model = saved_model_pb2.SavedModel()
        saved_model.ParseFromString((self.model_dir / "saved_model.pb").read_bytes())
        return saved_model.meta_graphs[0].signature_def[SIGNATURE]
