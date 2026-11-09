"""TensorFlowのSavedModelのフォルダ(usecase/imodel_folder.pyのIModelFolderの実装).

仕様はフォルダのfeature_spec.yaml。入力の形は、変数を読み込まずにsaved_model.pbから取る。
"""

from pathlib import Path
from typing import List

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
        # TensorFlowの読み込みは重いため、使うときにimportする
        from tensorflow.core.protobuf import saved_model_pb2

        saved_model = saved_model_pb2.SavedModel()
        saved_model.ParseFromString((self.model_dir / "saved_model.pb").read_bytes())
        inputs = saved_model.meta_graphs[0].signature_def[SIGNATURE].inputs
        if len(inputs) != 1:
            raise ValueError(f"モデルの入力が1つではありません: {list(inputs)}")
        [tensor] = inputs.values()
        return [dim.size for dim in tensor.tensor_shape.dim]

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
