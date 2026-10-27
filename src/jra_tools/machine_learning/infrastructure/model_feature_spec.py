"""推論モデルのフォルダに、学習に使った特徴量の仕様を保存・読み込みする.

仕様はモデルと一緒に持つ。どの特徴量で入力を作るかは、jra-toolsの版ではなく
モデルで決まる。
"""

from pathlib import Path
from typing import List

from jra_tools.machine_learning.entity.feature_spec import (
    FeatureSpec,
    load_feature_spec,
    save_feature_spec,
)

# モデルのフォルダ(SavedModel)に置く仕様のファイル名
MODEL_SPEC_FILE = "feature_spec.yaml"
# 推論で呼び出すシグネチャ
SIGNATURE = "serving_default"


def save_model_feature_spec(spec: FeatureSpec, model_dir: Path | str) -> Path:
    """学習に使った仕様を、モデルのフォルダに書き出す.

    Args:
        spec (FeatureSpec): 学習に使った特徴量の仕様
        model_dir (Path | str): モデルのフォルダ(SavedModel)

    Returns:
        Path: 書き出した仕様のファイル

    """
    path = Path(model_dir) / MODEL_SPEC_FILE
    spec.check_input_shape(saved_model_input_shape(model_dir))
    save_feature_spec(spec, path)
    return path


def load_model_feature_spec(model_dir: Path | str) -> FeatureSpec:
    """モデルのフォルダから仕様を読み込み、モデルの入力の形と照合する.

    Args:
        model_dir (Path | str): モデルのフォルダ(SavedModel)

    Raises:
        FileNotFoundError: モデルのフォルダに仕様がない場合
        ValueError: 仕様とモデルの入力の形が一致しない場合

    Returns:
        FeatureSpec: モデルの特徴量の仕様

    """
    path = Path(model_dir) / MODEL_SPEC_FILE
    if not path.exists():
        raise FileNotFoundError(f"モデルのフォルダに特徴量の仕様がありません: {path}")
    spec = load_feature_spec(path)
    spec.check_input_shape(saved_model_input_shape(model_dir))
    return spec


def saved_model_input_shape(model_dir: Path | str) -> List[int]:
    """SavedModelの入力の形を、変数を読み込まずにsaved_model.pbから取得する.

    Args:
        model_dir (Path | str): モデルのフォルダ(SavedModel)

    Raises:
        ValueError: 入力が1つではない場合

    Returns:
        List[int]: 入力の形。先頭はバッチの次元(-1)

    """
    # TensorFlowの読み込みは重いため、使うときにimportする
    from tensorflow.core.protobuf import saved_model_pb2

    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString((Path(model_dir) / "saved_model.pb").read_bytes())
    inputs = saved_model.meta_graphs[0].signature_def[SIGNATURE].inputs
    if len(inputs) != 1:
        raise ValueError(f"モデルの入力が1つではありません: {list(inputs)}")
    [tensor] = inputs.values()
    return [dim.size for dim in tensor.tensor_shape.dim]
