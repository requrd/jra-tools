"""推論モデルと一緒に持つ特徴量の仕様を読み書きする手続き.

どの特徴量で入力を作るかは、jra-toolsの版ではなくモデルで決まる。読むときも書くときも、
仕様の形(max_horses・次元数)がモデルの入力の形と一致することを確かめる。
"""

from pathlib import Path

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.usecase.imodel_folder import IModelFolder


def load_model_feature_spec(folder: IModelFolder) -> FeatureSpec:
    """モデルのフォルダの仕様を読み、モデルの入力の形と照合する.

    Raises:
        FileNotFoundError: フォルダに仕様がない場合
        ValueError: 仕様とモデルの入力の形が一致しない場合

    """
    spec = folder.read_spec()
    spec.check_input_shape(folder.input_shape())
    return spec


def save_model_feature_spec(folder: IModelFolder, spec: FeatureSpec) -> Path:
    """仕様がモデルの入力の形と一致することを確かめてから、フォルダに書く.

    Raises:
        ValueError: 仕様とモデルの入力の形が一致しない場合

    """
    spec.check_input_shape(folder.input_shape())
    return folder.write_spec(spec)
