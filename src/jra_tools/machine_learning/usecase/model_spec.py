"""推論モデルと一緒に持つ特徴量の仕様を読み書きする手続き.

どの特徴量で入力を作るかは、jra-toolsの版ではなくモデルで決まる。読むときも書くときも、
仕様の形(max_horses・次元数)がモデルの入力の形と一致することを確かめる。
書くときは、モデルの出力の名前(1着→5着の順)を仕様に写す。
"""

import dataclasses
from pathlib import Path
from typing import Sequence

from jra_tools.machine_learning.entity.dataset import LABEL_ORDERS
from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.usecase.imodel_folder import IModelFolder


def load_model_feature_spec(folder: IModelFolder) -> FeatureSpec:
    """モデルのフォルダの仕様を読み、モデルの入力の形と照合する.

    Raises:
        FileNotFoundError: フォルダに仕様がない場合
        ValueError: 仕様が固定されていない、または仕様とモデルの入力の形が一致しない場合

    """
    spec = folder.read_spec()
    spec.check_pinned()
    spec.check_input_shape(folder.input_shape())
    return spec


def save_model_feature_spec(
    folder: IModelFolder, spec: FeatureSpec, output_names: Sequence[str]
) -> Path:
    """出力の名前を写した仕様を、モデルの入力の形と照合してから、フォルダに書く.

    Args:
        folder (IModelFolder): モデルのフォルダ
        spec (FeatureSpec): 学習に使った(固定した)仕様
        output_names (Sequence[str]): モデルの出力の名前(1着→5着の順。Kerasのmodel.output_names)

    Raises:
        ValueError: 仕様が固定されていない、出力の数が違う、または入力の形が一致しない場合

    """
    spec.check_pinned()
    if len(output_names) != LABEL_ORDERS:
        raise ValueError(
            f"出力の名前は1着→{LABEL_ORDERS}着の{LABEL_ORDERS}つです: {list(output_names)}"
        )
    spec.check_input_shape(folder.input_shape())
    return folder.write_spec(
        dataclasses.replace(spec, output_names=tuple(output_names))
    )
