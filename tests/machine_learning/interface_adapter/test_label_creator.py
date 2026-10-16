"""LabelCreatorのユニットテスト."""

import os
import pickle
from typing import List

from jrdb_model import KaisaiData

from jra_tools.machine_learning.interface_adapter.label_creator import (
    LabelCreator,
)
from jra_tools.machine_learning.usecase.icreator import ICreator


def test_label_creator(sample_kaisais: List[KaisaiData]):
    """ラベルクリエイターのユニットテスト.

    Args:
        sample_kaisais (List[KaisaiData]): 開催データ一覧

    """
    try:
        creator = LabelCreator(sample_kaisais)
        creator.save("label_testfile")

        assert isinstance(creator, ICreator)
        with open("label_testfile.dump", mode="rb") as f:
            labels = pickle.load(f)
        assert len(labels) == 5

    finally:
        if os.path.exists("label_testfile.dump"):
            os.remove("label_testfile.dump")
