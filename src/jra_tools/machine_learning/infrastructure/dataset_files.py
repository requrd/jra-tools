"""学習用のデータセットを、ファイルに書き出す.

jra-tools 0.6.0までのORMの経路(InputCreator・LabelCreator・PayoutCreator)と同じ形式:
x_{name}.dump(入力)、{name}.dump(正解ラベルのリスト、pickle)、payout_{name}.dump(払戻)。
各レースのレースキーをracekeys_{name}.npyに書く。
"""

import pickle
from pathlib import Path

import numpy as np

from jra_tools.machine_learning.entity.dataset import TableDataset


def save_table_dataset(
    dataset: TableDataset, name: str, directory: Path | str = "."
) -> None:
    """データセットを、ディレクトリに4つのファイルで書き出す.

    Args:
        dataset (TableDataset): 学習用のデータセット
        name (str): ファイル名(拡張子を除く)
        directory (Path | str): 書き出すディレクトリ(既定は今のディレクトリ)

    """
    root = Path(directory)
    with open(root / f"x_{name}.dump", mode="wb") as f:
        dataset.x.dump(f)
    with open(root / f"{name}.dump", mode="wb") as f:
        pickle.dump(dataset.labels, f)
    with open(root / f"payout_{name}.dump", "wb") as f:
        dataset.payout.dump(f)
    np.save(root / f"racekeys_{name}.npy", dataset.racekeys)
