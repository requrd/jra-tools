"""ラベルを作成する."""

import pickle
from functools import reduce
from typing import List

import numpy as np
from jrdb_model import KaisaiData


class LabelCreator:
    """ラベル作成クラス."""

    def __init__(self, kaisais: List[KaisaiData]):
        """開催データから1着〜5着の正解ラベルを作る."""
        self.kaisais = kaisais
        num_race = reduce(lambda x, y: x + len(y.races), kaisais, 0)
        num_max_horse = 18
        self.t_icchaku = np.zeros([num_race, num_max_horse])
        self.t_nichaku = np.zeros([num_race, num_max_horse])
        self.t_sanchaku = np.zeros([num_race, num_max_horse])
        self.t_yonchaku = np.zeros([num_race, num_max_horse])
        self.t_gochaku = np.zeros([num_race, num_max_horse])
        self.labels = [
            self.t_icchaku,
            self.t_nichaku,
            self.t_sanchaku,
            self.t_yonchaku,
            self.t_gochaku,
        ]
        labels_by_order = dict(enumerate(self.labels, start=1))
        w_num = 0
        for kaisai in kaisais:
            for race in kaisai.races:
                for horse in race.racehorses:
                    if horse.result is None:
                        raise ValueError(f"成績がありません: {horse.racehorsekey}")
                    label = labels_by_order.get(horse.result.order_of_arrival)
                    if label is not None:
                        label[w_num, horse.num - 1] = 1
                w_num += 1

    def save(self, name: str):
        """ラベルデータをローカルに保存する.

        Args:
            name (str): ファイル名

        """
        with open(f"{name}.dump", mode="wb") as f:
            pickle.dump(self.labels, f)
