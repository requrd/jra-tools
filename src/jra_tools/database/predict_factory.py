"""推論データを生成する."""

from typing import TYPE_CHECKING, List, Optional

from jrdb_model import (
    KaisaiData,
    PredictData,
)

from jra_tools.machine_learning.entity.feature_spec import FeatureSpec
from jra_tools.machine_learning.interface_adapter.input_creator import InputCreator

if TYPE_CHECKING:
    from keras import Model


class PredictFactory:
    """推論データ生成クラス."""

    def __init__(
        self,
        kaisais: List[KaisaiData],
        model: "Model",
        spec: Optional[FeatureSpec] = None,
    ):
        """開催データとKerasモデルを検証して保持する.

        Args:
            kaisais (List[KaisaiData]): 開催データ一覧
            model (Model): 推論モデル
            spec (Optional[FeatureSpec]): モデルの特徴量の仕様(load_model_feature_specで
                モデルのフォルダから読み込んだもの). 省略時は現行の推論モデルの仕様

        """
        # Keras(TensorFlow)は読み込みに時間とメモリがかかる。jra_toolsは__init__.pyで
        # 全モジュールを読み込むため、推論しない利用者が読み込まないよう、ここでimportする。
        from keras import Model

        for kaisai in kaisais:
            if not isinstance(kaisai, KaisaiData):
                raise TypeError("開催データではありません")
        self.kaisais = kaisais

        if not isinstance(model, Model):
            raise TypeError("Keras V3のモデルではありません")
        self.model = model
        self.spec = spec

    def create(self) -> List[PredictData]:
        """推論データを生成する.

        Returns:
            List[PredictData]: 未登録の推論データ一覧

        """
        input_data = InputCreator(self.kaisais, self.spec)
        preds = self.model.predict(input_data.x_data)
        w = 0
        entities = []
        for k in self.kaisais:
            for r in k.races:
                for h in r.racehorses:
                    hn = h.num - 1
                    pp1 = preds["activation_4"][w][hn]
                    pp2 = preds["activation_5"][w][hn]
                    pp3 = preds["activation_6"][w][hn]
                    rentai_rate = 1 - ((1 - pp1) * (1 - pp2))
                    fukusho_rate = 1 - ((1 - pp1) * (1 - pp2) * (1 - pp3))
                    entities.append(
                        PredictData(
                            racehorsekey=h.racehorsekey,
                            pp_icchaku=float(pp1),
                            pp_nichaku=float(pp2),
                            pp_sanchaku=float(pp3),
                            rentai_rate=rentai_rate,
                            fukusho_rate=fukusho_rate,
                            tansho_odds=1 / pp1,
                            fukusho_odds=1 / fukusho_rate,
                        )
                    )
                w += 1
        return entities
