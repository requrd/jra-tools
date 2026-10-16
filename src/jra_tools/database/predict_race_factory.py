"""推論レースデータを生成する."""

from dataclasses import dataclass
from typing import List

from jrdb_model import (
    BangumiData,
    KaisaiData,
    PredictRaceData,
    RacehorseData,
)


@dataclass
class PredictRaceFactory:
    """推論レースデータファクトリクラス."""

    kaisais: List[KaisaiData]

    def create(self) -> List[PredictRaceData]:
        """開催データ一覧から推論レースデータを生成する.

        推論データが登録済みでないと失敗する.

        Returns:
            List[PredictRaceData]: 推論レースデータ一覧

        """
        return [
            PredictRaceData(
                racekey=race.racekey,
                umaren=umarenOddses(race),
                wide=wideOddses(race),
                wakuren=wakurenOddses(race),
            )
            for kaisai in self.kaisais
            for race in kaisai.races
        ]


def umarenOddses(race: BangumiData):
    """レース内の全組み合わせの推定馬連オッズを返す."""
    return oddses(race, "umaren")


def umarenOdds(horse1, horse2):
    """2頭の推定馬連オッズを返す."""
    p1 = 1 / umatanOdds(horse1, horse2)
    p2 = 1 / umatanOdds(horse2, horse1)
    p = p1 + p2
    return 1 / p


def umatanOdds(horse1, horse2):
    """horse1が1着、horse2が2着となる推定馬単オッズを返す."""
    p1 = horse1.predict.pp_icchaku
    p2 = horse2.predict.pp_nichaku / (1 - horse1.predict.pp_nichaku)
    p = p1 * p2
    return 1 / p


def wideOddses(race: BangumiData):
    """レース内の全組み合わせの推定ワイドオッズを返す."""
    return oddses(race, "wide")


def oddses(race: BangumiData, baken: str):
    """レース内の全組み合わせの推定オッズを返す.

    bakenが"wide"ならワイド、それ以外は馬連として計算する.
    """
    odds_dict = {}
    for i, horse1 in enumerate(race.racehorses):
        for j in range(i + 1, len(race.racehorses)):
            horse2 = race.racehorses[j]
            odds_dict.update(
                {
                    f"{horse1.num}-{horse2.num}": wideOdds(horse1, horse2)
                    if baken == "wide"
                    else umarenOdds(horse1, horse2)
                }
            )
    return odds_dict


def wideOdds(horse1: RacehorseData, horse2: RacehorseData):
    """2頭の推定ワイドオッズを返す."""
    p = horse1.predict.pp_icchaku * horse2.predict.pp_nichaku / (
        1 - horse1.predict.pp_nichaku
    ) + horse1.predict.pp_icchaku * horse2.predict.pp_sanchaku / (
        1 - horse1.predict.pp_sanchaku
    )
    p += horse1.predict.pp_nichaku * horse2.predict.pp_icchaku / (
        1 - horse1.predict.pp_icchaku
    ) + horse1.predict.pp_nichaku * horse2.predict.pp_sanchaku / (
        1 - horse1.predict.pp_sanchaku
    )
    p += horse1.predict.pp_sanchaku * horse2.predict.pp_icchaku / (
        1 - horse1.predict.pp_icchaku
    ) + horse1.predict.pp_sanchaku * horse2.predict.pp_nichaku / (
        1 - horse1.predict.pp_nichaku
    )
    return 1 / p


def wakurenOddses(race: BangumiData):
    """全枠番の組み合わせの推定枠連オッズを返す."""
    odds_dict = {}
    for i in range(8):
        for j in range(i, 8):
            gate1 = i + 1
            gate2 = j + 1
            odds_dict.update({f"{gate1}-{gate2}": wakurenOdds(race, gate1, gate2)})
    return odds_dict


def wakurenOdds(race: BangumiData, gate1: int, gate2: int):
    """2つの枠番の推定枠連オッズを返す.

    どちらかの枠に馬がいない場合は0を返す.
    """
    horses1 = [horse for horse in race.racehorses if horse.waku == gate1]
    if gate1 == gate2:
        return wakurenSameOdds(horses1)
    horses2 = [horse for horse in race.racehorses if horse.waku == gate2]
    pp = 0
    if len(horses1) == 0 or len(horses2) == 0:
        return 0
    for horse1 in horses1:
        for horse2 in horses2:
            pp += 1 / umarenOdds(horse1, horse2)
    return 1 / pp


def wakurenSameOdds(horses: List[RacehorseData]):
    """同じ枠同士の推定枠連オッズを返す.

    枠内が2頭未満の場合は0を返す.
    """
    if len(horses) <= 1:
        return 0
    pp = 0
    for i, horse1 in enumerate(horses):
        for j in range(i + 1, len(horses)):
            horse2 = horses[j]
            pp += 1 / umarenOdds(horse1, horse2)
    return 1 / pp
