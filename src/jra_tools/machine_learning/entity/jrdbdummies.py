"""JRDBのカテゴリ値をone-hotベクトル(ダミー変数)に変換する."""

# 各メソッドの数値はJRDBのコード表に対応する区切り値なので定数化しない
# ruff: noqa: PLR2004

import numpy as np


class CategoryGetter:
    """JRDBの各カテゴリ値をone-hotベクトルに変換する."""

    # 開催データ
    def getKaisaikbn(self, value):
        """開催区分をone-hotベクトルに変換する."""
        return self.categories(3, value)

    def getDayofweek(self, value):
        """曜日をone-hotベクトルに変換する."""
        types = {"日": 1, "月": 2, "火": 3, "水": 4, "木": 5, "金": 6, "土": 7}
        return self.categories(7, types[value])

    def getTenko(self, value):
        """天候をone-hotベクトルに変換する."""
        return self.categories(6, value)

    def getBabaabst(self, value):
        """馬場状態(大分類)をone-hotベクトルに変換する."""
        return self.categories(4, value)

    def getBabadetail(self, baba):
        """馬場状態(詳細)をone-hotベクトルに変換する."""
        return self.categories(3, baba)

    def getTurfkind(self, kind):
        """芝種類をone-hotベクトルに変換する."""
        return self.categories(3, kind)

    def getTennatsu(self, ten):
        """転圧をone-hotベクトルに変換する."""
        return self.categories(2, ten)

    def getStopfreeze(self, st):
        """凍結防止剤をone-hotベクトルに変換する."""
        return self.categories(2, st)

    # 番組データ
    def getDistance(self, value):
        """距離をone-hotベクトルに変換する."""
        if 2400 < value <= 2500:
            return self.categories(12, 9)
        if 2500 < value <= 3000:
            return self.categories(12, 10)
        if 3000 < value <= 3200:
            return self.categories(12, 11)
        if value > 3200:
            return self.categories(12, 12)
        return self.categories(12, (value // 200) - 4)

    def getTdscode(self, tds):
        """芝ダ障害コードをone-hotベクトルに変換する."""
        return self.categories(3, tds)

    def getRightleft(self, rl):
        """右左をone-hotベクトルに変換する."""
        return self.categories(4, rl)

    def getInout(self, inout):
        """内外をone-hotベクトルに変換する."""
        return self.categories(4, inout)

    def getShubetsu(self, s):
        """種別をone-hotベクトルに変換する."""
        return self.categories(5, s)

    def getJoken(self, value):
        """条件をone-hotベクトルに変換する."""
        types = ["04", "05", "08", "09", "10", "15", "16", "A1", "A2", "A3", "OP"]
        return self.categories(11, types.index(value) + 1)

    def getJuryo(self, j):
        """重量をone-hotベクトルに変換する."""
        return self.categories(4, j)

    def getGrade(self, g):
        """グレードをone-hotベクトルに変換する."""
        return self.categories(5, g)

    def getCourse(self, c):
        """コースをone-hotベクトルに変換する."""
        return self.categories(5, c)

    # RaceHorse
    # 汎用メソッド利用
    def getNum(self, num):
        """馬番をone-hotベクトルに変換する."""
        return self.categories(18, num)

    def getLegtype(self, leg_type):
        """脚質をone-hotベクトルに変換する."""
        return self.categories(7, int(leg_type))

    def getOmotekiseicode(self, omotekisei_code):
        """重適性コードをone-hotベクトルに変換する."""
        return self.categories(4, omotekisei_code)

    def getBrinkers(self, brinkers):
        """ブリンカーをone-hotベクトルに変換する."""
        return self.categories(4, brinkers)

    def getMinarai(self, minarai):
        """見習い区分をone-hotベクトルに変換する."""
        return self.categories(4, minarai)

    def getWaku(self, waku):
        """枠番をone-hotベクトルに変換する."""
        return self.categories(8, waku)

    def getTurfadjustcode(self, turf_adjust_code):
        """芝適性コードをone-hotベクトルに変換する."""
        return self.categories(5, turf_adjust_code)

    def getTorikeshi(self, torikeshi):
        """取消フラグをone-hotベクトルに変換する."""
        return self.categories(2, torikeshi)

    def getSex(self, sex):
        """性別をone-hotベクトルに変換する."""
        return self.categories(3, sex)

    def getBanushikaicode(self, banushikai_code):
        """馬主会コードをone-hotベクトルに変換する."""
        return self.categories(11, banushikai_code)

    def getYuso(self, yuso):
        """輸送区分をone-hotベクトルに変換する."""
        return self.categories(5, yuso)

    def getKokyuflg(self, kokyu_flg):
        """降級フラグをone-hotベクトルに変換する."""
        return self.categories(3, kokyu_flg)

    def getTurfdartsteepleflg(self, turf_dart_steeple_flg):
        """芝ダ障害フラグをone-hotベクトルに変換する."""
        return self.categories(3, turf_dart_steeple_flg)

    def getDistanceflg(self, distance_flg):
        """距離フラグをone-hotベクトルに変換する."""
        return self.categories(2, distance_flg)

    def getClassflg(self, class_flg):
        """クラスフラグをone-hotベクトルに変換する."""
        return self.categories(4, class_flg)

    def getTenkyuflg(self, tenkyu_flg):
        """転厩フラグをone-hotベクトルに変換する."""
        return self.categories(4, tenkyu_flg)

    def getKyoseiflg(self, kyosei_flg):
        """去勢フラグをone-hotベクトルに変換する."""
        return self.categories(4, kyosei_flg)

    def getTraincoursekind(self, train_course_kind):
        """調教コース種別をone-hotベクトルに変換する."""
        return self.categories(6, train_course_kind)

    def getSaka(self, saka):
        """坂路調教の有無をone-hotベクトルに変換する."""
        return self.categories(2, saka)

    def getWood(self, wood):
        """ウッド調教の有無をone-hotベクトルに変換する."""
        return self.categories(2, wood)

    def getDart(self, dart):
        """ダート調教の有無をone-hotベクトルに変換する."""
        return self.categories(2, dart)

    def getTurf(self, turf):
        """芝調教の有無をone-hotベクトルに変換する."""
        return self.categories(2, turf)

    def getPool(self, pool):
        """プール調教の有無をone-hotベクトルに変換する."""
        return self.categories(2, pool)

    def getSteeple(self, steeple):
        """障害調教の有無をone-hotベクトルに変換する."""
        return self.categories(2, steeple)

    def getPolitruck(self, politruck):
        """ポリトラック調教の有無をone-hotベクトルに変換する."""
        return self.categories(2, politruck)

    def getTraindistance(self, train_distance):
        """調教距離をone-hotベクトルに変換する."""
        return self.categories(5, train_distance)

    def getTrainjuten(self, train_juten):
        """調教重点をone-hotベクトルに変換する."""
        return self.categories(5, train_juten)

    def getOikirikind(self, oikiri_kind):
        """追切種類をone-hotベクトルに変換する."""
        return self.categories(4, oikiri_kind)

    def getZensoIjokbn(self, zenso_ijo_kbn):
        """前走異常区分をone-hotベクトルに変換する."""
        return self.categories(7, zenso_ijo_kbn)

    def getZensoCourseposition(self, zenso_course_position):
        """前走コース取りをone-hotベクトルに変換する."""
        return self.categories(9, zenso_course_position)

    def getZensoRacelegtype(self, zenso_race_leg_type):
        """前走レース脚質をone-hotベクトルに変換する."""
        return self.categories(7, zenso_race_leg_type)

    def getZensoCorner4courseposition(self, zenso_corner4_course_position):
        """前走4角コース取りをone-hotベクトルに変換する."""
        return self.categories(8, zenso_corner4_course_position)

    def getUmakigocode(self, uk):
        """馬記号コードをone-hotベクトルに変換する."""
        return self.categories(28, uk)

    # 独自メソッド利用
    def getBacode(self, value):
        """場コードをone-hotベクトルに変換する."""
        if (value < 1) or (value > 10):
            value = 11
        return self.categories(11, value)

    def getDistanceadjust(self, value):
        """距離適性をone-hotベクトルに変換する."""
        value = value + 1 if 0 <= value <= 6 else 8
        return self.categories(8, value)

    def getJockeyshirushi(self, value):
        """騎手印をone-hotベクトルに変換する."""
        value = value + 1 if 0 <= value <= 6 else 8
        return self.categories(8, value)

    def getDartadjustcode(self, value):
        """ダート適性コードをone-hotベクトルに変換する."""
        value = value + 1 if 0 <= value <= 3 else 5
        return self.categories(5, value)

    def getGekisotype(self, value):
        """激走タイプをone-hotベクトルに変換する."""
        types = {"00": 1, "A1": 2, "A2": 3, "A3": 4, "A4": 5, "B1": 6, "B2": 7}
        return self.categories(7, types[value])

    def getRestreasoncode(self, value):
        """休養理由分類コードをone-hotベクトルに変換する."""
        if 0 <= value <= 7:
            return self.categories(11, value + 1)
        if 11 <= value <= 16:
            return self.categories(11, value - 2)
        if value == 21:
            return self.categories(11, 15)

    def getNorikaeflg(self, value):
        """乗替フラグをone-hotベクトルに変換する."""
        value = value + 1 if 0 <= value <= 1 else 3
        return self.categories(3, value)

    def getHobokusakirank(self, value):
        """放牧先ランクをone-hotベクトルに変換する."""
        types = {"0": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6}
        return self.categories(6, types[value])

    def getTraintype(self, value):
        """調教タイプをone-hotベクトルに変換する."""
        return self.categories(11, int(value))

    def getTrainvolhyoka(self, value):
        """調教量評価をone-hotベクトルに変換する."""
        types = {"A": 1, "B": 2, "C": 3, "D": 4, " ": 5}
        return self.categories(5, types[value])

    # 汎用メソッド

    def categories(self, category_count, value):
        """値を長さcategory_countのone-hotベクトルに変換する.

        valueは1始まりのカテゴリ番号.
        """
        cat = np.zeros([category_count])
        cat[value - 1] = 1
        return cat
