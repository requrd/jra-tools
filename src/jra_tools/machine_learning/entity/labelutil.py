"""馬券の馬番の組み合わせとカテゴリ番号を相互に変換する."""

import numpy as np

HORSES = 18


def _create_umaren_ref() -> np.ndarray:
    ref = np.zeros([HORSES, HORSES])
    s = 1
    for i in range(HORSES):
        for j in range(i + 1, HORSES):
            ref[i, j] = s
            s += 1
    return ref


def _create_umatan_ref() -> np.ndarray:
    ref = np.zeros([HORSES, HORSES])
    s = 1
    for i in range(HORSES):
        for j in range(HORSES):
            if i != j:
                ref[i, j] = s
                s += 1
    return ref


def _create_sanrenpuku_ref() -> np.ndarray:
    ref = np.zeros([HORSES, HORSES, HORSES])
    s = 1
    for i in range(HORSES):
        for j in range(i + 1, HORSES):
            for k in range(j + 1, HORSES):
                ref[i, j, k] = s
                s += 1
    return ref


def _create_sanrentan_ref() -> np.ndarray:
    ref = np.zeros([HORSES, HORSES, HORSES])
    s = 1
    for i in range(HORSES):
        for j in range(HORSES):
            if i != j:
                for k in range(HORSES):
                    if k not in (i, j):
                        ref[i, j, k] = s
                        s += 1
    return ref


class Labelref:
    """18頭立てを前提とした券種別のカテゴリ番号の対応表."""

    def __init__(self):
        """馬連・馬単・三連複・三連単の対応表を作る."""
        self.umaren_ref = _create_umaren_ref()
        self.umatan_ref = _create_umatan_ref()
        self.sanrenpuku_ref = _create_sanrenpuku_ref()
        self.sanrentan_ref = _create_sanrentan_ref()

    def getUmarenNumbers(self, n):
        """馬連のカテゴリ番号を馬番の組(昇順)に変換する."""
        one = 0
        two = 0
        if n == 1:
            one = 1
            two = 2
        else:
            for i in range(18):
                for j in range(i + 1, 18):
                    if self.umaren_ref[i, j] == n:
                        one = i + 1
                        two = j + 1
                        break
        return (one, two)

    def getUmarenCatnumber(self, n, m):
        """馬番の組(n < m)を馬連のカテゴリ番号に変換する."""
        return int(self.umaren_ref[n - 1, m - 1])

    def getUmatanNumbers(self, n):
        """馬単のカテゴリ番号を1着・2着の馬番に変換する."""
        one = 0
        two = 0
        if n == 1:
            one = 1
            two = 2
        else:
            for i in range(18):
                for j in range(18):
                    if self.umatan_ref[i, j] == n:
                        one = i + 1
                        two = j + 1
                        break
        return (one, two)

    def getUmatanCatnumber(self, n, m):
        """1着n・2着mの馬番を馬単のカテゴリ番号に変換する."""
        return int(self.umatan_ref[n - 1, m - 1])

    def getSanrenpukuNumbers(self, n):
        """三連複のカテゴリ番号を馬番の組(昇順)に変換する."""
        one = 0
        two = 0
        three = 0
        if n == 1:
            one = 1
            two = 2
            three = 3
        else:
            for i in range(18):
                for j in range(i + 1, 18):
                    for k in range(j + 1, 18):
                        if self.sanrenpuku_ref[i, j, k] == n:
                            one = i + 1
                            two = j + 1
                            three = k + 1
                            break
        return (one, two, three)

    def getSanrenpukuCatnumber(self, n, m, l):
        """馬番の組(n < m < l)を三連複のカテゴリ番号に変換する."""
        return int(self.sanrenpuku_ref[n - 1, m - 1, l - 1])

    def getSanrentanNumbers(self, n):
        """三連単のカテゴリ番号を1着・2着・3着の馬番に変換する."""
        one = 0
        two = 0
        three = 0
        if n == 1:
            one = 1
            two = 2
            three = 3
        else:
            for i in range(18):
                for j in range(18):
                    if i != j:
                        for k in range(18):
                            if self.sanrentan_ref[i, j, k] == n:
                                one = i + 1
                                two = j + 1
                                three = k + 1
                                break
        return (one, two, three)

    def getSanrentanCatnumber(self, n, m, l):
        """1着n・2着m・3着lの馬番を三連単のカテゴリ番号に変換する."""
        return int(self.sanrentan_ref[n - 1, m - 1, l - 1])
