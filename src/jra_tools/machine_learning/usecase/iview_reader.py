"""ビューを読むためのインターフェース(usecase/view_dataset.pyが使う).

1回の読み込み(1年分)は1つのセッションで行う。セッションの中では、ビューの定義が
読み終わるまで変わらないこと(ロックしてから照合し、読む)を実装が保つ。
"""

from contextlib import AbstractContextManager
from typing import List, Optional, Protocol, Sequence

import pandas as pd


class IViewSession(Protocol):
    """1回の読み込みのセッション(1トランザクション)."""

    def exists(self, view: str) -> bool:
        """ビューがあるか."""

    def lock(self, views: Sequence[str]) -> None:
        """読み終わるまで、ビューの定義を変えさせない(ほかの読み込みは止めない)."""

    def columns(self, view: str) -> List[str]:
        """ビューの列の名前."""

    def digest(self, view: str) -> str:
        """ビューの定義のハッシュ(依存するビュー・関数の定義を含む)."""

    def read(
        self,
        view: str,
        columns: Sequence[str],
        labels: Optional[str],
        start: int,
        end: int,
    ) -> pd.DataFrame:
        """期間内(開催日、YYYYMMDD、両端を含む)の出走を1頭1行で読む.

        Args:
            view (str): 入力のビュー
            columns (Sequence[str]): 入力のビューから読む列
            labels (Optional[str]): ラベルのビュー。あれば、入力のビューに左結合して
                LABEL_PRESENT(ラベルのビューに行があるか)とLABEL_COLUMNSのキー以外の列を足す
            start (int): 開始日
            end (int): 終了日

        Returns:
            pd.DataFrame: 開催日・レースキー・馬番の順

        """


class IViewReader(Protocol):
    """ビューの読み手."""

    def session(self) -> AbstractContextManager[IViewSession]:
        """読み込みのセッションを始める."""
