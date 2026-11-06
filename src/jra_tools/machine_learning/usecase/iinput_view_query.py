"""入力のビューを確かめるための問い合わせのインターフェース."""

from typing import List, Protocol, runtime_checkable


@runtime_checkable
class IInputViewQuery(Protocol):
    """1つの入力のビューと期間について、確かめる材料を返す."""

    def columns(self) -> List[str]:
        """ビューの列の名前(定義の順)."""

    def duplicated_keys(self) -> List[str]:
        """期間内で2行以上ある出走馬のキー."""

    def uncovered_keys(self) -> List[str]:
        """model層の出走馬(期間内)のうち、ビューに出ないもののキー."""

    def null_keys(self, column: str) -> List[str]:
        """期間内で、列が値なしの出走馬のキー."""
