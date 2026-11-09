"""カテゴリの列の決まり: code_categoriesの行から控えを作り、コードを位置にする.

変換の決まりの正はDBのcode_categories(model・features・sandbox)。1コード1行で
(item, code, category_no, label)。列の名前 = 項目の名前(item)、列の値 = コード(文字列)。

- 項目は、探す先を合わせて名前で引く。0か所・2か所以上にあれば止める(s1)
- 控えは「コード→番号」。one-hotの次元は最大の番号 + 例外の列1
- 控えにないコード(学習のあとに増えたコードなど)・NULLは例外の列
"""

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# code_categoriesを探す先の既定(入口の引数で替えられる)
DEFAULT_CATEGORY_SOURCES = (
    "model.code_categories",
    "features.code_categories",
    "sandbox.code_categories",
)

# code_categoriesの行: (探す先, item, code, category_no)
CategoryRow = Tuple[str, str, str, int]


def pick_categories(
    items: Sequence[str], rows: Iterable[CategoryRow]
) -> Dict[str, Dict[str, int]]:
    """項目ごとの「コード→番号」を、探す先の行から1つに決める.

    Args:
        items (Sequence[str]): 項目の名前(仕様のカテゴリの列名)
        rows (Iterable[CategoryRow]): 探す先から読んだ、itemsの行

    Returns:
        Dict[str, Dict[str, int]]: 項目 → (コード → 番号)

    Raises:
        ValueError: 項目がどこにもない、または2か所以上にある場合

    """
    by_item: Dict[str, Dict[str, Dict[str, int]]] = {}
    for source, item, code, number in rows:
        by_item.setdefault(item, {}).setdefault(source, {})[str(code)] = int(number)
    missing = [item for item in items if item not in by_item]
    if missing:
        raise ValueError(f"code_categoriesに項目がありません: {missing}")
    duplicated = {
        item: sorted(by_item[item]) for item in items if len(by_item[item]) > 1
    }
    if duplicated:
        raise ValueError(
            f"同じ項目が2か所以上のcode_categoriesにあります(名前を分けてください): {duplicated}"
        )
    return {item: next(iter(by_item[item].values())) for item in items}


def positions(values: pd.Series, categories: Mapping[str, int]) -> np.ndarray:
    """コードを、one-hotで1にする位置(0から)にする.

    番号nの位置はn - 1。控えにないコード・NULLは例外の列(最大の番号の位置)。

    Args:
        values (pd.Series): コード(文字列。NULLを含んでよい)
        categories (Mapping[str, int]): 控えの「コード→番号」

    Returns:
        np.ndarray: 位置(int)

    """
    exception = max(categories.values())
    numbers = values.map(lambda v: _number(v, categories, exception))
    return numbers.to_numpy(dtype=int) - 1


def _number(value: object, categories: Mapping[str, int], exception: int) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return exception + 1
    return categories.get(str(value), exception + 1)
