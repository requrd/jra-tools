"""ビューから読んだ表と、Parquet(メタデータ付き)のバイト列との変換.

行の並び(開催日・レースキー・馬番の順)を保つ。Kerasのvalidation_splitは並びの末尾を
検証に使うので、並びが変わると検証の期間が変わる。
"""

import io
from typing import Dict, Mapping, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# メタデータのキーの接頭辞(pandasのメタデータと分ける)
_PREFIX = "jra_tools."


def frame_to_parquet(frame: pd.DataFrame, metadata: Mapping[str, str]) -> bytes:
    """表をParquetのバイト列にする. メタデータを付ける."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    existing = table.schema.metadata or {}
    extra = {f"{_PREFIX}{k}".encode(): v.encode() for k, v in metadata.items()}
    table = table.replace_schema_metadata({**existing, **extra})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def parquet_to_frame(data: bytes) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Parquetのバイト列から、表とメタデータを読む."""
    table = pq.read_table(io.BytesIO(data))
    metadata = {
        key.decode()[len(_PREFIX) :]: value.decode()
        for key, value in (table.schema.metadata or {}).items()
        if key.decode().startswith(_PREFIX)
    }
    return table.to_pandas(), metadata
