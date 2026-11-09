"""ファイルの読み書き(Parquet・YAML). 形式は扱わない."""

from pathlib import Path
from typing import Any, Optional

import yaml


class Directory:
    """1つのフォルダのファイル(interface_adapter/frame_store.pyのIBinaryFiles)."""

    def __init__(self, path: Path | str):
        """フォルダを受け取る. なければ書くときに作る."""
        self.path = Path(path)

    def read_bytes(self, name: str) -> Optional[bytes]:
        """ファイルの中身. なければNone."""
        file = self.path / name
        return file.read_bytes() if file.exists() else None

    def write_bytes(self, name: str, data: bytes) -> None:
        """ファイルに書く(途中で止まっても壊れたファイルを残さないよう、書いてから置き換える)."""
        self.path.mkdir(parents=True, exist_ok=True)
        temporary = self.path / f".{name}.tmp"
        temporary.write_bytes(data)
        temporary.replace(self.path / name)


def read_yaml(path: Path | str) -> Any:
    """YAMLファイルを読む."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data: Any, path: Path | str) -> None:
    """YAMLファイルに書く(キーの順を保つ)."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
