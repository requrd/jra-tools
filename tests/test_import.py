"""パッケージの読み込みのテスト."""

import subprocess
import sys


def test_import_does_not_load_keras():
    """推論しない利用者のため、jra_toolsのimportでKeras(TensorFlow)を読み込まないことを確認する.

    読み込み済みのモジュールの影響を受けないよう、別のプロセスでimportする。
    """
    code = (
        "import sys, jra_tools; "
        "sys.exit('keras' in sys.modules or 'tensorflow' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False)  # noqa: S603

    assert result.returncode == 0
