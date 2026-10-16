"""converterのユニットテスト."""

from unittest.mock import MagicMock

import pytest

from jra_tools.machine_learning.usecase.converter import Converter

mock_load = MagicMock()
mock_period = MagicMock()


class MockQuery:
    """IQueryのモック."""

    load = mock_load
    period = mock_period


@pytest.fixture
def mock_query(sample_kaisais):
    """サンプルの開催データを返すクエリ."""
    mock_query = MockQuery()
    mock_load.return_value = sample_kaisais
    yield mock_query
    mock_load.reset_mock()


mock_save = MagicMock()


class MockFactory:
    """ICreatorのモック."""

    def __init__(self, kaisais):
        """開催データを保持する."""
        self.kaisais = kaisais

    save = mock_save


@pytest.fixture
def mock_creator_factory():
    """saveの呼び出しを記録するCreatorのファクトリ."""
    yield MockFactory
    mock_save.reset_mock()


def test_save_with_name(mock_query, mock_creator_factory):
    """指定したファイル名で保存されることを確認する."""
    converter = Converter(mock_query, [mock_creator_factory])
    converter.save("test_file")

    mock_load.assert_called_once_with()
    mock_save.assert_called_once_with("test_file")


def test_save_without_name(mock_query, mock_creator_factory):
    """ファイル名を省略するとクエリの期間名で保存されることを確認する."""
    converter = Converter(mock_query, [mock_creator_factory])
    converter.save()

    mock_load.assert_called_once_with()
    mock_save.assert_called_once_with(mock_period)
