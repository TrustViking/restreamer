from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.pipeline.operator_notifier import OperatorNotifier


class TestOperatorNotifierTelegramFlag(unittest.TestCase):

    def test_to_telegram_false_does_not_call_sink(self) -> None:
        mock_sink = MagicMock()
        notifier = OperatorNotifier(telegram_sink=mock_sink)
        notifier.emit("test", to_telegram=False)
        mock_sink.assert_not_called()

    def test_to_telegram_true_calls_sink_once(self) -> None:
        mock_sink = MagicMock()
        notifier = OperatorNotifier(telegram_sink=mock_sink)
        notifier.emit("test2", to_telegram=True)
        mock_sink.assert_called_once_with("test2")
