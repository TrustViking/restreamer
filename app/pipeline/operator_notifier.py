from __future__ import annotations

import logging
from typing import Callable, Optional

from app.bootstrap.logging_config import get_console_logger


class OperatorNotifier:
    """Unified channel for operator-facing messages."""

    def __init__(
        self,
        *,
        telegram_sink: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._console_logger: logging.Logger = get_console_logger()
        self._telegram_sink: Optional[Callable[[str], None]] = telegram_sink
        self._diag_logger: logging.Logger = logging.getLogger(__name__)

    def emit(self, text: str, *, to_telegram: bool = True) -> None:
        """Fan-out: console logger + optional Telegram sink."""
        self._console_logger.info(text)
        if to_telegram and self._telegram_sink is not None:
            try:
                self._telegram_sink(text)
            except Exception:
                self._diag_logger.debug(
                    "operator_notifier_telegram_sink_failed text=%s", text
                )
