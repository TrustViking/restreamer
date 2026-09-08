from __future__ import annotations

import logging
import time
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
        snippet: str = text[:80].replace("\n", " ")
        t0: float = time.perf_counter()
        self._console_logger.info(text)
        t1: float = time.perf_counter()
        sink_used: bool = False
        if to_telegram and self._telegram_sink is not None:
            sink_used = True
            try:
                self._telegram_sink(text)
            except Exception:
                self._diag_logger.debug(
                    "operator_notifier_telegram_sink_failed text=%s", text
                )
        t2: float = time.perf_counter()
        self._diag_logger.debug(
            "operator_notifier_emit console_ms=%.1f sink_ms=%.1f sink_used=%s text=%s",
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            sink_used,
            snippet,
        )
