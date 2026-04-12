from __future__ import annotations

import logging
from collections.abc import Callable

from app.bootstrap.logging_config import get_console_logger


class OperatorNotifier:
    def __init__(
        self,
        *,
        telegram_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._console_logger: logging.Logger = get_console_logger()
        self._telegram_sink: Callable[[str], None] | None = telegram_sink
        self._diagnostic_logger: logging.Logger = logging.getLogger(__name__)

    def emit(self, text: str, *, to_telegram: bool = True) -> None:
        self._console_logger.info(text)
        if not to_telegram or self._telegram_sink is None:
            return
        try:
            self._telegram_sink(text)
        except Exception as error:
            self._diagnostic_logger.debug(
                "operator_notifier_sink_failed text=%s error=%s",
                text,
                error,
            )
