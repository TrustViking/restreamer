from __future__ import annotations

import logging

from app.observability.runtime_analytics import log_preflight_status
from app.planning.debug_self_checks import run_debug_self_checks


def run_bootstrap_preflight(*, logger: logging.Logger, debug_enabled: bool) -> None:
    if not debug_enabled:
        log_preflight_status(logger=logger, self_check_skipped=True)
        return
    logger.info("Bootstrap preflight: запуск debug self-check.")
    run_debug_self_checks()
    logger.info("Bootstrap preflight: debug self-check завершен.")
    log_preflight_status(logger=logger, self_check_skipped=False)
