from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from app.core.constants import (
    LOGGER_NAME_DEFAULT,
    LOGGER_NAME_ENV_VAR,
)
from app.paths._root import PROJECT_ROOT

LOG_FILE_ENV_VAR: str = "LOG_FILE"
CONSOLE_LOGGER_SUFFIX: str = "console"
_CURRENT_DETAILED_LOG_FILE_PATH: Optional[Path] = None
_CURRENT_SCREEN_LOG_FILE_PATH: Optional[Path] = None


def resolve_base_logger_name() -> str:
    preferred: str = os.getenv(LOGGER_NAME_ENV_VAR, "").strip()
    if preferred:
        return preferred
    return LOGGER_NAME_DEFAULT


def get_logger(module_name: str) -> logging.Logger:
    base_name: str = resolve_base_logger_name()
    normalized: str = str(module_name or "").strip()
    if not normalized or normalized == "__main__":
        return logging.getLogger(base_name)
    if normalized == base_name or normalized.startswith(f"{base_name}."):
        return logging.getLogger(normalized)
    if normalized.startswith("app."):
        normalized = normalized[4:]
        if normalized:
            return logging.getLogger(f"{base_name}.{normalized}")
        return logging.getLogger(base_name)
    if normalized.startswith("__"):
        return logging.getLogger(base_name)
    return logging.getLogger(f"{base_name}.{normalized}")


def get_console_logger() -> logging.Logger:
    """Return the dedicated console logger for operator-facing messages."""
    base_name: str = resolve_base_logger_name()
    return logging.getLogger(f"{base_name}.{CONSOLE_LOGGER_SUFFIX}")


def resolve_logger_name() -> str:
    return resolve_base_logger_name()


def resolve_logger_name_meta() -> Tuple[str, str, bool]:
    preferred_raw: str | None = os.getenv(LOGGER_NAME_ENV_VAR)
    preferred_cleaned: str = (preferred_raw or "").strip()
    if preferred_cleaned:
        return (preferred_cleaned, f"env:{LOGGER_NAME_ENV_VAR}", True)
    env_present: bool = preferred_raw is not None
    return (LOGGER_NAME_DEFAULT, "default", env_present)


def resolve_log_dir() -> Path:
    env_log_dir_raw: str = str(os.getenv("LOG_DIR", "") or "").strip()
    project_root: Path = PROJECT_ROOT
    if env_log_dir_raw:
        candidate_path: Path = Path(env_log_dir_raw)
        if not candidate_path.is_absolute():
            candidate_path = (project_root / candidate_path).resolve()
        return candidate_path
    return project_root / "logs"


def resolve_log_file_path() -> Path:
    env_log_file_raw: str = str(os.getenv(LOG_FILE_ENV_VAR, "") or "").strip()
    if env_log_file_raw:
        candidate_path: Path = Path(env_log_file_raw)
        if not candidate_path.is_absolute():
            candidate_path = (resolve_log_dir() / candidate_path).resolve()
        return candidate_path

    log_dir_path: Path = resolve_log_dir()
    log_filename: str = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{LOGGER_NAME_DEFAULT}.log"
    return log_dir_path / log_filename


def _resolve_log_file_pair_from_detailed_path(
    detailed_log_file_path: Path,
) -> Tuple[Path, Path]:
    suffix: str = detailed_log_file_path.suffix or ".log"
    stem: str = detailed_log_file_path.stem
    screen_log_file_path: Path = detailed_log_file_path.with_name(
        f"{stem}_operator{suffix}"
    )
    return detailed_log_file_path, screen_log_file_path


def resolve_log_file_paths(*, entrypoint_label: str = LOGGER_NAME_DEFAULT) -> Tuple[Path, Path]:
    env_log_file_raw: str = str(os.getenv(LOG_FILE_ENV_VAR, "") or "").strip()
    if env_log_file_raw:
        candidate_path: Path = Path(env_log_file_raw)
        if not candidate_path.is_absolute():
            candidate_path = (resolve_log_dir() / candidate_path).resolve()
        return _resolve_log_file_pair_from_detailed_path(candidate_path)

    log_dir_path: Path = resolve_log_dir()
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_log_file_path: Path = (
        log_dir_path / f"{timestamp}_{entrypoint_label}_detailed.log"
    )
    screen_log_file_path: Path = (
        log_dir_path / f"{timestamp}_{entrypoint_label}_operator.log"
    )
    return detailed_log_file_path, screen_log_file_path


def get_current_log_file_paths() -> Tuple[Optional[Path], Optional[Path]]:
    return _CURRENT_DETAILED_LOG_FILE_PATH, _CURRENT_SCREEN_LOG_FILE_PATH


class _ConsoleChannelFilter(logging.Filter):
    """Filter for the base/root logger's screen handlers.

    - Blocks ALL records from the console logger (any level, including
      WARNING/ERROR) - the console logger has its own screen handlers,
      so letting its records through the parent's screen handlers would
      cause duplicates.
    - For all other loggers: passes WARNING+ only, blocks INFO/DEBUG.
    """

    def __init__(self, console_logger_name: str) -> None:
        super().__init__()
        self._console_prefix: str = console_logger_name + "."
        self._console_exact: str = console_logger_name

    def filter(self, record: logging.LogRecord) -> bool:
        name: str = record.name
        if name == self._console_exact or name.startswith(self._console_prefix):
            return False
        if record.levelno < logging.WARNING:
            return False
        # Block informational warnings from operator screen - detailed logs keep them.
        warning_category: str = str(getattr(record, "warning_category", "") or "")
        if warning_category == "informational":
            return False
        return True


class _ConsoleOnlyFilter(logging.Filter):
    """Filter for the console logger's own handlers.

    Passes only records that originated from the console logger.
    This prevents WARNING/ERROR records from non-console loggers
    from appearing twice on screen (they already pass through
    the base/root logger's screen handlers via _ConsoleChannelFilter).
    """

    def __init__(self, console_logger_name: str) -> None:
        super().__init__()
        self._console_prefix: str = console_logger_name + "."
        self._console_exact: str = console_logger_name

    def filter(self, record: logging.LogRecord) -> bool:
        name: str = record.name
        return name == self._console_exact or name.startswith(self._console_prefix)


def _remove_and_close_handlers(logger: logging.Logger) -> None:
    """Remove all handlers from *logger* and close each one.

    Closing is important on Windows where FileHandler keeps the file
    descriptor open until explicitly closed.
    """
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def setup_logging(debug: bool) -> None:
    global _CURRENT_DETAILED_LOG_FILE_PATH, _CURRENT_SCREEN_LOG_FILE_PATH
    base_logger_name: str = resolve_base_logger_name()

    detailed_log_file_path: Path
    screen_log_file_path: Path
    detailed_log_file_path, screen_log_file_path = resolve_log_file_paths(
        entrypoint_label=base_logger_name
    )
    _CURRENT_DETAILED_LOG_FILE_PATH = detailed_log_file_path
    _CURRENT_SCREEN_LOG_FILE_PATH = screen_log_file_path
    detailed_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    screen_log_file_path.parent.mkdir(parents=True, exist_ok=True)

    stream_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    base_logger: logging.Logger = logging.getLogger(base_logger_name)
    base_logger.setLevel(logging.DEBUG)
    base_logger.propagate = False
    _remove_and_close_handlers(base_logger)

    stream_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    base_logger.addHandler(stream_handler)

    file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(detailed_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    base_logger.addHandler(file_handler)

    screen_file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(screen_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    screen_file_handler.setLevel(logging.INFO)
    screen_file_handler.setFormatter(stream_formatter)
    base_logger.addHandler(screen_file_handler)

    # --- Console channel: operator-facing messages on screen ---
    console_logger_name: str = f"{base_logger_name}.{CONSOLE_LOGGER_SUFFIX}"
    console_logger: logging.Logger = logging.getLogger(console_logger_name)
    _remove_and_close_handlers(console_logger)
    console_logger.setLevel(logging.INFO)
    console_logger.propagate = True  # detailed file handler receives via parent

    console_formatter: logging.Formatter = logging.Formatter("%(message)s")

    console_stream_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    console_stream_handler.setLevel(logging.INFO)
    console_stream_handler.setFormatter(console_formatter)
    console_only_filter: _ConsoleOnlyFilter = _ConsoleOnlyFilter(console_logger_name)
    console_stream_handler.addFilter(console_only_filter)
    console_logger.addHandler(console_stream_handler)

    console_file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(screen_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    console_file_handler.setLevel(logging.INFO)
    console_file_handler.setFormatter(console_formatter)
    console_file_handler.addFilter(console_only_filter)
    console_logger.addHandler(console_file_handler)

    # Filter base logger's screen handlers: WARNING+ only, console records blocked
    channel_filter: _ConsoleChannelFilter = _ConsoleChannelFilter(console_logger_name)
    stream_handler.addFilter(channel_filter)
    screen_file_handler.addFilter(channel_filter)

    base_logger.info(
        "Logging initialized console_channel=enabled detailed_file_level=DEBUG operator_filter=WARNING+ detailed_file=%s operator_file=%s",
        str(detailed_log_file_path),
        str(screen_log_file_path),
    )

    logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
    logging.getLogger("oauthlib").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.http").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def setup_bot_logging(*, debug: bool = False) -> None:
    """Configure logging for the Telegram bot entrypoint.

    Uses the same three-tier scheme as setup_logging():
    - detailed file: DEBUG (all records)
    - screen stdout + screen file: WARNING+ only (via filter)
    - console channel: operator-facing INFO messages

    Key difference from setup_logging(): handlers are attached to
    the root logger so that third-party libraries (aiogram, etc.)
    are also captured in the detailed file.
    """
    base_logger_name: str = resolve_base_logger_name()
    entrypoint_label: str = f"{base_logger_name}_bot"

    detailed_log_file_path: Path
    screen_log_file_path: Path
    detailed_log_file_path, screen_log_file_path = resolve_log_file_paths(
        entrypoint_label=entrypoint_label
    )
    detailed_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    screen_log_file_path.parent.mkdir(parents=True, exist_ok=True)

    stream_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # --- Defensive reset: clear base logger to avoid cross-contamination ---
    base_logger: logging.Logger = logging.getLogger(base_logger_name)
    _remove_and_close_handlers(base_logger)
    base_logger.setLevel(logging.NOTSET)
    base_logger.propagate = True

    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    _remove_and_close_handlers(root_logger)

    stream_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    root_logger.addHandler(stream_handler)

    detailed_file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(detailed_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    detailed_file_handler.setLevel(logging.DEBUG)
    detailed_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(detailed_file_handler)

    screen_file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(screen_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    screen_file_handler.setLevel(logging.INFO)
    screen_file_handler.setFormatter(stream_formatter)
    root_logger.addHandler(screen_file_handler)

    # --- Console channel (same scheme as setup_logging) ---
    console_logger_name: str = f"{base_logger_name}.{CONSOLE_LOGGER_SUFFIX}"
    console_logger: logging.Logger = logging.getLogger(console_logger_name)
    _remove_and_close_handlers(console_logger)
    console_logger.setLevel(logging.INFO)
    console_logger.propagate = True

    console_formatter: logging.Formatter = logging.Formatter("%(message)s")

    console_stream_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    console_stream_handler.setLevel(logging.INFO)
    console_stream_handler.setFormatter(console_formatter)
    console_only_filter: _ConsoleOnlyFilter = _ConsoleOnlyFilter(console_logger_name)
    console_stream_handler.addFilter(console_only_filter)
    console_logger.addHandler(console_stream_handler)

    console_file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(screen_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    console_file_handler.setLevel(logging.INFO)
    console_file_handler.setFormatter(console_formatter)
    console_file_handler.addFilter(console_only_filter)
    console_logger.addHandler(console_file_handler)

    # Filter root logger's screen handlers: WARNING+ only, console records blocked
    channel_filter: _ConsoleChannelFilter = _ConsoleChannelFilter(console_logger_name)
    stream_handler.addFilter(channel_filter)
    screen_file_handler.addFilter(channel_filter)

    root_logger.info(
        "Bot logging initialized console_channel=enabled detailed_file_level=DEBUG operator_filter=WARNING+ detailed_file=%s operator_file=%s",
        str(detailed_log_file_path),
        str(screen_log_file_path),
    )

    logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
    logging.getLogger("oauthlib").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.http").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    class _AiogramFetchUpdateDowngradeFilter(logging.Filter):
        """Downgrade transient aiogram polling errors from ERROR to WARNING.

        ServerDisconnectedError and Request timeout error during getUpdates
        are normal transient network glitches in long-running polling and
        should not pollute the operator-facing error stream.
        """

        _PATTERNS: tuple[str, ...] = (
            "Failed to fetch updates",
        )

        def filter(self, record: logging.LogRecord) -> bool:
            if record.name != "aiogram.dispatcher":
                return True
            if record.levelno < logging.ERROR:
                return True
            message: str = record.getMessage()
            for pattern in self._PATTERNS:
                if pattern in message:
                    record.levelno = logging.WARNING
                    record.levelname = "WARNING"
                    return True
            return True

    logging.getLogger("aiogram.dispatcher").addFilter(_AiogramFetchUpdateDowngradeFilter())
