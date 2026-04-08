from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from app.core.constants import LOGGER_NAME_DEFAULT, LOGGER_NAME_ENV_VAR
from app.paths._root import PROJECT_ROOT

LEGACY_LOGGER_NAME_ENV_VAR: str = "STREAMERTG_LOGGER_NAME"
LOG_FILE_ENV_VAR: str = "RESTREAMER_LOG_FILE"


def resolve_base_logger_name() -> str:
    preferred: str = os.getenv(LOGGER_NAME_ENV_VAR, "").strip()
    if preferred:
        return preferred
    legacy: str = os.getenv(LEGACY_LOGGER_NAME_ENV_VAR, "").strip()
    if legacy:
        return legacy
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


def resolve_logger_name() -> str:
    return resolve_base_logger_name()


def resolve_logger_name_meta() -> Tuple[str, str, bool]:
    preferred_raw: Optional[str] = os.getenv(LOGGER_NAME_ENV_VAR)
    preferred_cleaned: str = (preferred_raw or "").strip()
    if preferred_cleaned:
        return (preferred_cleaned, f"env:{LOGGER_NAME_ENV_VAR}", True)

    legacy_raw: Optional[str] = os.getenv(LEGACY_LOGGER_NAME_ENV_VAR)
    legacy_cleaned: str = (legacy_raw or "").strip()
    if legacy_cleaned:
        return (legacy_cleaned, f"env:{LEGACY_LOGGER_NAME_ENV_VAR}", True)

    env_present: bool = preferred_raw is not None or legacy_raw is not None
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
        f"{stem}_screen{suffix}"
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
        log_dir_path / f"{timestamp}_{entrypoint_label}_screen.log"
    )
    return detailed_log_file_path, screen_log_file_path


def setup_logging(debug: bool) -> None:
    base_logger_name: str = resolve_base_logger_name()

    detailed_log_file_path: Path
    screen_log_file_path: Path
    detailed_log_file_path, screen_log_file_path = resolve_log_file_paths(
        entrypoint_label=base_logger_name
    )
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
    for handler in list(base_logger.handlers):
        base_logger.removeHandler(handler)

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

    base_logger.info(
        "Logging initialized stream_level=INFO detailed_file_level=DEBUG screen_file_level=INFO detailed_file=%s screen_file=%s",
        str(detailed_log_file_path),
        str(screen_log_file_path),
    )

    logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
    logging.getLogger("oauthlib").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.http").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
