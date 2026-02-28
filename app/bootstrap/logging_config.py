from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from app.core.constants import LOGGER_NAME_DEFAULT, LOGGER_NAME_ENV_VAR

LEGACY_LOGGER_NAME_ENV_VAR: str = "STREAMERTG_LOGGER_NAME"


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
    env_log_dir_raw: str = str(os.getenv("STG_LOG_DIR", "") or "").strip()
    script_dir: Path = Path(__file__).resolve().parents[2]
    if env_log_dir_raw:
        candidate_path: Path = Path(env_log_dir_raw)
        if not candidate_path.is_absolute():
            candidate_path = (script_dir / candidate_path).resolve()
        return candidate_path
    windows_default: Path = Path(r"D:\_projects\restreamer\logs")
    if windows_default.exists():
        return windows_default
    return script_dir / "logs"


def setup_logging(debug: bool) -> None:
    base_logger_name: str = resolve_base_logger_name()
    app_level: int = logging.DEBUG if debug else logging.INFO

    log_dir_path: Path = resolve_log_dir()
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_filename: str = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_restreamer.log"
    log_file_path: Path = log_dir_path / log_filename

    formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    base_logger: logging.Logger = logging.getLogger(base_logger_name)
    base_logger.setLevel(app_level)
    base_logger.propagate = False
    for handler in list(base_logger.handlers):
        base_logger.removeHandler(handler)

    stream_handler: logging.StreamHandler = logging.StreamHandler()
    stream_handler.setLevel(app_level)
    stream_handler.setFormatter(formatter)
    base_logger.addHandler(stream_handler)

    file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(log_file_path),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(app_level)
    file_handler.setFormatter(formatter)
    base_logger.addHandler(file_handler)

    base_logger.info(
        "Logging initialized level=%s file=%s",
        logging.getLevelName(app_level),
        str(log_file_path),
    )

    logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
    logging.getLogger("oauthlib").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.http").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
