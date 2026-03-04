from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import yaml

from app.config.settings import AppConfig
from app.config.template_loader import load_templates_from_path
from app.config.validators import (
    must_get_env,
    normalize_google_doc_share_mode,
    normalize_now_tz_mode,
    normalize_processing_mode,
    resolve_llm_provider_from_env,
    setting_as_bool,
    setting_as_float,
    setting_as_int,
    setting_as_optional_str,
    setting_as_str,
    validate_app_settings,
)
from app.paths import get_project_paths


def load_app_settings_from_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"App config file does not exist: {path}")
    raw_text: str = path.read_text(encoding="utf-8")
    try:
        payload: Any = yaml.safe_load(raw_text)
    except Exception as error:
        raise RuntimeError(f"Failed to parse app config file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"App config root must be a mapping in {path}")
    app_payload: Any = payload.get("app")
    if not isinstance(app_payload, dict):
        raise RuntimeError(f"App config must contain mapping key 'app' in {path}")
    app_settings: Dict[str, Any] = cast(Dict[str, Any], app_payload)
    validate_app_settings(app_settings)
    return app_settings


def load_google_auth_mode_from_env() -> str:
    raw_value: str = os.getenv("GOOGLE_AUTH_MODE", "").strip().lower()
    if not raw_value:
        return "oauth"
    if raw_value in {"oauth", "service_account"}:
        return raw_value
    raise RuntimeError(
        "Invalid GOOGLE_AUTH_MODE. Allowed values: 'oauth', 'service_account'. "
        f"Current value: {raw_value!r}"
    )


def load_google_service_account_path() -> Optional[Path]:
    raw_value: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH", "").strip()
    if not raw_value:
        return None
    return Path(raw_value)


def load_google_oauth_credentials_path() -> Path:
    raw_value: str = os.getenv("GOOGLE_OAUTH_CREDENTIALS_PATH", "").strip()
    return Path(raw_value) if raw_value else get_project_paths().oauth_credentials_path


def load_google_oauth_token_path() -> Path:
    raw_value: str = os.getenv("GOOGLE_OAUTH_TOKEN_PATH", "").strip()
    return Path(raw_value) if raw_value else get_project_paths().oauth_token_path


def warn_ignored_google_auth_mode_in_config(
    app_settings: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    google_payload: Any = app_settings.get("google")
    if not isinstance(google_payload, dict):
        return
    for key, value in google_payload.items():
        normalized_key: str = str(key or "").strip().lower()
        if "auth" in normalized_key and "mode" in normalized_key:
            (logger or logging.getLogger(__name__)).warning(
                "auth_mode in config is ignored; use GOOGLE_AUTH_MODE env; ignored_value=%r",
                value,
            )
            return


def validate_google_service_account_path_requirement(
    *,
    google_enabled: bool,
    google_auth_mode: str,
    service_account_path_raw: str,
    service_account_path: Optional[Path],
    summarize_error: Optional[Callable[[Exception], str]] = None,
) -> None:
    if not google_enabled:
        return
    if google_auth_mode != "service_account":
        return
    resolved_path: str = str(service_account_path_raw or "").strip()
    if not resolved_path or service_account_path is None:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}"
        )
    if not service_account_path.exists():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}. File not found."
        )
    if not service_account_path.is_file():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}. Path is not a file."
        )
    try:
        with service_account_path.open("r", encoding="utf-8-sig") as file_obj:
            file_obj.read(1)
    except Exception as error:
        format_error: str = summarize_error(error) if summarize_error else str(error)
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}. File is not readable: {format_error}"
        ) from error


def _load_bool_env(name: str, default: bool) -> bool:
    raw_value: str = str(os.getenv(name, "")).strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"Invalid boolean for {name}: {raw_value!r}. Allowed: 1/0,true/false,yes/no,on/off."
    )


def _load_int_env(name: str, default: int, min_value: int = 0) -> int:
    raw_value: str = str(os.getenv(name, "")).strip()
    if not raw_value:
        return default
    try:
        value: int = int(raw_value)
    except Exception as error:
        raise RuntimeError(f"Invalid integer for {name}: {raw_value!r}") from error
    if value < min_value:
        raise RuntimeError(f"{name} must be >= {min_value}, got {value}")
    return value


def _load_float_env(name: str, default: float, min_value: float = 0.0) -> float:
    raw_value: str = str(os.getenv(name, "")).strip()
    if not raw_value:
        return default
    try:
        value: float = float(raw_value)
    except Exception as error:
        raise RuntimeError(f"Invalid float for {name}: {raw_value!r}") from error
    if value < min_value:
        raise RuntimeError(f"{name} must be >= {min_value}, got {value}")
    return value


def load_config_from_env(
    *,
    logger: logging.Logger,
    summarize_error: Optional[Callable[[Exception], str]] = None,
) -> AppConfig:
    project_paths = get_project_paths()
    app_config_path: Path = (
        Path(os.getenv("APP_CONFIG_PATH", "").strip())
        if os.getenv("APP_CONFIG_PATH", "").strip()
        else project_paths.runtime_config_path
    )
    app_settings: Dict[str, Any] = load_app_settings_from_path(app_config_path)
    warn_ignored_google_auth_mode_in_config(app_settings, logger=logger)

    templates = load_templates_from_path(project_paths.templates_path)

    processing_mode: str = normalize_processing_mode(
        setting_as_str(app_settings, "processing.mode"),
        source="app.processing.mode",
    )
    config_now_tz_raw: str = setting_as_optional_str(app_settings, "processing.now_tz") or ""
    env_now_tz_raw: str = os.getenv("STG_NOW_TZ", "").strip()
    if env_now_tz_raw and config_now_tz_raw and env_now_tz_raw != config_now_tz_raw:
        logger.warning(
            "STG_NOW_TZ overrides processing.now_tz: config=%r env=%r",
            config_now_tz_raw,
            env_now_tz_raw,
        )
    now_tz_mode_input: str = env_now_tz_raw or config_now_tz_raw or "kyiv"
    now_tz_mode: str = normalize_now_tz_mode(now_tz_mode_input, source="now timezone mode")

    llm_provider: str = resolve_llm_provider_from_env(logger=logger)
    openai_model_primary: str = os.getenv("STG_OPENAI_MODEL_PRIMARY", "").strip() or "gpt-5.1"
    openai_model_fallback: str = os.getenv("STG_OPENAI_MODEL_FALLBACK", "").strip() or "gpt-5-mini"
    openai_timeout_sec: float = _load_float_env("STG_OPENAI_TIMEOUT_SEC", 120.0, min_value=1.0)
    openai_max_output_tokens: int = _load_int_env("STG_OPENAI_MAX_OUTPUT_TOKENS", 1000, min_value=1)
    openai_pre_delay_sec: float = _load_float_env("STG_OPENAI_PRE_DELAY_SEC", 5.0, min_value=0.0)
    llm_source_desc_max_chars: int = _load_int_env("STG_LLM_SOURCE_DESC_MAX_CHARS", 2000, min_value=200)
    llm_run_if_single_source: bool = _load_bool_env("STG_LLM_RUN_IF_SINGLE_SOURCE", False)
    if llm_provider != "openai":
        raise RuntimeError(
            f"Only OpenAI is supported now. Unsupported llm_provider={llm_provider!r}."
        )

    google_auth_mode: str = load_google_auth_mode_from_env()
    google_service_account_path_raw: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH", "").strip()
    if os.getenv("GOOGLE_CREDENTIALS_PATH", "").strip() or os.getenv("GOOGLE_TOKEN_PATH", "").strip():
        logger.warning(
            "GOOGLE_CREDENTIALS_PATH/GOOGLE_TOKEN_PATH are deprecated and ignored. "
            "Use GOOGLE_OAUTH_CREDENTIALS_PATH/GOOGLE_OAUTH_TOKEN_PATH."
        )

    google_enabled: bool = setting_as_bool(app_settings, "google.enabled")
    config_kwargs: Dict[str, Any] = {
        "telegram_bot_token": must_get_env("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": must_get_env("TELEGRAM_CHAT_ID"),
        "telegram_enabled": setting_as_bool(app_settings, "telegram.enabled"),
        "google_enabled": google_enabled,
        "google_service_account_path": Path(google_service_account_path_raw) if google_service_account_path_raw else None,
        "google_drive_folder_id": setting_as_str(app_settings, "google.drive_folder_id") or None,
        "google_drive_preview_folder_id": setting_as_str(app_settings, "google.drive_preview_folder_id") or setting_as_str(app_settings, "google.drive_folder_id") or None,
        "google_drive_preview_path_template": setting_as_str(app_settings, "google.drive_preview_path_template"),
        "google_doc_share_mode": normalize_google_doc_share_mode(setting_as_str(app_settings, "google.doc_share_mode")),
        "google_sheets_id": setting_as_str(app_settings, "google.sheets_id"),
        "google_sheets_range": setting_as_str(app_settings, "google.sheets_range"),
        "google_form_url": setting_as_str(app_settings, "google.form_url"),
        "google_contacts": setting_as_str(app_settings, "google.contacts"),
        "local_image_dir_template": setting_as_str(app_settings, "paths.local_image_dir_template"),
        "local_doc_dir_template": setting_as_optional_str(app_settings, "paths.local_doc_dir_template"),
        "timezone_kiev": setting_as_str(app_settings, "timezones.kiev"),
        "timezone_cet": setting_as_str(app_settings, "timezones.cet"),
        "telegram_symbol_separator": setting_as_str(app_settings, "telegram.symbol_separator"),
        "telegram_separator_repeat_count": setting_as_int(app_settings, "telegram.separator_repeat_count"),
        "telegram_symbol_broadcast": setting_as_str(app_settings, "telegram.symbol_broadcast"),
        "telegram_symbol_alert": setting_as_str(app_settings, "telegram.symbol_alert"),
        "telegram_symbol_form": setting_as_str(app_settings, "telegram.symbol_form"),
        "telegram_symbol_description": setting_as_str(app_settings, "telegram.symbol_description"),
        "telegram_symbol_pin": setting_as_str(app_settings, "telegram.symbol_pin"),
        "telegram_symbol_done": setting_as_str(app_settings, "telegram.symbol_done"),
        "telegram_flag_uk": setting_as_str(app_settings, "telegram.flag_uk"),
        "telegram_flag_en": setting_as_str(app_settings, "telegram.flag_en"),
        "telegram_flag_ru": setting_as_str(app_settings, "telegram.flag_ru"),
        "telegram_flag_other": setting_as_str(app_settings, "telegram.flag_other"),
        "telegram_flag_repeat_count": setting_as_int(app_settings, "telegram.flag_repeat_count"),
        "telegram_language_name_uk": setting_as_str(app_settings, "telegram.language_name_uk"),
        "telegram_language_name_en": setting_as_str(app_settings, "telegram.language_name_en"),
        "telegram_language_name_ru": setting_as_str(app_settings, "telegram.language_name_ru"),
        "telegram_language_name_other": setting_as_str(app_settings, "telegram.language_name_other"),
        "processing_mode": processing_mode,
        "now_tz_mode": now_tz_mode,
        "llm_provider": llm_provider,
        "openai_model_primary": openai_model_primary,
        "openai_model_fallback": openai_model_fallback,
        "openai_timeout_sec": openai_timeout_sec,
        "openai_max_output_tokens": openai_max_output_tokens,
        "openai_pre_delay_sec": openai_pre_delay_sec,
        "llm_source_desc_max_chars": llm_source_desc_max_chars,
        "llm_run_if_single_source": llm_run_if_single_source,
        "preview_filename_max_stem": setting_as_int(app_settings, "files.preview_filename_max_stem"),
        "stg_templates_path": project_paths.templates_path,
        "telegram_use_audit": setting_as_bool(app_settings, "telegram.use_audit"),
        "templates": templates,
    }

    validate_google_service_account_path_requirement(
        google_enabled=google_enabled,
        google_auth_mode=google_auth_mode,
        service_account_path_raw=google_service_account_path_raw,
        service_account_path=cast(Optional[Path], config_kwargs["google_service_account_path"]),
        summarize_error=summarize_error,
    )
    return AppConfig(**config_kwargs)
