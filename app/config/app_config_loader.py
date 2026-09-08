from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import yaml

from app.config.env_reader import EnvReader
from app.config.settings import (
    AppConfig,
    CleanupConfig,
    GoogleConfig,
    LlmConfig,
    PathsConfig,
    ProcessingConfig,
    TelegramConfig,
    TimezoneConfig,
    YtDlpConfig,
)
from app.config.template_loader import load_templates_from_path
from app.config.validators import (
    normalize_google_doc_share_mode,
    normalize_now_tz_mode,
    normalize_processing_mode,
    setting_as_bool,
    setting_as_float,
    setting_as_int,
    setting_as_optional_str,
    setting_as_str,
    validate_app_settings,
)
from app.llm.models.model_identity import (
    DEFAULT_OPENAI_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SERVICE_TIER,
    REASONING_EFFORT_VALUES,
    SERVICE_TIER_VALUES,
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


def _parse_env_bool(raw_value: str, *, env_name: str) -> bool:
    """Parse common boolean-like env values. Raises RuntimeError on invalid input."""
    normalized_value: str = raw_value.strip().lower()
    if normalized_value in {"1", "true", "yes", "on"}:
        return True
    if normalized_value in {"0", "false", "no", "off", ""}:
        return False
    raise RuntimeError(
        f"Invalid {env_name} value: {raw_value!r}. "
        "Allowed: true/false, 1/0, yes/no, on/off."
    )


def load_google_auth_mode(
    app_settings: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve Google auth mode: GOOGLE_AUTH_MODE env > GOOGLE_AUTH_MODE_SERVICE_ACCOUNT env > YAML > default 'oauth'."""
    env_mode: str = os.getenv("GOOGLE_AUTH_MODE", "").strip().lower()
    if env_mode:
        if env_mode in {"oauth", "service_account"}:
            return env_mode
        raise RuntimeError(
            "Invalid GOOGLE_AUTH_MODE. Allowed values: 'oauth', 'service_account'. "
            f"Current value: {env_mode!r}"
        )

    env_service_account_toggle: str = os.getenv(
        "GOOGLE_AUTH_MODE_SERVICE_ACCOUNT",
        "",
    ).strip()
    if env_service_account_toggle:
        if _parse_env_bool(
            env_service_account_toggle,
            env_name="GOOGLE_AUTH_MODE_SERVICE_ACCOUNT",
        ):
            return "service_account"
        return "oauth"

    if isinstance(app_settings, dict):
        google_payload: Any = app_settings.get("google")
        if isinstance(google_payload, dict):
            yaml_value: str = str(
                google_payload.get("auth_mode") or ""
            ).strip().lower()
            if yaml_value in {"oauth", "service_account"}:
                return yaml_value

    return "oauth"


# Legacy alias kept for backward compatibility (used by google/auth.py import)
def load_google_auth_mode_from_env() -> str:
    return load_google_auth_mode()


def load_google_service_account_path() -> Optional[Path]:
    raw_value: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH", "").strip()
    if raw_value:
        return Path(raw_value)
    # Fallback: secrets/service_account.json via ProjectPaths
    default_path: Path = get_project_paths().service_account_path
    if default_path.exists():
        return default_path
    return None


def load_google_oauth_credentials_path() -> Path:
    raw_value: str = os.getenv("GOOGLE_OAUTH_CREDENTIALS_PATH", "").strip()
    return Path(raw_value) if raw_value else get_project_paths().oauth_credentials_path


def load_google_oauth_token_path() -> Path:
    raw_value: str = os.getenv("GOOGLE_OAUTH_TOKEN_PATH", "").strip()
    return Path(raw_value) if raw_value else get_project_paths().oauth_token_path


def warn_ignored_google_ids_in_config(
    app_settings: Dict[str, Any],
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    google_payload: Any = app_settings.get("google")
    if not isinstance(google_payload, dict):
        return
    ignored_keys: tuple[str, ...] = (
        "drive_folder_id",
        "drive_preview_folder_id",
        "sheets_id",
    )
    present_keys: list[str] = [key for key in ignored_keys if key in google_payload]
    if not present_keys:
        return
    (logger or logging.getLogger(__name__)).warning(
        "google.%s in app config are ignored; use env vars GOOGLE_DRIVE_FOLDER_ID, GOOGLE_DRIVE_PREVIEW_FOLDER_ID, GOOGLE_SHEETS_ID",
        ",google.".join(present_keys),
    )


def validate_google_service_account_path_requirement(
    *,
    google_enabled: bool,
    google_auth_mode: str,
    service_account_path: Optional[Path],
    summarize_error: Optional[Callable[[Exception], str]] = None,
) -> None:
    if not google_enabled:
        return
    if google_auth_mode != "service_account":
        return
    if service_account_path is None:
        raise RuntimeError(
            "Service account JSON is required when GOOGLE_AUTH_MODE=service_account. "
            "Place the file at secrets/service_account.json or set GOOGLE_SERVICE_ACCOUNT_PATH env var."
        )
    if not service_account_path.exists():
        raise RuntimeError(
            "Service account JSON not found. "
            f"Resolved path: {str(service_account_path)!r}. "
            "Place the file at secrets/service_account.json or set GOOGLE_SERVICE_ACCOUNT_PATH env var."
        )
    if not service_account_path.is_file():
        raise RuntimeError(
            "Service account path is not a file. "
            f"Resolved path: {str(service_account_path)!r}."
        )
    try:
        with service_account_path.open("r", encoding="utf-8-sig") as file_obj:
            file_obj.read(1)
    except Exception as error:
        format_error: str = summarize_error(error) if summarize_error else str(error)
        raise RuntimeError(
            "Service account JSON is not readable. "
            f"Resolved path: {str(service_account_path)!r}. Error: {format_error}"
        ) from error


def _resolve_template_path(
    template_value: Optional[str],
    *,
    base_dir: Path,
) -> Optional[str]:
    raw_value: str = str(template_value or "").strip()
    if not raw_value:
        return None
    candidate: Path = Path(raw_value)
    if candidate.is_absolute():
        return raw_value
    return str((base_dir / candidate).resolve())


def _build_processing_config(
    app_settings: Dict[str, Any],
    *,
    logger: logging.Logger,
) -> ProcessingConfig:
    """Build processing config from YAML settings + env overrides."""
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
    return ProcessingConfig(
        mode=processing_mode,
        now_tz_mode=now_tz_mode,
    )


def _build_llm_config(app_settings: Dict[str, Any]) -> LlmConfig:
    """Build LLM config from YAML (primary) with env overrides."""
    llm_payload: Any = app_settings.get("llm")
    if not isinstance(llm_payload, dict):
        llm_payload = {}

    yaml_model: str = str(llm_payload.get("model") or "").strip() or DEFAULT_OPENAI_MODEL
    yaml_fallback_model: str = (
        str(llm_payload.get("fallback_model") or "").strip() or DEFAULT_OPENAI_MODEL
    )
    yaml_reasoning_effort: str = (
        str(llm_payload.get("reasoning_effort") or "").strip().lower()
        or DEFAULT_REASONING_EFFORT
    )
    yaml_service_tier: str = (
        str(llm_payload.get("service_tier") or "").strip().lower() or DEFAULT_SERVICE_TIER
    )
    yaml_timeout: float = float(llm_payload.get("timeout_sec", 120.0) or 120.0)
    yaml_max_output: int = int(llm_payload.get("max_output_tokens", 2000) or 2000)
    yaml_pre_delay: float = float(llm_payload.get("pre_delay_sec", 5.0) or 5.0)

    openai_model: str = os.getenv("OPENAI_MODEL", "").strip() or yaml_model
    openai_fallback_model: str = (
        os.getenv("OPENAI_FALLBACK_MODEL", "").strip() or yaml_fallback_model
    )
    openai_reasoning_effort: str = (
        os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower() or yaml_reasoning_effort
    )
    if openai_reasoning_effort not in REASONING_EFFORT_VALUES:
        raise RuntimeError(
            "Config key llm.reasoning_effort must be one of "
            f"{sorted(REASONING_EFFORT_VALUES)}: {openai_reasoning_effort!r}"
        )
    openai_service_tier: str = (
        os.getenv("OPENAI_SERVICE_TIER", "").strip().lower() or yaml_service_tier
    )
    if openai_service_tier not in SERVICE_TIER_VALUES:
        raise RuntimeError(
            "Config key llm.service_tier must be one of "
            f"{sorted(SERVICE_TIER_VALUES)}: {openai_service_tier!r}"
        )
    openai_timeout_sec: float = EnvReader.float(
        "STG_OPENAI_TIMEOUT_SEC", yaml_timeout, min_value=1.0,
    )
    openai_max_output_tokens: int = EnvReader.int(
        "STG_OPENAI_MAX_OUTPUT_TOKENS", yaml_max_output, min_value=1,
    )
    openai_pre_delay_sec: float = EnvReader.float(
        "STG_OPENAI_PRE_DELAY_SEC", yaml_pre_delay, min_value=0.0,
    )
    llm_source_desc_max_chars: int = 2000
    return LlmConfig(
        provider="openai",
        model=openai_model,
        fallback_model=openai_fallback_model,
        reasoning_effort=openai_reasoning_effort,
        service_tier=openai_service_tier,
        timeout_sec=openai_timeout_sec,
        max_output_tokens=openai_max_output_tokens,
        pre_delay_sec=openai_pre_delay_sec,
        source_desc_max_chars=llm_source_desc_max_chars,
    )


def _build_cleanup_config(app_settings: Dict[str, Any]) -> CleanupConfig:
    """Build cleanup config from YAML (primary) with env override."""
    cleanup_payload: Any = app_settings.get("cleanup")
    if not isinstance(cleanup_payload, dict):
        cleanup_payload = {}
    yaml_max_age: int = int(cleanup_payload.get("max_age_days", 3) or 3)
    cleanup_max_age_days: int = EnvReader.int(
        "STG_CLEANUP_MAX_AGE_DAYS",
        yaml_max_age,
        min_value=1,
    )
    return CleanupConfig(max_age_days=cleanup_max_age_days)


def _build_ytdlp_config(app_settings: Dict[str, Any]) -> YtDlpConfig:
    """Build YtDlp config from YAML (primary) with defaults."""
    ytdlp_payload: Any = app_settings.get("ytdlp")
    if not isinstance(ytdlp_payload, dict):
        ytdlp_payload = {}
    auto_update: bool = bool(ytdlp_payload.get("auto_update", True))
    interval_days: int = int(ytdlp_payload.get("update_check_interval_days", 7))
    cookies_warn_age_days: int = int(ytdlp_payload.get("cookies_warn_age_days", 7))
    deno_auto_update: bool = bool(ytdlp_payload.get("deno_auto_update", True))
    deno_update_interval_days: int = int(
        ytdlp_payload.get("deno_update_interval_days", 7)
    )
    return YtDlpConfig(
        auto_update=auto_update,
        update_check_interval_days=interval_days,
        cookies_warn_age_days=cookies_warn_age_days,
        deno_auto_update=deno_auto_update,
        deno_update_interval_days=deno_update_interval_days,
    )


def _build_google_config(
    app_settings: Dict[str, Any],
    *,
    logger: logging.Logger,
    summarize_error: Optional[Callable[[Exception], str]],
) -> GoogleConfig:
    """Build Google config from YAML settings + env + validation."""
    google_auth_mode: str = load_google_auth_mode(app_settings)
    if os.getenv("GOOGLE_CREDENTIALS_PATH", "").strip() or os.getenv("GOOGLE_TOKEN_PATH", "").strip():
        logger.warning(
            "GOOGLE_CREDENTIALS_PATH/GOOGLE_TOKEN_PATH are deprecated and ignored. "
            "Use GOOGLE_OAUTH_CREDENTIALS_PATH/GOOGLE_OAUTH_TOKEN_PATH."
        )

    google_enabled: bool = setting_as_bool(app_settings, "google.enabled")
    google_service_account_path: Optional[Path] = load_google_service_account_path()
    validate_google_service_account_path_requirement(
        google_enabled=google_enabled,
        google_auth_mode=google_auth_mode,
        service_account_path=google_service_account_path,
        summarize_error=summarize_error,
    )
    drive_folder_id: str = EnvReader.str_required("GOOGLE_DRIVE_FOLDER_ID")
    drive_preview_folder_id: Optional[str] = EnvReader.str_optional(
        "GOOGLE_DRIVE_PREVIEW_FOLDER_ID",
    )
    if drive_preview_folder_id:
        logger.info(
            "google_drive_preview_folder_id source=env value=%s",
            drive_preview_folder_id,
        )
    else:
        drive_preview_folder_id = drive_folder_id
        logger.info(
            "google_drive_preview_folder_id source=fallback_to_drive_folder_id value=%s",
            drive_preview_folder_id,
        )
    return GoogleConfig(
        enabled=google_enabled,
        auth_mode=google_auth_mode,
        service_account_path=google_service_account_path,
        drive_folder_id=drive_folder_id,
        drive_preview_folder_id=drive_preview_folder_id,
        drive_preview_path_template=setting_as_str(app_settings, "google.drive_preview_path_template"),
        doc_share_mode=normalize_google_doc_share_mode(setting_as_str(app_settings, "google.doc_share_mode")),
        sheets_id=EnvReader.str_required("GOOGLE_SHEETS_ID"),
        sheets_range=setting_as_str(app_settings, "google.sheets_range"),
        form_url=setting_as_str(app_settings, "google.form_url"),
        contacts=setting_as_str(app_settings, "google.contacts"),
    )


def _build_telegram_config(app_settings: Dict[str, Any]) -> TelegramConfig:
    """Build Telegram config from YAML settings + env secrets."""
    return TelegramConfig(
        bot_token=EnvReader.str_required("TELEGRAM_BOT_TOKEN"),
        chat_id=EnvReader.str_required("TELEGRAM_CHAT_ID"),
        enabled=setting_as_bool(app_settings, "telegram.enabled"),
        use_audit=setting_as_bool(app_settings, "telegram.use_audit"),
        symbol_separator=setting_as_str(app_settings, "telegram.symbol_separator"),
        separator_repeat_count=setting_as_int(app_settings, "telegram.separator_repeat_count"),
        symbol_separator_start=setting_as_str(app_settings, "telegram.symbol_separator_start"),
        separator_start_repeat_count=setting_as_int(app_settings, "telegram.separator_start_repeat_count"),
        symbol_broadcast=setting_as_str(app_settings, "telegram.symbol_broadcast"),
        symbol_alert=setting_as_str(app_settings, "telegram.symbol_alert"),
        symbol_form=setting_as_str(app_settings, "telegram.symbol_form"),
        symbol_description=setting_as_str(app_settings, "telegram.symbol_description"),
        symbol_pin=setting_as_str(app_settings, "telegram.symbol_pin"),
        symbol_done=setting_as_str(app_settings, "telegram.symbol_done"),
        flag_repeat_count=setting_as_int(app_settings, "telegram.flag_repeat_count"),
        send_delay_seconds=EnvReader.float(
            "TELEGRAM_SEND_DELAY_SECONDS",
            default=setting_as_float(app_settings, "telegram.send_delay_seconds"),
            min_value=0.0,
        ),
        max_retries=EnvReader.int(
            "TELEGRAM_MAX_RETRIES",
            default=setting_as_int(app_settings, "telegram.max_retries"),
            min_value=0,
        ),
    )


def _build_paths_config(
    app_settings: Dict[str, Any],
    *,
    entrypoint_dir: Path,
    templates_path: Path,
) -> PathsConfig:
    """Build paths config from YAML settings."""
    return PathsConfig(
        local_image_dir_template=str(
            _resolve_template_path(
                setting_as_str(app_settings, "paths.local_image_dir_template"),
                base_dir=entrypoint_dir,
            )
            or ""
        ),
        local_doc_dir_template=_resolve_template_path(
            setting_as_optional_str(app_settings, "paths.local_doc_dir_template"),
            base_dir=entrypoint_dir,
        ),
        templates_path=templates_path,
        preview_filename_max_stem=setting_as_int(app_settings, "files.preview_filename_max_stem"),
    )


def _build_timezone_config(app_settings: Dict[str, Any]) -> TimezoneConfig:
    """Build timezone config from YAML settings."""
    return TimezoneConfig(
        kiev=setting_as_str(app_settings, "timezones.kiev"),
        cet=setting_as_str(app_settings, "timezones.cet"),
    )


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
    warn_ignored_google_ids_in_config(app_settings, logger=logger)

    templates = load_templates_from_path(project_paths.templates_path)
    entrypoint_dir: Path = project_paths.entrypoint_path.parent
    return AppConfig(
        processing=_build_processing_config(app_settings, logger=logger),
        cleanup=_build_cleanup_config(app_settings),
        ytdlp=_build_ytdlp_config(app_settings),
        llm=_build_llm_config(app_settings),
        google=_build_google_config(
            app_settings,
            logger=logger,
            summarize_error=summarize_error,
        ),
        telegram=_build_telegram_config(app_settings),
        paths=_build_paths_config(
            app_settings,
            entrypoint_dir=entrypoint_dir,
            templates_path=project_paths.templates_path,
        ),
        timezones=_build_timezone_config(app_settings),
        templates=templates,
    )
