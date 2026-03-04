from __future__ import annotations

import logging
import os
from datetime import timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo


def must_get_setting_value(payload: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"Config key is missing: {dotted_key}")
        current = current[key]
    return current


def setting_as_bool(payload: Dict[str, Any], dotted_key: str) -> bool:
    value: Any = must_get_setting_value(payload, dotted_key)
    if isinstance(value, bool):
        return value
    raw: str = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Config key must be bool: {dotted_key}={value!r}")


def setting_as_int(payload: Dict[str, Any], dotted_key: str) -> int:
    value: Any = must_get_setting_value(payload, dotted_key)
    try:
        return int(value)
    except Exception as error:
        raise RuntimeError(f"Config key must be int: {dotted_key}={value!r}") from error


def setting_as_float(payload: Dict[str, Any], dotted_key: str) -> float:
    value: Any = must_get_setting_value(payload, dotted_key)
    try:
        return float(value)
    except Exception as error:
        raise RuntimeError(f"Config key must be float: {dotted_key}={value!r}") from error


def setting_as_str(payload: Dict[str, Any], dotted_key: str) -> str:
    value: Any = must_get_setting_value(payload, dotted_key)
    return str(value).strip()


def setting_as_optional_str(payload: Dict[str, Any], dotted_key: str) -> Optional[str]:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    value: str = str(current).strip()
    return value or None


def resolve_llm_provider_from_env(*, logger: Optional[logging.Logger] = None) -> str:
    raw_value: str = os.getenv("STG_LLM_PROVIDER", "openai").strip().lower()
    if raw_value in {"gpt", "openai"}:
        return "openai"
    raise RuntimeError(
        "Only OpenAI is supported in this application. "
        f"Unsupported STG_LLM_PROVIDER={raw_value!r}."
    )


def normalize_processing_mode(value: str, *, source: str) -> str:
    raw_value: str = str(value or "").strip().lower()
    normalized_input: str = raw_value[2:] if raw_value.startswith("--") else raw_value
    if normalized_input != "audit":
        raise RuntimeError(
            f"Unsupported {source}={value!r}. Supported mode: audit."
        )
    return "audit"


def normalize_audit_mode(value: str, *, source: str) -> str:
    raw_value: str = str(value or "").strip().lower()
    normalized_input: str = raw_value[2:] if raw_value.startswith("--") else raw_value
    alias_to_mode: Dict[str, str] = {
        "nomerge": "nomerge",
        "no-merge": "nomerge",
        "no_merge": "nomerge",
        "merge": "merge",
        "unite": "unite",
    }
    normalized_mode: Optional[str] = alias_to_mode.get(normalized_input)
    if normalized_mode is None:
        raise RuntimeError(
            f"Unsupported {source}={value!r}. Supported audit modes: nomerge, merge, unite."
        )
    return normalized_mode


def normalize_now_tz_mode(value: str, *, source: str) -> str:
    raw_value: str = str(value or "").strip().lower()
    alias_to_mode: Dict[str, str] = {
        "kyiv": "kyiv",
        "kiev": "kyiv",
        "fixed_gmt_plus_2": "fixed_gmt_plus_2",
        "fixedgmtplus2": "fixed_gmt_plus_2",
        "gmt+2": "fixed_gmt_plus_2",
        "utc+2": "fixed_gmt_plus_2",
    }
    normalized_mode: Optional[str] = alias_to_mode.get(raw_value)
    if normalized_mode is None:
        raise RuntimeError(
            f"Unsupported {source}={value!r}. Supported: kyiv, fixed_gmt_plus_2."
        )
    return normalized_mode


def now_filter_timezone(now_tz_mode: str, kiev_tz: ZoneInfo) -> timezone | ZoneInfo:
    if now_tz_mode == "fixed_gmt_plus_2":
        return timezone(timedelta(hours=2))
    return kiev_tz


def validate_app_settings(payload: Dict[str, Any]) -> None:
    checks: List[Tuple[str, str]] = [
        ("processing.mode", "str"),
        ("google.enabled", "bool"),
        ("google.drive_folder_id", "str"),
        ("google.drive_preview_folder_id", "str"),
        ("google.drive_preview_path_template", "str"),
        ("google.doc_share_mode", "str"),
        ("google.sheets_id", "str"),
        ("google.sheets_range", "str"),
        ("google.form_url", "str"),
        ("google.contacts", "str"),
        ("paths.local_image_dir_template", "str"),
        ("timezones.kiev", "str"),
        ("timezones.cet", "str"),
        ("telegram.enabled", "bool"),
        ("telegram.use_audit", "bool"),
        ("telegram.symbol_separator", "str"),
        ("telegram.separator_repeat_count", "int"),
        ("telegram.symbol_broadcast", "str"),
        ("telegram.symbol_alert", "str"),
        ("telegram.symbol_form", "str"),
        ("telegram.symbol_description", "str"),
        ("telegram.symbol_pin", "str"),
        ("telegram.symbol_done", "str"),
        ("telegram.flag_uk", "str"),
        ("telegram.flag_en", "str"),
        ("telegram.flag_ru", "str"),
        ("telegram.flag_other", "str"),
        ("telegram.flag_repeat_count", "int"),
        ("telegram.language_name_uk", "str"),
        ("telegram.language_name_en", "str"),
        ("telegram.language_name_ru", "str"),
        ("telegram.language_name_other", "str"),
        ("files.preview_filename_max_stem", "int"),
    ]
    errors: List[str] = []
    for key, key_type in checks:
        try:
            if key_type == "bool":
                setting_as_bool(payload, key)
            elif key_type == "int":
                setting_as_int(payload, key)
            elif key_type == "float":
                setting_as_float(payload, key)
            else:
                value: str = setting_as_str(payload, key)
                if not value:
                    errors.append(f"{key}: must not be empty")
        except Exception as error:
            errors.append(f"{key}: {error}")
    if errors:
        joined: str = "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(
            "Invalid app config. Please fix these keys:\n"
            f"{joined}\n"
            "You can start from app/config/runtime/app_config.example.yaml."
        )


def normalize_google_doc_share_mode(mode_raw_input: str) -> str:
    mode_raw: str = str(mode_raw_input or "").strip().lower()
    alias_to_mode: Dict[str, str] = {
        "private": "private",
        "off": "private",
        "none": "private",
        "anyone_reader": "anyone_reader",
        "reader": "anyone_reader",
        "read": "anyone_reader",
        "view": "anyone_reader",
        "anyone_commenter": "anyone_commenter",
        "commenter": "anyone_commenter",
        "comment": "anyone_commenter",
        "anyone_writer": "anyone_writer",
        "writer": "anyone_writer",
        "edit": "anyone_writer",
        "editable": "anyone_writer",
    }
    normalized_mode: Optional[str] = alias_to_mode.get(mode_raw)
    if normalized_mode is None:
        supported_modes: str = ", ".join(
            ["private", "anyone_reader", "anyone_commenter", "anyone_writer"]
        )
        raise RuntimeError(
            f"Unsupported google.doc_share_mode={mode_raw!r}. Supported: {supported_modes}"
        )
    return normalized_mode


def describe_google_doc_share_mode(share_mode: str) -> str:
    descriptions: Dict[str, str] = {
        "private": "private: only explicitly granted users can access the document",
        "anyone_reader": "anyone_reader: anyone with the link can view",
        "anyone_commenter": "anyone_commenter: anyone with the link can comment",
        "anyone_writer": "anyone_writer: anyone with the link can edit",
    }
    if share_mode not in descriptions:
        raise RuntimeError(f"Unsupported google_doc_share_mode={share_mode!r}.")
    return descriptions[share_mode]


def must_get_env(name: str) -> str:
    value: Optional[str] = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Env var {name} is required.")
    return value.strip()
