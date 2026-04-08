from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppTemplates

LOGGER = _get_logger_impl(__name__)


def must_get_template_value(payload: Dict[str, Any], dotted_key: str) -> str:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"Template key is missing: {dotted_key}")
        current = current[key]
    value: str = str(current or "").strip()
    if not value:
        raise RuntimeError(f"Template key is empty: {dotted_key}")
    return value


def must_get_template_object(payload: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"Template key is missing: {dotted_key}")
        current = current[key]
    return current


def _get_optional_template_value(
    payload: Dict[str, Any],
    dotted_key: str,
    *,
    default_value: str = "",
) -> str:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            LOGGER.warning(
                "Optional template key is missing: %s. Using default value.",
                dotted_key,
            )
            return default_value
        current = current[key]
    value: str = str(current or "").strip()
    if not value:
        LOGGER.warning(
            "Optional template key is empty: %s. Using default value.",
            dotted_key,
        )
        return default_value
    return value


def _get_optional_template_object(
    payload: Dict[str, Any],
    dotted_key: str,
    *,
    default_value: Any,
) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            LOGGER.warning(
                "Optional template key is missing: %s. Using default value.",
                dotted_key,
            )
            return default_value
        current = current[key]
    return current


def _normalize_string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized_mapping: dict[str, str] = {}
    for key, item in value.items():
        normalized_mapping[str(key)] = str(item)
    return normalized_mapping


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _normalize_table_labels(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    normalized_mapping: dict[str, list[str]] = {}
    for key, item in value.items():
        if not isinstance(item, list):
            continue
        normalized_mapping[str(key)] = [str(part) for part in item]
    return normalized_mapping


def load_templates_from_path(path: Path) -> AppTemplates:
    if not path.exists():
        raise RuntimeError(f"Templates file does not exist: {path}")
    raw_text: str = path.read_text(encoding="utf-8")
    try:
        payload: Any = yaml.safe_load(raw_text)
    except Exception as error:
        raise RuntimeError(f"Failed to parse templates file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Templates root must be a mapping in {path}")

    google_doc_table_labels_raw: Any = must_get_template_object(
        payload, "google_doc.table_labels"
    )
    google_doc_bold_line_prefixes_raw: Any = must_get_template_object(
        payload, "google_doc.bold_line_prefixes"
    )
    llm_merge_contracts_raw: Any = _get_optional_template_object(
        payload,
        "llm.merge_contracts",
        default_value={},
    )
    llm_merge_retry_reinforcements_raw: Any = _get_optional_template_object(
        payload,
        "llm.merge_retry_reinforcements",
        default_value={},
    )

    google_doc_table_labels: dict[str, list[str]] = _normalize_table_labels(
        google_doc_table_labels_raw
    )
    google_doc_bold_line_prefixes: list[str] = _normalize_string_list(
        google_doc_bold_line_prefixes_raw
    )

    if not isinstance(llm_merge_contracts_raw, dict):
        LOGGER.warning(
            "Optional template key llm.merge_contracts has invalid type %s. Using empty mapping.",
            type(llm_merge_contracts_raw).__name__,
        )
        llm_merge_contracts_raw = {}
    if not isinstance(llm_merge_retry_reinforcements_raw, dict):
        LOGGER.warning(
            "Optional template key llm.merge_retry_reinforcements has invalid type %s. Using empty mapping.",
            type(llm_merge_retry_reinforcements_raw).__name__,
        )
        llm_merge_retry_reinforcements_raw = {}

    try:
        google_doc_table_labels_json: str = json.dumps(
            google_doc_table_labels_raw, ensure_ascii=False
        )
        google_doc_bold_line_prefixes_json: str = json.dumps(
            google_doc_bold_line_prefixes_raw, ensure_ascii=False
        )
        llm_merge_contracts_json: str = json.dumps(
            llm_merge_contracts_raw,
            ensure_ascii=False,
        )
        llm_merge_retry_reinforcements_json: str = json.dumps(
            llm_merge_retry_reinforcements_raw,
            ensure_ascii=False,
        )
    except Exception as error:
        raise RuntimeError(f"Templates JSON conversion failed: {error}") from error

    return AppTemplates(
        google_doc_header=must_get_template_value(payload, "google_doc.header"),
        google_doc_table_labels_json=google_doc_table_labels_json,
        google_doc_table_labels=google_doc_table_labels,
        google_doc_bold_line_prefixes_json=google_doc_bold_line_prefixes_json,
        google_doc_bold_line_prefixes=google_doc_bold_line_prefixes,
        telegram_header=must_get_template_value(payload, "telegram.header"),
        telegram_language_block=must_get_template_value(payload, "telegram.language_block"),
        telegram_language_merged_block=must_get_template_value(
            payload, "telegram.language_merged_block"
        ),
        telegram_key_form_reminder=must_get_template_value(
            payload, "telegram.key_form_reminder"
        ),
        telegram_sparkle_separator=must_get_template_value(
            payload, "telegram.sparkle_separator"
        ),
        telegram_post_header=must_get_template_value(payload, "telegram.post_header"),
        telegram_language_digest_header=must_get_template_value(
            payload, "telegram.language_digest_header"
        ),
        llm_merge_title_description_prompt=must_get_template_value(
            payload, "llm.merge_title_description_prompt"
        ),
        llm_startup_ping_prompt=must_get_template_value(payload, "llm.startup_ping_prompt"),
        llm_merge_structural_rules=_get_optional_template_value(
            payload,
            "llm.merge_structural_rules",
            default_value="",
        ),
        llm_merge_contracts_json=llm_merge_contracts_json,
        llm_merge_retry_reinforcements_json=llm_merge_retry_reinforcements_json,
        llm_merge_contracts=llm_merge_contracts_raw,
        llm_merge_retry_reinforcements=llm_merge_retry_reinforcements_raw,
        files_preview_name_template=must_get_template_value(
            payload, "files.preview_name_template"
        ),
        files_doc_title_template=must_get_template_value(payload, "files.doc_title_template"),
        common_no_description_text=must_get_template_value(
            payload, "common.no_description_text"
        ),
        common_single_mode_message=must_get_template_value(
            payload, "common.single_mode_message"
        ),
    )
