from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from app.config.settings import AppTemplates


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
    google_doc_language_headings_raw: Any = must_get_template_object(
        payload, "google_doc.language_headings"
    )
    llm_language_names_raw: Any = must_get_template_object(payload, "llm.language_names")
    files_language_codes_raw: Any = must_get_template_object(
        payload, "files.language_codes"
    )

    try:
        google_doc_table_labels_json: str = json.dumps(
            google_doc_table_labels_raw, ensure_ascii=False
        )
        google_doc_bold_line_prefixes_json: str = json.dumps(
            google_doc_bold_line_prefixes_raw, ensure_ascii=False
        )
        google_doc_language_headings_json: str = json.dumps(
            google_doc_language_headings_raw, ensure_ascii=False
        )
        llm_language_names_json: str = json.dumps(llm_language_names_raw, ensure_ascii=False)
        files_language_codes_json: str = json.dumps(
            files_language_codes_raw, ensure_ascii=False
        )
    except Exception as error:
        raise RuntimeError(f"Templates JSON conversion failed: {error}") from error

    return AppTemplates(
        google_doc_header=must_get_template_value(payload, "google_doc.header"),
        google_doc_table_labels_json=google_doc_table_labels_json,
        google_doc_bold_line_prefixes_json=google_doc_bold_line_prefixes_json,
        google_doc_language_headings_json=google_doc_language_headings_json,
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
        llm_language_names_json=llm_language_names_json,
        files_preview_name_template=must_get_template_value(
            payload, "files.preview_name_template"
        ),
        files_doc_title_template=must_get_template_value(payload, "files.doc_title_template"),
        files_language_codes_json=files_language_codes_json,
        common_no_description_text=must_get_template_value(
            payload, "common.no_description_text"
        ),
        common_single_mode_message=must_get_template_value(
            payload, "common.single_mode_message"
        ),
    )
