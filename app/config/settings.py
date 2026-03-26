from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class AppTemplates:
    google_doc_header: str
    google_doc_table_labels_json: str
    google_doc_table_labels: dict[str, list[str]]
    google_doc_bold_line_prefixes_json: str
    google_doc_bold_line_prefixes: list[str]
    google_doc_language_headings_json: str
    google_doc_language_headings: dict[str, str]
    telegram_header: str
    telegram_language_block: str
    telegram_language_merged_block: str
    telegram_key_form_reminder: str
    telegram_sparkle_separator: str
    telegram_post_header: str
    telegram_language_digest_header: str
    llm_merge_title_description_prompt: str
    llm_startup_ping_prompt: str
    llm_language_names_json: str
    llm_language_names: dict[str, str]
    llm_merge_structural_rules: str
    llm_merge_contracts_json: str
    llm_merge_retry_reinforcements_json: str
    llm_merge_contracts: dict[str, Any]
    llm_merge_retry_reinforcements: dict[str, Any]
    files_preview_name_template: str
    files_doc_title_template: str
    files_language_codes_json: str
    files_language_codes: dict[str, str]
    common_no_description_text: str
    common_single_mode_message: str

# Compatibility dataclass: keep current runtime behavior in Phase 1.
@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_enabled: bool
    google_enabled: bool
    google_service_account_path: Optional[Path]
    google_drive_folder_id: Optional[str]
    google_drive_preview_folder_id: Optional[str]
    google_drive_preview_path_template: str
    google_doc_share_mode: str
    google_sheets_id: str
    google_sheets_range: str
    google_form_url: str
    google_contacts: str
    local_image_dir_template: str
    local_doc_dir_template: Optional[str]
    timezone_kiev: str
    timezone_cet: str
    telegram_symbol_separator: str
    telegram_separator_repeat_count: int
    telegram_symbol_broadcast: str
    telegram_symbol_alert: str
    telegram_symbol_form: str
    telegram_symbol_description: str
    telegram_symbol_pin: str
    telegram_symbol_done: str
    telegram_flag_uk: str
    telegram_flag_en: str
    telegram_flag_ru: str
    telegram_flag_other: str
    telegram_flag_repeat_count: int
    telegram_language_name_uk: str
    telegram_language_name_en: str
    telegram_language_name_ru: str
    telegram_language_name_other: str
    processing_mode: str
    now_tz_mode: str
    llm_provider: str
    llm_model: str
    openai_model: str
    openai_timeout_sec: float
    openai_max_output_tokens: int
    openai_pre_delay_sec: float
    llm_source_desc_max_chars: int
    llm_run_if_single_source: bool
    preview_filename_max_stem: int
    stg_templates_path: Path
    telegram_use_audit: bool
    templates: AppTemplates
