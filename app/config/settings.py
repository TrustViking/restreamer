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
    telegram_header: str
    telegram_language_block: str
    telegram_language_merged_block: str
    telegram_key_form_reminder: str
    telegram_sparkle_separator: str
    telegram_post_header: str
    telegram_language_digest_header: str
    llm_merge_title_description_prompt: str
    llm_startup_ping_prompt: str
    llm_merge_structural_rules: str
    llm_merge_contracts_json: str
    llm_merge_retry_reinforcements_json: str
    llm_merge_contracts: dict[str, Any]
    llm_merge_retry_reinforcements: dict[str, Any]
    files_preview_name_template: str
    files_doc_title_template: str
    common_no_description_text: str
    common_single_mode_message: str


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str
    enabled: bool
    use_audit: bool
    symbol_separator: str
    separator_repeat_count: int
    symbol_separator_start: str
    separator_start_repeat_count: int
    symbol_broadcast: str
    symbol_alert: str
    symbol_form: str
    symbol_description: str
    symbol_pin: str
    symbol_done: str
    flag_repeat_count: int
    send_delay_seconds: float
    max_retries: int


@dataclass(frozen=True)
class GoogleConfig:
    enabled: bool
    service_account_path: Optional[Path]
    drive_folder_id: Optional[str]
    drive_preview_folder_id: Optional[str]
    drive_preview_path_template: str
    doc_share_mode: str
    sheets_id: str
    sheets_range: str
    form_url: str
    contacts: str


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    timeout_sec: float
    max_output_tokens: int
    pre_delay_sec: float
    source_desc_max_chars: int


@dataclass(frozen=True)
class ProcessingConfig:
    mode: str
    now_tz_mode: str


@dataclass(frozen=True)
class CleanupConfig:
    max_age_days: int


@dataclass(frozen=True)
class PathsConfig:
    local_image_dir_template: str
    local_doc_dir_template: Optional[str]
    templates_path: Path
    preview_filename_max_stem: int


@dataclass(frozen=True)
class TimezoneConfig:
    kiev: str
    cet: str


@dataclass(frozen=True)
class AppConfig:
    telegram: TelegramConfig
    google: GoogleConfig
    llm: LlmConfig
    processing: ProcessingConfig
    cleanup: CleanupConfig
    paths: PathsConfig
    timezones: TimezoneConfig
    templates: AppTemplates
