from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class AppTemplates:
    google_doc_header: str
    google_doc_table_labels_json: str
    google_doc_bold_line_prefixes_json: str
    google_doc_language_headings_json: str
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
    llm_merge_structural_rules: str
    llm_merge_contracts_json: str
    llm_merge_retry_reinforcements_json: str
    llm_merge_contracts: dict[str, Any]
    llm_merge_retry_reinforcements: dict[str, Any]
    files_preview_name_template: str
    files_doc_title_template: str
    files_language_codes_json: str
    common_no_description_text: str
    common_single_mode_message: str


@dataclass(frozen=True)
class OpenAIOrgUsageConfig:
    admin_api_key: str
    project_id: str
    models: Tuple[str, ...]
    monthly_budget_usd: Optional[float]
    timeout_sec: float
    verbose_log: bool


@dataclass(frozen=True)
class GoogleSettings:
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
class TelegramSettings:
    bot_token: str
    chat_id: str
    enabled: bool
    use_audit: bool
    symbol_separator: str
    separator_repeat_count: int
    symbol_broadcast: str
    symbol_alert: str
    symbol_form: str
    symbol_description: str
    symbol_pin: str
    symbol_done: str
    flag_uk: str
    flag_en: str
    flag_ru: str
    flag_other: str
    flag_repeat_count: int
    language_name_uk: str
    language_name_en: str
    language_name_ru: str
    language_name_other: str


@dataclass(frozen=True)
class OpenAISettings:
    model: str
    timeout_sec: float
    max_output_tokens: int
    pre_delay_sec: float


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    model: str
    source_desc_max_chars: int
    run_if_single_source: bool


@dataclass(frozen=True)
class PathSettings:
    local_image_dir_template: str
    local_doc_dir_template: Optional[str]
    templates_path: Path


@dataclass(frozen=True)
class TimezoneSettings:
    kiev: str
    cet: str
    now_tz_mode: str


@dataclass(frozen=True)
class FileNamingSettings:
    preview_filename_max_stem: int


@dataclass(frozen=True)
class AppSettings:
    google: GoogleSettings
    telegram: TelegramSettings
    openai: OpenAISettings
    llm: LlmSettings
    paths: PathSettings
    timezones: TimezoneSettings
    files: FileNamingSettings
    processing_mode: str
    templates: AppTemplates


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


def to_app_settings(config: AppConfig) -> AppSettings:
    return AppSettings(
        google=GoogleSettings(
            enabled=config.google_enabled,
            service_account_path=config.google_service_account_path,
            drive_folder_id=config.google_drive_folder_id,
            drive_preview_folder_id=config.google_drive_preview_folder_id,
            drive_preview_path_template=config.google_drive_preview_path_template,
            doc_share_mode=config.google_doc_share_mode,
            sheets_id=config.google_sheets_id,
            sheets_range=config.google_sheets_range,
            form_url=config.google_form_url,
            contacts=config.google_contacts,
        ),
        telegram=TelegramSettings(
            bot_token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
            enabled=config.telegram_enabled,
            use_audit=config.telegram_use_audit,
            symbol_separator=config.telegram_symbol_separator,
            separator_repeat_count=config.telegram_separator_repeat_count,
            symbol_broadcast=config.telegram_symbol_broadcast,
            symbol_alert=config.telegram_symbol_alert,
            symbol_form=config.telegram_symbol_form,
            symbol_description=config.telegram_symbol_description,
            symbol_pin=config.telegram_symbol_pin,
            symbol_done=config.telegram_symbol_done,
            flag_uk=config.telegram_flag_uk,
            flag_en=config.telegram_flag_en,
            flag_ru=config.telegram_flag_ru,
            flag_other=config.telegram_flag_other,
            flag_repeat_count=config.telegram_flag_repeat_count,
            language_name_uk=config.telegram_language_name_uk,
            language_name_en=config.telegram_language_name_en,
            language_name_ru=config.telegram_language_name_ru,
            language_name_other=config.telegram_language_name_other,
        ),
        openai=OpenAISettings(
            model=config.openai_model,
            timeout_sec=config.openai_timeout_sec,
            max_output_tokens=config.openai_max_output_tokens,
            pre_delay_sec=config.openai_pre_delay_sec,
        ),
        llm=LlmSettings(
            provider=config.llm_provider,
            model=config.llm_model,
            source_desc_max_chars=config.llm_source_desc_max_chars,
            run_if_single_source=config.llm_run_if_single_source,
        ),
        paths=PathSettings(
            local_image_dir_template=config.local_image_dir_template,
            local_doc_dir_template=config.local_doc_dir_template,
            templates_path=config.stg_templates_path,
        ),
        timezones=TimezoneSettings(
            kiev=config.timezone_kiev,
            cet=config.timezone_cet,
            now_tz_mode=config.now_tz_mode,
        ),
        files=FileNamingSettings(
            preview_filename_max_stem=config.preview_filename_max_stem,
        ),
        processing_mode=config.processing_mode,
        templates=config.templates,
    )
