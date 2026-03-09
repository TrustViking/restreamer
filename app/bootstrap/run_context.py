from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.config.settings import AppConfig
from app.core.branching import audit_branch_labels

if TYPE_CHECKING:
    from app.observability.startup_summary import LlmSummarySnapshot


@dataclass(frozen=True)
class StartupContext:
    run_id: str
    argv_list: list[str]
    args_audit_mode: str
    args_debug: bool
    args_dry_run: bool
    processing_mode: str
    config_processing_mode_raw: str
    project_root: Path
    entrypoint_path: Path
    runtime_config_path: Path
    templates_path: Path
    secrets_env_path: Path
    oauth_credentials_path: Path
    oauth_token_path: Path


@dataclass(frozen=True)
class RunContext:
    run_id: str
    processing_mode: str
    audit_mode: str
    config_processing_mode: str
    audit_branches: list[str]
    debug_enabled: bool
    dry_run: bool
    google_enabled: bool
    telegram_enabled: bool
    llm_provider: str
    llm_primary_provider: str
    llm_fallback_provider: str
    llm_effective_primary_model: str
    llm_effective_fallback_model: str
    llm_merge_stage_model: str
    llm_packaging_stage_model: str
    llm_base_url: str
    llm_usage_reporting_mode: str
    sheet_id: str
    sheet_range: str
    sheets_link_writeback: bool
    local_doc_export_enabled: bool
    strip_chapter_timestamps: bool


def build_startup_context(
    *,
    run_id: str,
    argv_list: list[str],
    args_audit_mode: str,
    args_debug: bool,
    args_dry_run: bool,
    processing_mode: str,
    config_processing_mode_raw: str,
    project_root: Path,
    entrypoint_path: Path,
    runtime_config_path: Path,
    templates_path: Path,
    secrets_env_path: Path,
    oauth_credentials_path: Path,
    oauth_token_path: Path,
) -> StartupContext:
    return StartupContext(
        run_id=run_id,
        argv_list=argv_list,
        args_audit_mode=args_audit_mode,
        args_debug=args_debug,
        args_dry_run=args_dry_run,
        processing_mode=processing_mode,
        config_processing_mode_raw=config_processing_mode_raw,
        project_root=project_root,
        entrypoint_path=entrypoint_path,
        runtime_config_path=runtime_config_path,
        templates_path=templates_path,
        secrets_env_path=secrets_env_path,
        oauth_credentials_path=oauth_credentials_path,
        oauth_token_path=oauth_token_path,
    )


def build_run_context(
    *,
    run_id: str,
    processing_mode: str,
    audit_mode: str,
    config: AppConfig,
    llm_summary: LlmSummarySnapshot,
    debug_enabled: bool,
    dry_run: bool,
    sheets_link_writeback: bool,
    strip_chapter_timestamps: bool,
) -> RunContext:
    audit_branches: list[str] = audit_branch_labels(audit_mode=audit_mode)
    return RunContext(
        run_id=run_id,
        processing_mode=processing_mode,
        audit_mode=audit_mode,
        config_processing_mode=str(config.processing_mode or "").strip(),
        audit_branches=audit_branches,
        debug_enabled=debug_enabled,
        dry_run=dry_run,
        google_enabled=config.google_enabled,
        telegram_enabled=config.telegram_enabled,
        llm_provider=config.llm_provider,
        llm_primary_provider=llm_summary.primary_provider,
        llm_fallback_provider=llm_summary.fallback_provider,
        llm_effective_primary_model=llm_summary.effective_primary_model,
        llm_effective_fallback_model=llm_summary.effective_fallback_model,
        llm_merge_stage_model=llm_summary.effective_primary_model,
        llm_packaging_stage_model=llm_summary.effective_fallback_model,
        llm_base_url=llm_summary.base_url,
        llm_usage_reporting_mode=llm_summary.usage_reporting_mode,
        sheet_id=config.google_sheets_id,
        sheet_range=config.google_sheets_range,
        sheets_link_writeback=sheets_link_writeback,
        local_doc_export_enabled=bool(str(config.local_doc_dir_template or "").strip()),
        strip_chapter_timestamps=strip_chapter_timestamps,
    )
