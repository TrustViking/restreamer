from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.paths import ProjectPaths


@dataclass(frozen=True)
class StartupContext:
    run_id: str
    argv_list: list[str]
    args_audit_mode: str
    args_debug: bool
    args_dry_run: bool
    processing_mode: str
    config_processing_mode_raw: str
    paths: ProjectPaths

    @property
    def project_root(self) -> Path:
        return self.paths.project_root

    @property
    def entrypoint_path(self) -> Path:
        return self.paths.entrypoint_path

    @property
    def runtime_config_path(self) -> Path:
        return self.paths.runtime_config_path

    @property
    def templates_path(self) -> Path:
        return self.paths.templates_path

    @property
    def secrets_env_path(self) -> Path:
        return self.paths.secrets_env_path

    @property
    def oauth_credentials_path(self) -> Path:
        return self.paths.oauth_credentials_path

    @property
    def oauth_token_path(self) -> Path:
        return self.paths.oauth_token_path


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
    llm_model: str
    llm_usage_reporting_mode: str
    sheet_id: str
    sheet_range: str
    sheets_link_writeback: bool
    local_doc_export_enabled: bool
    strip_chapter_timestamps: bool
    llm_model_configured: str = ""
    llm_provider_model: str = ""
