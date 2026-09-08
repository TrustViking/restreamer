from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config.settings import AppConfig
from app.bootstrap.run_context import RunContext, StartupContext
from app.core.branching import audit_branch_labels
from app.llm.models.model_identity import build_effective_llm_model_identity


@dataclass(frozen=True)
class LlmSummarySnapshot:
    provider: str
    configured_model: str
    provider_model: str
    effective_model: str
    usage_reporting_mode: str

    @property
    def model(self) -> str:
        return self.effective_model


def _summary_effective_model(llm_summary: LlmSummarySnapshot) -> str:
    return str(
        getattr(llm_summary, "effective_model", getattr(llm_summary, "model", "")) or ""
    ).strip()


def _summary_configured_model(llm_summary: LlmSummarySnapshot) -> str:
    effective_model: str = _summary_effective_model(llm_summary)
    return str(getattr(llm_summary, "configured_model", effective_model) or "").strip()


def _summary_provider_model(llm_summary: LlmSummarySnapshot) -> str:
    effective_model: str = _summary_effective_model(llm_summary)
    return str(getattr(llm_summary, "provider_model", effective_model) or "").strip()


def build_llm_summary_snapshot(config: AppConfig) -> LlmSummarySnapshot:
    model_identity = build_effective_llm_model_identity(config)
    return LlmSummarySnapshot(
        provider=model_identity.provider,
        configured_model=model_identity.configured_model,
        provider_model=model_identity.provider_model,
        effective_model=model_identity.effective_model,
        usage_reporting_mode="openai_run_local",
    )


def _safe_console_text(text: str) -> str:
    stdout_encoding: str = getattr(sys.stdout, "encoding", None) or "utf-8"
    normalized: str = str(text)
    try:
        normalized.encode(stdout_encoding, errors="strict")
        return normalized
    except Exception:
        try:
            return normalized.encode(stdout_encoding, errors="backslashreplace").decode(
                stdout_encoding,
                errors="strict",
            )
        except Exception:
            return normalized.encode("utf-8", errors="backslashreplace").decode(
                "utf-8",
                errors="strict",
            )


def _safe_console_print(text: str) -> None:
    print(_safe_console_text(text))


def _log_startup_banner(
    *,
    logger: logging.Logger,
    startup_context: StartupContext,
    llm_summary: Optional[LlmSummarySnapshot],
) -> None:
    config_processing_mode_for_banner: str = (
        startup_context.config_processing_mode_raw
        if startup_context.config_processing_mode_raw
        else "missing"
    )
    audit_branches: str = ",".join(
        audit_branch_labels(audit_mode=startup_context.args_audit_mode)
    )
    logger.info("=== STARTUP BANNER BEGIN ===")
    logger.info("run_id=%s", startup_context.run_id)
    logger.info("argv=%s", repr(list(sys.argv)))
    logger.info("cwd=%s", os.getcwd())
    logger.info("python=%s", sys.executable)
    logger.info(
        "args.audit_mode=%s args.debug=%s args.dry_run=%s",
        startup_context.args_audit_mode,
        startup_context.args_debug,
        startup_context.args_dry_run,
    )
    logger.info("config_processing_mode=%s", config_processing_mode_for_banner)
    logger.info("resolved_processing_mode=%s", startup_context.processing_mode)
    logger.info(
        "resolved_audit_mode=%s branches=%s",
        startup_context.args_audit_mode,
        audit_branches,
    )
    logger.info("project_root=%s", startup_context.project_root)
    logger.info(
        "entrypoint_path=%s",
        Path(sys.argv[0]).resolve() if sys.argv else startup_context.entrypoint_path,
    )
    logger.info("runtime_config_path=%s", startup_context.runtime_config_path)
    logger.info("templates_path=%s", startup_context.templates_path)
    logger.info("secrets_env_path=%s", startup_context.secrets_env_path)
    logger.info("oauth_credentials_path=%s", startup_context.oauth_credentials_path)
    logger.info("oauth_token_path=%s", startup_context.oauth_token_path)
    if llm_summary is not None:
        logger.info(
            "llm_provider=%s llm_model_effective=%s llm_model_configured=%s llm_provider_model=%s llm_usage_reporting_mode=%s",
            llm_summary.provider or "unknown",
            _summary_effective_model(llm_summary) or "unknown",
            _summary_configured_model(llm_summary) or "unknown",
            _summary_provider_model(llm_summary) or "unknown",
            llm_summary.usage_reporting_mode or "unknown",
        )
    logger.info("=== STARTUP BANNER END ===")


def _log_startup_dump(
    *,
    logger: logging.Logger,
    startup_context: StartupContext,
    llm_summary: Optional[LlmSummarySnapshot],
) -> None:
    def _log_run_startup_line(text: str) -> None:
        logger.info(text)
        if startup_context.args_debug:
            _safe_console_print(text)

    _log_run_startup_line("Run startup dump begin")
    _log_run_startup_line(f"Run argv(sys): {repr(list(sys.argv))}")
    _log_run_startup_line(f"Run argv(main): {repr(startup_context.argv_list)}")
    _log_run_startup_line(
        "Run args: "
        f"args.audit_mode={startup_context.args_audit_mode} "
        f"args.debug={startup_context.args_debug} "
        f"args.dry_run={startup_context.args_dry_run}"
    )
    _log_run_startup_line(
        "Run resolved: "
        f"processing_mode={startup_context.processing_mode} "
        f"audit_mode={startup_context.args_audit_mode}"
    )
    _log_run_startup_line(f"cwd={os.getcwd()}")
    _invoked_script: Path = (
        Path(sys.argv[0]).resolve() if sys.argv else startup_context.entrypoint_path
    )
    _log_run_startup_line(f"script_path={_invoked_script}")
    _log_run_startup_line(f"project_root={startup_context.project_root}")
    _log_run_startup_line(f"runtime_config_path={startup_context.runtime_config_path}")
    _log_run_startup_line(f"templates_path={startup_context.templates_path}")
    _log_run_startup_line(f"secrets_env_path={startup_context.secrets_env_path}")
    _log_run_startup_line(
        f"config_processing_mode={startup_context.config_processing_mode_raw or '<empty>'}"
    )
    if llm_summary is not None:
        _log_run_startup_line(
            "Run llm: "
            f"provider={llm_summary.provider or 'unknown'} "
            f"effective_model={_summary_effective_model(llm_summary) or 'unknown'} "
            f"configured_model={_summary_configured_model(llm_summary) or 'unknown'} "
            f"provider_model={_summary_provider_model(llm_summary) or 'unknown'} "
            f"usage_reporting_mode={llm_summary.usage_reporting_mode or 'unknown'}"
        )
    _log_run_startup_line("Run startup dump end")


def log_startup_summary(
    logger: logging.Logger,
    startup_context: StartupContext,
    llm_summary: Optional[LlmSummarySnapshot] = None,
) -> None:
    _log_startup_banner(
        logger=logger,
        startup_context=startup_context,
        llm_summary=llm_summary,
    )
    _log_startup_dump(
        logger=logger,
        startup_context=startup_context,
        llm_summary=llm_summary,
    )


def log_config_summary(
    logger: logging.Logger,
    config: AppConfig,
    llm_summary: LlmSummarySnapshot,
    run_context: RunContext,
) -> None:
    mode_label: str = f"{run_context.processing_mode}:{run_context.audit_mode}"
    logger.info(
        "run_id=%s Config loaded successfully for mode=%s.",
        run_context.run_id,
        mode_label,
    )
    logger.info(
        "run_id=%s Config summary: google=%s telegram=%s templates=%s",
        run_context.run_id,
        "enabled" if config.google.enabled else "disabled",
        "enabled" if config.telegram.enabled else "disabled",
        str(config.paths.templates_path),
    )
    logger.info(
        "run_id=%s Config summary: sheets=%s range=%s",
        run_context.run_id,
        config.google.sheets_id,
        config.google.sheets_range,
    )
    logger.info(
        "run_id=%s Config summary: config_processing_mode=%s resolved_processing_mode=%s resolved_audit_mode=%s now_tz_mode=%s llm_provider=%s llm_model_effective=%s llm_model_configured=%s llm_provider_model=%s llm_usage_reporting_mode=%s openai_timeout_sec=%.1f openai_max_output_tokens=%d llm_source_desc_max_chars=%d openai_pre_delay_sec=%.1f",
        run_context.run_id,
        config.processing.mode,
        run_context.processing_mode,
        run_context.audit_mode,
        config.processing.now_tz_mode,
        llm_summary.provider,
        _summary_effective_model(llm_summary),
        _summary_configured_model(llm_summary),
        _summary_provider_model(llm_summary),
        llm_summary.usage_reporting_mode,
        config.llm.timeout_sec,
        config.llm.max_output_tokens,
        config.llm.source_desc_max_chars,
        config.llm.pre_delay_sec,
    )
    logger.info(
        "run_id=%s LLM summary: provider=%s effective_model=%s configured_model=%s provider_model=%s usage_reporting_mode=%s",
        run_context.run_id,
        llm_summary.provider,
        _summary_effective_model(llm_summary),
        _summary_configured_model(llm_summary),
        _summary_provider_model(llm_summary),
        llm_summary.usage_reporting_mode,
    )
    logger.info(
        "run_id=%s sheets_link_writeback=%s",
        run_context.run_id,
        "enabled" if run_context.sheets_link_writeback else "disabled",
    )
    logger.info(
        "run_id=%s strip_chapter_timestamps=%s",
        run_context.run_id,
        "enabled" if run_context.strip_chapter_timestamps else "disabled",
    )
    logger.info(
        "run_id=%s local_doc_export_enabled=%s",
        run_context.run_id,
        "true" if bool(str(config.paths.local_doc_dir_template or "").strip()) else "false",
    )

