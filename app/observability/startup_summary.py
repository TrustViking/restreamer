from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from app.config.settings import AppConfig


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


def log_startup_summary(
    logger: logging.Logger,
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
) -> None:
    env_stg_merge_semantics: str = (
        os.getenv("STG_MERGE_SEMANTICS", "").strip() or "<unset>"
    )
    config_processing_mode_for_banner: str = (
        config_processing_mode_raw if config_processing_mode_raw else "missing"
    )
    audit_branches: str = "nomerge,merge" if args_audit_mode == "unite" else args_audit_mode
    logger.info("=== STARTUP BANNER BEGIN ===")
    logger.info("run_id=%s", run_id)
    logger.info("argv=%s", repr(list(sys.argv)))
    logger.info("cwd=%s", os.getcwd())
    logger.info("python=%s", sys.executable)
    logger.info(
        "args.audit_mode=%s args.debug=%s args.dry_run=%s",
        args_audit_mode,
        args_debug,
        args_dry_run,
    )
    logger.info("config_processing_mode=%s", config_processing_mode_for_banner)
    logger.info("env.STG_MERGE_SEMANTICS=%s", env_stg_merge_semantics)
    logger.info("resolved_processing_mode=%s", processing_mode)
    logger.info("resolved_audit_mode=%s branches=%s", args_audit_mode, audit_branches)
    logger.info("project_root=%s", project_root)
    logger.info("entrypoint_path=%s", entrypoint_path)
    logger.info("runtime_config_path=%s", runtime_config_path)
    logger.info("templates_path=%s", templates_path)
    logger.info("secrets_env_path=%s", secrets_env_path)
    logger.info("oauth_credentials_path=%s", oauth_credentials_path)
    logger.info("oauth_token_path=%s", oauth_token_path)
    logger.info("=== STARTUP BANNER END ===")

    def _log_run_startup_line(text: str) -> None:
        logger.info(text)
        if args_debug:
            _safe_console_print(text)

    _log_run_startup_line("Run startup dump begin")
    _log_run_startup_line(f"Run argv(sys): {repr(list(sys.argv))}")
    _log_run_startup_line(f"Run argv(main): {repr(argv_list)}")
    _log_run_startup_line(
        "Run args: "
        f"args.audit_mode={args_audit_mode} "
        f"args.debug={args_debug} "
        f"args.dry_run={args_dry_run}"
    )
    _log_run_startup_line(f"Run resolved: processing_mode={processing_mode} audit_mode={args_audit_mode}")
    _log_run_startup_line(f"cwd={os.getcwd()}")
    _log_run_startup_line(f"script_path={entrypoint_path}")
    _log_run_startup_line(f"project_root={project_root}")
    _log_run_startup_line(f"runtime_config_path={runtime_config_path}")
    _log_run_startup_line(f"templates_path={templates_path}")
    _log_run_startup_line(f"secrets_env_path={secrets_env_path}")
    _log_run_startup_line(
        f"config_processing_mode={config_processing_mode_raw or '<empty>'}"
    )
    _log_run_startup_line("Run startup dump end")


def log_config_summary(
    logger: logging.Logger,
    config: AppConfig,
    *,
    mode_label: str,
    resolved_processing_mode: Optional[str] = None,
    resolved_audit_mode: Optional[str] = None,
    run_id: Optional[str] = None,
    sheets_link_writeback_enabled: bool,
    strip_chapter_timestamps_enabled: bool,
) -> None:
    logger.info("run_id=%s Config loaded successfully for mode=%s.", run_id, mode_label)
    logger.info(
        "run_id=%s Config summary: google=%s telegram=%s templates=%s",
        run_id,
        "enabled" if config.google_enabled else "disabled",
        "enabled" if config.telegram_enabled else "disabled",
        str(config.stg_templates_path),
    )
    logger.info(
        "run_id=%s Config summary: sheets=%s range=%s",
        run_id,
        config.google_sheets_id,
        config.google_sheets_range,
    )
    logger.info(
        "run_id=%s Config summary: config_processing_mode=%s resolved_processing_mode=%s resolved_audit_mode=%s now_tz_mode=%s llm_provider=%s openai_primary=%s openai_fallback=%s openai_timeout_sec=%.1f openai_max_output_tokens=%d llm_source_desc_max_chars=%d llm_run_if_single_source=%s openai_pre_delay_sec=%.1f",
        run_id,
        config.processing_mode,
        str(resolved_processing_mode or config.processing_mode),
        str(resolved_audit_mode or "unknown"),
        config.now_tz_mode,
        config.llm_provider,
        config.openai_model_primary,
        config.openai_model_fallback,
        config.openai_timeout_sec,
        config.openai_max_output_tokens,
        config.llm_source_desc_max_chars,
        config.llm_run_if_single_source,
        config.openai_pre_delay_sec,
    )
    logger.info(
        "run_id=%s sheets_link_writeback=%s",
        run_id,
        "enabled" if sheets_link_writeback_enabled else "disabled",
    )
    logger.info(
        "run_id=%s strip_chapter_timestamps=%s",
        run_id,
        "enabled" if strip_chapter_timestamps_enabled else "disabled",
    )
    logger.info(
        "run_id=%s local_doc_export_enabled=%s",
        run_id,
        "true" if bool(str(config.local_doc_dir_template or "").strip()) else "false",
    )
