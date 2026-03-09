from __future__ import annotations

import secrets
import sys
import time
from datetime import datetime
from typing import List, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from app.bootstrap.preflight import run_bootstrap_preflight
from app.bootstrap.cli import build_cli_parser
from app.bootstrap.logging_config import (
    get_logger,
    resolve_logger_name_meta,
    setup_logging,
)
from app.bootstrap.run_context import build_run_context, build_startup_context
from app.bootstrap.runtime_services import build_entrypoint_runtime_services
from app.config.app_config_loader import (
    load_config_from_env as _load_config_from_env_impl,
)
from app.config.settings import AppConfig
from app.config.validators import (
    describe_google_doc_share_mode,
    normalize_audit_mode,
    normalize_processing_mode,
)
from app.core.env_flags import (
    sheets_link_writeback_enabled_from_env,
    strip_chapter_timestamps_enabled_from_env,
)
from app.core.error_summary import summarize_error
from app.llm.openai_client import reset_run_local_openai_usage
from app.observability.final_summary import (
    FinalRunSummaryContext,
    emit_final_run_summary,
)
from app.observability.openai_usage import (
    log_openai_limits_and_usage,
    log_run_local_openai_usage,
)
from app.observability.runtime_analytics import (
    log_run_context,
    log_run_started,
    log_stage_timing,
    record_stage_duration,
    setup_runtime_analytics,
)
from app.observability.startup_summary import (
    build_llm_summary_snapshot,
    log_startup_summary,
    log_config_summary,
)
from app.paths import get_project_paths


LOGGER = get_logger(__name__)


def _load_config_from_env() -> AppConfig:
    return _load_config_from_env_impl(
        logger=LOGGER,
        summarize_error=summarize_error,
    )


def _load_zoneinfo(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        raise RuntimeError(
            "Timezone database is unavailable for this Python environment. "
            "Install tzdata in the active venv: "
            r"'.venv_restreamer\Scripts\python.exe -m pip install tzdata'. "
            f"Missing zone: {name}"
        ) from error


def _log_exit_code(*, logger, exit_code: int) -> None:
    logger.info("Exit code: %d", int(exit_code))


def _is_openai_usage_reporting_enabled(*, llm_summary) -> bool:
    return "openai" in llm_summary.providers_used


def _apply_llm_usage_reset(*, logger, llm_summary) -> None:
    provider_name: str = llm_summary.provider
    if _is_openai_usage_reporting_enabled(llm_summary=llm_summary):
        reset_run_local_openai_usage()
        logger.info("llm_usage_reset_applied provider=%s", provider_name)
        return
    logger.info(
        "llm_usage_reset_skipped provider=%s reason=provider_not_openai",
        provider_name,
    )


def _log_llm_usage_reports(*, logger, llm_summary) -> None:
    provider_name: str = llm_summary.provider
    logger.info("llm_usage_report_start provider=%s", provider_name)
    if not _is_openai_usage_reporting_enabled(llm_summary=llm_summary):
        logger.info(
            "llm_usage_report_skipped provider=%s reason=provider_not_openai",
            provider_name,
        )
        logger.info(
            "llm_org_usage_report_skipped provider=%s reason=provider_not_openai",
            provider_name,
        )
        logger.info(
            "llm_usage_report_completed provider=%s status=skipped_provider_logs_only",
            provider_name,
        )
        return
    try:
        log_run_local_openai_usage(logger)
    except Exception:
        logger.exception("Run-local OpenAI usage report failed")
    try:
        log_openai_limits_and_usage(logger, summarize_error=summarize_error)
    except Exception:
        logger.exception("OpenAI usage report failed")
    logger.info(
        "llm_usage_report_completed provider=%s status=completed",
        provider_name,
    )


def main(argv: Sequence[str]) -> int:
    project_paths = get_project_paths()
    load_dotenv(dotenv_path=project_paths.secrets_env_path)
    argv_list: List[str] = list(argv)
    run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    run_summary_started_at: float = 0.0

    parser = build_cli_parser()
    args = parser.parse_args(argv_list)

    setup_logging(debug=bool(args.debug))
    setup_runtime_analytics(logger=LOGGER, debug_enabled=bool(args.debug))
    run_bootstrap_preflight(logger=LOGGER, debug_enabled=bool(args.debug))

    config: AppConfig = _load_config_from_env()
    llm_summary = build_llm_summary_snapshot(config)
    sheets_link_writeback_enabled: bool = sheets_link_writeback_enabled_from_env()
    strip_chapter_timestamps_enabled: bool = strip_chapter_timestamps_enabled_from_env()
    _apply_llm_usage_reset(logger=LOGGER, llm_summary=llm_summary)
    config_processing_mode_raw: str = str(config.processing_mode or "").strip()
    processing_mode: str = normalize_processing_mode(
        config_processing_mode_raw or "audit",
        source="runtime processing mode",
    )
    audit_mode: str = normalize_audit_mode(args.audit_mode, source="CLI audit mode")
    startup_context = build_startup_context(
        run_id=run_id,
        argv_list=argv_list,
        args_audit_mode=audit_mode,
        args_debug=bool(args.debug),
        args_dry_run=bool(args.dry_run),
        processing_mode=processing_mode,
        config_processing_mode_raw=config_processing_mode_raw,
        project_root=project_paths.project_root,
        entrypoint_path=project_paths.entrypoint_path,
        runtime_config_path=project_paths.runtime_config_path,
        templates_path=project_paths.templates_path,
        secrets_env_path=project_paths.secrets_env_path,
        oauth_credentials_path=project_paths.oauth_credentials_path,
        oauth_token_path=project_paths.oauth_token_path,
    )
    run_context = build_run_context(
        run_id=run_id,
        processing_mode=processing_mode,
        audit_mode=audit_mode,
        config=config,
        llm_summary=llm_summary,
        debug_enabled=bool(args.debug),
        dry_run=bool(args.dry_run),
        sheets_link_writeback=sheets_link_writeback_enabled,
        strip_chapter_timestamps=strip_chapter_timestamps_enabled,
    )

    log_startup_summary(LOGGER, startup_context, llm_summary)
    log_run_context(LOGGER, run_context)
    log_config_summary(
        LOGGER,
        config,
        llm_summary,
        run_context,
    )
    LOGGER.debug(
        "Google Doc share mode resolved: %s (%s)",
        config.google_doc_share_mode,
        describe_google_doc_share_mode(config.google_doc_share_mode),
    )

    runtime_services = build_entrypoint_runtime_services(
        logger=LOGGER,
        config=config,
        kiev_tz=_load_zoneinfo(config.timezone_kiev),
        cet_tz=_load_zoneinfo(config.timezone_cet),
        resolve_logger_name_meta=resolve_logger_name_meta,
    )
    batch_runner = runtime_services.batch_runner

    exit_code: int = 0
    try:
        log_run_started(
            logger=LOGGER,
            processing_mode=processing_mode,
            audit_mode=audit_mode,
            debug_enabled=bool(args.debug),
            dry_run=bool(args.dry_run),
        )
        batch_runner.run(
            dry_run=bool(args.dry_run),
            audit_mode=audit_mode,
            run_id=run_id,
            llm_summary=llm_summary,
        )
        exit_code = 0
    except Exception as error:
        LOGGER.error("FATAL: %s", summarize_error(error))
        exit_code = 1
    finally:
        run_summary_started_at = time.perf_counter()
        _log_llm_usage_reports(logger=LOGGER, llm_summary=llm_summary)
        try:
            batch_runner.log_last_merge_run_summary()
        except Exception:
            LOGGER.exception("Merge run summary report failed")
        try:
            merge_summary = batch_runner.last_merge_run_summary
            run_summary_ms: int = int(
                round((time.perf_counter() - run_summary_started_at) * 1000.0)
            )
            record_stage_duration(stage_name="run_summary", elapsed_ms=run_summary_ms)
            log_stage_timing(
                logger=LOGGER,
                stage_name="run_summary",
                elapsed_ms=run_summary_ms,
                scope="run",
            )
            emit_final_run_summary(
                logger=LOGGER,
                summary_context=FinalRunSummaryContext(
                    processing_mode=processing_mode,
                    audit_mode=audit_mode,
                    exit_code=exit_code,
                    merge_run_summary=merge_summary,
                    run_summary_ms=run_summary_ms,
                ),
            )
        except Exception:
            LOGGER.exception("Runtime analytics summary failed")
        _log_exit_code(logger=LOGGER, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
