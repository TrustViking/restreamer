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
from app.config.app_config_loader import load_config_from_env as _load_config_from_env_impl
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
from app.ingest.youtube_metadata import YouTubeMetadataFetcher, YtDlpYouTubeMetadataFetcher
from app.llm.openai_client import reset_run_local_openai_usage
from app.net.http_client import HttpClient
from app.observability.openai_usage import (
    log_openai_limits_and_usage,
    log_run_local_openai_usage,
)
from app.observability.runtime_analytics import (
    log_run_context,
    log_run_completed,
    log_run_started,
    log_stage_timing,
    record_stage_duration,
    setup_runtime_analytics,
)
from app.observability.startup_summary import log_config_summary, log_startup_summary
from app.paths import get_project_paths
from app.paths.name_builder import NamePathBuilder
from app.pipeline.batch_runner import BatchRunner
from app.telegram.bot_client import TelegramBotClient


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
            r"'.venv_streamertg\Scripts\python.exe -m pip install tzdata'. "
            f"Missing zone: {name}"
        ) from error


def _log_exit_code(*, logger, exit_code: int) -> None:
    logger.info("Exit code: %d", int(exit_code))


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
    reset_run_local_openai_usage()
    run_bootstrap_preflight(logger=LOGGER, debug_enabled=bool(args.debug))

    config: AppConfig = _load_config_from_env()
    config_processing_mode_raw: str = str(config.processing_mode or "").strip()
    processing_mode: str = normalize_processing_mode(
        config_processing_mode_raw or "audit",
        source="runtime processing mode",
    )
    audit_mode: str = normalize_audit_mode(args.audit_mode, source="CLI audit mode")

    log_startup_summary(
        LOGGER,
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
    log_run_context(
        logger=LOGGER,
        run_id=run_id,
        processing_mode=processing_mode,
        audit_mode=audit_mode,
        config_processing_mode=config_processing_mode_raw,
        audit_branches=(["nomerge", "merge"] if audit_mode == "unite" else [audit_mode]),
        debug_enabled=bool(args.debug),
        dry_run=bool(args.dry_run),
        google_enabled=config.google_enabled,
        telegram_enabled=config.telegram_enabled,
        llm_provider=config.llm_provider,
        openai_primary=config.openai_model_primary,
        openai_fallback=config.openai_model_fallback,
        sheet_id=config.google_sheets_id,
        sheet_range=config.google_sheets_range,
        sheets_link_writeback=sheets_link_writeback_enabled_from_env(),
        local_doc_export_enabled=bool(str(config.local_doc_dir_template or "").strip()),
        strip_chapter_timestamps=strip_chapter_timestamps_enabled_from_env(),
    )

    mode_label: str = f"audit:{audit_mode}"
    log_config_summary(
        LOGGER,
        config,
        mode_label=mode_label,
        resolved_processing_mode=processing_mode,
        resolved_audit_mode=audit_mode,
        run_id=run_id,
        sheets_link_writeback_enabled=sheets_link_writeback_enabled_from_env(),
        strip_chapter_timestamps_enabled=strip_chapter_timestamps_enabled_from_env(),
    )
    LOGGER.debug(
        "Google Doc share mode resolved: %s (%s)",
        config.google_doc_share_mode,
        describe_google_doc_share_mode(config.google_doc_share_mode),
    )

    metadata_fetcher: YouTubeMetadataFetcher = YtDlpYouTubeMetadataFetcher()
    http_client: HttpClient = HttpClient()
    telegram_client: TelegramBotClient = TelegramBotClient(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )
    name_builder: NamePathBuilder = NamePathBuilder(
        local_image_dir_template=config.local_image_dir_template,
        local_doc_dir_template=config.local_doc_dir_template,
        preview_name_template=config.templates.files_preview_name_template,
        doc_title_template=config.templates.files_doc_title_template,
        language_codes_json=config.templates.files_language_codes_json,
        max_filename_stem=config.preview_filename_max_stem,
    )
    batch_runner: BatchRunner = BatchRunner(
        logger=LOGGER,
        config=config,
        metadata_fetcher=metadata_fetcher,
        http_client=http_client,
        telegram_client=telegram_client,
        name_builder=name_builder,
        kiev_tz=_load_zoneinfo(config.timezone_kiev),
        cet_tz=_load_zoneinfo(config.timezone_cet),
        resolve_logger_name_meta=resolve_logger_name_meta,
    )

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
        )
        exit_code = 0
    except Exception as error:
        LOGGER.error("FATAL: %s", summarize_error(error))
        exit_code = 1
    finally:
        run_summary_started_at = time.perf_counter()
        try:
            log_run_local_openai_usage(LOGGER)
        except Exception:
            LOGGER.exception("Run-local OpenAI usage report failed")
        try:
            log_openai_limits_and_usage(LOGGER, summarize_error=summarize_error)
        except Exception:
            LOGGER.exception("OpenAI usage report failed")
        try:
            batch_runner.log_last_merge_run_summary()
        except Exception:
            LOGGER.exception("Merge run summary report failed")
        try:
            merge_summary = batch_runner.last_merge_run_summary
            run_summary_ms: int = int(round((time.perf_counter() - run_summary_started_at) * 1000.0))
            record_stage_duration(stage_name="run_summary", elapsed_ms=run_summary_ms)
            log_stage_timing(
                logger=LOGGER,
                stage_name="run_summary",
                elapsed_ms=run_summary_ms,
                scope="run",
            )
            log_run_completed(
                logger=LOGGER,
                processing_mode=processing_mode,
                audit_mode=audit_mode,
                exit_code=exit_code,
                primary_success=merge_summary.primary_success if merge_summary is not None else 0,
                validation_rejected=(
                    merge_summary.validation_rejected if merge_summary is not None else 0
                ),
                primary_retry_used=(
                    merge_summary.primary_retry_used if merge_summary is not None else 0
                ),
                fallback_success=(
                    merge_summary.fallback_success if merge_summary is not None else 0
                ),
                final_failure=(
                    merge_summary.final_failure if merge_summary is not None else 0
                ),
                paragraph_recovery_used=(
                    merge_summary.paragraph_recovery_used
                    if merge_summary is not None
                    else 0
                ),
                run_summary_ms=run_summary_ms,
            )
        except Exception:
            LOGGER.exception("Runtime analytics summary failed")
        _log_exit_code(logger=LOGGER, exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
