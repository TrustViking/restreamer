from __future__ import annotations

import argparse
import logging
import secrets
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from app.bootstrap.cli import build_cli_parser
from app.bootstrap.cleanup import run_daily_cleanup
from app.bootstrap.logging_config import (
    get_logger,
    resolve_logger_name_meta,
    setup_logging,
)
from app.bootstrap.preflight import run_bootstrap_preflight
from app.bootstrap.run_context import (
    RunContext,
    StartupContext,
)
from app.config.app_config_loader import (
    load_config_from_env as _load_config_from_env_impl,
)
from app.config.settings import AppConfig
from app.core.branching import audit_branch_labels
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
from app.ingest.youtube_metadata import YtDlpYouTubeMetadataFetcher
from app.llm.llm_client import reset_run_local_openai_usage
from app.net.http_client import HttpClient
from app.observability.final_summary import (
    FinalRunSummaryContext,
    emit_final_run_summary,
)
from app.observability.openai_usage import log_run_local_openai_usage
from app.observability.runtime_analytics import (
    log_run_context,
    log_run_started,
    log_stage_timing,
    record_stage_duration,
    setup_runtime_analytics,
)
from app.observability.startup_summary import (
    LlmSummarySnapshot,
    build_llm_summary_snapshot,
    log_config_summary,
    log_startup_summary,
)
from app.paths import ProjectPaths, get_project_paths
from app.paths.name_builder import NamePathBuilder
from app.pipeline.batch_runner import BatchRunner
from app.pipeline.operator_notifier import OperatorNotifier
from app.telegram.bot_client import TelegramBotClient
from app.telegram_bot.group_registry import handle_group_migration


LOGGER: logging.Logger = get_logger(__name__)


def _normalize_chat_id(raw_chat_id: object | None) -> str | None:
    normalized_chat_id: str = str(raw_chat_id or "").strip()
    if not normalized_chat_id:
        return None
    return normalized_chat_id


def _load_config_from_env(*, logger: logging.Logger) -> AppConfig:
    return _load_config_from_env_impl(
        logger=logger,
        summarize_error=summarize_error,
    )


def _load_zoneinfo(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        python_executable: str = str(sys.executable or "").strip() or "python"
        raise RuntimeError(
            "Timezone database is unavailable for this Python environment. "
            "Install tzdata in the active venv: "
            f"'{python_executable} -m pip install tzdata'. "
            f"Missing zone: {name}"
        ) from error


def _log_exit_code(*, logger: logging.Logger, exit_code: int) -> None:
    logger.info("Exit code: %d", int(exit_code))


def _is_openai_usage_reporting_enabled(*, llm_summary: LlmSummarySnapshot) -> bool:
    return str(llm_summary.provider or "").strip().lower() == "openai"


def _apply_llm_usage_reset(
    *,
    logger: logging.Logger,
    llm_summary: LlmSummarySnapshot,
) -> None:
    provider_name: str = str(getattr(llm_summary, "provider", "") or "").strip()
    if _is_openai_usage_reporting_enabled(llm_summary=llm_summary):
        reset_run_local_openai_usage()
        logger.info("llm_usage_reset_applied provider=%s", provider_name)
        return
    logger.info(
        "llm_usage_reset_skipped provider=%s reason=provider_not_openai",
        provider_name,
    )


def _log_llm_usage_reports(
    *,
    logger: logging.Logger,
    llm_summary: LlmSummarySnapshot,
) -> None:
    provider_name: str = str(getattr(llm_summary, "provider", "") or "").strip()
    effective_model: str = str(
        getattr(llm_summary, "effective_model", getattr(llm_summary, "model", "")) or ""
    ).strip() or "unknown"
    usage_reporting_mode: str = str(
        getattr(llm_summary, "usage_reporting_mode", "") or ""
    ).strip() or "unknown"
    logger.info(
        "llm_usage_report_start provider=%s effective_model=%s usage_reporting_mode=%s",
        provider_name,
        effective_model,
        usage_reporting_mode,
    )
    try:
        log_run_local_openai_usage(logger, effective_model=effective_model)
    except Exception:
        logger.exception("Run-local OpenAI usage report failed")
    logger.info(
        "llm_usage_report_completed provider=%s effective_model=%s status=completed",
        provider_name,
        effective_model,
    )

class RestreamerApplication:
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        telegram_chat_id_override: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._logger: logging.Logger = logger or LOGGER
        self._telegram_chat_id_override: str | None = _normalize_chat_id(
            telegram_chat_id_override,
        )
        self._progress_callback: Callable[[str], None] | None = progress_callback

    def run(self, argv: Sequence[str]) -> int:
        project_paths: ProjectPaths = get_project_paths()
        load_dotenv(dotenv_path=project_paths.secrets_env_path)
        argv_list: list[str] = list(argv)
        run_id: str = self._build_run_id()

        parser: argparse.ArgumentParser = build_cli_parser()
        args: argparse.Namespace = parser.parse_args(argv_list)
        debug_enabled: bool = bool(args.debug)
        dry_run: bool = bool(args.dry_run)

        self._configure_runtime(debug_enabled=debug_enabled)

        config: AppConfig = _load_config_from_env(logger=self._logger)
        run_daily_cleanup(
            logger=self._logger,
            project_root=project_paths.project_root,
            max_age_days=config.cleanup.max_age_days,
            local_image_dir_template=config.paths.local_image_dir_template,
            local_doc_dir_template=config.paths.local_doc_dir_template,
        )
        llm_summary: LlmSummarySnapshot = build_llm_summary_snapshot(config)
        sheets_link_writeback_enabled: bool = sheets_link_writeback_enabled_from_env()
        strip_chapter_timestamps_enabled: bool = (
            strip_chapter_timestamps_enabled_from_env()
        )
        _apply_llm_usage_reset(logger=self._logger, llm_summary=llm_summary)

        config_processing_mode_raw: str = str(config.processing.mode or "").strip()
        processing_mode: str = normalize_processing_mode(
            config_processing_mode_raw or "audit",
            source="runtime processing mode",
        )
        audit_mode: str = normalize_audit_mode(args.audit_mode, source="CLI audit mode")

        startup_context: StartupContext = StartupContext(
            run_id=run_id,
            argv_list=argv_list,
            args_audit_mode=audit_mode,
            args_debug=debug_enabled,
            args_dry_run=dry_run,
            processing_mode=processing_mode,
            config_processing_mode_raw=config_processing_mode_raw,
            paths=project_paths,
        )
        effective_model: str = str(
            getattr(llm_summary, "effective_model", getattr(llm_summary, "model", ""))
            or ""
        ).strip()
        configured_model: str = str(
            getattr(llm_summary, "configured_model", effective_model) or ""
        ).strip()
        provider_model: str = str(
            getattr(llm_summary, "provider_model", effective_model) or ""
        ).strip()
        run_context: RunContext = RunContext(
            run_id=run_id,
            processing_mode=processing_mode,
            audit_mode=audit_mode,
            config_processing_mode=str(config.processing.mode or "").strip(),
            audit_branches=audit_branch_labels(audit_mode=audit_mode),
            debug_enabled=debug_enabled,
            dry_run=dry_run,
            google_enabled=config.google.enabled,
            telegram_enabled=config.telegram.enabled,
            llm_provider=config.llm.provider,
            llm_model=effective_model,
            llm_model_configured=configured_model,
            llm_provider_model=provider_model,
            llm_usage_reporting_mode=llm_summary.usage_reporting_mode,
            sheet_id=config.google.sheets_id,
            sheet_range=config.google.sheets_range,
            sheets_link_writeback=sheets_link_writeback_enabled,
            local_doc_export_enabled=bool(
                str(config.paths.local_doc_dir_template or "").strip()
            ),
            strip_chapter_timestamps=strip_chapter_timestamps_enabled,
        )

        log_startup_summary(self._logger, startup_context, llm_summary)
        log_run_context(self._logger, run_context)
        log_config_summary(self._logger, config, llm_summary, run_context)
        self._logger.debug(
            "Google Doc share mode resolved: %s (%s)",
            config.google.doc_share_mode,
            describe_google_doc_share_mode(config.google.doc_share_mode),
        )

        batch_runner: BatchRunner = self._build_batch_runner(config=config)

        return self._run_batch(
            batch_runner=batch_runner,
            run_id=run_id,
            llm_summary=llm_summary,
            processing_mode=processing_mode,
            audit_mode=audit_mode,
            debug_enabled=debug_enabled,
            dry_run=dry_run,
        )

    def _build_run_id(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)

    def _configure_runtime(self, *, debug_enabled: bool) -> None:
        setup_logging(debug=debug_enabled)
        setup_runtime_analytics(logger=self._logger, debug_enabled=debug_enabled)
        run_bootstrap_preflight(logger=self._logger, debug_enabled=debug_enabled)

    def _build_batch_runner(self, *, config: AppConfig) -> BatchRunner:
        metadata_fetcher: YtDlpYouTubeMetadataFetcher = YtDlpYouTubeMetadataFetcher()
        http_client: HttpClient = HttpClient()
        effective_chat_id: str = (
            self._telegram_chat_id_override
            if self._telegram_chat_id_override is not None
            else str(config.telegram.chat_id)
        )
        chat_id_source: str = (
            "telegram_event"
            if self._telegram_chat_id_override is not None
            else "config.telegram.chat_id"
        )
        self._logger.info(
            "telegram_publish_target_resolved chat_id=%s source=%s",
            effective_chat_id,
            chat_id_source,
        )
        telegram_client: TelegramBotClient = TelegramBotClient(
            bot_token=config.telegram.bot_token,
            chat_id=effective_chat_id,
            send_delay_seconds=config.telegram.send_delay_seconds,
            max_retries=config.telegram.max_retries,
        )
        self._logger.info(
            "telegram_client_config send_delay_seconds=%.2f max_retries=%d",
            config.telegram.send_delay_seconds,
            config.telegram.max_retries,
        )

        def _on_chat_migrated(old_id: str, new_id: str) -> None:
            handle_group_migration(
                logger=self._logger,
                old_chat_id=old_id,
                new_chat_id=new_id,
            )

        telegram_client.set_migration_callback(_on_chat_migrated)
        name_builder: NamePathBuilder = NamePathBuilder(
            local_image_dir_template=config.paths.local_image_dir_template,
            local_doc_dir_template=config.paths.local_doc_dir_template,
            preview_name_template=config.templates.files_preview_name_template,
            doc_title_template=config.templates.files_doc_title_template,
            max_filename_stem=config.paths.preview_filename_max_stem,
        )
        telegram_sink: Callable[[str], None] | None = None
        if self._progress_callback is not None:
            telegram_sink = self._progress_callback
        elif config.telegram.enabled:
            telegram_sink = telegram_client.send_text
        notifier: OperatorNotifier = OperatorNotifier(telegram_sink=telegram_sink)
        return BatchRunner(
            logger=self._logger,
            config=config,
            metadata_fetcher=metadata_fetcher,
            http_client=http_client,
            telegram_client=telegram_client,
            name_builder=name_builder,
            kiev_tz=_load_zoneinfo(config.timezones.kiev),
            cet_tz=_load_zoneinfo(config.timezones.cet),
            resolve_logger_name_meta=resolve_logger_name_meta,
            notifier=notifier,
        )

    def _run_batch(
        self,
        *,
        batch_runner: BatchRunner,
        run_id: str,
        llm_summary: LlmSummarySnapshot,
        processing_mode: str,
        audit_mode: str,
        debug_enabled: bool,
        dry_run: bool,
    ) -> int:
        exit_code: int = 0
        try:
            log_run_started(
                logger=self._logger,
                processing_mode=processing_mode,
                audit_mode=audit_mode,
                debug_enabled=debug_enabled,
                dry_run=dry_run,
            )
            batch_runner.run(
                dry_run=dry_run,
                audit_mode=audit_mode,
                run_id=run_id,
                llm_summary=llm_summary,
            )
            exit_code = 0
        except Exception as error:
            self._logger.error("FATAL: %s", summarize_error(error))
            exit_code = 1
        finally:
            self._finalize_run(
                batch_runner=batch_runner,
                llm_summary=llm_summary,
                processing_mode=processing_mode,
                audit_mode=audit_mode,
                exit_code=exit_code,
            )
        return exit_code

    def _finalize_run(
        self,
        *,
        batch_runner: BatchRunner,
        llm_summary: LlmSummarySnapshot,
        processing_mode: str,
        audit_mode: str,
        exit_code: int,
    ) -> None:
        run_summary_started_at: float = time.perf_counter()
        _log_llm_usage_reports(logger=self._logger, llm_summary=llm_summary)
        try:
            batch_runner.log_last_merge_run_summary()
        except Exception:
            self._logger.exception("Merge run summary report failed")

        try:
            merge_summary: object | None = batch_runner.last_merge_run_summary
            run_summary_ms: int = int(
                round((time.perf_counter() - run_summary_started_at) * 1000.0)
            )
            record_stage_duration(stage_name="run_summary", elapsed_ms=run_summary_ms)
            log_stage_timing(
                logger=self._logger,
                stage_name="run_summary",
                elapsed_ms=run_summary_ms,
                scope="run",
            )
            emit_final_run_summary(
                logger=self._logger,
                summary_context=FinalRunSummaryContext(
                    processing_mode=processing_mode,
                    audit_mode=audit_mode,
                    exit_code=exit_code,
                    merge_run_summary=merge_summary,
                    run_summary_ms=run_summary_ms,
                    llm_provider=str(getattr(llm_summary, "provider", "") or "").strip(),
                    llm_effective_model=str(
                        getattr(
                            llm_summary,
                            "effective_model",
                            getattr(llm_summary, "model", ""),
                        )
                        or ""
                    ).strip(),
                    llm_configured_model=str(
                        getattr(
                            llm_summary,
                            "configured_model",
                            getattr(
                                llm_summary,
                                "effective_model",
                                getattr(llm_summary, "model", ""),
                            ),
                        )
                        or ""
                    ).strip(),
                    llm_provider_model=str(
                        getattr(
                            llm_summary,
                            "provider_model",
                            getattr(
                                llm_summary,
                                "effective_model",
                                getattr(llm_summary, "model", ""),
                            ),
                        )
                        or ""
                    ).strip(),
                ),
            )
        except Exception:
            self._logger.exception("Runtime analytics summary failed")
        _log_exit_code(logger=self._logger, exit_code=exit_code)
