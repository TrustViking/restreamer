from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.core.env_flags import sheets_link_normalize_report_limit_from_env
from app.core.models import PlannedVideo, PreparedVideo
from app.ingest.youtube_metadata import YouTubeMetadataFetcher
from app.llm import MergeRunSummary
from app.net.http_client import HttpClient
from app.observability.runtime_analytics import (
    get_branch_date_summary,
    log_date_started,
    log_docs_publish_summary,
    log_error_event,
    log_merge_summary,
    log_planning_completed,
    log_sheet_loaded,
    log_stage_timing,
    log_telegram_publish_summary,
    log_warning_operational,
    record_branch_completed,
    record_branch_failed,
    record_branch_started,
    record_branch_total_ms,
    record_date_branch_execution,
    record_docs_created,
    record_docs_failed,
    record_planning_completed,
    record_sheet_loaded,
    record_stage_duration,
    record_telegram_failed,
    record_telegram_sent,
    record_telegram_skipped,
)
from app.observability.startup_health import log_section, run_startup_health_checks
from app.paths.name_builder import NamePathBuilder
from app.planning import log_link_normalization_report, planned_video_time_key
from app.planning.batch_planner import (
    build_prepared_videos,
    derive_planned_videos,
    materialize_prepared_previews,
)
from app.planning.sheet_loader import BatchSheetState, load_sheet_state
from app.telegram.bot_client import TelegramBotClient

from .daily_doc_publish import publish_daily_document
from .daily_telegram_publish import publish_daily_telegram
from .runtime_services import BatchServices, build_runtime_services
from .slot_processing import SlotProcessResult, process_slot


@dataclass(frozen=True)
class AuditBranch:
    name: str
    processing_mode: str
    llm_merge_enabled: bool


@dataclass(frozen=True)
class PreparedRunContext:
    services: BatchServices
    sheet_state: BatchSheetState
    prepared_videos: List[PreparedVideo]
    llm_merge_available: bool


class BatchRunner:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        config: AppConfig,
        metadata_fetcher: YouTubeMetadataFetcher,
        http_client: HttpClient,
        telegram_client: TelegramBotClient,
        name_builder: NamePathBuilder,
        kiev_tz: ZoneInfo,
        cet_tz: ZoneInfo,
        resolve_logger_name_meta: Any,
    ) -> None:
        self._logger = logger
        self._config = config
        self._metadata_fetcher = metadata_fetcher
        self._http_client = http_client
        self._telegram_client = telegram_client
        self._name_builder = name_builder
        self._kiev_tz = kiev_tz
        self._cet_tz = cet_tz
        self._resolve_logger_name_meta = resolve_logger_name_meta
        self._last_merge_run_summary: Optional[MergeRunSummary] = None

    @property
    def last_merge_run_summary(self) -> Optional[MergeRunSummary]:
        return self._last_merge_run_summary

    def log_last_merge_run_summary(self) -> None:
        if self._last_merge_run_summary is None:
            return
        self._last_merge_run_summary.log_summary(self._logger)

    def _log_section(self, title: str) -> None:
        log_section(logger=self._logger, title=title)

    def _group_processed_videos_by_date(
        self,
        *,
        processed: List[PlannedVideo],
    ) -> Dict[str, List[PlannedVideo]]:
        videos_by_date: Dict[str, List[PlannedVideo]] = {}
        for item in processed:
            videos_by_date.setdefault(item.date_key, []).append(item)
        return videos_by_date

    def _group_date_videos_by_slot_time(
        self,
        *,
        date_videos_all: List[PlannedVideo],
    ) -> Dict[str, List[PlannedVideo]]:
        slots_by_time: Dict[str, List[PlannedVideo]] = {}
        for item in date_videos_all:
            slots_by_time.setdefault(planned_video_time_key(item), []).append(item)
        return slots_by_time

    def _resolve_audit_branches(
        self,
        *,
        audit_mode: str,
        llm_merge_available: bool,
    ) -> List[AuditBranch]:
        if audit_mode == "unite":
            return [
                AuditBranch(
                    name="nomerge",
                    processing_mode="nomerge",
                    llm_merge_enabled=False,
                ),
                AuditBranch(
                    name="merge",
                    processing_mode="merge",
                    llm_merge_enabled=llm_merge_available,
                ),
            ]
        if audit_mode == "merge":
            return [
                AuditBranch(
                    name="merge",
                    processing_mode="merge",
                    llm_merge_enabled=llm_merge_available,
                )
            ]
        return [
            AuditBranch(
                name="nomerge",
                processing_mode="nomerge",
                llm_merge_enabled=False,
            )
        ]

    def run(self, *, dry_run: bool, audit_mode: str, run_id: str) -> None:
        run_started_at: float = time.perf_counter()
        if not self._config.google_enabled:
            raise RuntimeError("Для batch режима GOOGLE_ENABLED должен быть включен.")

        merge_run_summary: MergeRunSummary = MergeRunSummary()
        self._last_merge_run_summary = merge_run_summary
        branch_failures: List[str] = []
        branches_for_log: str = "nomerge,merge" if audit_mode == "unite" else audit_mode
        self._logger.info(
            "audit_start audit_mode=%s branches=%s run_id=%s dry_run=%s",
            audit_mode,
            branches_for_log,
            run_id,
            dry_run,
        )

        self._log_section("Runtime Services")
        services: BatchServices = build_runtime_services(config=self._config)
        try:
            prepared_context: PreparedRunContext = self._prepare_run_context(
                services=services,
                dry_run=dry_run,
                audit_mode=audit_mode,
                run_id=run_id,
            )
            if not prepared_context.prepared_videos:
                log_link_normalization_report(
                    logger=self._logger,
                    normalization_candidates=prepared_context.sheet_state.link_normalization_candidates,
                    writeback_enabled=prepared_context.sheet_state.sheets_link_writeback_enabled,
                    report_limit=sheets_link_normalize_report_limit_from_env(),
                )
                log_warning_operational(
                    self._logger,
                    "No videos to process after shared preparation.",
                    reason_code="no_videos_after_shared_preparation",
                )
                return

            branches: List[AuditBranch] = self._resolve_audit_branches(
                audit_mode=audit_mode,
                llm_merge_available=prepared_context.llm_merge_available,
            )
            processed_by_branch: Dict[str, Dict[str, List[PlannedVideo]]] = self._build_processed_by_branch(
                branches=branches,
                prepared_videos=prepared_context.prepared_videos,
                merge_semantics=prepared_context.sheet_state.merge_semantics,
            )

            self._log_section("Process Slots")
            self._execute_branch_dates(
                services=services,
                branches=branches,
                processed_by_branch=processed_by_branch,
                dry_run=dry_run,
                merge_run_summary=merge_run_summary,
                audit_mode=audit_mode,
                branch_failures=branch_failures,
            )

            log_link_normalization_report(
                logger=self._logger,
                normalization_candidates=prepared_context.sheet_state.link_normalization_candidates,
                writeback_enabled=prepared_context.sheet_state.sheets_link_writeback_enabled,
                report_limit=sheets_link_normalize_report_limit_from_env(),
            )
            if branch_failures:
                raise RuntimeError("; ".join(branch_failures))
        finally:
            total_run_ms: int = int(round((time.perf_counter() - run_started_at) * 1000.0))
            record_stage_duration(stage_name="total_run", elapsed_ms=total_run_ms)
            log_stage_timing(
                logger=self._logger,
                stage_name="total_run",
                elapsed_ms=total_run_ms,
                scope="run",
            )
            self._logger.info(
                "audit_done audit_mode=%s overall_status=%s run_id=%s",
                audit_mode,
                "ok" if not branch_failures else "failed",
                run_id,
            )

    def _prepare_run_context(
        self,
        *,
        services: BatchServices,
        dry_run: bool,
        audit_mode: str,
        run_id: str,
    ) -> PreparedRunContext:
        self._log_section("Startup Health Check")
        startup_health_started_at: float = time.perf_counter()
        llm_merge_available: bool = run_startup_health_checks(
            logger=self._logger,
            config=self._config,
            services=services,
            telegram_client=self._telegram_client,
            resolve_logger_name_meta=self._resolve_logger_name_meta,
            dry_run=dry_run,
            resolved_audit_mode=audit_mode,
            run_id=run_id,
        )
        startup_health_ms: int = int(round((time.perf_counter() - startup_health_started_at) * 1000.0))
        record_stage_duration(stage_name="startup_health", elapsed_ms=startup_health_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="startup_health",
            elapsed_ms=startup_health_ms,
            scope="run",
        )

        self._log_section("Sheet Load")
        sheet_load_started_at: float = time.perf_counter()
        sheet_state: BatchSheetState = load_sheet_state(
            logger=self._logger,
            config=self._config,
            services=services,
            kiev_tz=self._kiev_tz,
            run_id=run_id,
        )
        sheet_load_ms: int = int(round((time.perf_counter() - sheet_load_started_at) * 1000.0))
        record_stage_duration(stage_name="sheet_load", elapsed_ms=sheet_load_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="sheet_load",
            elapsed_ms=sheet_load_ms,
            scope="run",
        )
        record_sheet_loaded(rows=len(sheet_state.rows))
        log_sheet_loaded(logger=self._logger, rows=len(sheet_state.rows))

        self._log_section("Shared Preparation")
        shared_preparation_started_at: float = time.perf_counter()
        prepared_videos: List[PreparedVideo] = build_prepared_videos(
            logger=self._logger,
            config=self._config,
            services=services,
            sheet_state=sheet_state,
            metadata_fetcher=self._metadata_fetcher,
            http_client=self._http_client,
            kiev_tz=self._kiev_tz,
        )
        prepared_videos = materialize_prepared_previews(
            logger=self._logger,
            config=self._config,
            drive_client=services.drive_client,
            name_builder=self._name_builder,
            prepared_videos=prepared_videos,
            dry_run=dry_run,
        )
        shared_preparation_ms: int = int(
            round((time.perf_counter() - shared_preparation_started_at) * 1000.0)
        )
        record_stage_duration(
            stage_name="shared_preparation",
            elapsed_ms=shared_preparation_ms,
        )
        log_stage_timing(
            logger=self._logger,
            stage_name="shared_preparation",
            elapsed_ms=shared_preparation_ms,
            scope="run",
        )

        rows_skipped: int = max(0, len(sheet_state.rows) - len(prepared_videos))
        prepared_dates_count: int = len({item.date_key for item in prepared_videos})
        prepared_slots_count: int = len(
            {
                f"{item.date_key}_{item.scheduled_at_kiev.strftime('%H%M')}"
                for item in prepared_videos
            }
        )
        record_planning_completed(
            planned_items=len(prepared_videos),
            rows_skipped=rows_skipped,
        )
        log_planning_completed(
            logger=self._logger,
            planned_items=len(prepared_videos),
            rows_skipped=rows_skipped,
            dates=prepared_dates_count,
            slots=prepared_slots_count,
        )
        return PreparedRunContext(
            services=services,
            sheet_state=sheet_state,
            prepared_videos=prepared_videos,
            llm_merge_available=llm_merge_available,
        )

    def _build_processed_by_branch(
        self,
        *,
        branches: List[AuditBranch],
        prepared_videos: List[PreparedVideo],
        merge_semantics: str,
    ) -> Dict[str, Dict[str, List[PlannedVideo]]]:
        processed_by_branch: Dict[str, Dict[str, List[PlannedVideo]]] = {}
        planning_started_at: float = time.perf_counter()
        for branch in branches:
            self._logger.info(
                "audit_branch_plan branch=%s processing_mode=%s shared_prepared_videos=%d",
                branch.name,
                branch.processing_mode,
                len(prepared_videos),
            )
            processed_by_branch[branch.name] = self._group_processed_videos_by_date(
                processed=derive_planned_videos(
                    logger=self._logger,
                    prepared_videos=prepared_videos,
                    processing_mode=branch.processing_mode,
                    merge_semantics=merge_semantics,
                )
            )
        planning_ms: int = int(round((time.perf_counter() - planning_started_at) * 1000.0))
        record_stage_duration(stage_name="planning", elapsed_ms=planning_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="planning",
            elapsed_ms=planning_ms,
            scope="run",
        )
        return processed_by_branch

    def _execute_branch_dates(
        self,
        *,
        services: BatchServices,
        branches: List[AuditBranch],
        processed_by_branch: Dict[str, Dict[str, List[PlannedVideo]]],
        dry_run: bool,
        merge_run_summary: MergeRunSummary,
        audit_mode: str,
        branch_failures: List[str],
    ) -> None:
        date_keys: List[str] = sorted(
            {
                date_key
                for branch_videos in processed_by_branch.values()
                for date_key in branch_videos.keys()
            }
        )
        for date_key in date_keys:
            for branch in branches:
                self._execute_single_branch_date(
                    services=services,
                    branch=branch,
                    date_key=date_key,
                    processed_by_branch=processed_by_branch,
                    dry_run=dry_run,
                    merge_run_summary=merge_run_summary,
                    audit_mode=audit_mode,
                    branch_failures=branch_failures,
                )

    def _execute_single_branch_date(
        self,
        *,
        services: BatchServices,
        branch: AuditBranch,
        date_key: str,
        processed_by_branch: Dict[str, Dict[str, List[PlannedVideo]]],
        dry_run: bool,
        merge_run_summary: MergeRunSummary,
        audit_mode: str,
        branch_failures: List[str],
    ) -> None:
        date_videos_all: List[PlannedVideo] = processed_by_branch.get(branch.name, {}).get(date_key, [])
        if not date_videos_all:
            self._logger.info(
                "audit_branch_skip branch=%s date_key=%s reason=empty_branch_items",
                branch.name,
                date_key,
            )
            return
        branch_started_at: float = time.perf_counter()
        self._logger.info("audit_branch_start branch=%s", branch.name)
        try:
            self._run_branch_for_date(
                services=services,
                branch=branch,
                date_key=date_key,
                date_videos_all=date_videos_all,
                dry_run=dry_run,
                merge_run_summary=merge_run_summary,
            )
            branch_total_ms: int = int(round((time.perf_counter() - branch_started_at) * 1000.0))
            record_branch_total_ms(
                date_key=date_key,
                branch_label=branch.name,
                elapsed_ms=branch_total_ms,
            )
            log_stage_timing(
                logger=self._logger,
                stage_name="branch_total",
                elapsed_ms=branch_total_ms,
                scope="branch",
                branch_label=branch.name,
                date_key=date_key,
            )
            self._logger.info(
                "audit_branch_done branch=%s status=ok date_key=%s elapsed_ms=%d",
                branch.name,
                date_key,
                branch_total_ms,
            )
            if audit_mode == "unite":
                self._log_audit_branch_compare(date_key=date_key)
        except Exception as error:
            branch_failures.append(f"branch={branch.name} date={date_key} failed: {error}")
            self._logger.error(
                "audit_branch_done branch=%s status=failed date_key=%s reason=%s",
                branch.name,
                date_key,
                error,
            )
            log_error_event(
                self._logger,
                "branch=%s date=%s failed: %s",
                branch.name,
                date_key,
                error,
                reason_code="branch_date_failed",
            )

    def _run_branch_for_date(
        self,
        *,
        services: BatchServices,
        branch: AuditBranch,
        date_key: str,
        date_videos_all: List[PlannedVideo],
        dry_run: bool,
        merge_run_summary: MergeRunSummary,
    ) -> None:
        slot_processing_started_at: float = time.perf_counter()
        slots_by_time: Dict[str, List[PlannedVideo]] = self._group_date_videos_by_slot_time(
            date_videos_all=date_videos_all
        )
        self._logger.info(
            "[%s] Date branch started: %s slots=%d items=%d",
            branch.name,
            date_key,
            len(slots_by_time),
            len(date_videos_all),
        )
        record_branch_started(branch_label=branch.name)
        record_date_branch_execution(date_key=date_key, branch_label=branch.name)
        log_date_started(
            logger=self._logger,
            date_key=f"{date_key}/{branch.name}",
            slot_count=len(slots_by_time),
            item_count=len(date_videos_all),
        )

        merge_snapshot_before: tuple[int, int, int, int, int, int] = (
            merge_run_summary.primary_success,
            merge_run_summary.validation_rejected,
            merge_run_summary.primary_retry_used,
            merge_run_summary.fallback_success,
            merge_run_summary.final_failure,
            merge_run_summary.paragraph_recovery_used,
        )
        slot_results: List[SlotProcessResult] = []
        for slot_time_key in sorted(slots_by_time.keys()):
            slot_results.append(
                process_slot(
                    logger=self._logger,
                    config=self._config,
                    videos=slots_by_time[slot_time_key],
                    date_key=date_key,
                    slot_time_key=slot_time_key,
                    llm_merge_enabled=branch.llm_merge_enabled,
                    cet_tz=self._cet_tz,
                    merge_run_summary=merge_run_summary,
                    branch_label=branch.name,
                )
            )
        slot_processing_ms: int = int(round((time.perf_counter() - slot_processing_started_at) * 1000.0))
        record_stage_duration(stage_name="slot_processing", elapsed_ms=slot_processing_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="slot_processing",
            elapsed_ms=slot_processing_ms,
            scope="branch",
            branch_label=branch.name,
            date_key=date_key,
        )

        merge_snapshot_after: tuple[int, int, int, int, int, int] = (
            merge_run_summary.primary_success,
            merge_run_summary.validation_rejected,
            merge_run_summary.primary_retry_used,
            merge_run_summary.fallback_success,
            merge_run_summary.final_failure,
            merge_run_summary.paragraph_recovery_used,
        )
        log_merge_summary(
            logger=self._logger,
            groups=len(slot_results),
            primary_success=merge_snapshot_after[0] - merge_snapshot_before[0],
            validation_rejected=merge_snapshot_after[1] - merge_snapshot_before[1],
            primary_retry_used=merge_snapshot_after[2] - merge_snapshot_before[2],
            fallback_success=merge_snapshot_after[3] - merge_snapshot_before[3],
            final_failure=merge_snapshot_after[4] - merge_snapshot_before[4],
            paragraph_recovery_used=merge_snapshot_after[5] - merge_snapshot_before[5],
        )

        self._log_section(f"Publish Daily Docs [{branch.name}]")
        doc_publish_started_at: float = time.perf_counter()
        try:
            doc_publish_result = publish_daily_document(
                logger=self._logger,
                config=self._config,
                docs_client=services.docs_client,
                drive_client=services.drive_client,
                report_writer=services.report_writer,
                name_builder=self._name_builder,
                slot_results=slot_results,
                date_key=date_key,
                dry_run=dry_run,
                processing_mode=branch.processing_mode,
                kiev_tz=self._kiev_tz,
                branch_label=branch.name,
            )
        except Exception:
            record_branch_failed(branch_label=branch.name)
            record_docs_failed(count=1, date_key=date_key, branch_label=branch.name)
            log_docs_publish_summary(logger=self._logger, created=0, failed=1)
            raise
        doc_publish_ms: int = int(round((time.perf_counter() - doc_publish_started_at) * 1000.0))
        record_stage_duration(stage_name="doc_publish", elapsed_ms=doc_publish_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="doc_publish",
            elapsed_ms=doc_publish_ms,
            scope="branch",
            branch_label=branch.name,
            date_key=date_key,
        )
        docs_created_count: int = 0 if dry_run else 1
        record_docs_created(count=docs_created_count, date_key=date_key, branch_label=branch.name)
        log_docs_publish_summary(logger=self._logger, created=docs_created_count, failed=0)

        self._log_section(f"Publish Daily Telegram [{branch.name}]")
        telegram_publish_started_at: float = time.perf_counter()
        try:
            telegram_result = publish_daily_telegram(
                logger=self._logger,
                config=self._config,
                telegram_client=self._telegram_client,
                templates=self._config.templates,
                slot_results=slot_results,
                date_key=date_key,
                doc_url=doc_publish_result.doc_url,
                header_context=doc_publish_result.header_context,
                dry_run=dry_run,
                processing_mode=branch.processing_mode,
                branch_label=branch.name,
            )
        except Exception:
            record_branch_failed(branch_label=branch.name)
            record_telegram_failed(count=1, date_key=date_key, branch_label=branch.name)
            log_telegram_publish_summary(logger=self._logger, sent=0, failed=1, skipped=0)
            raise
        telegram_publish_ms: int = int(
            round((time.perf_counter() - telegram_publish_started_at) * 1000.0)
        )
        record_stage_duration(stage_name="telegram_publish", elapsed_ms=telegram_publish_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="telegram_publish",
            elapsed_ms=telegram_publish_ms,
            scope="branch",
            branch_label=branch.name,
            date_key=date_key,
        )
        record_telegram_sent(count=telegram_result.sent_count, date_key=date_key, branch_label=branch.name)
        record_telegram_failed(count=telegram_result.failed_count, date_key=date_key, branch_label=branch.name)
        record_telegram_skipped(count=telegram_result.skipped_count, date_key=date_key, branch_label=branch.name)
        log_telegram_publish_summary(
            logger=self._logger,
            sent=telegram_result.sent_count,
            failed=telegram_result.failed_count,
            skipped=telegram_result.skipped_count,
        )
        record_branch_completed(branch_label=branch.name)

    def _log_audit_branch_compare(self, *, date_key: str) -> None:
        merge_state = get_branch_date_summary(date_key=date_key, branch_label="merge")
        nomerge_state = get_branch_date_summary(date_key=date_key, branch_label="nomerge")
        if merge_state is None and nomerge_state is None:
            return
        comparison_status: str = (
            "complete" if merge_state is not None and nomerge_state is not None else "incomplete"
        )
        log_method = self._logger.info if comparison_status == "complete" else self._logger.debug
        log_method(
            "audit_branch_compare date_key=%s merge_branch_executed=%s nomerge_branch_executed=%s merge_doc_created=%s nomerge_doc_created=%s merge_telegram_sent=%s nomerge_telegram_sent=%s merge_contract_failures=%d nomerge_contract_failures=%d merge_recovered_degradations=%d nomerge_recovered_degradations=%d merge_models_used=%s nomerge_models_used=%s comparison_status=%s",
            date_key,
            "yes" if merge_state is not None else "no",
            "yes" if nomerge_state is not None else "no",
            "yes" if merge_state is not None and merge_state.docs_created > 0 else "no",
            "yes" if nomerge_state is not None and nomerge_state.docs_created > 0 else "no",
            "yes" if merge_state is not None and merge_state.telegram_sent > 0 else "no",
            "yes" if nomerge_state is not None and nomerge_state.telegram_sent > 0 else "no",
            merge_state.contract_failures if merge_state is not None else 0,
            nomerge_state.contract_failures if nomerge_state is not None else 0,
            merge_state.contract_recovered if merge_state is not None else 0,
            nomerge_state.contract_recovered if nomerge_state is not None else 0,
            ",".join(sorted(merge_state.models_used)) if merge_state is not None and merge_state.models_used else "none",
            ",".join(sorted(nomerge_state.models_used)) if nomerge_state is not None and nomerge_state.models_used else "none",
            comparison_status,
        )
