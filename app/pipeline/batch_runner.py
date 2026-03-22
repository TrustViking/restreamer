from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.core.env_flags import sheets_link_normalize_report_limit_from_env
from app.core.models import LanguageMergeAttempt, PlannedVideo, PreparedVideo
from app.ingest.youtube_metadata import YouTubeMetadataFetcher
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.llm.merges.merge_run_summary import MergeRunSummary
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
    record_branch_model_used,
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
from app.observability.startup_summary import LlmSummarySnapshot
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
from .slot_processing import SlotProcessResult, process_slot, resolve_merge_artifact_status


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


_MERGE_DOC_PUBLISHABLE_STATUSES: frozenset[str] = frozenset(
    {"full", "partial", "fallback_only"}
)


def _resolve_merge_doc_gate(
    *,
    branch_name: str,
    merge_artifact_status: str,
) -> tuple[bool, str]:
    if branch_name != BRANCH_MERGE:
        return True, "nomerge_branch_publish_mode"
    if merge_artifact_status in _MERGE_DOC_PUBLISHABLE_STATUSES:
        return True, f"{merge_artifact_status}_merge_artifact_present"
    return False, "no_publishable_merge_artifact"


def _resolve_merge_telegram_gate(
    *,
    branch_name: str,
    real_merge_blocks: int,
) -> tuple[bool, str]:
    if branch_name != BRANCH_MERGE:
        return True, "nomerge_branch_publish_mode"
    if real_merge_blocks > 0:
        return True, "real_merge_blocks_present"
    return False, "no_real_merge_blocks"


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
        if audit_mode == "audit":
            return [
                AuditBranch(name=BRANCH_NOMERGE, processing_mode="nomerge", llm_merge_enabled=False),
                AuditBranch(name=BRANCH_MERGE, processing_mode="merge", llm_merge_enabled=llm_merge_available),
            ]
        if audit_mode == "merge":
            return [AuditBranch(name=BRANCH_MERGE, processing_mode="merge", llm_merge_enabled=llm_merge_available)]
        return [AuditBranch(name=BRANCH_NOMERGE, processing_mode="nomerge", llm_merge_enabled=False)]

    def run(
        self,
        *,
        dry_run: bool,
        audit_mode: str,
        run_id: str,
        llm_summary: LlmSummarySnapshot,
    ) -> None:
        run_started_at: float = time.perf_counter()
        if not self._config.google_enabled:
            raise RuntimeError("Для batch режима GOOGLE_ENABLED должен быть включен.")

        merge_run_summary: MergeRunSummary = MergeRunSummary()
        self._last_merge_run_summary = merge_run_summary
        branch_failures: List[str] = []
        branches_for_log: str = ",".join(self._resolve_branch_labels_for_log(audit_mode=audit_mode))
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
                llm_summary=llm_summary,
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

    def _resolve_branch_labels_for_log(self, *, audit_mode: str) -> List[str]:
        return [branch.name for branch in self._resolve_audit_branches(audit_mode=audit_mode, llm_merge_available=True)]

    def _write_merge_reject_debug_artifacts(
        self,
        *,
        slot_results: List[SlotProcessResult],
        date_key: str,
        processing_mode: str,
        branch_label: str,
    ) -> None:
        if branch_label != BRANCH_MERGE:
            return
        for slot_result in slot_results:
            for language, merge_attempt in slot_result.merge_audit_by_language.items():
                if not self._should_write_merge_reject_debug_artifact(
                    merge_attempt=merge_attempt
                ):
                    continue
                source_count: int = len(slot_result.language_groups.get(language, ()))
                json_path = self._name_builder.build_merge_reject_debug_json_path(
                    date_key=date_key,
                    slot_key=slot_result.slot_key,
                    language=language,
                    processing_mode=processing_mode,
                    source_count=source_count,
                )
                if json_path is None:
                    continue
                artifact_payload: Dict[str, Any] = {
                    "case_metadata": {
                        "date_key": date_key,
                        "slot_key": slot_result.slot_key,
                        "language": language,
                        "source_count": source_count,
                        "merge_mode": "expanded" if source_count >= 3 else "compact",
                        "processing_mode": processing_mode,
                        "publish_source_label": str(
                            merge_attempt.publish_source_label or ""
                        ).strip(),
                    },
                    "attempts": [
                        {
                            "attempt_index": rejected_attempt.attempt_index,
                            "model": rejected_attempt.model_name,
                            "reject_reasons": list(rejected_attempt.reject_reasons),
                            "title": rejected_attempt.title,
                            "description": rejected_attempt.description,
                            "raw_response_text": rejected_attempt.raw_response_text,
                        }
                        for rejected_attempt in merge_attempt.rejected_attempts
                    ],
                }
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(
                    json.dumps(artifact_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._logger.info(
                    '[%s] merge_reject_debug_json_written date_key=%s slot_key=%s language=%s attempts=%d path="%s"',
                    branch_label,
                    date_key,
                    slot_result.slot_key,
                    language,
                    len(merge_attempt.rejected_attempts),
                    str(json_path),
                )

    def _should_write_merge_reject_debug_artifact(
        self,
        *,
        merge_attempt: LanguageMergeAttempt,
    ) -> bool:
        return (
            str(merge_attempt.publish_source_label or "").strip() == "merge_failed"
            and bool(merge_attempt.rejected_attempts)
        )

    def _prepare_run_context(
        self,
        *,
        services: BatchServices,
        dry_run: bool,
        audit_mode: str,
        run_id: str,
        llm_summary: LlmSummarySnapshot,
    ) -> PreparedRunContext:
        self._log_section("Startup Health Check")
        startup_health_started_at: float = time.perf_counter()
        llm_merge_available: bool = run_startup_health_checks(
            logger=self._logger,
            config=self._config,
            llm_summary=llm_summary,
            services=services,
            telegram_client=self._telegram_client,
            resolve_logger_name_meta=self._resolve_logger_name_meta,
            dry_run=dry_run,
            resolved_audit_mode=audit_mode,
            run_id=run_id,
        )
        startup_health_ms: int = int(round((time.perf_counter() - startup_health_started_at) * 1000.0))
        record_stage_duration(stage_name="startup_health", elapsed_ms=startup_health_ms)
        log_stage_timing(logger=self._logger, stage_name="startup_health", elapsed_ms=startup_health_ms, scope="run")

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
        log_stage_timing(logger=self._logger, stage_name="sheet_load", elapsed_ms=sheet_load_ms, scope="run")
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
        shared_preparation_ms: int = int(round((time.perf_counter() - shared_preparation_started_at) * 1000.0))
        record_stage_duration(stage_name="shared_preparation", elapsed_ms=shared_preparation_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="shared_preparation",
            elapsed_ms=shared_preparation_ms,
            scope="run",
        )

        rows_skipped: int = max(0, len(sheet_state.rows) - len(prepared_videos))
        prepared_dates_count: int = len({item.date_key for item in prepared_videos})
        prepared_slots_count: int = len({f"{item.date_key}_{item.scheduled_at_kiev.strftime('%H%M')}" for item in prepared_videos})
        record_planning_completed(planned_items=len(prepared_videos), rows_skipped=rows_skipped)
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
        log_stage_timing(logger=self._logger, stage_name="planning", elapsed_ms=planning_ms, scope="run")
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
        date_keys: List[str] = sorted({date_key for branch_videos in processed_by_branch.values() for date_key in branch_videos.keys()})
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
            self._logger.info("audit_branch_skip branch=%s date_key=%s reason=empty_branch_items", branch.name, date_key)
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
            record_branch_total_ms(date_key=date_key, branch_label=branch.name, elapsed_ms=branch_total_ms)
            log_stage_timing(
                logger=self._logger,
                stage_name="branch_total",
                elapsed_ms=branch_total_ms,
                scope="branch",
                branch_label=branch.name,
                date_key=date_key,
            )
            self._logger.info("audit_branch_done branch=%s status=ok date_key=%s elapsed_ms=%d", branch.name, date_key, branch_total_ms)
            if audit_mode == "audit":
                self._log_audit_branch_compare(date_key=date_key)
        except LlmModelConfigurationError as error:
            branch_failures.append(f"branch={branch.name} date={date_key} failed: {error}")
            record_branch_failed(branch_label=branch.name)
            self._logger.error(
                "audit_branch_done branch=%s status=fatal_model_config date_key=%s reason_code=%s reason=%s",
                branch.name,
                date_key,
                error.reason_code,
                error.detail,
            )
            raise
        except Exception as error:
            branch_failures.append(f"branch={branch.name} date={date_key} failed: {error}")
            self._logger.error("audit_branch_done branch=%s status=failed date_key=%s reason=%s", branch.name, date_key, error)
            log_error_event(self._logger, "branch=%s date=%s failed: %s", branch.name, date_key, error, reason_code="branch_date_failed")

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
        slots_by_time: Dict[str, List[PlannedVideo]] = self._group_date_videos_by_slot_time(date_videos_all=date_videos_all)
        self._logger.info("[%s] Date branch started: %s slots=%d items=%d", branch.name, date_key, len(slots_by_time), len(date_videos_all))
        record_branch_started(branch_label=branch.name)
        record_date_branch_execution(date_key=date_key, branch_label=branch.name)
        log_date_started(logger=self._logger, date_key=f"{date_key}/{branch.name}", slot_count=len(slots_by_time), item_count=len(date_videos_all))

        merge_snapshot_before: tuple[int, int, int, int, int, int, int, int, int] = (
            merge_run_summary.merge_success,
            merge_run_summary.validation_rejected,
            merge_run_summary.retry_used,
            merge_run_summary.final_failure,
            merge_run_summary.paragraph_recovery_used,
            merge_run_summary.real_merge_blocks,
            merge_run_summary.merge_candidate_blocks,
            merge_run_summary.fallback_merge_blocks,
            merge_run_summary.partial_merge_artifacts,
        )
        slot_results: List[SlotProcessResult] = []
        for slot_time_key in sorted(slots_by_time.keys()):
            try:
                processed_slot_result = process_slot(
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
            except LlmModelConfigurationError as error:
                self._logger.error(
                    "[%s] slot_process_fatal_model_config date_key=%s slot_key=%s reason_code=%s model=%s provider=%s reason=%s",
                    branch.name,
                    date_key,
                    f"{date_key}_{slot_time_key}",
                    error.reason_code,
                    error.model_name or "unknown",
                    error.provider_name or "unknown",
                    error.detail,
                )
                raise
            for merge_attempt in processed_slot_result.merge_audit_by_language.values():
                if str(merge_attempt.model_name or "").strip():
                    record_branch_model_used(date_key=date_key, branch_label=branch.name, model_name=merge_attempt.model_name)
            slot_results.append(processed_slot_result)
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

        real_merge_blocks: int = sum(slot.real_merge_blocks for slot in slot_results)
        merge_candidate_blocks: int = sum(slot.merge_candidate_blocks for slot in slot_results)
        fallback_merge_blocks: int = sum(slot.fallback_merge_blocks for slot in slot_results)
        merge_artifact_status: str = resolve_merge_artifact_status(
            merge_candidate_blocks=merge_candidate_blocks,
            real_merge_blocks=real_merge_blocks,
            fallback_merge_blocks=fallback_merge_blocks,
        )
        fallback_merge_targets: List[str] = [
            target
            for slot in slot_results
            for target in slot.fallback_merge_targets
        ]
        if branch.name == BRANCH_MERGE:
            if merge_artifact_status == "full":
                merge_run_summary.record_full_merge_artifact()
            elif merge_artifact_status == "partial":
                merge_run_summary.record_partial_merge_artifact()
        merge_snapshot_after: tuple[int, int, int, int, int, int, int, int, int] = (
            merge_run_summary.merge_success,
            merge_run_summary.validation_rejected,
            merge_run_summary.retry_used,
            merge_run_summary.final_failure,
            merge_run_summary.paragraph_recovery_used,
            merge_run_summary.real_merge_blocks,
            merge_run_summary.merge_candidate_blocks,
            merge_run_summary.fallback_merge_blocks,
            merge_run_summary.partial_merge_artifacts,
        )
        log_merge_summary(
            logger=self._logger,
            groups=len(slot_results),
            merge_success=merge_snapshot_after[0] - merge_snapshot_before[0],
            validation_rejected=merge_snapshot_after[1] - merge_snapshot_before[1],
            retry_used=merge_snapshot_after[2] - merge_snapshot_before[2],
            final_failure=merge_snapshot_after[3] - merge_snapshot_before[3],
            paragraph_recovery_used=merge_snapshot_after[4] - merge_snapshot_before[4],
            real_merge_blocks=merge_snapshot_after[5] - merge_snapshot_before[5],
            merge_candidate_blocks=merge_snapshot_after[6] - merge_snapshot_before[6],
            fallback_merge_blocks=merge_snapshot_after[7] - merge_snapshot_before[7],
            partial_merge_artifacts=merge_snapshot_after[8] - merge_snapshot_before[8],
        )
        self._logger.info(
            "[%s] merge_artifact_observability date_key=%s merge_artifact_status=%s merged_blocks=%d fallback_blocks=%d merge_candidate_blocks=%d fallback_targets=%s",
            branch.name,
            date_key,
            merge_artifact_status,
            real_merge_blocks,
            fallback_merge_blocks,
            merge_candidate_blocks,
            ",".join(fallback_merge_targets) or "none",
        )
        self._write_merge_reject_debug_artifacts(
            slot_results=slot_results,
            date_key=date_key,
            processing_mode=branch.processing_mode,
            branch_label=branch.name,
        )
        merge_doc_allowed, merge_doc_reason = _resolve_merge_doc_gate(
            branch_name=branch.name,
            merge_artifact_status=merge_artifact_status,
        )
        self._logger.info(
            "[%s] merge_doc_decision date_key=%s had_real_merge_blocks=%s real_merge_blocks=%d fallback_merge_blocks=%d merge_artifact_status=%s merge_doc_allowed=%s reason=%s",
            branch.name,
            date_key,
            "yes" if real_merge_blocks > 0 else "no",
            real_merge_blocks,
            fallback_merge_blocks,
            merge_artifact_status,
            "yes" if merge_doc_allowed else "no",
            merge_doc_reason,
        )
        if not merge_doc_allowed:
            self._logger.info(
                "[%s] merge_doc_result date_key=%s merge_doc_created=no reason=%s",
                branch.name,
                date_key,
                merge_doc_reason,
            )
            log_docs_publish_summary(logger=self._logger, created=0, failed=0)
            log_telegram_publish_summary(logger=self._logger, sent=0, failed=0, skipped=1)
            record_telegram_skipped(count=1, date_key=date_key, branch_label=branch.name)
            record_branch_completed(branch_label=branch.name)
            self._logger.info(
                "[%s] merge_artifact_skipped date_key=%s reason=%s",
                branch.name,
                date_key,
                merge_doc_reason,
            )
            return

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
        log_stage_timing(logger=self._logger, stage_name="doc_publish", elapsed_ms=doc_publish_ms, scope="branch", branch_label=branch.name, date_key=date_key)
        docs_created_count: int = 0 if dry_run else 1
        self._logger.info(
            "[%s] merge_doc_result date_key=%s merge_doc_created=%s reason=%s",
            branch.name,
            date_key,
            "yes" if doc_publish_result.google_doc_created else "no",
            "published" if doc_publish_result.google_doc_created else ("dry_run" if dry_run else "not_created"),
        )
        record_docs_created(count=docs_created_count, date_key=date_key, branch_label=branch.name)
        log_docs_publish_summary(logger=self._logger, created=docs_created_count, failed=0)

        merge_telegram_allowed, merge_telegram_reason = _resolve_merge_telegram_gate(
            branch_name=branch.name,
            real_merge_blocks=real_merge_blocks,
        )
        self._logger.info(
            "[%s] merge_telegram_decision date_key=%s had_real_merge_blocks=%s real_merge_blocks=%d fallback_merge_blocks=%d merge_artifact_status=%s merge_telegram_allowed=%s reason=%s",
            branch.name,
            date_key,
            "yes" if real_merge_blocks > 0 else "no",
            real_merge_blocks,
            fallback_merge_blocks,
            merge_artifact_status,
            "yes" if merge_telegram_allowed else "no",
            merge_telegram_reason,
        )
        if not merge_telegram_allowed:
            self._logger.info(
                "[%s] merge_telegram_result date_key=%s telegram_sent=no reason=%s",
                branch.name,
                date_key,
                merge_telegram_reason,
            )
            log_telegram_publish_summary(
                logger=self._logger,
                sent=0,
                failed=0,
                skipped=1,
            )
            record_telegram_skipped(
                count=1,
                date_key=date_key,
                branch_label=branch.name,
            )
            record_branch_completed(branch_label=branch.name)
            return

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
        telegram_publish_ms: int = int(round((time.perf_counter() - telegram_publish_started_at) * 1000.0))
        record_stage_duration(stage_name="telegram_publish", elapsed_ms=telegram_publish_ms)
        log_stage_timing(logger=self._logger, stage_name="telegram_publish", elapsed_ms=telegram_publish_ms, scope="branch", branch_label=branch.name, date_key=date_key)
        record_telegram_sent(count=telegram_result.sent_count, date_key=date_key, branch_label=branch.name)
        record_telegram_failed(count=telegram_result.failed_count, date_key=date_key, branch_label=branch.name)
        record_telegram_skipped(count=telegram_result.skipped_count, date_key=date_key, branch_label=branch.name)
        log_telegram_publish_summary(logger=self._logger, sent=telegram_result.sent_count, failed=telegram_result.failed_count, skipped=telegram_result.skipped_count)
        record_branch_completed(branch_label=branch.name)

    def _log_audit_branch_compare(self, *, date_key: str) -> None:
        merge_state = get_branch_date_summary(date_key=date_key, branch_label=BRANCH_MERGE)
        nomerge_state = get_branch_date_summary(date_key=date_key, branch_label=BRANCH_NOMERGE)
        if merge_state is None and nomerge_state is None:
            return
        comparison_status: str = "complete" if merge_state is not None and nomerge_state is not None else "incomplete"
        log_method = self._logger.info if comparison_status == "complete" else self._logger.debug
        log_method(
            "audit_branch_compare date_key=%s merge_executed=%s nomerge_executed=%s merge_doc_created=%s nomerge_doc_created=%s merge_telegram_sent=%s nomerge_telegram_sent=%s merge_contract_failures=%d nomerge_contract_failures=%d merge_models_used=%s nomerge_models_used=%s comparison_status=%s",
            date_key,
            "yes" if merge_state is not None else "no",
            "yes" if nomerge_state is not None else "no",
            "yes" if merge_state is not None and merge_state.docs_created > 0 else "no",
            "yes" if nomerge_state is not None and nomerge_state.docs_created > 0 else "no",
            "yes" if merge_state is not None and merge_state.telegram_sent > 0 else "no",
            "yes" if nomerge_state is not None and nomerge_state.telegram_sent > 0 else "no",
            merge_state.contract_failures if merge_state is not None else 0,
            nomerge_state.contract_failures if nomerge_state is not None else 0,
            ",".join(sorted(merge_state.models_used)) if merge_state is not None and merge_state.models_used else "none",
            ",".join(sorted(nomerge_state.models_used)) if nomerge_state is not None and nomerge_state.models_used else "none",
            comparison_status,
        )
