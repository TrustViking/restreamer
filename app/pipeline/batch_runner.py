from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.core.env_flags import sheets_link_normalize_report_limit_from_env
from app.core.models import PlannedVideo, PreparedVideo
from app.core.text_utils import format_date_key_for_display
from app.ingest.youtube_metadata import YouTubeMetadataFetcher
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.net.http_client import HttpClient
from app.observability.runtime_analytics import (
    log_error_event,
    log_planning_completed,
    log_sheet_loaded,
    log_stage_timing,
    log_warning_operational,
    record_branch_failed,
    record_branch_total_ms,
    record_planning_completed,
    record_sheet_loaded,
    record_stage_duration,
)
from app.observability.startup_health import (
    LoggerNameMetaResolver,
    log_section,
    run_startup_health_checks,
)
from app.observability.startup_summary import LlmSummarySnapshot
from app.paths.name_builder import NamePathBuilder
from app.planning import log_link_normalization_report
from app.planning.batch_planner import (
    build_prepared_videos,
    derive_planned_videos,
    materialize_prepared_previews,
)
from app.planning.sheet_loader import BatchSheetState, load_sheet_state
from app.telegram.bot_client import TelegramBotClient

from .branch_executor import BranchExecutor
from .debug_artifacts import DebugArtifactWriter
from .operator_notifier import OperatorNotifier
from .runtime_services import BatchServices, build_runtime_services


@dataclass(frozen=True)
class AuditBranch:
    name: str
    processing_mode: str
    llm_merge_enabled: bool


@dataclass(frozen=True)
class PreparedRunContext:
    services: BatchServices
    sheet_state: BatchSheetState
    prepared_videos: list[PreparedVideo]
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
        resolve_logger_name_meta: LoggerNameMetaResolver,
        notifier: OperatorNotifier,
    ) -> None:
        self._logger: logging.Logger = logger
        self._config: AppConfig = config
        self._metadata_fetcher: YouTubeMetadataFetcher = metadata_fetcher
        self._http_client: HttpClient = http_client
        self._telegram_client: TelegramBotClient = telegram_client
        self._name_builder: NamePathBuilder = name_builder
        self._kiev_tz: ZoneInfo = kiev_tz
        self._cet_tz: ZoneInfo = cet_tz
        self._resolve_logger_name_meta: LoggerNameMetaResolver = resolve_logger_name_meta
        self._notifier: OperatorNotifier = notifier
        self._last_merge_run_summary: MergeRunSummary | None = None
        self._debug_writer: DebugArtifactWriter = DebugArtifactWriter(
            logger=logger,
            name_builder=name_builder,
        )
        self._branch_executor = BranchExecutor(
            logger=logger,
            config=config,
            telegram_client=telegram_client,
            name_builder=name_builder,
            kiev_tz=kiev_tz,
            cet_tz=cet_tz,
            debug_artifact_writer=self._debug_writer,
            notifier=notifier,
        )

    @property
    def last_merge_run_summary(self) -> MergeRunSummary | None:
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
        processed: list[PlannedVideo],
    ) -> dict[str, list[PlannedVideo]]:
        videos_by_date: dict[str, list[PlannedVideo]] = {}
        for item in processed:
            videos_by_date.setdefault(item.date_key, []).append(item)
        return videos_by_date

    def _resolve_audit_branches(
        self,
        *,
        audit_mode: str,
        llm_merge_available: bool,
    ) -> list[AuditBranch]:
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
        if not self._config.google.enabled:
            raise RuntimeError("Для batch режима GOOGLE_ENABLED должен быть включен.")

        merge_run_summary: MergeRunSummary = MergeRunSummary()
        self._last_merge_run_summary = merge_run_summary
        branch_failures: list[str] = []
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

            branches: list[AuditBranch] = self._resolve_audit_branches(
                audit_mode=audit_mode,
                llm_merge_available=prepared_context.llm_merge_available,
            )
            processed_by_branch: dict[str, dict[str, list[PlannedVideo]]] = (
                self._build_processed_by_branch(
                    branches=branches,
                    prepared_videos=prepared_context.prepared_videos,
                )
            )

            self._log_section("Process Slots")
            self._notifier.emit("✅ Обработка слотов...", to_telegram=not dry_run)
            self._execute_branch_dates(
                services=services,
                branches=branches,
                processed_by_branch=processed_by_branch,
                dry_run=dry_run,
                merge_run_summary=merge_run_summary,
                audit_mode=audit_mode,
                branch_failures=branch_failures,
                run_id=run_id,
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
            if branch_failures:
                overall_status = "failed"
            elif self._last_merge_run_summary is not None and self._last_merge_run_summary.final_failure > 0:
                overall_status = "partial"
            else:
                overall_status = "ok"
            self._logger.info(
                "audit_done audit_mode=%s overall_status=%s run_id=%s",
                audit_mode,
                overall_status,
                run_id,
            )

    def _resolve_branch_labels_for_log(self, *, audit_mode: str) -> list[str]:
        return [
            branch.name
            for branch in self._resolve_audit_branches(
                audit_mode=audit_mode,
                llm_merge_available=True,
            )
        ]

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
        self._notifier.emit(
            "✅ Проверка сервисов и конфигурации: OK",
            to_telegram=not dry_run,
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
        log_stage_timing(logger=self._logger, stage_name="sheet_load", elapsed_ms=sheet_load_ms, scope="run")
        record_sheet_loaded(rows=len(sheet_state.rows))
        log_sheet_loaded(logger=self._logger, rows=len(sheet_state.rows))
        self._notifier.emit(
            f"✅ Загружено строк из таблицы: {len(sheet_state.rows)}",
            to_telegram=not dry_run,
        )

        self._log_section("Shared Preparation")
        shared_preparation_started_at: float = time.perf_counter()
        prepared_videos: list[PreparedVideo] = build_prepared_videos(
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
        self._notifier.emit(
            f"✅ Подготовлено ссылок видео: {len(prepared_videos)}",
            to_telegram=not dry_run,
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
        branches: list[AuditBranch],
        prepared_videos: list[PreparedVideo],
    ) -> dict[str, dict[str, list[PlannedVideo]]]:
        processed_by_branch: dict[str, dict[str, list[PlannedVideo]]] = {}
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
        branches: list[AuditBranch],
        processed_by_branch: dict[str, dict[str, list[PlannedVideo]]],
        dry_run: bool,
        merge_run_summary: MergeRunSummary,
        audit_mode: str,
        branch_failures: list[str],
        run_id: str = "",
    ) -> None:
        date_keys: list[str] = sorted(
            {
                date_key
                for branch_videos in processed_by_branch.values()
                for date_key in branch_videos.keys()
            }
        )
        stage_count: int = len(branches)
        for date_key in date_keys:
            failures_before: int = len(branch_failures)
            merge_failures_before: int = merge_run_summary.final_failure
            for stage_index, branch in enumerate(branches, start=1):
                self._execute_single_branch_date(
                    services=services,
                    branch=branch,
                    date_key=date_key,
                    processed_by_branch=processed_by_branch,
                    dry_run=dry_run,
                    merge_run_summary=merge_run_summary,
                    audit_mode=audit_mode,
                    branch_failures=branch_failures,
                    run_id=run_id,
                    stage_index=stage_index,
                    stage_count=stage_count,
                )
            has_date_failures: bool = (
                len(branch_failures) > failures_before
                or merge_run_summary.final_failure > merge_failures_before
            )
            if has_date_failures:
                date_done_message: str = (
                    f"⚠️ Дата {format_date_key_for_display(date_key)} завершена частично"
                )
            else:
                date_done_message = f"✅ Дата {format_date_key_for_display(date_key)} завершена"
            self._notifier.emit(
                date_done_message,
                to_telegram=not dry_run,
            )

    def _execute_single_branch_date(
        self,
        *,
        services: BatchServices,
        branch: AuditBranch,
        date_key: str,
        processed_by_branch: dict[str, dict[str, list[PlannedVideo]]],
        dry_run: bool,
        merge_run_summary: MergeRunSummary,
        audit_mode: str,
        branch_failures: list[str],
        run_id: str = "",
        stage_index: int,
        stage_count: int,
    ) -> None:
        date_videos_all: list[PlannedVideo] = processed_by_branch.get(
            branch.name,
            {},
        ).get(date_key, [])
        if not date_videos_all:
            self._logger.info("audit_branch_skip branch=%s date_key=%s reason=empty_branch_items", branch.name, date_key)
            return
        branch_started_at: float = time.perf_counter()
        self._logger.info("audit_branch_start branch=%s", branch.name)
        try:
            self._branch_executor.execute(
                services=services,
                branch=branch,
                date_key=date_key,
                date_videos=date_videos_all,
                dry_run=dry_run,
                merge_run_summary=merge_run_summary,
                run_id=run_id,
                stage_index=stage_index,
                stage_count=stage_count,
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
            branch_status: str = "ok"
            if branch.name == BRANCH_MERGE and merge_run_summary.final_failure > 0:
                branch_status = "partial"
            self._logger.info(
                "audit_branch_done branch=%s status=%s date_key=%s elapsed_ms=%d",
                branch.name,
                branch_status,
                date_key,
                branch_total_ms,
            )
            if audit_mode == "audit":
                self._debug_writer.log_audit_branch_compare(date_key=date_key)
        except LlmModelConfigurationError as error:
            branch_failures.append(f"branch={branch.name} date={date_key} failed: {error}")
            record_branch_failed(branch_label=branch.name)
            self._notifier.emit(
                f"❌ Ошибка ветки {branch.name}, дата {format_date_key_for_display(date_key)}",
                to_telegram=not dry_run,
            )
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
            self._notifier.emit(
                f"❌ Ошибка ветки {branch.name}, дата {format_date_key_for_display(date_key)}",
                to_telegram=not dry_run,
            )
            self._logger.error("audit_branch_done branch=%s status=failed date_key=%s reason=%s", branch.name, date_key, error)
            log_error_event(self._logger, "branch=%s date=%s failed: %s", branch.name, date_key, error, reason_code="branch_date_failed")

