from __future__ import annotations

import logging
import time
from typing import Optional

from app.bootstrap.run_context import RunContext
# Backward-compatible re-exports — DO NOT REMOVE
from app.observability.analytics_formatters import (  # noqa: F401
    _branch_date_key,
    _format_branch_summary,
    _format_reason_counts,
    _has_failed_branch,
    _resolve_run_status,
)
from app.observability.analytics_state import (  # noqa: F401
    WARNING_CATEGORY_INFORMATIONAL,
    WARNING_CATEGORY_OPERATIONAL,
    BranchAnalyticsState,
    BranchDateAnalyticsState,
    RuntimeAnalyticsState,
)


class _RuntimeAnalyticsCounter(logging.Handler):
    def __init__(self, state: RuntimeAnalyticsState) -> None:
        super().__init__(level=logging.WARNING)
        self._state: RuntimeAnalyticsState = state

    def emit(self, record: logging.LogRecord) -> None:
        reason_code: str = str(getattr(record, "reason_code", "") or "").strip()
        if record.levelno >= logging.ERROR:
            self._state.errors += 1
            if reason_code:
                self._state.error_reason_counts[reason_code] = (
                    self._state.error_reason_counts.get(reason_code, 0) + 1
                )
            return
        if record.levelno >= logging.WARNING:
            warning_category: str = str(
                getattr(record, "warning_category", WARNING_CATEGORY_OPERATIONAL)
                or WARNING_CATEGORY_OPERATIONAL
            ).strip().lower()
            if reason_code:
                self._state.warning_reason_counts[reason_code] = (
                    self._state.warning_reason_counts.get(reason_code, 0) + 1
                )
            if warning_category == WARNING_CATEGORY_INFORMATIONAL:
                self._state.warnings_informational += 1
                return
            self._state.warnings_operational += 1


def log_warning_operational(
    logger: logging.Logger,
    message: str,
    *args: object,
    reason_code: Optional[str] = None,
) -> None:
    logger.warning(
        message,
        *args,
        extra={
            "warning_category": WARNING_CATEGORY_OPERATIONAL,
            "reason_code": str(reason_code or "").strip(),
        },
    )


def log_warning_informational(
    logger: logging.Logger,
    message: str,
    *args: object,
    reason_code: Optional[str] = None,
) -> None:
    logger.warning(
        message,
        *args,
        extra={
            "warning_category": WARNING_CATEGORY_INFORMATIONAL,
            "reason_code": str(reason_code or "").strip(),
        },
    )


def log_error_event(
    logger: logging.Logger,
    message: str,
    *args: object,
    reason_code: Optional[str] = None,
) -> None:
    logger.error(
        message,
        *args,
        extra={"reason_code": str(reason_code or "").strip()},
    )


class RuntimeAnalyticsCollector:
    def __init__(self, *, debug_enabled: bool) -> None:
        self._state: RuntimeAnalyticsState = RuntimeAnalyticsState(debug_enabled=debug_enabled)
        self._state.malformed_tail_cleanup_keys = set()
        self._handler: _RuntimeAnalyticsCounter = _RuntimeAnalyticsCounter(self._state)

    @property
    def state(self) -> RuntimeAnalyticsState:
        return self._state

    @property
    def handler(self) -> logging.Handler:
        return self._handler

    def _get_branch_date_state(
        self,
        *,
        date_key: str,
        branch_label: str,
    ) -> BranchDateAnalyticsState:
        state_key: str = _branch_date_key(date_key=date_key, branch_label=branch_label)
        return self._state.branch_date_results.setdefault(state_key, BranchDateAnalyticsState())

    def record_stage_duration(self, *, stage_name: str, elapsed_ms: int) -> None:
        self._state.stage_durations_ms[stage_name] = max(
            0,
            self._state.stage_durations_ms.get(stage_name, 0) + int(elapsed_ms),
        )

    def log_stage_timing(
        self,
        *,
        logger: logging.Logger,
        stage_name: str,
        elapsed_ms: int,
        scope: str = "run",
        branch_label: str = "n/a",
        date_key: str = "n/a",
        slot_key: str = "n/a",
    ) -> None:
        logger.info(
            "stage_timing stage=%s scope=%s branch=%s date_key=%s slot_key=%s elapsed_ms=%d",
            stage_name,
            scope,
            branch_label,
            date_key,
            slot_key,
            elapsed_ms,
        )

    def record_slot_total_ms(self, *, slot_key: str, elapsed_ms: int) -> None:
        self._state.slot_total_ms_by_key[slot_key] = int(elapsed_ms)

    def record_branch_total_ms(self, *, date_key: str, branch_label: str, elapsed_ms: int) -> None:
        branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        branch_date_state.branch_total_ms = int(elapsed_ms)

    def record_branch_model_used(self, *, date_key: str, branch_label: str, model_name: str) -> None:
        branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        cleaned_model_name: str = str(model_name or "").strip()
        if cleaned_model_name:
            branch_date_state.models_used.add(cleaned_model_name)

    def record_contract_result(
        self,
        *,
        date_key: str,
        branch_label: str,
        contract_status: str,
    ) -> None:
        branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        normalized_status: str = str(contract_status or "").strip().lower()
        if normalized_status == "degraded_recovered":
            self._state.degraded_recovered_count += 1
            self._state.content_contract_recovered += 1
            branch_date_state.contract_recovered += 1
            return
        if normalized_status in {"degraded_unrecovered", "fail"}:
            self._state.degraded_unrecovered_count += 1
            self._state.content_contract_failures += 1
            self._state.content_contract_unrecovered += 1
            branch_date_state.contract_failures += 1
            branch_date_state.contract_unrecovered += 1

    def get_branch_date_summary(
        self,
        *,
        date_key: str,
        branch_label: str,
    ) -> Optional[BranchDateAnalyticsState]:
        return self._state.branch_date_results.get(
            _branch_date_key(date_key=date_key, branch_label=branch_label)
        )

    def record_sheet_loaded(self, *, rows: int) -> None:
        self._state.rows_loaded = rows

    def record_planning_completed(self, *, planned_items: int, rows_skipped: int) -> None:
        self._state.planned_items = planned_items
        self._state.rows_skipped = max(0, rows_skipped)

    def record_date_processed(self) -> None:
        self._state.date_branch_executions += 1

    def record_date_branch_execution(self, *, date_key: str, branch_label: str) -> None:
        self._state.unique_dates_processed.add(str(date_key))
        self._state.date_branch_executions += 1

    def record_branch_started(self, *, branch_label: str) -> None:
        branch_state: BranchAnalyticsState = self._state.branch_results.setdefault(
            branch_label,
            BranchAnalyticsState(),
        )
        branch_state.started = True

    def record_branch_completed(self, *, branch_label: str) -> None:
        branch_state: BranchAnalyticsState = self._state.branch_results.setdefault(
            branch_label,
            BranchAnalyticsState(),
        )
        branch_state.started = True
        branch_state.completed = True

    def record_branch_failed(self, *, branch_label: str) -> None:
        branch_state: BranchAnalyticsState = self._state.branch_results.setdefault(
            branch_label,
            BranchAnalyticsState(),
        )
        branch_state.started = True
        branch_state.failed = True

    def record_docs_created(self, *, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
        self._state.docs_created += max(0, count)
        if date_key and branch_label:
            branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
                date_key=date_key,
                branch_label=branch_label,
            )
            branch_date_state.docs_created += max(0, count)

    def record_docs_failed(self, *, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
        self._state.docs_failed += max(0, count)
        if date_key and branch_label:
            branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
                date_key=date_key,
                branch_label=branch_label,
            )
            branch_date_state.docs_failed += max(0, count)

    def record_telegram_sent(self, *, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
        self._state.telegram_sent += max(0, count)
        if date_key and branch_label:
            branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
                date_key=date_key,
                branch_label=branch_label,
            )
            branch_date_state.telegram_sent += max(0, count)

    def record_telegram_failed(self, *, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
        self._state.telegram_failed += max(0, count)
        if date_key and branch_label:
            branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
                date_key=date_key,
                branch_label=branch_label,
            )
            branch_date_state.telegram_failed += max(0, count)

    def record_telegram_skipped(self, *, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
        self._state.telegram_skipped += max(0, count)
        if date_key and branch_label:
            branch_date_state: BranchDateAnalyticsState = self._get_branch_date_state(
                date_key=date_key,
                branch_label=branch_label,
            )
            branch_date_state.telegram_skipped += max(0, count)

    def record_merge_final_failure(self, *, count: int = 1) -> None:
        self._state.merge_final_failure += max(0, count)

    def record_merge_validation_rejected(self, *, count: int = 1) -> None:
        self._state.merge_validation_rejected += max(0, count)

    def record_publish_gate_blocked(self, *, language: str, target: str) -> None:
        # publish_gate_blocked_count считает срабатывания publish-gate ПО target-каналам.
        # Один язык, заблокированный и в doc, и в telegram, даёт count=2 при единственном
        # языке в publish_gate_blocked_languages — это by design, не баг отчётности.
        self._state.publish_gate_blocked_count += 1
        normalized_language: str = str(language or "unknown").strip().lower() or "unknown"
        self._state.publish_gate_blocked_languages.add(normalized_language)

    def record_malformed_tail_url_cleanup(self, count: int, *, event_key: Optional[str] = None) -> None:
        if (
            event_key
            and self._state.malformed_tail_cleanup_keys is not None
            and event_key in self._state.malformed_tail_cleanup_keys
        ):
            return
        if event_key and self._state.malformed_tail_cleanup_keys is not None:
            self._state.malformed_tail_cleanup_keys.add(event_key)
        self._state.malformed_tail_url_fragments_dropped += max(0, count)
    def log_run_started(
        self,
        *,
        logger: logging.Logger,
        processing_mode: str,
        audit_mode: str,
        debug_enabled: bool,
        dry_run: bool,
    ) -> None:
        logger.info(
            "Run started: processing_mode=%s audit_mode=%s debug=%s dry_run=%s",
            processing_mode,
            audit_mode,
            "yes" if debug_enabled else "no",
            "yes" if dry_run else "no",
        )

    def log_run_context(
        self,
        logger: logging.Logger,
        run_context: RunContext,
    ) -> None:
        logger.info(
            "run_context run_id=%s processing_mode=%s audit_mode=%s config_processing_mode=%s audit_branches=%s debug=%s dry_run=%s google_enabled=%s telegram_enabled=%s llm_provider=%s llm_model_effective=%s llm_model_configured=%s llm_provider_model=%s llm_usage_reporting_mode=%s sheet_id=%s sheet_range=%s sheets_link_writeback=%s local_doc_export_enabled=%s strip_chapter_timestamps=%s",
            run_context.run_id,
            run_context.processing_mode,
            run_context.audit_mode,
            run_context.config_processing_mode or "default",
            ",".join(run_context.audit_branches) if run_context.audit_branches else "none",
            "yes" if run_context.debug_enabled else "no",
            "yes" if run_context.dry_run else "no",
            "yes" if run_context.google_enabled else "no",
            "yes" if run_context.telegram_enabled else "no",
            run_context.llm_provider or "unknown",
            run_context.llm_model or "unknown",
            run_context.llm_model_configured or "unknown",
            run_context.llm_provider_model or "unknown",
            run_context.llm_usage_reporting_mode or "unknown",
            run_context.sheet_id or "unknown",
            run_context.sheet_range or "unknown",
            "yes" if run_context.sheets_link_writeback else "no",
            "yes" if run_context.local_doc_export_enabled else "no",
            "yes" if run_context.strip_chapter_timestamps else "no",
        )

    def log_preflight_status(self, *, logger: logging.Logger, self_check_skipped: bool) -> None:
        logger.info(
            "Preflight completed: self_check=%s",
            "skipped" if self_check_skipped else "completed",
        )

    def log_sheet_loaded(self, *, logger: logging.Logger, rows: int) -> None:
        logger.info("Sheet loaded: rows=%d", rows)

    def log_planning_completed(
        self,
        *,
        logger: logging.Logger,
        planned_items: int,
        rows_skipped: int,
        dates: int,
        slots: int,
    ) -> None:
        logger.info(
            "Planning completed: planned_items=%d skipped_rows=%d dates=%d slots=%d",
            planned_items,
            rows_skipped,
            dates,
            slots,
        )

    def log_date_started(
        self,
        *,
        logger: logging.Logger,
        date_key: str,
        slot_count: int,
        item_count: int,
    ) -> None:
        logger.info(
            "Date started: %s slots=%d items=%d",
            date_key,
            slot_count,
            item_count,
        )

    def log_merge_summary(
        self,
        *,
        logger: logging.Logger,
        groups: int,
        merge_success: int,
        validation_rejected: int,
        retry_used: int,
        final_failure: int,
        paragraph_recovery_used: int,
        real_merge_blocks: int = 0,
        merge_candidate_blocks: int = 0,
        fallback_merge_blocks: int = 0,
        partial_merge_artifacts: int = 0,
    ) -> None:
        logger.info(
            "Merge summary: groups=%d merge_success=%d validation_rejected=%d retry_used=%d final_failure=%d paragraph_recovery_used=%d real_merge_blocks=%d merge_candidate_blocks=%d fallback_merge_blocks=%d partial_merge_artifacts=%d",
            groups,
            merge_success,
            validation_rejected,
            retry_used,
            final_failure,
            paragraph_recovery_used,
            real_merge_blocks,
            merge_candidate_blocks,
            fallback_merge_blocks,
            partial_merge_artifacts,
        )

    def log_docs_publish_summary(
        self,
        *,
        logger: logging.Logger,
        created: int,
        failed: int,
    ) -> None:
        logger.info("Publish summary (docs): created=%d failed=%d", created, failed)

    def log_telegram_publish_summary(
        self,
        *,
        logger: logging.Logger,
        sent: int,
        failed: int,
        skipped: int,
    ) -> None:
        logger.info(
            "Publish summary (telegram): sent=%d failed=%d skipped=%d",
            sent,
            failed,
            skipped,
        )

    def emit_summary(
        self,
        *,
        logger: logging.Logger,
        processing_mode: str,
        audit_mode: str,
        exit_code: int,
        llm_provider: str = "",
        llm_effective_model: str = "",
        llm_configured_model: str = "",
        llm_provider_model: str = "",
        merge_success: int = 0,
        validation_rejected: int = 0,
        retry_used: int = 0,
        final_failure: int = 0,
        paragraph_recovery_used: int = 0,
        merge_candidate_blocks: int = 0,
        fallback_merge_blocks: int = 0,
        partial_merge_artifacts: int = 0,
        full_merge_artifacts: int = 0,
        run_summary_ms: int = 0,
    ) -> None:
        status: str = _resolve_run_status(
            exit_code,
            self._state,
            fallback_merge_blocks=fallback_merge_blocks,
            partial_merge_artifacts=partial_merge_artifacts,
        )
        branch_summary: str = _format_branch_summary(
            audit_mode=audit_mode,
            state=self._state,
        )
        logger.info(
            "run_final_summary processing_mode=%s audit_mode=%s status=%s llm_provider=%s llm_model_effective=%s llm_model_configured=%s llm_provider_model=%s warnings_total=%d warnings_operational=%d warnings_informational=%d errors_total=%d warning_reason_codes=%s error_reason_codes=%s degraded_recovered_count=%d degraded_unrecovered_count=%d rows_processed=%d rows_skipped=%d planned_items=%d unique_dates_processed=%d date_branch_executions=%d docs_created=%d docs_failed=%d telegram_sent=%d telegram_failed=%d telegram_skipped=%d merge_success=%d validation_rejected=%d retry_used=%d final_failure=%d merge_final_failure_recorded=%d merge_validation_rejected_recorded=%d paragraph_recovery_used=%d merge_candidate_blocks=%d fallback_merge_blocks=%d partial_merge_artifacts=%d full_merge_artifacts=%d content_contract_failures=%d content_contract_recovered=%d content_contract_unrecovered=%d malformed_tail_urls_dropped=%d startup_health_ms=%d sheet_load_ms=%d shared_preparation_ms=%d planning_ms=%d slot_processing_ms=%d doc_publish_ms=%d telegram_publish_ms=%d run_summary_ms=%d total_run_ms=%d publish_gate_blocked_count=%d publish_gate_blocked_languages=%s branch_summary=%s",
            processing_mode,
            audit_mode,
            status,
            llm_provider or "unknown",
            llm_effective_model or "unknown",
            llm_configured_model or "unknown",
            llm_provider_model or "unknown",
            self._state.warnings_operational + self._state.warnings_informational,
            self._state.warnings_operational,
            self._state.warnings_informational,
            self._state.errors,
            _format_reason_counts(self._state.warning_reason_counts),
            _format_reason_counts(self._state.error_reason_counts),
            self._state.degraded_recovered_count,
            self._state.degraded_unrecovered_count,
            self._state.planned_items,
            self._state.rows_skipped,
            self._state.planned_items,
            len(self._state.unique_dates_processed),
            self._state.date_branch_executions,
            self._state.docs_created,
            self._state.docs_failed,
            self._state.telegram_sent,
            self._state.telegram_failed,
            self._state.telegram_skipped,
            merge_success,
            validation_rejected,
            retry_used,
            final_failure,
            self._state.merge_final_failure,
            self._state.merge_validation_rejected,
            paragraph_recovery_used,
            merge_candidate_blocks,
            fallback_merge_blocks,
            partial_merge_artifacts,
            full_merge_artifacts,
            self._state.content_contract_failures,
            self._state.content_contract_recovered,
            self._state.content_contract_unrecovered,
            self._state.malformed_tail_url_fragments_dropped,
            self._state.stage_durations_ms.get("startup_health", 0),
            self._state.stage_durations_ms.get("sheet_load", 0),
            self._state.stage_durations_ms.get("shared_preparation", 0),
            self._state.stage_durations_ms.get("planning", 0),
            self._state.stage_durations_ms.get("slot_processing", 0),
            self._state.stage_durations_ms.get("doc_publish", 0),
            self._state.stage_durations_ms.get("telegram_publish", 0),
            int(run_summary_ms),
            int(round((time.perf_counter() - self._state.run_started_at) * 1000.0)),
            self._state.publish_gate_blocked_count,
            ",".join(sorted(self._state.publish_gate_blocked_languages)) or "none",
            branch_summary,
        )

_ACTIVE_COLLECTOR: Optional[RuntimeAnalyticsCollector] = None
_FALLBACK_COLLECTOR: RuntimeAnalyticsCollector = RuntimeAnalyticsCollector(
    debug_enabled=False
)


def setup_runtime_analytics(
    *,
    logger: logging.Logger,
    debug_enabled: bool,
) -> RuntimeAnalyticsState:
    global _ACTIVE_COLLECTOR

    if _ACTIVE_COLLECTOR is not None:
        try:
            logger.removeHandler(_ACTIVE_COLLECTOR.handler)
        except Exception:
            pass
    _ACTIVE_COLLECTOR = RuntimeAnalyticsCollector(debug_enabled=debug_enabled)
    logger.addHandler(_ACTIVE_COLLECTOR.handler)
    return _ACTIVE_COLLECTOR.state


def get_runtime_analytics_state() -> Optional[RuntimeAnalyticsState]:
    return _ACTIVE_COLLECTOR.state if _ACTIVE_COLLECTOR is not None else None


def record_stage_duration(*, stage_name: str, elapsed_ms: int) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_stage_duration(stage_name=stage_name, elapsed_ms=elapsed_ms)


def log_stage_timing(
    *,
    logger: logging.Logger,
    stage_name: str,
    elapsed_ms: int,
    scope: str = "run",
    branch_label: str = "n/a",
    date_key: str = "n/a",
    slot_key: str = "n/a",
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_stage_timing(
        logger=logger,
        stage_name=stage_name,
        elapsed_ms=elapsed_ms,
        scope=scope,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )


def record_slot_total_ms(*, slot_key: str, elapsed_ms: int) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_slot_total_ms(slot_key=slot_key, elapsed_ms=elapsed_ms)


def record_branch_total_ms(*, date_key: str, branch_label: str, elapsed_ms: int) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_branch_total_ms(
        date_key=date_key,
        branch_label=branch_label,
        elapsed_ms=elapsed_ms,
    )


def record_branch_model_used(*, date_key: str, branch_label: str, model_name: str) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_branch_model_used(
        date_key=date_key,
        branch_label=branch_label,
        model_name=model_name,
    )


def record_contract_result(
    *,
    date_key: str,
    branch_label: str,
    contract_status: str,
) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_contract_result(
        date_key=date_key,
        branch_label=branch_label,
        contract_status=contract_status,
    )


def get_branch_date_summary(
    *,
    date_key: str,
    branch_label: str,
) -> Optional[BranchDateAnalyticsState]:
    if _ACTIVE_COLLECTOR is None:
        return None
    return _ACTIVE_COLLECTOR.get_branch_date_summary(
        date_key=date_key,
        branch_label=branch_label,
    )


def record_sheet_loaded(*, rows: int) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_sheet_loaded(rows=rows)


def record_planning_completed(*, planned_items: int, rows_skipped: int) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_planning_completed(
        planned_items=planned_items,
        rows_skipped=rows_skipped,
    )


def record_date_processed() -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_date_processed()


def record_date_branch_execution(*, date_key: str, branch_label: str) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_date_branch_execution(
        date_key=date_key,
        branch_label=branch_label,
    )


def record_branch_started(*, branch_label: str) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_branch_started(branch_label=branch_label)


def record_branch_completed(*, branch_label: str) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_branch_completed(branch_label=branch_label)


def record_branch_failed(*, branch_label: str) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_branch_failed(branch_label=branch_label)


def record_docs_created(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_docs_created(
        count=count,
        date_key=date_key,
        branch_label=branch_label,
    )


def record_docs_failed(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_docs_failed(
        count=count,
        date_key=date_key,
        branch_label=branch_label,
    )


def record_telegram_sent(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_telegram_sent(
        count=count,
        date_key=date_key,
        branch_label=branch_label,
    )


def record_telegram_failed(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_telegram_failed(
        count=count,
        date_key=date_key,
        branch_label=branch_label,
    )


def record_telegram_skipped(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_telegram_skipped(
        count=count,
        date_key=date_key,
        branch_label=branch_label,
    )


def record_merge_final_failure(*, count: int = 1) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_merge_final_failure(count=count)


def record_merge_validation_rejected(*, count: int = 1) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_merge_validation_rejected(count=count)


def record_publish_gate_blocked(*, language: str, target: str) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_publish_gate_blocked(language=language, target=target)


def get_state_snapshot() -> Optional[RuntimeAnalyticsState]:
    # NOTE: Возвращает живой объект state, а не его копию. Имя сохранено как
    # `snapshot` для семантической однородности с другими модулями; вызывающий
    # код должен использовать его только для чтения, без мутаций.
    if _ACTIVE_COLLECTOR is None:
        return None
    return _ACTIVE_COLLECTOR._state


def record_malformed_tail_url_cleanup(count: int, *, event_key: Optional[str] = None) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.record_malformed_tail_url_cleanup(count, event_key=event_key)

def log_run_started(
    *,
    logger: logging.Logger,
    processing_mode: str,
    audit_mode: str,
    debug_enabled: bool,
    dry_run: bool,
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_run_started(
        logger=logger,
        processing_mode=processing_mode,
        audit_mode=audit_mode,
        debug_enabled=debug_enabled,
        dry_run=dry_run,
    )


def log_run_context(
    logger: logging.Logger,
    run_context: RunContext,
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_run_context(logger, run_context)


def log_preflight_status(*, logger: logging.Logger, self_check_skipped: bool) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_preflight_status(
        logger=logger,
        self_check_skipped=self_check_skipped,
    )


def log_sheet_loaded(*, logger: logging.Logger, rows: int) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_sheet_loaded(logger=logger, rows=rows)


def log_planning_completed(
    *,
    logger: logging.Logger,
    planned_items: int,
    rows_skipped: int,
    dates: int,
    slots: int,
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_planning_completed(
        logger=logger,
        planned_items=planned_items,
        rows_skipped=rows_skipped,
        dates=dates,
        slots=slots,
    )


def log_date_started(
    *,
    logger: logging.Logger,
    date_key: str,
    slot_count: int,
    item_count: int,
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_date_started(
        logger=logger,
        date_key=date_key,
        slot_count=slot_count,
        item_count=item_count,
    )


def log_merge_summary(
    *,
    logger: logging.Logger,
    groups: int,
    merge_success: int,
    validation_rejected: int,
    retry_used: int,
    final_failure: int,
    paragraph_recovery_used: int,
    real_merge_blocks: int = 0,
    merge_candidate_blocks: int = 0,
    fallback_merge_blocks: int = 0,
    partial_merge_artifacts: int = 0,
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_merge_summary(
        logger=logger,
        groups=groups,
        merge_success=merge_success,
        validation_rejected=validation_rejected,
        retry_used=retry_used,
        final_failure=final_failure,
        paragraph_recovery_used=paragraph_recovery_used,
        real_merge_blocks=real_merge_blocks,
        merge_candidate_blocks=merge_candidate_blocks,
        fallback_merge_blocks=fallback_merge_blocks,
        partial_merge_artifacts=partial_merge_artifacts,
    )


def log_docs_publish_summary(
    *,
    logger: logging.Logger,
    created: int,
    failed: int,
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_docs_publish_summary(
        logger=logger,
        created=created,
        failed=failed,
    )


def log_telegram_publish_summary(
    *,
    logger: logging.Logger,
    sent: int,
    failed: int,
    skipped: int,
) -> None:
    collector: RuntimeAnalyticsCollector = _ACTIVE_COLLECTOR or _FALLBACK_COLLECTOR
    collector.log_telegram_publish_summary(
        logger=logger,
        sent=sent,
        failed=failed,
        skipped=skipped,
    )


def log_run_completed(
    *,
    logger: logging.Logger,
    processing_mode: str,
    audit_mode: str,
    exit_code: int,
    llm_provider: str = "",
    llm_effective_model: str = "",
    llm_configured_model: str = "",
    llm_provider_model: str = "",
    merge_success: int = 0,
    validation_rejected: int = 0,
    retry_used: int = 0,
    final_failure: int = 0,
    paragraph_recovery_used: int = 0,
    merge_candidate_blocks: int = 0,
    fallback_merge_blocks: int = 0,
    partial_merge_artifacts: int = 0,
    full_merge_artifacts: int = 0,
    run_summary_ms: int = 0,
) -> None:
    if _ACTIVE_COLLECTOR is None:
        return
    _ACTIVE_COLLECTOR.emit_summary(
        logger=logger,
        processing_mode=processing_mode,
        audit_mode=audit_mode,
        exit_code=exit_code,
        llm_provider=llm_provider,
        llm_effective_model=llm_effective_model,
        llm_configured_model=llm_configured_model,
        llm_provider_model=llm_provider_model,
        merge_success=merge_success,
        validation_rejected=validation_rejected,
        retry_used=retry_used,
        final_failure=final_failure,
        paragraph_recovery_used=paragraph_recovery_used,
        merge_candidate_blocks=merge_candidate_blocks,
        fallback_merge_blocks=fallback_merge_blocks,
        partial_merge_artifacts=partial_merge_artifacts,
        full_merge_artifacts=full_merge_artifacts,
        run_summary_ms=run_summary_ms,
    )
