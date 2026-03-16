from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Dict, List, Optional, Set

from app.bootstrap.run_context import RunContext
from app.core.branching import (
    BRANCH_MERGE,
    BRANCH_NOMERGE,
)

WARNING_CATEGORY_INFORMATIONAL: str = "informational"
WARNING_CATEGORY_OPERATIONAL: str = "operational"


@dataclass
class BranchAnalyticsState:
    started: bool = False
    completed: bool = False
    failed: bool = False


@dataclass
class BranchDateAnalyticsState:
    docs_created: int = 0
    docs_failed: int = 0
    telegram_sent: int = 0
    telegram_failed: int = 0
    telegram_skipped: int = 0
    contract_failures: int = 0
    contract_recovered: int = 0
    contract_unrecovered: int = 0
    branch_total_ms: int = 0
    models_used: Set[str] = field(default_factory=set)


@dataclass
class RuntimeAnalyticsState:
    debug_enabled: bool
    run_started_at: float = field(default_factory=time.perf_counter)
    warnings_operational: int = 0
    warnings_informational: int = 0
    errors: int = 0
    warning_reason_counts: Dict[str, int] = field(default_factory=dict)
    error_reason_counts: Dict[str, int] = field(default_factory=dict)
    rows_loaded: int = 0
    planned_items: int = 0
    rows_skipped: int = 0
    unique_dates_processed: Set[str] = field(default_factory=set)
    date_branch_executions: int = 0
    docs_created: int = 0
    docs_failed: int = 0
    telegram_sent: int = 0
    telegram_failed: int = 0
    telegram_skipped: int = 0
    malformed_tail_url_fragments_dropped: int = 0
    malformed_tail_cleanup_keys: Set[str] | None = None
    branch_results: Dict[str, BranchAnalyticsState] = field(default_factory=dict)
    branch_date_results: Dict[str, BranchDateAnalyticsState] = field(default_factory=dict)
    stage_durations_ms: Dict[str, int] = field(default_factory=dict)
    slot_total_ms_by_key: Dict[str, int] = field(default_factory=dict)
    degraded_recovered_count: int = 0
    degraded_unrecovered_count: int = 0
    content_contract_failures: int = 0
    content_contract_recovered: int = 0
    content_contract_unrecovered: int = 0


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


_ACTIVE_STATE: Optional[RuntimeAnalyticsState] = None
_ACTIVE_HANDLER: Optional[_RuntimeAnalyticsCounter] = None


def setup_runtime_analytics(
    *,
    logger: logging.Logger,
    debug_enabled: bool,
) -> RuntimeAnalyticsState:
    global _ACTIVE_HANDLER, _ACTIVE_STATE

    if _ACTIVE_HANDLER is not None:
        try:
            logger.removeHandler(_ACTIVE_HANDLER)
        except Exception:
            pass
    state: RuntimeAnalyticsState = RuntimeAnalyticsState(debug_enabled=debug_enabled)
    state.malformed_tail_cleanup_keys = set()
    handler = _RuntimeAnalyticsCounter(state)
    logger.addHandler(handler)
    _ACTIVE_STATE = state
    _ACTIVE_HANDLER = handler
    return state


def get_runtime_analytics_state() -> Optional[RuntimeAnalyticsState]:
    return _ACTIVE_STATE


def _branch_date_key(*, date_key: str, branch_label: str) -> str:
    return f"{date_key}|{branch_label}"


def _get_branch_date_state(*, date_key: str, branch_label: str) -> Optional[BranchDateAnalyticsState]:
    if _ACTIVE_STATE is None:
        return None
    state_key: str = _branch_date_key(date_key=date_key, branch_label=branch_label)
    return _ACTIVE_STATE.branch_date_results.setdefault(state_key, BranchDateAnalyticsState())


def record_stage_duration(*, stage_name: str, elapsed_ms: int) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.stage_durations_ms[stage_name] = max(
        0,
        _ACTIVE_STATE.stage_durations_ms.get(stage_name, 0) + int(elapsed_ms),
    )


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
    logger.info(
        "stage_timing stage=%s scope=%s branch=%s date_key=%s slot_key=%s elapsed_ms=%d",
        stage_name,
        scope,
        branch_label,
        date_key,
        slot_key,
        elapsed_ms,
    )


def record_slot_total_ms(*, slot_key: str, elapsed_ms: int) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.slot_total_ms_by_key[slot_key] = int(elapsed_ms)


def record_branch_total_ms(*, date_key: str, branch_label: str, elapsed_ms: int) -> None:
    branch_date_state: Optional[BranchDateAnalyticsState] = _get_branch_date_state(
        date_key=date_key,
        branch_label=branch_label,
    )
    if branch_date_state is None:
        return
    branch_date_state.branch_total_ms = int(elapsed_ms)


def record_branch_model_used(*, date_key: str, branch_label: str, model_name: str) -> None:
    branch_date_state: Optional[BranchDateAnalyticsState] = _get_branch_date_state(
        date_key=date_key,
        branch_label=branch_label,
    )
    if branch_date_state is None:
        return
    cleaned_model_name: str = str(model_name or "").strip()
    if cleaned_model_name:
        branch_date_state.models_used.add(cleaned_model_name)


def record_contract_result(
    *,
    date_key: str,
    branch_label: str,
    contract_status: str,
) -> None:
    if _ACTIVE_STATE is None:
        return
    branch_date_state: Optional[BranchDateAnalyticsState] = _get_branch_date_state(
        date_key=date_key,
        branch_label=branch_label,
    )
    if branch_date_state is None:
        return
    normalized_status: str = str(contract_status or "").strip().lower()
    if normalized_status == "degraded_recovered":
        _ACTIVE_STATE.degraded_recovered_count += 1
        _ACTIVE_STATE.content_contract_recovered += 1
        branch_date_state.contract_recovered += 1
        return
    if normalized_status in {"degraded_unrecovered", "fail"}:
        _ACTIVE_STATE.degraded_unrecovered_count += 1
        _ACTIVE_STATE.content_contract_failures += 1
        _ACTIVE_STATE.content_contract_unrecovered += 1
        branch_date_state.contract_failures += 1
        branch_date_state.contract_unrecovered += 1


def get_branch_date_summary(
    *,
    date_key: str,
    branch_label: str,
) -> Optional[BranchDateAnalyticsState]:
    if _ACTIVE_STATE is None:
        return None
    return _ACTIVE_STATE.branch_date_results.get(
        _branch_date_key(date_key=date_key, branch_label=branch_label)
    )


def record_sheet_loaded(*, rows: int) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.rows_loaded = rows


def record_planning_completed(*, planned_items: int, rows_skipped: int) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.planned_items = planned_items
    _ACTIVE_STATE.rows_skipped = max(0, rows_skipped)


def record_date_processed() -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.date_branch_executions += 1


def record_date_branch_execution(*, date_key: str, branch_label: str) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.unique_dates_processed.add(str(date_key))
    _ACTIVE_STATE.date_branch_executions += 1


def record_branch_started(*, branch_label: str) -> None:
    if _ACTIVE_STATE is None:
        return
    branch_state: BranchAnalyticsState = _ACTIVE_STATE.branch_results.setdefault(
        branch_label,
        BranchAnalyticsState(),
    )
    branch_state.started = True


def record_branch_completed(*, branch_label: str) -> None:
    if _ACTIVE_STATE is None:
        return
    branch_state: BranchAnalyticsState = _ACTIVE_STATE.branch_results.setdefault(
        branch_label,
        BranchAnalyticsState(),
    )
    branch_state.started = True
    branch_state.completed = True


def record_branch_failed(*, branch_label: str) -> None:
    if _ACTIVE_STATE is None:
        return
    branch_state: BranchAnalyticsState = _ACTIVE_STATE.branch_results.setdefault(
        branch_label,
        BranchAnalyticsState(),
    )
    branch_state.started = True
    branch_state.failed = True


def record_docs_created(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.docs_created += max(0, count)
    if date_key and branch_label:
        branch_date_state: Optional[BranchDateAnalyticsState] = _get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        if branch_date_state is not None:
            branch_date_state.docs_created += max(0, count)


def record_docs_failed(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.docs_failed += max(0, count)
    if date_key and branch_label:
        branch_date_state = _get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        if branch_date_state is not None:
            branch_date_state.docs_failed += max(0, count)


def record_telegram_sent(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.telegram_sent += max(0, count)
    if date_key and branch_label:
        branch_date_state = _get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        if branch_date_state is not None:
            branch_date_state.telegram_sent += max(0, count)


def record_telegram_failed(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.telegram_failed += max(0, count)
    if date_key and branch_label:
        branch_date_state = _get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        if branch_date_state is not None:
            branch_date_state.telegram_failed += max(0, count)


def record_telegram_skipped(*, count: int, date_key: Optional[str] = None, branch_label: Optional[str] = None) -> None:
    if _ACTIVE_STATE is None:
        return
    _ACTIVE_STATE.telegram_skipped += max(0, count)
    if date_key and branch_label:
        branch_date_state = _get_branch_date_state(
            date_key=date_key,
            branch_label=branch_label,
        )
        if branch_date_state is not None:
            branch_date_state.telegram_skipped += max(0, count)


def record_malformed_tail_url_cleanup(count: int, *, event_key: Optional[str] = None) -> None:
    if _ACTIVE_STATE is None:
        return
    if (
        event_key
        and _ACTIVE_STATE.malformed_tail_cleanup_keys is not None
        and event_key in _ACTIVE_STATE.malformed_tail_cleanup_keys
    ):
        return
    if event_key and _ACTIVE_STATE.malformed_tail_cleanup_keys is not None:
        _ACTIVE_STATE.malformed_tail_cleanup_keys.add(event_key)
    _ACTIVE_STATE.malformed_tail_url_fragments_dropped += max(0, count)


def log_run_started(
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


def log_preflight_status(*, logger: logging.Logger, self_check_skipped: bool) -> None:
    logger.info(
        "Preflight completed: self_check=%s",
        "skipped" if self_check_skipped else "completed",
    )


def log_sheet_loaded(*, logger: logging.Logger, rows: int) -> None:
    logger.info("Sheet loaded: rows=%d", rows)


def log_planning_completed(
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
    *,
    logger: logging.Logger,
    created: int,
    failed: int,
) -> None:
    logger.info("Publish summary (docs): created=%d failed=%d", created, failed)


def log_telegram_publish_summary(
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


def _has_failed_branch(state: RuntimeAnalyticsState) -> bool:
    return any(branch_state.failed for branch_state in state.branch_results.values())


def _resolve_run_status(exit_code: int, state: RuntimeAnalyticsState) -> str:
    if exit_code != 0 or state.errors > 0 or _has_failed_branch(state):
        return "failed"
    if (
        state.warnings_operational > 0
        or state.docs_failed > 0
        or state.telegram_failed > 0
        or state.malformed_tail_url_fragments_dropped > 0
    ):
        return "partial"
    return "success"


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
    if _ACTIVE_STATE is None:
        return
    status: str = _resolve_run_status(exit_code, _ACTIVE_STATE)
    branch_summary: str = _format_branch_summary(
        audit_mode=audit_mode,
        state=_ACTIVE_STATE,
    )
    logger.info(
        "run_final_summary processing_mode=%s audit_mode=%s status=%s llm_provider=%s llm_model_effective=%s llm_model_configured=%s llm_provider_model=%s warnings_total=%d warnings_operational=%d warnings_informational=%d errors_total=%d warning_reason_codes=%s error_reason_codes=%s degraded_recovered_count=%d degraded_unrecovered_count=%d rows_processed=%d rows_skipped=%d planned_items=%d unique_dates_processed=%d date_branch_executions=%d docs_created=%d docs_failed=%d telegram_sent=%d telegram_failed=%d telegram_skipped=%d merge_success=%d validation_rejected=%d retry_used=%d final_failure=%d paragraph_recovery_used=%d merge_candidate_blocks=%d fallback_merge_blocks=%d partial_merge_artifacts=%d full_merge_artifacts=%d content_contract_failures=%d content_contract_recovered=%d content_contract_unrecovered=%d malformed_tail_urls_dropped=%d startup_health_ms=%d sheet_load_ms=%d shared_preparation_ms=%d planning_ms=%d slot_processing_ms=%d doc_publish_ms=%d telegram_publish_ms=%d run_summary_ms=%d total_run_ms=%d branch_summary=%s",
        processing_mode,
        audit_mode,
        status,
        llm_provider or "unknown",
        llm_effective_model or "unknown",
        llm_configured_model or "unknown",
        llm_provider_model or "unknown",
        _ACTIVE_STATE.warnings_operational + _ACTIVE_STATE.warnings_informational,
        _ACTIVE_STATE.warnings_operational,
        _ACTIVE_STATE.warnings_informational,
        _ACTIVE_STATE.errors,
        _format_reason_counts(_ACTIVE_STATE.warning_reason_counts),
        _format_reason_counts(_ACTIVE_STATE.error_reason_counts),
        _ACTIVE_STATE.degraded_recovered_count,
        _ACTIVE_STATE.degraded_unrecovered_count,
        _ACTIVE_STATE.planned_items,
        _ACTIVE_STATE.rows_skipped,
        _ACTIVE_STATE.planned_items,
        len(_ACTIVE_STATE.unique_dates_processed),
        _ACTIVE_STATE.date_branch_executions,
        _ACTIVE_STATE.docs_created,
        _ACTIVE_STATE.docs_failed,
        _ACTIVE_STATE.telegram_sent,
        _ACTIVE_STATE.telegram_failed,
        _ACTIVE_STATE.telegram_skipped,
        merge_success,
        validation_rejected,
        retry_used,
        final_failure,
        paragraph_recovery_used,
        merge_candidate_blocks,
        fallback_merge_blocks,
        partial_merge_artifacts,
        full_merge_artifacts,
        _ACTIVE_STATE.content_contract_failures,
        _ACTIVE_STATE.content_contract_recovered,
        _ACTIVE_STATE.content_contract_unrecovered,
        _ACTIVE_STATE.malformed_tail_url_fragments_dropped,
        _ACTIVE_STATE.stage_durations_ms.get("startup_health", 0),
        _ACTIVE_STATE.stage_durations_ms.get("sheet_load", 0),
        _ACTIVE_STATE.stage_durations_ms.get("shared_preparation", 0),
        _ACTIVE_STATE.stage_durations_ms.get("planning", 0),
        _ACTIVE_STATE.stage_durations_ms.get("slot_processing", 0),
        _ACTIVE_STATE.stage_durations_ms.get("doc_publish", 0),
        _ACTIVE_STATE.stage_durations_ms.get("telegram_publish", 0),
        int(run_summary_ms),
        int(round((time.perf_counter() - _ACTIVE_STATE.run_started_at) * 1000.0)),
        branch_summary,
    )


def _format_branch_summary(
    *,
    audit_mode: str,
    state: RuntimeAnalyticsState,
) -> str:
    if not state.branch_results:
        return "<not_run>"
    if audit_mode == "nomerge":
        branch_state: BranchAnalyticsState = state.branch_results.get(
            audit_mode,
            BranchAnalyticsState(),
        )
        return (
            f"{audit_mode}:"
            f"{'failed' if branch_state.failed else ('success' if branch_state.completed else 'not_run')}"
        )
    branch_labels: tuple[str, ...]
    if audit_mode == "merge":
        branch_labels = (BRANCH_MERGE,)
    else:
        branch_labels = (BRANCH_NOMERGE, BRANCH_MERGE)
    branch_parts: list[str] = []
    for branch_label in branch_labels:
        branch_state: BranchAnalyticsState = state.branch_results.get(
            branch_label,
            BranchAnalyticsState(),
        )
        branch_parts.append(
            f"{branch_label}:"
            f"{'failed' if branch_state.failed else ('success' if branch_state.completed else 'not_run')}"
        )
    return ",".join(branch_parts)


def _format_reason_counts(reason_counts: Dict[str, int]) -> str:
    if not reason_counts:
        return "none"
    ordered_items: List[tuple[str, int]] = sorted(reason_counts.items())
    return ",".join(f"{reason_code}:{count}" for reason_code, count in ordered_items)
