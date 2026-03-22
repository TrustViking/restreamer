from __future__ import annotations

import logging
from dataclasses import dataclass

from app.llm.merges.merge_run_summary import MergeRunSummary
from app.observability.runtime_analytics import log_run_completed


@dataclass(frozen=True)
class FinalRunSummaryContext:
    processing_mode: str
    audit_mode: str
    exit_code: int
    merge_run_summary: MergeRunSummary | None
    run_summary_ms: int
    llm_provider: str
    llm_effective_model: str
    llm_configured_model: str
    llm_provider_model: str


def emit_final_run_summary(
    *,
    logger: logging.Logger,
    summary_context: FinalRunSummaryContext,
) -> None:
    merge_summary: MergeRunSummary | None = summary_context.merge_run_summary
    log_run_completed(
        logger=logger,
        processing_mode=summary_context.processing_mode,
        audit_mode=summary_context.audit_mode,
        exit_code=summary_context.exit_code,
        llm_provider=summary_context.llm_provider,
        llm_effective_model=summary_context.llm_effective_model,
        llm_configured_model=summary_context.llm_configured_model,
        llm_provider_model=summary_context.llm_provider_model,
        merge_success=merge_summary.merge_success if merge_summary is not None else 0,
        validation_rejected=merge_summary.validation_rejected if merge_summary is not None else 0,
        retry_used=merge_summary.retry_used if merge_summary is not None else 0,
        final_failure=merge_summary.final_failure if merge_summary is not None else 0,
        paragraph_recovery_used=(
            merge_summary.paragraph_recovery_used if merge_summary is not None else 0
        ),
        merge_candidate_blocks=(
            merge_summary.merge_candidate_blocks if merge_summary is not None else 0
        ),
        fallback_merge_blocks=(
            merge_summary.fallback_merge_blocks if merge_summary is not None else 0
        ),
        partial_merge_artifacts=(
            merge_summary.partial_merge_artifacts if merge_summary is not None else 0
        ),
        full_merge_artifacts=(
            merge_summary.full_merge_artifacts if merge_summary is not None else 0
        ),
        run_summary_ms=summary_context.run_summary_ms,
    )
