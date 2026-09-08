from __future__ import annotations

from typing import Dict, List

from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.observability.analytics_state import (
    BranchAnalyticsState,
    RuntimeAnalyticsState,
)


def _branch_date_key(*, date_key: str, branch_label: str) -> str:
    return f"{date_key}|{branch_label}"


def _has_failed_branch(state: RuntimeAnalyticsState) -> bool:
    return any(branch_state.failed for branch_state in state.branch_results.values())


def _resolve_run_status(
    exit_code: int,
    state: RuntimeAnalyticsState,
    *,
    fallback_merge_blocks: int = 0,
    partial_merge_artifacts: int = 0,
) -> str:
    if exit_code != 0:
        return "failed"
    if _has_failed_branch(state):
        return "failed"
    if state.docs_failed > 0 or state.telegram_failed > 0:
        return "failed"
    if (
        state.errors > 0
        or state.warnings_operational > 0
        or state.malformed_tail_url_fragments_dropped > 0
        or state.merge_final_failure > 0
        or state.publish_gate_blocked_count > 0
        or state.telegram_skipped > 0
        or fallback_merge_blocks > 0
        or partial_merge_artifacts > 0
    ):
        return "partial"
    return "success"


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
        if branch_state.failed:
            branch_status = "failed"
        elif branch_state.completed:
            # Merge-ветка может завершиться без краха, но с фактическими
            # деградациями публикации (final_failure либо publish-gate fallback).
            # В этом случае summary должен отражать `partial`, чтобы
            # соответствовать общему status=partial.
            if branch_label == BRANCH_MERGE and (
                state.merge_final_failure > 0
                or state.publish_gate_blocked_count > 0
            ):
                branch_status = "partial"
            else:
                branch_status = "success"
        else:
            branch_status = "not_run"
        branch_parts.append(f"{branch_label}:{branch_status}")
    return ",".join(branch_parts)


def _format_reason_counts(reason_counts: Dict[str, int]) -> str:
    if not reason_counts:
        return "none"
    ordered_items: List[tuple[str, int]] = sorted(reason_counts.items())
    return ",".join(f"{reason_code}:{count}" for reason_code, count in ordered_items)
