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
