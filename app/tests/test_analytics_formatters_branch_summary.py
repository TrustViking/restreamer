from __future__ import annotations

import unittest

from app.core.branching import BRANCH_MERGE
from app.observability.analytics_formatters import _format_branch_summary
from app.observability.analytics_state import BranchAnalyticsState, RuntimeAnalyticsState


def _state(**kwargs) -> RuntimeAnalyticsState:
    s = RuntimeAnalyticsState(debug_enabled=False)
    for key, value in kwargs.items():
        setattr(s, key, value)
    return s


class TestFormatBranchSummaryMergePartial(unittest.TestCase):

    def test_partial_via_final_failure(self) -> None:
        state = _state(
            branch_results={BRANCH_MERGE: BranchAnalyticsState(completed=True)},
            merge_final_failure=1,
        )
        result = _format_branch_summary(audit_mode="merge", state=state)
        self.assertEqual(result, "merge:partial")

    def test_partial_via_publish_gate_blocked(self) -> None:
        state = _state(
            branch_results={BRANCH_MERGE: BranchAnalyticsState(completed=True)},
            publish_gate_blocked_count=1,
        )
        result = _format_branch_summary(audit_mode="merge", state=state)
        self.assertEqual(result, "merge:partial")

    def test_validation_rejected_alone_does_not_imply_partial(self) -> None:
        state = _state(
            branch_results={BRANCH_MERGE: BranchAnalyticsState(completed=True)},
            merge_validation_rejected=1,
        )
        result = _format_branch_summary(audit_mode="merge", state=state)
        self.assertEqual(result, "merge:success")

    def test_success_when_no_failures(self) -> None:
        state = _state(
            branch_results={BRANCH_MERGE: BranchAnalyticsState(completed=True)},
        )
        result = _format_branch_summary(audit_mode="merge", state=state)
        self.assertEqual(result, "merge:success")

    def test_failed_branch(self) -> None:
        state = _state(
            branch_results={BRANCH_MERGE: BranchAnalyticsState(failed=True)},
        )
        result = _format_branch_summary(audit_mode="merge", state=state)
        self.assertEqual(result, "merge:failed")

    def test_not_run_when_empty_branch_results(self) -> None:
        # No branch_results at all → top-level guard returns "<not_run>"
        state = _state()
        result = _format_branch_summary(audit_mode="merge", state=state)
        self.assertEqual(result, "<not_run>")

    def test_not_run_when_merge_branch_missing_from_results(self) -> None:
        # branch_results present but merge key absent → BranchAnalyticsState() default
        state = _state(branch_results={"nomerge": BranchAnalyticsState(completed=True)})
        result = _format_branch_summary(audit_mode="merge", state=state)
        self.assertEqual(result, "merge:not_run")
