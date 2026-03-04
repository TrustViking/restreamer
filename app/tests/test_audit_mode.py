from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.bootstrap.cli import build_cli_parser
from app.config.validators import normalize_audit_mode, normalize_processing_mode
from app.observability import runtime_analytics
from app.pipeline.batch_runner import BatchRunner


class AuditModeCliTests(unittest.TestCase):
    def test_legacy_top_level_flags_are_rejected(self) -> None:
        parser = build_cli_parser()
        for legacy_flag in ("--merge", "--nomerge", "--audit"):
            with self.assertRaises(SystemExit):
                parser.parse_args([legacy_flag])

    def test_cli_accepts_all_supported_audit_modes(self) -> None:
        parser = build_cli_parser()
        for audit_mode in ("nomerge", "merge", "unite"):
            args = parser.parse_args(["--audit-mode", audit_mode, "--dry-run"])
            self.assertEqual(audit_mode, args.audit_mode)

    def test_processing_mode_is_audit_only_and_audit_mode_is_normalized(self) -> None:
        self.assertEqual("audit", normalize_processing_mode("audit", source="test"))
        self.assertEqual("nomerge", normalize_audit_mode("no-merge", source="test"))
        self.assertEqual("merge", normalize_audit_mode("merge", source="test"))
        self.assertEqual("unite", normalize_audit_mode("unite", source="test"))
        for invalid_value in ("merge", "nomerge", "other"):
            with self.assertRaises(RuntimeError):
                normalize_processing_mode(invalid_value, source="test")


class AuditModeRunnerTests(unittest.TestCase):
    def _build_runner(self) -> BatchRunner:
        logger = logging.getLogger("audit-mode-runner")
        config = SimpleNamespace(google_enabled=True, templates=SimpleNamespace())
        return BatchRunner(
            logger=logger,
            config=config,
            metadata_fetcher=MagicMock(),
            http_client=MagicMock(),
            telegram_client=MagicMock(),
            name_builder=MagicMock(),
            kiev_tz=MagicMock(),
            cet_tz=MagicMock(),
            resolve_logger_name_meta=MagicMock(return_value=("logger", "test", False)),
        )

    def test_unite_builds_branch_plan_in_nomerge_then_merge_order(self) -> None:
        runner = self._build_runner()
        branches = runner._resolve_audit_branches(  # type: ignore[attr-defined]
            audit_mode="unite",
            llm_merge_available=True,
        )
        self.assertEqual(["nomerge", "merge"], [branch.name for branch in branches])
        self.assertEqual(["nomerge", "merge"], [branch.processing_mode for branch in branches])
        self.assertFalse(branches[0].llm_merge_enabled)
        self.assertTrue(branches[1].llm_merge_enabled)

    def test_single_branch_plans_are_precise(self) -> None:
        runner = self._build_runner()
        nomerge_branch = runner._resolve_audit_branches(  # type: ignore[attr-defined]
            audit_mode="nomerge",
            llm_merge_available=True,
        )
        merge_branch = runner._resolve_audit_branches(  # type: ignore[attr-defined]
            audit_mode="merge",
            llm_merge_available=False,
        )
        self.assertEqual(["nomerge"], [branch.name for branch in nomerge_branch])
        self.assertEqual(["merge"], [branch.name for branch in merge_branch])
        self.assertFalse(merge_branch[0].llm_merge_enabled)

    def test_unite_reuses_shared_preparation_once(self) -> None:
        runner = self._build_runner()
        prepared_videos = [SimpleNamespace(date_key="010130", scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1000"))]
        sheet_state = SimpleNamespace(
            rows=[object()],
            link_normalization_candidates=[],
            sheets_link_writeback_enabled=False,
            merge_semantics="override",
        )
        planned_merge = [SimpleNamespace(date_key="010130")]
        planned_nomerge = [SimpleNamespace(date_key="010130")]

        with patch("app.pipeline.batch_runner.build_runtime_services", return_value=SimpleNamespace(
            drive_client=MagicMock(),
            docs_client=MagicMock(),
            report_writer=MagicMock(),
            sheets_client=MagicMock(),
            factory=MagicMock(),
        )), patch("app.pipeline.batch_runner.run_startup_health_checks", return_value=True), patch(
            "app.pipeline.batch_runner.load_sheet_state",
            return_value=sheet_state,
        ) as load_sheet_state_mock, patch(
            "app.pipeline.batch_runner.build_prepared_videos",
            return_value=prepared_videos,
        ) as build_prepared_mock, patch(
            "app.pipeline.batch_runner.materialize_prepared_previews",
            return_value=prepared_videos,
        ) as materialize_mock, patch(
            "app.pipeline.batch_runner.derive_planned_videos",
            side_effect=[planned_nomerge, planned_merge],
        ) as derive_mock, patch.object(BatchRunner, "_run_branch_for_date", return_value=None):
            runner.run(dry_run=True, audit_mode="unite", run_id="run")

        self.assertEqual(1, load_sheet_state_mock.call_count)
        self.assertEqual(1, build_prepared_mock.call_count)
        self.assertEqual(1, materialize_mock.call_count)
        self.assertEqual(2, derive_mock.call_count)

    def test_incomplete_audit_branch_compare_is_debug_only(self) -> None:
        runtime_analytics.setup_runtime_analytics(
            logger=logging.getLogger("audit-mode-compare"),
            debug_enabled=False,
        )
        runtime_analytics.record_docs_created(count=1, date_key="010130", branch_label="nomerge")
        runner = self._build_runner()
        runner._logger = MagicMock()

        runner._log_audit_branch_compare(date_key="010130")  # type: ignore[attr-defined]

        runner._logger.info.assert_not_called()
        runner._logger.debug.assert_called_once()


if __name__ == "__main__":
    unittest.main()
