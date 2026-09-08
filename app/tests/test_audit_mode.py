from __future__ import annotations

import logging
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from app.bootstrap.cli import build_cli_parser
from app.config.validators import normalize_audit_mode, normalize_processing_mode
from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.core.models import PlannedVideo, VideoMetadata
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.observability.startup_health import StartupHealthResult
from app.pipeline.batch_runner import AuditBranch, BatchRunner
from app.pipeline.operator_notifier import OperatorNotifier
from app.pipeline.slot_processing import process_slot


class AuditModeCliTests(unittest.TestCase):
    def test_cli_accepts_supported_modes(self) -> None:
        parser = build_cli_parser()
        for audit_mode in ("nomerge", "merge", "audit"):
            args = parser.parse_args(["--audit-mode", audit_mode, "--dry-run"])
            self.assertEqual(audit_mode, args.audit_mode)

    def test_processing_mode_and_audit_mode_are_normalized(self) -> None:
        self.assertEqual("audit", normalize_processing_mode("audit", source="test"))
        self.assertEqual("nomerge", normalize_audit_mode("nomerge", source="test"))
        self.assertEqual("merge", normalize_audit_mode("merge", source="test"))
        self.assertEqual("audit", normalize_audit_mode("audit", source="test"))


class AuditModeRunnerTests(unittest.TestCase):
    def _build_runner(self) -> BatchRunner:
        logger = logging.getLogger("audit-mode-runner")
        config = SimpleNamespace(
            google=SimpleNamespace(enabled=True),
            templates=SimpleNamespace(),
        )
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
            notifier=OperatorNotifier(),
        )

    def test_audit_runs_nomerge_then_merge(self) -> None:
        runner = self._build_runner()
        branches = runner._resolve_audit_branches(audit_mode="audit", llm_merge_available=True)  # type: ignore[attr-defined]
        self.assertEqual([BRANCH_NOMERGE, BRANCH_MERGE], [branch.name for branch in branches])
        self.assertEqual(["nomerge", "merge"], [branch.processing_mode for branch in branches])
        self.assertFalse(branches[0].llm_merge_enabled)
        self.assertTrue(branches[1].llm_merge_enabled)

    def test_merge_only_runs_single_branch(self) -> None:
        runner = self._build_runner()
        branches = runner._resolve_audit_branches(audit_mode="merge", llm_merge_available=False)  # type: ignore[attr-defined]
        self.assertEqual([BRANCH_MERGE], [branch.name for branch in branches])
        self.assertFalse(branches[0].llm_merge_enabled)

    def test_audit_reuses_shared_preparation_once(self) -> None:
        runner = self._build_runner()
        prepared_videos = [
            SimpleNamespace(
                date_key="010130",
                scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1000"),
                saved_preview_url="",
                row_number=1,
                language="",
            )
        ]
        sheet_state = SimpleNamespace(
            rows=[object()],
            link_normalization_candidates=[],
            sheets_link_writeback_enabled=False,
            merge_semantics="override",
        )
        planned_nomerge = [SimpleNamespace(date_key="010130")]
        planned_merge = [SimpleNamespace(date_key="010130")]

        with patch("app.pipeline.batch_runner.build_runtime_services", return_value=SimpleNamespace(
            drive_client=MagicMock(),
            docs_client=MagicMock(),
            report_writer=MagicMock(),
            sheets_client=MagicMock(),
            factory=MagicMock(),
        )), patch("app.pipeline.batch_runner.run_startup_health_checks", return_value=StartupHealthResult(llm_merge_enabled=True, failed_checks=())), patch(
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
        ) as derive_mock, patch("app.pipeline.batch_runner.BranchExecutor.execute", return_value=None):
            runner.run(
                dry_run=True,
                audit_mode="audit",
                run_id="run",
                llm_summary=SimpleNamespace(
                    provider="openai",
                    model="gpt-5.1",
                    usage_reporting_mode="openai_run_local",
                ),
            )

        self.assertEqual(1, load_sheet_state_mock.call_count)
        self.assertEqual(1, build_prepared_mock.call_count)
        self.assertEqual(1, materialize_mock.call_count)
        self.assertEqual(2, derive_mock.call_count)

    def test_fatal_model_config_error_stops_remaining_slot_processing(self) -> None:
        runner = self._build_runner()
        date_videos_all = [SimpleNamespace()]
        fatal_error = LlmModelConfigurationError(
            provider_name="openai",
            model_name="gpt-5.4",
            reason_code="openai_model_access_denied",
            detail="Project does not have access to model `gpt-5.4`",
            status_code=403,
            api_error_code="access_denied",
            api_error_param="model",
        )
        with patch.object(
            runner._branch_executor,
            "_group_date_videos_by_time_and_language",
            return_value={("0900", "uk"): [SimpleNamespace()], ("1000", "en"): [SimpleNamespace()]},
        ), patch(
            "app.pipeline.branch_executor.process_slot",
            side_effect=[fatal_error, MagicMock()],
        ) as process_slot_mock:
            with self.assertRaises(LlmModelConfigurationError):
                runner._branch_executor.execute(
                    services=SimpleNamespace(),
                    branch=AuditBranch(
                        name=BRANCH_MERGE,
                        processing_mode="merge",
                        llm_merge_enabled=True,
                    ),
                    date_key="010130",
                    date_videos=date_videos_all,
                    dry_run=True,
                    merge_run_summary=MergeRunSummary(),
                )
        self.assertEqual(1, process_slot_mock.call_count)


class SlotMergePolicyTests(unittest.TestCase):
    def _planned_video(self, *, description: str, language: str = "en", row_number: int = 1) -> PlannedVideo:
        return PlannedVideo(
            row_number=row_number,
            original_link="https://youtube.com/watch?v=test",
            normalized_link="https://youtube.com/watch?v=test",
            scheduled_at_kiev=datetime(2026, 3, 9, 13, 50 + row_number),
            date_key="090326",
            date_display="09.03.2026",
            language=language,
            metadata=VideoMetadata(
                url="https://youtube.com/watch?v=test",
                title=f"title-{row_number}",
                description=description,
                thumbnail_url="",
                youtube_language=language,
            ),
            thumbnail=MagicMock(),
            local_thumbnail_path=None,
        )

    def test_single_source_merge_skips_llm(self) -> None:
        logger = logging.getLogger("single-source-skip")
        config = SimpleNamespace(
            llm=SimpleNamespace(
                run_if_single_source=False,
                source_desc_max_chars=2000,
                provider="openai",
            ),
            google=SimpleNamespace(form_url="", contacts=""),
            templates=SimpleNamespace(common_no_description_text="No description"),
        )
        with patch("app.pipeline.slot_processing.attempt_llm_merge_with_audit") as merge_mock:
            result = process_slot(
                logger=logger,
                config=config,
                videos=[self._planned_video(description="one source", row_number=1)],
                date_key="090326",
                slot_time_key="1350",
                llm_merge_enabled=True,
                cet_tz=ZoneInfo("Europe/Berlin"),
                merge_run_summary=MergeRunSummary(),
                branch_label=BRANCH_MERGE,
            )
        self.assertFalse(result.merged_content_by_language)
        merge_mock.assert_not_called()

    def test_multi_source_merge_uses_single_llm_call(self) -> None:
        logger = logging.getLogger("multi-source-merge")
        config = SimpleNamespace(
            llm=SimpleNamespace(
                run_if_single_source=False,
                source_desc_max_chars=2000,
                provider="openai",
            ),
            google=SimpleNamespace(form_url="", contacts=""),
            templates=SimpleNamespace(common_no_description_text="No description"),
        )
        merge_result = SimpleNamespace(
            merged=SimpleNamespace(title="Merged", description="Body"),
            model_name="gpt-5.1",
            used_model_names=("gpt-5.1",),
            raw_response_text="{}",
            error_summary=None,
            generator_model_name="gpt-5.1",
            polish_model_name=None,
            polish_accepted=None,
            title_source="llm",
            hook_source="llm",
            hashtags_source="llm",
            body_source="main_merge",
        )
        with patch("app.pipeline.slot_processing.attempt_llm_merge_with_audit", return_value=merge_result) as merge_mock, patch(
            "app.pipeline.slot_processing.enforce_openai_merged_paragraphs",
            side_effect=lambda **kwargs: kwargs["merged_content"],
        ):
            process_slot(
                logger=logger,
                config=config,
                videos=[
                    self._planned_video(description="first", row_number=1),
                    self._planned_video(description="second", row_number=2),
                ],
                date_key="090326",
                slot_time_key="1350",
                llm_merge_enabled=True,
                cet_tz=ZoneInfo("Europe/Berlin"),
                merge_run_summary=MergeRunSummary(),
                branch_label=BRANCH_MERGE,
            )
        merge_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()

