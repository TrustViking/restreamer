from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from app.core.models import LanguageMergeAttempt, RejectedMergeAttempt
from app.llm.merge_run_summary import MergeRunSummary
from app.pipeline.batch_runner import AuditBranch, BatchRunner
from app.pipeline.daily_doc_publish import DailyDocumentPublishResult
from app.pipeline.daily_telegram_publish import DailyTelegramPublishResult
from app.pipeline.slot_processing import SlotProcessResult, resolve_merge_artifact_status


class MergeArtifactGateTests(unittest.TestCase):
    def _runner(self) -> BatchRunner:
        return BatchRunner(
            logger=logging.getLogger("merge-artifact-gate-tests"),
            config=SimpleNamespace(
                llm_model="gpt-5.1",
                google_enabled=True,
                templates=SimpleNamespace(),
            ),
            metadata_fetcher=MagicMock(),
            http_client=MagicMock(),
            telegram_client=MagicMock(),
            name_builder=MagicMock(),
            kiev_tz=ZoneInfo("Europe/Kiev"),
            cet_tz=ZoneInfo("Europe/Berlin"),
            resolve_logger_name_meta=MagicMock(),
        )

    def _slot_result(
        self,
        *,
        real_merge_blocks: int,
        merge_candidate_blocks: int = 0,
        fallback_merge_blocks: int = 0,
        merge_artifact_status: str = "none",
        fallback_merge_targets: tuple[str, ...] = (),
    ) -> SlotProcessResult:
        return SlotProcessResult(
            slot_key="100326_1800",
            slot_time_key="1800",
            header_context={"time_kiev": "18:00"},
            day_videos=[SimpleNamespace(date_key="100326", scheduled_at_kiev=SimpleNamespace(time=lambda: None), row_number=1)],
            language_groups={"uk": [], "en": [], "ru": [], "other": []},
            merged_content_by_language={},
            merge_audit_by_language={},
            real_merge_blocks=real_merge_blocks,
            merge_candidate_blocks=merge_candidate_blocks,
            fallback_merge_blocks=fallback_merge_blocks,
            merge_artifact_status=merge_artifact_status,
            fallback_merge_targets=fallback_merge_targets,
        )

    def test_merge_artifact_is_skipped_when_no_real_merge_blocks_exist(self) -> None:
        runner = self._runner()
        branch = AuditBranch(name="merge", processing_mode="merge", llm_merge_enabled=True)
        merge_run_summary = MergeRunSummary()
        with patch("app.pipeline.batch_runner.process_slot", return_value=self._slot_result(real_merge_blocks=0)), patch(
            "app.pipeline.batch_runner.publish_daily_document"
        ) as publish_doc_mock, patch(
            "app.pipeline.batch_runner.publish_daily_telegram"
        ) as publish_telegram_mock, self.assertLogs(level="INFO") as captured:
            runner._run_branch_for_date(
                services=SimpleNamespace(
                    docs_client=MagicMock(),
                    drive_client=MagicMock(),
                    report_writer=MagicMock(),
                ),
                branch=branch,
                date_key="100326",
                date_videos_all=[
                    SimpleNamespace(
                        date_key="100326",
                        scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1800"),
                    )
                ],
                dry_run=True,
                merge_run_summary=merge_run_summary,
            )
        publish_doc_mock.assert_not_called()
        publish_telegram_mock.assert_not_called()
        self.assertIn("merge_doc_created=no", "\n".join(captured.output))
        self.assertIn("reason=no_publishable_merge_artifact", "\n".join(captured.output))

    def test_fallback_only_status_is_resolved_explicitly(self) -> None:
        self.assertEqual(
            "fallback_only",
            resolve_merge_artifact_status(
                merge_candidate_blocks=2,
                real_merge_blocks=0,
                fallback_merge_blocks=2,
            ),
        )

    def test_merge_artifact_is_created_when_real_merge_block_exists(self) -> None:
        runner = self._runner()
        branch = AuditBranch(name="merge", processing_mode="merge", llm_merge_enabled=True)
        merge_run_summary = MergeRunSummary()
        with patch(
            "app.pipeline.batch_runner.process_slot",
            return_value=self._slot_result(
                real_merge_blocks=1,
                merge_candidate_blocks=1,
                merge_artifact_status="full",
            ),
        ), patch(
            "app.pipeline.batch_runner.publish_daily_document",
            return_value=DailyDocumentPublishResult(
                doc_title="doc",
                doc_url="DRY_RUN_DOC_URL",
                header_context={"time_kiev": "18:00"},
                slot_keys=["100326_1800"],
            ),
        ) as publish_doc_mock, patch(
            "app.pipeline.batch_runner.publish_daily_telegram",
            return_value=DailyTelegramPublishResult(sent_count=0, failed_count=0, skipped_count=1),
        ) as publish_telegram_mock:
            runner._run_branch_for_date(
                services=SimpleNamespace(
                    docs_client=MagicMock(),
                    drive_client=MagicMock(),
                    report_writer=MagicMock(),
                ),
                branch=branch,
                date_key="100326",
                date_videos_all=[
                    SimpleNamespace(
                        date_key="100326",
                        scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1800"),
                    )
                ],
                dry_run=True,
                merge_run_summary=merge_run_summary,
            )
        publish_doc_mock.assert_called_once()
        publish_telegram_mock.assert_called_once()

    def test_partial_merge_artifact_is_logged_honestly(self) -> None:
        runner = self._runner()
        branch = AuditBranch(name="merge", processing_mode="merge", llm_merge_enabled=True)
        merge_run_summary = MergeRunSummary()
        with patch(
            "app.pipeline.batch_runner.process_slot",
            return_value=self._slot_result(
                real_merge_blocks=1,
                merge_candidate_blocks=2,
                fallback_merge_blocks=1,
                merge_artifact_status="partial",
                fallback_merge_targets=("100326_1800:en",),
            ),
        ), patch(
            "app.pipeline.batch_runner.publish_daily_document",
            return_value=DailyDocumentPublishResult(
                doc_title="doc",
                doc_url="DRY_RUN_DOC_URL",
                header_context={"time_kiev": "18:00"},
                slot_keys=["100326_1800"],
            ),
        ), patch(
            "app.pipeline.batch_runner.publish_daily_telegram",
            return_value=DailyTelegramPublishResult(sent_count=0, failed_count=0, skipped_count=1),
        ), self.assertLogs(level="INFO") as captured:
            runner._run_branch_for_date(
                services=SimpleNamespace(
                    docs_client=MagicMock(),
                    drive_client=MagicMock(),
                    report_writer=MagicMock(),
                ),
                branch=branch,
                date_key="100326",
                date_videos_all=[
                    SimpleNamespace(
                        date_key="100326",
                        scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1800"),
                    )
                ],
                dry_run=True,
                merge_run_summary=merge_run_summary,
            )
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_artifact_status=partial", joined_logs)
        self.assertIn("fallback_merge_blocks=1", joined_logs)
        self.assertIn("fallback_targets=100326_1800:en", joined_logs)

    def test_fallback_only_merge_artifact_publishes_doc_but_skips_telegram(self) -> None:
        runner = self._runner()
        branch = AuditBranch(name="merge", processing_mode="merge", llm_merge_enabled=True)
        merge_run_summary = MergeRunSummary()
        with patch(
            "app.pipeline.batch_runner.process_slot",
            return_value=self._slot_result(
                real_merge_blocks=0,
                merge_candidate_blocks=2,
                fallback_merge_blocks=2,
                merge_artifact_status="fallback_only",
                fallback_merge_targets=("100326_1800:en", "100326_1800:uk"),
            ),
        ), patch(
            "app.pipeline.batch_runner.publish_daily_document",
            return_value=DailyDocumentPublishResult(
                doc_title="doc",
                doc_url="DRY_RUN_DOC_URL",
                header_context={"time_kiev": "18:00"},
                slot_keys=["100326_1800"],
            ),
        ) as publish_doc_mock, patch(
            "app.pipeline.batch_runner.publish_daily_telegram"
        ) as publish_telegram_mock, self.assertLogs(level="INFO") as captured:
            runner._run_branch_for_date(
                services=SimpleNamespace(
                    docs_client=MagicMock(),
                    drive_client=MagicMock(),
                    report_writer=MagicMock(),
                ),
                branch=branch,
                date_key="100326",
                date_videos_all=[
                    SimpleNamespace(
                        date_key="100326",
                        scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1800"),
                    )
                ],
                dry_run=True,
                merge_run_summary=merge_run_summary,
            )
        publish_doc_mock.assert_called_once()
        publish_telegram_mock.assert_not_called()
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_artifact_status=fallback_only", joined_logs)
        self.assertIn("merge_doc_allowed=yes", joined_logs)
        self.assertIn("reason=fallback_only_merge_artifact_present", joined_logs)
        self.assertIn("merge_telegram_allowed=no", joined_logs)
        self.assertIn("reason=no_real_merge_blocks", joined_logs)

    def test_nomerge_logging_reason_is_honest(self) -> None:
        runner = self._runner()
        branch = AuditBranch(name="nomerge", processing_mode="nomerge", llm_merge_enabled=False)
        merge_run_summary = MergeRunSummary()
        with patch("app.pipeline.batch_runner.process_slot", return_value=self._slot_result(real_merge_blocks=0)), patch(
            "app.pipeline.batch_runner.publish_daily_document",
            return_value=DailyDocumentPublishResult(
                doc_title="doc",
                doc_url="DRY_RUN_DOC_URL",
                header_context={"time_kiev": "18:00"},
                slot_keys=["100326_1800"],
            ),
        ), patch(
            "app.pipeline.batch_runner.publish_daily_telegram",
            return_value=DailyTelegramPublishResult(sent_count=0, failed_count=0, skipped_count=1),
        ), self.assertLogs(level="INFO") as captured:
            runner._run_branch_for_date(
                services=SimpleNamespace(
                    docs_client=MagicMock(),
                    drive_client=MagicMock(),
                    report_writer=MagicMock(),
                ),
                branch=branch,
                date_key="100326",
                date_videos_all=[
                    SimpleNamespace(
                        date_key="100326",
                        scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1800"),
                    )
                ],
                dry_run=True,
                merge_run_summary=merge_run_summary,
            )
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("reason=nomerge_branch_publish_mode", joined_logs)
        self.assertNotIn("reason=real_merge_blocks_present", joined_logs)

    def test_merge_failed_case_writes_debug_json_artifact_even_without_real_merge_blocks(self) -> None:
        runner = self._runner()
        branch = AuditBranch(name="merge", processing_mode="merge", llm_merge_enabled=True)
        merge_run_summary = MergeRunSummary()
        merge_attempt = LanguageMergeAttempt(
            language="en",
            model_name="gpt-5.1",
            raw_response_text='{"title":"Rejected","description":"Rejected description."}',
            merged=None,
            error_summary="description validation failed",
            publish_source_label="merge_failed",
            rejected_attempts=(
                RejectedMergeAttempt(
                    attempt_index=1,
                    model_name="gpt-5.1",
                    reject_reasons=("too_few_expanded_bullets",),
                    title="Rejected title",
                    description="Rejected description.",
                    raw_response_text='{"title":"Rejected title","description":"Rejected description."}',
                ),
            ),
        )
        slot_result = SlotProcessResult(
            slot_key="100326_1800",
            slot_time_key="1800",
            header_context={"time_kiev": "18:00"},
            day_videos=[SimpleNamespace(date_key="100326", scheduled_at_kiev=SimpleNamespace(time=lambda: None), row_number=1)],
            language_groups={"uk": [], "en": [SimpleNamespace(), SimpleNamespace(), SimpleNamespace()], "ru": [], "other": []},
            merged_content_by_language={},
            merge_audit_by_language={"en": merge_attempt},
            real_merge_blocks=0,
            merge_candidate_blocks=1,
            fallback_merge_blocks=1,
            merge_artifact_status="fallback_only",
            fallback_merge_targets=("100326_1800:en",),
        )
        with TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "merge_reject_debug_json" / "case.json"
            runner._name_builder.build_merge_reject_debug_json_path.return_value = json_path
            with patch(
                "app.pipeline.batch_runner.process_slot",
                return_value=slot_result,
            ), patch(
                "app.pipeline.batch_runner.publish_daily_document"
            ) as publish_doc_mock, patch(
                "app.pipeline.batch_runner.publish_daily_telegram"
            ) as publish_telegram_mock, self.assertLogs(level="INFO") as captured:
                runner._run_branch_for_date(
                    services=SimpleNamespace(
                        docs_client=MagicMock(),
                        drive_client=MagicMock(),
                        report_writer=MagicMock(),
                    ),
                    branch=branch,
                    date_key="100326",
                    date_videos_all=[
                        SimpleNamespace(
                            date_key="100326",
                            scheduled_at_kiev=SimpleNamespace(strftime=lambda _: "1800"),
                        )
                    ],
                    dry_run=True,
                    merge_run_summary=merge_run_summary,
                )
            publish_doc_mock.assert_called_once()
            publish_telegram_mock.assert_not_called()
            self.assertTrue(json_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual("100326_1800", payload["case_metadata"]["slot_key"])
            self.assertEqual("en", payload["case_metadata"]["language"])
            self.assertEqual(3, payload["case_metadata"]["source_count"])
            self.assertEqual("expanded", payload["case_metadata"]["merge_mode"])
            self.assertEqual("Rejected title", payload["attempts"][0]["title"])
            self.assertEqual(["too_few_expanded_bullets"], payload["attempts"][0]["reject_reasons"])
            logs: str = "\n".join(captured.output)
            self.assertIn("merge_reject_debug_json_written", logs)
            self.assertIn("merge_artifact_status=fallback_only", logs)
            self.assertIn("merge_telegram_allowed=no", logs)


if __name__ == "__main__":
    unittest.main()
