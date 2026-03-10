from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from app.llm.merge_run_summary import MergeRunSummary
from app.pipeline.batch_runner import AuditBranch, BatchRunner
from app.pipeline.daily_doc_publish import DailyDocumentPublishResult
from app.pipeline.daily_telegram_publish import DailyTelegramPublishResult
from app.pipeline.slot_processing import SlotProcessResult


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

    def _slot_result(self, *, real_merge_blocks: int) -> SlotProcessResult:
        return SlotProcessResult(
            slot_key="100326_1800",
            slot_time_key="1800",
            header_context={"time_kiev": "18:00"},
            day_videos=[SimpleNamespace(date_key="100326", scheduled_at_kiev=SimpleNamespace(time=lambda: None), row_number=1)],
            language_groups={"uk": [], "en": [], "ru": [], "other": []},
            merged_content_by_language={},
            merge_audit_by_language={},
            real_merge_blocks=real_merge_blocks,
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
        self.assertIn("reason=no_real_merge_blocks", "\n".join(captured.output))

    def test_merge_artifact_is_created_when_real_merge_block_exists(self) -> None:
        runner = self._runner()
        branch = AuditBranch(name="merge", processing_mode="merge", llm_merge_enabled=True)
        merge_run_summary = MergeRunSummary()
        with patch("app.pipeline.batch_runner.process_slot", return_value=self._slot_result(real_merge_blocks=1)), patch(
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


if __name__ == "__main__":
    unittest.main()
