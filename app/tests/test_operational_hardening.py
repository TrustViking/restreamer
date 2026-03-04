from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.llm.merge_run_summary import MergeRunSummary
from app.observability.content_contract import analyze_content_contract
from app.observability import runtime_analytics
from app.observability.runtime_analytics import log_warning_informational, log_warning_operational
from app.observability.startup_health import run_startup_health_checks
from app.observability.startup_summary import log_startup_summary
from app.paths import get_project_paths
from app.planning.link_normalization import handle_normalized_link_writeback
from app.pipeline.slot_processing import process_slot
from restreamer import _log_exit_code


class OperationalHardeningTests(unittest.TestCase):
    def test_runtime_summary_uses_audit_mode_aware_branch_summary(self) -> None:
        logger = logging.getLogger("operational-hardening-runtime")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label="nomerge")
        runtime_analytics.record_branch_completed(branch_label="nomerge")
        runtime_analytics.record_branch_started(branch_label="merge")
        runtime_analytics.record_branch_failed(branch_label="merge")
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="unite",
                exit_code=1,
                primary_success=0,
                validation_rejected=1,
                primary_retry_used=1,
                fallback_success=0,
                final_failure=1,
                paragraph_recovery_used=0,
            )
        text: str = "\n".join(captured.output)
        self.assertIn("audit_mode=unite", text)
        self.assertIn("branch_summary=nomerge:success,merge:failed", text)
        self.assertIn("validation_rejected=1", text)
        self.assertIn("primary_retry_used=1", text)
        self.assertIn("final_failure=1", text)

    def test_informational_warning_does_not_downgrade_success_status(self) -> None:
        logger = logging.getLogger("operational-hardening-informational")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label="nomerge")
        runtime_analytics.record_branch_completed(branch_label="nomerge")
        log_warning_informational(logger, "Informational warning only")
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="nomerge",
                exit_code=0,
                primary_success=0,
                validation_rejected=0,
                primary_retry_used=0,
                fallback_success=0,
                final_failure=0,
                paragraph_recovery_used=0,
            )
        self.assertIn("status=success", "\n".join(captured.output))

    def test_operational_warning_downgrades_status(self) -> None:
        logger = logging.getLogger("operational-hardening-operational")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label="merge")
        runtime_analytics.record_branch_completed(branch_label="merge")
        log_warning_operational(logger, "Operational warning")
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="merge",
                exit_code=0,
                primary_success=0,
                validation_rejected=0,
                primary_retry_used=0,
                fallback_success=0,
                final_failure=0,
                paragraph_recovery_used=0,
            )
        self.assertIn("status=partial", "\n".join(captured.output))

    def test_merge_run_summary_uses_honest_counter_names(self) -> None:
        logger = logging.getLogger("operational-hardening-merge-summary")
        summary = MergeRunSummary()
        summary.record_primary_retry_used()
        summary.record_validation_rejected()
        summary.record_fallback_success()
        summary.record_final_failure()
        with self.assertLogs(logger, level="INFO") as captured:
            summary.log_summary(logger)
        text: str = "\n".join(captured.output)
        self.assertIn("primary_retry_used=1", text)
        self.assertIn("fallback_success=1", text)
        self.assertNotIn("plain_fallback_ok", text)

    def test_startup_summary_reports_new_paths_and_audit_mode(self) -> None:
        logger = logging.getLogger("operational-hardening-startup")
        paths = get_project_paths()
        with patch("builtins.print") as print_mock, self.assertLogs(logger, level="INFO") as captured:
            log_startup_summary(
                logger,
                run_id="run",
                argv_list=["--audit-mode", "unite"],
                args_audit_mode="unite",
                args_debug=False,
                args_dry_run=False,
                processing_mode="audit",
                config_processing_mode_raw="audit",
                project_root=paths.project_root,
                entrypoint_path=paths.entrypoint_path,
                runtime_config_path=paths.runtime_config_path,
                templates_path=paths.templates_path,
                secrets_env_path=paths.secrets_env_path,
                oauth_credentials_path=paths.oauth_credentials_path,
                oauth_token_path=paths.oauth_token_path,
            )
        text: str = "\n".join(captured.output)
        self.assertFalse(print_mock.called)
        self.assertIn("resolved_audit_mode=unite branches=nomerge,merge", text)
        self.assertIn(str(paths.templates_path), text)
        self.assertIn(str(paths.secrets_env_path), text)

    def test_startup_health_logs_honest_merge_policy(self) -> None:
        logger = logging.getLogger("operational-hardening-health")
        config = SimpleNamespace(
            llm_provider="openai",
            openai_model_primary="gpt-5.1",
            openai_model_fallback="gpt-5-mini",
            openai_timeout_sec=30.0,
            openai_max_output_tokens=1000,
            llm_source_desc_max_chars=500,
            openai_pre_delay_sec=0.0,
            telegram_enabled=True,
            google_sheets_id="sheet-id",
        )
        services = SimpleNamespace(
            factory=SimpleNamespace(
                get_auth_mode=lambda: "oauth",
                get_oauth_paths=lambda: ("cred", "token"),
                get_runtime_principal_email=lambda: "user@example.com",
                get_google_project_info=lambda strict: ("id", "name"),
            ),
            drive_client=SimpleNamespace(
                get_file_owner_info=lambda file_id: "owner",
                ping_access=lambda: ("user", "user@example.com"),
            ),
            sheets_client=SimpleNamespace(ping_access=lambda spreadsheet_id: ("sheet-id", "title")),
            docs_client=SimpleNamespace(ping_access=lambda: "ok"),
        )
        telegram_client = SimpleNamespace(get_me=lambda: {"id": 1, "username": "bot"})
        with patch("app.observability.startup_health.os.getenv", return_value="token"), self.assertLogs(logger, level="INFO") as captured:
            run_startup_health_checks(
                logger=logger,
                config=config,
                services=services,
                telegram_client=telegram_client,
                resolve_logger_name_meta=lambda: ("logger", "test", False),
                dry_run=False,
                resolved_audit_mode="merge",
                run_id="run",
            )
        text: str = "\n".join(captured.output)
        self.assertIn("primary_attempts=2", text)
        self.assertIn("fallback_enabled=yes", text)
        self.assertNotIn("max_retries=0", text)

    def test_nomerge_branch_does_not_call_llm_merge(self) -> None:
        logger = logging.getLogger("operational-hardening-slot")
        video = SimpleNamespace(
            row_number=1,
            scheduled_at_kiev=datetime(2030, 1, 1, 10, 0, tzinfo=timezone.utc),
            metadata=SimpleNamespace(title="Video title", description="Video description"),
            row_characteristics=None,
            date_display="01.01.2030",
            normalized_link="https://www.youtube.com/watch?v=abcdefghijk",
            forced_block_language=None,
            language="en",
        )
        config = SimpleNamespace(
            llm_run_if_single_source=False,
            llm_source_desc_max_chars=3000,
            templates=SimpleNamespace(common_no_description_text="no description"),
            llm_provider="openai",
            google_form_url="form",
            google_contacts="contacts",
        )
        with patch("app.pipeline.slot_processing.planned_video_block_language", return_value="en"), patch(
            "app.pipeline.slot_processing.attempt_openai_merge_with_audit"
        ) as merge_mock:
            process_slot(
                logger=logger,
                config=config,
                videos=[video],
                date_key="010130",
                slot_time_key="1000",
                llm_merge_enabled=False,
                cet_tz=timezone.utc,
                merge_run_summary=SimpleNamespace(
                    primary_success=0,
                    validation_rejected=0,
                    primary_retry_used=0,
                    fallback_success=0,
                    final_failure=0,
                    paragraph_recovery_used=0,
                ),
                branch_label="nomerge",
            )
        self.assertEqual(0, merge_mock.call_count)

    def test_content_contract_does_not_expect_per_source_paragraphs_or_source_tail(self) -> None:
        snapshot, sanitization_result = analyze_content_contract(
            title_text="Final title",
            description_text="Paragraph one.\n\nParagraph two.\n\nWatch live.\n\nhttps://youtube.com/watch?v=abcdefghijk\n\n#one #two",
            source_videos=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
            mode="merge",
            title_mode="merged",
            tail_source="fallback_success",
        )
        self.assertEqual(2, snapshot.paragraph_count_min)
        self.assertEqual(4, snapshot.paragraph_count_max)
        self.assertTrue(snapshot.paragraph_count_valid)
        self.assertEqual(3, snapshot.links_allowed_max)
        self.assertEqual(1, snapshot.links_actual)
        self.assertTrue(snapshot.links_valid)
        self.assertNotIn("paragraph_count_invalid", snapshot.contract_reason_codes)
        self.assertNotIn("links_limit_exceeded", snapshot.contract_reason_codes)

    def test_python_logs_final_exit_code(self) -> None:
        logger = logging.getLogger("operational-hardening-exit-code")
        with self.assertLogs(logger, level="INFO") as captured:
            _log_exit_code(logger=logger, exit_code=1)
        self.assertIn("Exit code: 1", "\n".join(captured.output))

    def test_link_writeback_unchanged_does_not_raise_and_returns_outcome(self) -> None:
        logger = logging.getLogger("operational-hardening-link-writeback")
        outcome = handle_normalized_link_writeback(
            logger=logger,
            summarize_error=lambda error: str(error),
            sheets_client=SimpleNamespace(update_cell_string=lambda **kwargs: None),
            spreadsheet_id="sheet",
            sheet_name_for_writeback="Sheet1",
            row_number=2,
            links_column_index=1,
            old_link="https://youtu.be/abc",
            normalized_link="https://youtu.be/abc",
            writeback_enabled=True,
            normalization_candidates=[],
            links_column_label="Links/B",
        )
        self.assertEqual("unchanged", outcome.status)
        self.assertEqual("not_applicable", outcome.writeback)


if __name__ == "__main__":
    unittest.main()
