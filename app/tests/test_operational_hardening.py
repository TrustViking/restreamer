from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.bootstrap.run_context import StartupContext
from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.llm.merge_run_summary import MergeRunSummary
from app.observability import runtime_analytics
from app.observability.content_contract import analyze_content_contract
from app.observability.runtime_analytics import log_warning_informational, log_warning_operational
from app.observability.startup_health import run_startup_health_checks
from app.observability.startup_summary import log_startup_summary
from app.paths import get_project_paths
from app.planning.link_normalization import handle_normalized_link_writeback
from app.pipeline.slot_processing import process_slot
from restreamer import _log_exit_code


class OperationalHardeningTests(unittest.TestCase):
    def test_runtime_summary_uses_current_audit_mode_branch_summary(self) -> None:
        logger = logging.getLogger("operational-hardening-runtime")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label=BRANCH_NOMERGE)
        runtime_analytics.record_branch_completed(branch_label=BRANCH_NOMERGE)
        runtime_analytics.record_branch_started(branch_label=BRANCH_MERGE)
        runtime_analytics.record_branch_failed(branch_label=BRANCH_MERGE)
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="audit",
                exit_code=1,
                merge_success=0,
                validation_rejected=1,
                retry_used=1,
                final_failure=1,
                paragraph_recovery_used=0,
            )
        text: str = "\n".join(captured.output)
        self.assertIn("audit_mode=audit", text)
        self.assertIn("branch_summary=nomerge:success,merge:failed", text)
        self.assertIn("validation_rejected=1", text)
        self.assertIn("retry_used=1", text)
        self.assertIn("final_failure=1", text)

    def test_informational_warning_does_not_downgrade_success_status(self) -> None:
        logger = logging.getLogger("operational-hardening-informational")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label=BRANCH_NOMERGE)
        runtime_analytics.record_branch_completed(branch_label=BRANCH_NOMERGE)
        log_warning_informational(logger, "Informational warning only")
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="nomerge",
                exit_code=0,
                merge_success=0,
                validation_rejected=0,
                retry_used=0,
                final_failure=0,
                paragraph_recovery_used=0,
            )
        self.assertIn("status=success", "\n".join(captured.output))

    def test_operational_warning_downgrades_status(self) -> None:
        logger = logging.getLogger("operational-hardening-operational")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label=BRANCH_MERGE)
        runtime_analytics.record_branch_completed(branch_label=BRANCH_MERGE)
        log_warning_operational(logger, "Operational warning")
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="merge",
                exit_code=0,
                merge_success=1,
                validation_rejected=0,
                retry_used=0,
                final_failure=0,
                paragraph_recovery_used=0,
            )
        self.assertIn("status=partial", "\n".join(captured.output))

    def test_merge_run_summary_uses_honest_counter_names(self) -> None:
        logger = logging.getLogger("operational-hardening-merge-summary")
        summary = MergeRunSummary()
        summary.record_retry_used()
        summary.record_merge_success()
        summary.record_validation_rejected()
        summary.record_final_failure()
        with self.assertLogs(logger, level="INFO") as captured:
            summary.log_summary(logger)
        text: str = "\n".join(captured.output)
        self.assertIn("retry_used=1", text)
        self.assertIn("merge_success=1", text)
        self.assertNotIn("plain_fallback_ok", text)

    def test_startup_summary_reports_current_branches_and_paths(self) -> None:
        logger = logging.getLogger("operational-hardening-startup")
        paths = get_project_paths()
        with patch("builtins.print") as print_mock, self.assertLogs(logger, level="INFO") as captured:
            log_startup_summary(
                logger,
                StartupContext(
                    run_id="run",
                    argv_list=["--audit-mode", "audit"],
                    args_audit_mode="audit",
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
                ),
            )
        text: str = "\n".join(captured.output)
        self.assertFalse(print_mock.called)
        self.assertIn("resolved_audit_mode=audit branches=nomerge,merge", text)
        self.assertIn(str(paths.templates_path), text)
        self.assertIn(str(paths.secrets_env_path), text)

    def test_startup_health_logs_current_merge_policy(self) -> None:
        logger = logging.getLogger("operational-hardening-health")
        config = SimpleNamespace(
            llm_provider="openai",
            llm_model="gpt-5.1",
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
        with patch("app.observability.startup_health.os.getenv", return_value="token"), self.assertLogs(
            logger, level="INFO"
        ) as captured:
            run_startup_health_checks(
                logger=logger,
                config=config,
                llm_summary=SimpleNamespace(
                    provider="openai",
                    model="gpt-5.1",
                    usage_reporting_mode="openai_run_local+openai_org_snapshot",
                ),
                services=services,
                telegram_client=telegram_client,
                resolve_logger_name_meta=lambda: ("logger", "test", False),
                dry_run=False,
                resolved_audit_mode="merge",
                run_id="run",
            )
        text: str = "\n".join(captured.output)
        self.assertIn("LLM merge policy provider=openai max_attempts=2", text)
        self.assertIn("OpenAI merge policy model=gpt-5.1 timeout_sec=30.0", text)
        self.assertIn("LLM selection: audit branch execution=merge", text)
        self.assertNotIn("packaging_stage_enabled", text)

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
            "app.pipeline.slot_processing.attempt_llm_merge_with_audit"
        ) as merge_mock:
            merge_run_summary = MergeRunSummary()
            process_slot(
                logger=logger,
                config=config,
                videos=[video],
                date_key="010130",
                slot_time_key="1000",
                llm_merge_enabled=False,
                cet_tz=timezone.utc,
                merge_run_summary=merge_run_summary,
                branch_label="nomerge",
            )
        self.assertEqual(0, merge_mock.call_count)

    def test_content_contract_does_not_expect_per_source_paragraphs_or_source_tail(self) -> None:
        snapshot, sanitization_result = analyze_content_contract(
            title_text="Final title",
            description_text="Paragraph one.\n\nParagraph two.",
            source_videos=[SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), SimpleNamespace()],
            mode="merge",
            title_mode="merged",
            tail_source="fallback_success",
        )
        self.assertEqual(2, snapshot.paragraph_count_min)
        self.assertEqual(4, snapshot.paragraph_count_max)
        self.assertTrue(snapshot.paragraph_count_valid)
        self.assertEqual(3, snapshot.links_allowed_max)
        self.assertEqual(0, snapshot.links_actual)
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
