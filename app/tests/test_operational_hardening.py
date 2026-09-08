from __future__ import annotations

import logging
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from app.bootstrap.run_context import StartupContext
from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.observability import runtime_analytics
from app.observability.content_contract import analyze_content_contract
from app.observability.runtime_analytics import log_warning_informational, log_warning_operational
from app.observability.startup_health import run_startup_health_checks
from app.observability.startup_summary import log_startup_summary
from app.paths import get_project_paths
from app.planning.link_normalization import handle_normalized_link_writeback
from app.pipeline.slot_processing import process_slot
from app.application.application import _log_exit_code


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

    def test_language_merge_failure_keeps_run_partial_when_branch_and_publish_succeed(self) -> None:
        logger = logging.getLogger("operational-hardening-language-merge-warning")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label=BRANCH_MERGE)
        runtime_analytics.record_branch_completed(branch_label=BRANCH_MERGE)
        log_warning_operational(
            logger,
            "merge_llm_final_failure branch=merge date_key=010130 slot_key=010130_1000 language=en provider=openai stage=primary code=too_few_expanded_bullets fallback_used=no raw_response_received=yes reason=description validation failed: too_few_expanded_bullets raw_chars=123",
            reason_code="too_few_expanded_bullets",
        )
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="merge",
                exit_code=0,
                llm_provider="openai",
                llm_effective_model="gpt-5.2",
                llm_configured_model="gpt-5.2",
                llm_provider_model="gpt-5.1",
                merge_success=0,
                validation_rejected=1,
                retry_used=1,
                final_failure=1,
                paragraph_recovery_used=0,
                merge_candidate_blocks=2,
                fallback_merge_blocks=1,
                partial_merge_artifacts=1,
                full_merge_artifacts=0,
            )
        text: str = "\n".join(captured.output)
        self.assertIn("status=partial", text)
        self.assertIn("branch_summary=merge:success", text)
        self.assertIn("llm_model_effective=gpt-5.2", text)
        self.assertIn("llm_model_configured=gpt-5.2", text)
        self.assertIn("llm_provider_model=gpt-5.1", text)
        self.assertIn("errors_total=0", text)
        self.assertIn("error_reason_codes=none", text)
        self.assertIn("warning_reason_codes=too_few_expanded_bullets:1", text)
        self.assertIn("partial_merge_artifacts=1", text)
        self.assertIn("final_failure=1", text)

    def test_fatal_model_config_error_is_visible_in_final_summary(self) -> None:
        logger = logging.getLogger("operational-hardening-fatal-model-summary")
        runtime_analytics.setup_runtime_analytics(logger=logger, debug_enabled=False)
        runtime_analytics.record_branch_started(branch_label=BRANCH_MERGE)
        runtime_analytics.record_branch_failed(branch_label=BRANCH_MERGE)
        runtime_analytics.log_error_event(
            logger,
            "openai_model_configuration_error provider=openai model=gpt-5.4 reason_code=openai_model_access_denied",
            reason_code="openai_model_access_denied",
        )
        with self.assertLogs(logger, level="INFO") as captured:
            runtime_analytics.log_run_completed(
                logger=logger,
                processing_mode="audit",
                audit_mode="merge",
                exit_code=1,
                llm_provider="openai",
                llm_effective_model="gpt-5.4",
                llm_configured_model="gpt-5.4",
                llm_provider_model="gpt-5.4",
                merge_success=0,
                validation_rejected=0,
                retry_used=0,
                final_failure=0,
                paragraph_recovery_used=0,
            )
        text: str = "\n".join(captured.output)
        self.assertIn("status=failed", text)
        self.assertIn("llm_model_effective=gpt-5.4", text)
        self.assertIn("error_reason_codes=openai_model_access_denied:1", text)

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
                    paths=paths,
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
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-5.1",
                timeout_sec=30.0,
                max_output_tokens=1000,
                source_desc_max_chars=500,
                pre_delay_sec=0.0,
            ),
            telegram=SimpleNamespace(enabled=True),
            google=SimpleNamespace(sheets_id="sheet-id"),
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
                    usage_reporting_mode="openai_run_local",
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
        self.assertIn(
            "OpenAI merge policy effective_model=gpt-5.1 configured_model=gpt-5.1 provider_model=gpt-5.1 timeout_sec=30.0",
            text,
        )
        self.assertIn(
            "OpenAI model compatibility effective_model=gpt-5.1 configured_model=gpt-5.1 provider_model=gpt-5.1 model_family=gpt-5 reasoning_effort=enabled structured_output=json_schema temperature=disabled capability_source=heuristic_name_rules model_access=verified_at_startup_by_responses_probe",
            text,
        )
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
            llm=SimpleNamespace(
                run_if_single_source=False,
                source_desc_max_chars=3000,
                provider="openai",
            ),
            templates=SimpleNamespace(common_no_description_text="no description"),
            google=SimpleNamespace(form_url="form", contacts="contacts"),
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
