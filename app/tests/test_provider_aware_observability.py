from __future__ import annotations

import logging
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import promo
from app.application.application import (
    PipelineApplication,
    _apply_llm_usage_reset,
    _log_llm_usage_reports,
)
from app.application import application as application_module
from app.llm.llm_client import reset_run_local_openai_usage
from app.observability.openai_usage import log_run_local_openai_usage
from app.bootstrap.run_context import RunContext, StartupContext
from app.observability.runtime_analytics import log_run_context
from app.observability.startup_health import run_startup_health_checks
from app.observability.startup_summary import log_config_summary, log_startup_summary
from app.paths import ProjectPaths


class ProviderAwareUsageHooksTests(unittest.TestCase):
    def test_openai_usage_hooks_are_active(self) -> None:
        logger = logging.getLogger("provider-aware-openai-usage")
        with patch("app.application.application.reset_run_local_openai_usage") as reset_mock, patch(
            "app.application.application.log_run_local_openai_usage"
        ) as local_usage_mock, self.assertLogs(
            logger, level="INFO"
        ) as captured:
            llm_summary = SimpleNamespace(
                provider="openai",
                model="gpt-5.2",
                usage_reporting_mode="openai_run_local",
            )
            _apply_llm_usage_reset(logger=logger, llm_summary=llm_summary)
            _log_llm_usage_reports(logger=logger, llm_summary=llm_summary)
        self.assertEqual(1, reset_mock.call_count)
        self.assertEqual(1, local_usage_mock.call_count)
        text: str = "\n".join(captured.output)
        self.assertIn("llm_usage_reset_applied provider=openai", text)
        self.assertIn("llm_usage_report_start provider=openai effective_model=gpt-5.2", text)
        self.assertIn("llm_usage_report_completed provider=openai effective_model=gpt-5.2 status=completed", text)

    def test_non_openai_provider_skips_reset_but_keeps_usage_report_logging(self) -> None:
        logger = logging.getLogger("provider-aware-claude-usage")
        with patch("app.application.application.reset_run_local_openai_usage") as reset_mock, patch(
            "app.application.application.log_run_local_openai_usage"
        ) as local_usage_mock, self.assertLogs(
            logger, level="INFO"
        ) as captured:
            llm_summary = SimpleNamespace(
                provider="claude",
                model="claude-opus-4-6",
                usage_reporting_mode="openai_run_local",
            )
            _apply_llm_usage_reset(logger=logger, llm_summary=llm_summary)
            _log_llm_usage_reports(logger=logger, llm_summary=llm_summary)
        self.assertEqual(0, reset_mock.call_count)
        self.assertEqual(1, local_usage_mock.call_count)
        text: str = "\n".join(captured.output)
        self.assertIn("llm_usage_reset_skipped provider=claude reason=provider_not_openai", text)
        self.assertIn("llm_usage_report_start provider=claude effective_model=claude-opus-4-6", text)
        self.assertIn("llm_usage_report_completed provider=claude effective_model=claude-opus-4-6 status=completed", text)

    def test_run_local_usage_log_marks_effective_model_as_run_source_of_truth(self) -> None:
        logger = logging.getLogger("provider-aware-run-local-usage")
        usage_state = reset_run_local_openai_usage()
        usage_state.requests_sent = 2
        usage_state.structured_calls = 2
        usage_state.models_used.add("gpt-5.2")
        with self.assertLogs(logger, level="INFO") as captured:
            log_run_local_openai_usage(logger, effective_model="gpt-5.2")
        text: str = "\n".join(captured.output)
        self.assertIn("OPENAI RUN USAGE", text)
        self.assertIn("scope=run_local", text)
        self.assertIn("source_of_truth_for_run=yes", text)
        self.assertIn("effective_model=gpt-5.2", text)

class ProviderAwareSummaryTests(unittest.TestCase):
    def _run_context(
        self,
        *,
        provider: str,
        model: str,
        usage_reporting_mode: str,
    ) -> RunContext:
        return RunContext(
            run_id="run",
            processing_mode="audit",
            audit_mode="audit",
            config_processing_mode="audit",
            audit_branches=["nomerge", "merge"],
            debug_enabled=False,
            dry_run=False,
            google_enabled=False,
            telegram_enabled=False,
            llm_provider=provider,
            llm_model=model,
            llm_usage_reporting_mode=usage_reporting_mode,
            sheet_id="sheet-id",
            sheet_range="A:F",
            sheets_link_writeback=False,
            local_doc_export_enabled=True,
            strip_chapter_timestamps=False,
        )

    def _startup_context(self) -> StartupContext:
        return StartupContext(
            run_id="run",
            argv_list=[],
            args_audit_mode="audit",
            args_debug=False,
            args_dry_run=False,
            processing_mode="audit",
            config_processing_mode_raw="audit",
            paths=ProjectPaths(
                project_root=Path("."),
                entrypoint_path=Path("promo.py"),
                runtime_config_path=Path("app_config.yaml"),
                runtime_config_example_path=Path("app_config.example.yaml"),
                templates_path=Path("templates.yaml"),
                secrets_dir=Path("secrets"),
                secrets_env_path=Path(".env"),
                oauth_token_path=Path("token.json"),
                oauth_credentials_path=Path("oauth.json"),
                service_account_path=Path("service_account.json"),
                bundled_config_path=Path("app_config.yaml"),
                bundled_templates_path=Path("templates.yaml"),
            ),
        )

    def _config(self, provider: str) -> SimpleNamespace:
        model_name: str = "gpt-5.1" if provider == "openai" else "claude-opus-4-6"
        return SimpleNamespace(
            google=SimpleNamespace(
                enabled=False,
                sheets_id="sheet-id",
                sheets_range="A:F",
            ),
            telegram=SimpleNamespace(enabled=False),
            paths=SimpleNamespace(
                templates_path=Path("templates.yaml"),
                local_doc_dir_template="D:\\docs\\{date}",
            ),
            processing=SimpleNamespace(mode="audit", now_tz_mode="kyiv"),
            llm=SimpleNamespace(
                provider=provider,
                model=model_name,
                timeout_sec=30.0,
                max_output_tokens=1000,
                source_desc_max_chars=500,
                run_if_single_source=False,
                pre_delay_sec=0.0,
            ),
        )

    def test_openai_summary_fields_show_current_values(self) -> None:
        logger = logging.getLogger("provider-aware-openai-summary")
        config = self._config("openai")
        with self.assertLogs(logger, level="INFO") as captured:
            log_config_summary(
                logger,
                config,
                SimpleNamespace(
                    provider="openai",
                    model="gpt-5.1",
                    usage_reporting_mode="openai_run_local",
                ),
                self._run_context(
                    provider="openai",
                    model="gpt-5.1",
                    usage_reporting_mode="openai_run_local",
                ),
            )
            log_run_context(
                logger,
                self._run_context(
                    provider="openai",
                    model="gpt-5.1",
                    usage_reporting_mode="openai_run_local",
                ),
            )
        text: str = "\n".join(captured.output)
        self.assertIn("llm_provider=openai", text)
        self.assertIn("llm_model_effective=gpt-5.1", text)
        self.assertIn("llm_model_configured=gpt-5.1", text)
        self.assertIn("llm_usage_reporting_mode=openai_run_local", text)

    def test_non_openai_provider_summary_fields_show_current_values(self) -> None:
        logger = logging.getLogger("provider-aware-claude-summary")
        config = self._config("claude")
        with self.assertLogs(logger, level="INFO") as captured:
            log_config_summary(
                logger,
                config,
                SimpleNamespace(
                    provider="claude",
                    model="claude-opus-4-6",
                    usage_reporting_mode="openai_run_local",
                ),
                self._run_context(
                    provider="claude",
                    model="claude-opus-4-6",
                    usage_reporting_mode="openai_run_local",
                ),
            )
            log_startup_summary(
                logger,
                self._startup_context(),
                SimpleNamespace(
                    provider="claude",
                    model="claude-opus-4-6",
                    usage_reporting_mode="openai_run_local",
                ),
            )
        text: str = "\n".join(captured.output)
        self.assertIn("llm_provider=claude", text)
        self.assertIn("llm_model_effective=claude-opus-4-6", text)
        self.assertIn("llm_model_configured=claude-opus-4-6", text)
        self.assertIn("llm_usage_reporting_mode=openai_run_local", text)
        self.assertIn("resolved_audit_mode=audit branches=nomerge,merge", text)

    def test_startup_health_logs_current_merge_policy(self) -> None:
        logger = logging.getLogger("provider-aware-health-claude")
        config = SimpleNamespace(
            llm=SimpleNamespace(
                provider="claude",
                model="claude-opus-4-6",
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
                    provider="claude",
                    model="claude-opus-4-6",
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
        self.assertIn("LLM policy: provider=claude effective_model=claude-opus-4-6", text)
        self.assertIn("usage_reporting_mode=openai_run_local", text)
        self.assertIn("LLM selection: audit branch execution=merge", text)
        self.assertIn(
            "LLM usage reporting note: scope=organization_aggregate source_of_truth_for_run=no current_run_effective_model=claude-opus-4-6",
            text,
        )


class EntrypointRegressionTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            processing=SimpleNamespace(mode="audit"),
            cleanup=SimpleNamespace(max_age_days=7),
            google=SimpleNamespace(
                doc_share_mode="anyone_writer",
                enabled=False,
                sheets_id="sheet-id",
                sheets_range="A:F",
            ),
            telegram=SimpleNamespace(enabled=False),
            llm=SimpleNamespace(provider="openai"),
            paths=SimpleNamespace(
                local_doc_dir_template="D:\\docs\\{date}",
                local_image_dir_template="D:\\images\\{date}",
            ),
            timezones=SimpleNamespace(kiev="Europe/Kyiv", cet="Europe/Berlin"),
        )

    def _llm_summary(self) -> SimpleNamespace:
        return SimpleNamespace(
            provider="openai",
            model="gpt-5.1",
            usage_reporting_mode="openai_run_local",
        )

    def _run_main(self) -> tuple[int, MagicMock, MagicMock, MagicMock]:
        batch_runner = MagicMock()
        batch_runner.last_merge_run_summary = None
        batch_runner.log_last_merge_run_summary.return_value = None
        build_llm_summary_mock = MagicMock(return_value=self._llm_summary())
        sheets_flag_mock = MagicMock(return_value=True)
        strip_flag_mock = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch.object(application_module, "load_dotenv"))
            stack.enter_context(
                patch.object(
                    application_module,
                    "get_project_paths",
                    return_value=SimpleNamespace(
                        project_root=Path("."),
                        entrypoint_path=Path("promo.py"),
                        runtime_config_path=Path("app/config/runtime/app_config.yaml"),
                        templates_path=Path("templates.yaml"),
                        secrets_env_path=Path("secrets/.env"),
                        oauth_credentials_path=Path("secrets/credentials.json"),
                        oauth_token_path=Path("secrets/token.json"),
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    application_module,
                    "build_cli_parser",
                    return_value=SimpleNamespace(
                        parse_args=lambda argv: SimpleNamespace(
                            debug=False,
                            dry_run=True,
                            audit_mode="merge",
                        )
                    ),
                )
            )
            stack.enter_context(patch.object(application_module, "setup_logging"))
            stack.enter_context(patch.object(application_module, "setup_runtime_analytics"))
            stack.enter_context(patch.object(application_module, "run_bootstrap_preflight"))
            stack.enter_context(
                patch.object(application_module, "_load_config_from_env", return_value=self._config())
            )
            stack.enter_context(
                patch.object(application_module, "build_llm_summary_snapshot", build_llm_summary_mock)
            )
            stack.enter_context(
                patch.object(
                    application_module,
                    "sheets_link_writeback_enabled_from_env",
                    sheets_flag_mock,
                )
            )
            stack.enter_context(
                patch.object(
                    application_module,
                    "strip_chapter_timestamps_enabled_from_env",
                    strip_flag_mock,
                )
            )
            stack.enter_context(patch.object(application_module, "_apply_llm_usage_reset"))
            stack.enter_context(
                patch.object(application_module, "normalize_processing_mode", return_value="audit")
            )
            stack.enter_context(
                patch.object(application_module, "normalize_audit_mode", return_value="merge")
            )
            stack.enter_context(
                patch.object(application_module, "StartupContext", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch.object(application_module, "RunContext", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch.object(application_module, "audit_branch_labels", return_value=["merge"])
            )
            stack.enter_context(patch.object(application_module, "log_startup_summary"))
            stack.enter_context(patch.object(application_module, "log_run_context"))
            stack.enter_context(patch.object(application_module, "log_config_summary"))
            stack.enter_context(
                patch.object(application_module, "_load_zoneinfo", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch.object(
                    PipelineApplication,
                    "_build_batch_runner",
                    return_value=batch_runner,
                )
            )
            stack.enter_context(patch.object(application_module, "log_run_started"))
            stack.enter_context(patch.object(application_module, "_log_llm_usage_reports"))
            stack.enter_context(patch.object(application_module, "record_stage_duration"))
            stack.enter_context(patch.object(application_module, "log_stage_timing"))
            stack.enter_context(patch.object(application_module, "emit_final_run_summary"))
            stack.enter_context(patch.object(application_module, "_log_exit_code"))
            exit_code = promo.main([])

        return exit_code, build_llm_summary_mock, sheets_flag_mock, strip_flag_mock

    def test_entrypoint_builds_llm_summary_once(self) -> None:
        exit_code, build_llm_summary_mock, _, _ = self._run_main()
        self.assertEqual(0, exit_code)
        self.assertEqual(1, build_llm_summary_mock.call_count)

    def test_entrypoint_reads_env_derived_flags_once(self) -> None:
        exit_code, _, sheets_flag_mock, strip_flag_mock = self._run_main()
        self.assertEqual(0, exit_code)
        self.assertEqual(1, sheets_flag_mock.call_count)
        self.assertEqual(1, strip_flag_mock.call_count)


if __name__ == "__main__":
    unittest.main()
