from __future__ import annotations

import logging
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import restreamer
from app.bootstrap.run_context import RunContext, StartupContext
from app.core.branching import BRANCH_MERGE_MAIN, BRANCH_MERGE_MAIN_FALLBACK_PACKAGING
from app.observability.runtime_analytics import log_run_context
from app.observability.startup_health import run_startup_health_checks
from app.observability.startup_summary import log_config_summary, log_startup_summary
from restreamer import _apply_llm_usage_reset, _log_llm_usage_reports


class ProviderAwareUsageHooksTests(unittest.TestCase):
    def test_openai_usage_hooks_are_active(self) -> None:
        logger = logging.getLogger("provider-aware-openai-usage")
        config = SimpleNamespace(llm_provider="openai", openai_model_primary="gpt-5.1", openai_model_fallback="gpt-5-mini")
        with patch("restreamer.reset_run_local_openai_usage") as reset_mock, patch(
            "restreamer.log_run_local_openai_usage"
        ) as local_usage_mock, patch("restreamer.log_openai_limits_and_usage") as org_usage_mock, self.assertLogs(
            logger, level="INFO"
        ) as captured:
            _apply_llm_usage_reset(
                logger=logger,
                llm_summary=SimpleNamespace(provider="openai", providers_used=("openai",)),
            )
            _log_llm_usage_reports(
                logger=logger,
                llm_summary=SimpleNamespace(provider="openai", providers_used=("openai",)),
            )
        self.assertEqual(1, reset_mock.call_count)
        self.assertEqual(1, local_usage_mock.call_count)
        self.assertEqual(1, org_usage_mock.call_count)
        text: str = "\n".join(captured.output)
        self.assertIn("llm_usage_reset_applied provider=openai", text)
        self.assertIn("llm_usage_report_completed provider=openai status=completed", text)

    def test_deepseek_usage_hooks_are_skipped(self) -> None:
        logger = logging.getLogger("provider-aware-deepseek-usage")
        config = SimpleNamespace(llm_provider="deepseek", deepseek_model="deepseek-chat")
        with patch("restreamer.reset_run_local_openai_usage") as reset_mock, patch(
            "restreamer.log_run_local_openai_usage"
        ) as local_usage_mock, patch("restreamer.log_openai_limits_and_usage") as org_usage_mock, self.assertLogs(
            logger, level="INFO"
        ) as captured:
            _apply_llm_usage_reset(
                logger=logger,
                llm_summary=SimpleNamespace(provider="deepseek", providers_used=("deepseek",)),
            )
            _log_llm_usage_reports(
                logger=logger,
                llm_summary=SimpleNamespace(provider="deepseek", providers_used=("deepseek",)),
            )
        self.assertEqual(0, reset_mock.call_count)
        self.assertEqual(0, local_usage_mock.call_count)
        self.assertEqual(0, org_usage_mock.call_count)
        text: str = "\n".join(captured.output)
        self.assertIn("llm_usage_reset_skipped provider=deepseek reason=provider_not_openai", text)
        self.assertIn("llm_usage_report_skipped provider=deepseek reason=provider_not_openai", text)
        self.assertIn("llm_org_usage_report_skipped provider=deepseek reason=provider_not_openai", text)
        self.assertIn(
            "llm_usage_report_completed provider=deepseek status=skipped_provider_logs_only",
            text,
        )

    def test_openai_usage_hooks_remain_active_when_openai_is_only_fallback(self) -> None:
        logger = logging.getLogger("provider-aware-mixed-usage")
        config = SimpleNamespace(
            llm_provider="deepseek",
            llm_main_model="deepseek-chat",
            llm_fallback_model="gpt-5.1",
            deepseek_model="deepseek-chat",
            openai_model_primary="gpt-5.1",
            openai_model_fallback="gpt-5-mini",
        )
        with patch("restreamer.reset_run_local_openai_usage") as reset_mock, patch(
            "restreamer.log_run_local_openai_usage"
        ) as local_usage_mock, patch("restreamer.log_openai_limits_and_usage") as org_usage_mock, self.assertLogs(
            logger, level="INFO"
        ) as captured:
            llm_summary = SimpleNamespace(provider="deepseek", providers_used=("deepseek", "openai"))
            _apply_llm_usage_reset(logger=logger, llm_summary=llm_summary)
            _log_llm_usage_reports(logger=logger, llm_summary=llm_summary)
        self.assertEqual(1, reset_mock.call_count)
        self.assertEqual(1, local_usage_mock.call_count)
        self.assertEqual(1, org_usage_mock.call_count)
        text: str = "\n".join(captured.output)
        self.assertIn("llm_usage_reset_applied provider=deepseek", text)
        self.assertIn("llm_usage_report_completed provider=deepseek status=completed", text)


class ProviderAwareSummaryTests(unittest.TestCase):
    def _run_context(
        self,
        *,
        provider: str,
        primary_provider: str,
        fallback_provider: str,
        primary_model: str,
        fallback_model: str,
        base_url: str,
        usage_reporting_mode: str,
    ) -> RunContext:
        return RunContext(
            run_id="run",
            processing_mode="audit",
            audit_mode="merge",
            config_processing_mode="audit",
            audit_branches=["merge"],
            debug_enabled=False,
            dry_run=False,
            google_enabled=False,
            telegram_enabled=False,
            llm_provider=provider,
            llm_primary_provider=primary_provider,
            llm_fallback_provider=fallback_provider,
            llm_effective_primary_model=primary_model,
            llm_effective_fallback_model=fallback_model,
            llm_merge_stage_model=primary_model,
            llm_packaging_stage_model=fallback_model,
            llm_base_url=base_url,
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
            args_audit_mode="merge",
            args_debug=False,
            args_dry_run=False,
            processing_mode="audit",
            config_processing_mode_raw="audit",
            project_root=Path("."),
            entrypoint_path=Path("restreamer.py"),
            runtime_config_path=Path("app_config.yaml"),
            templates_path=Path("templates.yaml"),
            secrets_env_path=Path(".env"),
            oauth_credentials_path=Path("oauth.json"),
            oauth_token_path=Path("token.json"),
        )

    def _config(self, provider: str) -> SimpleNamespace:
        return SimpleNamespace(
            google_enabled=False,
            telegram_enabled=False,
            stg_templates_path=Path("templates.yaml"),
            google_sheets_id="sheet-id",
            google_sheets_range="A:F",
            processing_mode="audit",
            now_tz_mode="kyiv",
            llm_provider=provider,
            llm_main_model="gpt-5.1" if provider == "openai" else "deepseek-chat",
            llm_fallback_model="gpt-5-mini" if provider == "openai" else "deepseek-chat",
            openai_model_primary="gpt-5.1",
            openai_model_fallback="gpt-5-mini",
            openai_timeout_sec=30.0,
            deepseek_model="deepseek-chat",
            deepseek_base_url="https://api.deepseek.com/v1",
            deepseek_timeout_sec=45.0,
            deepseek_max_retries=2,
            openai_max_output_tokens=1000,
            llm_source_desc_max_chars=500,
            llm_run_if_single_source=False,
            openai_pre_delay_sec=0.0,
            local_doc_dir_template="D:\\docs\\{date}",
        )

    def test_openai_summary_fields_show_provider_aware_values(self) -> None:
        logger = logging.getLogger("provider-aware-openai-summary")
        config = self._config("openai")
        with self.assertLogs(logger, level="INFO") as captured:
            log_config_summary(
                logger,
                config,
                SimpleNamespace(
                    provider="openai",
                    primary_provider="openai",
                    fallback_provider="openai",
                    effective_primary_model="gpt-5.1",
                    effective_fallback_model="gpt-5-mini",
                    merge_stage_model="gpt-5.1",
                    packaging_stage_model="gpt-5-mini",
                    base_url="default_openai",
                    usage_reporting_mode="openai_run_local+openai_org_snapshot",
                    providers_used=("openai",),
                    is_mixed_provider=False,
                ),
                self._run_context(
                    provider="openai",
                    primary_provider="openai",
                    fallback_provider="openai",
                    primary_model="gpt-5.1",
                    fallback_model="gpt-5-mini",
                    base_url="default_openai",
                    usage_reporting_mode="openai_run_local+openai_org_snapshot",
                ),
            )
            log_run_context(
                logger,
                self._run_context(
                    provider="openai",
                    primary_provider="openai",
                    fallback_provider="openai",
                    primary_model="gpt-5.1",
                    fallback_model="gpt-5-mini",
                    base_url="default_openai",
                    usage_reporting_mode="openai_run_local+openai_org_snapshot",
                ),
            )
        text: str = "\n".join(captured.output)
        self.assertIn("llm_provider=openai", text)
        self.assertIn("llm_primary_provider=openai", text)
        self.assertIn("llm_fallback_provider=openai", text)
        self.assertIn("llm_effective_primary_model=gpt-5.1", text)
        self.assertIn("llm_effective_fallback_model=gpt-5-mini", text)
        self.assertIn("merge_stage_model=gpt-5.1", text)
        self.assertIn("packaging_stage_model=gpt-5-mini", text)
        self.assertIn("llm_base_url=default_openai", text)
        self.assertIn("llm_usage_reporting_mode=openai_run_local+openai_org_snapshot", text)

    def test_deepseek_summary_fields_show_provider_aware_values(self) -> None:
        logger = logging.getLogger("provider-aware-deepseek-summary")
        config = self._config("deepseek")
        with self.assertLogs(logger, level="INFO") as captured:
            log_config_summary(
                logger,
                config,
                SimpleNamespace(
                    provider="deepseek",
                    primary_provider="deepseek",
                    fallback_provider="openai",
                    effective_primary_model="deepseek-chat",
                    effective_fallback_model="gpt-5.1",
                    merge_stage_model="deepseek-chat",
                    packaging_stage_model="gpt-5.1",
                    base_url="primary=https://api.deepseek.com/v1;fallback=default_openai",
                    usage_reporting_mode="openai_run_local+openai_org_snapshot+provider_logs_only",
                    providers_used=("deepseek", "openai"),
                    is_mixed_provider=True,
                ),
                self._run_context(
                    provider="deepseek",
                    primary_provider="deepseek",
                    fallback_provider="openai",
                    primary_model="deepseek-chat",
                    fallback_model="gpt-5.1",
                    base_url="primary=https://api.deepseek.com/v1;fallback=default_openai",
                    usage_reporting_mode="openai_run_local+openai_org_snapshot+provider_logs_only",
                ),
            )
            log_startup_summary(
                logger,
                self._startup_context(),
                SimpleNamespace(
                    provider="deepseek",
                    primary_provider="deepseek",
                    fallback_provider="openai",
                    effective_primary_model="deepseek-chat",
                    effective_fallback_model="gpt-5.1",
                    merge_stage_model="deepseek-chat",
                    packaging_stage_model="gpt-5.1",
                    base_url="primary=https://api.deepseek.com/v1;fallback=default_openai",
                    usage_reporting_mode="openai_run_local+openai_org_snapshot+provider_logs_only",
                    providers_used=("deepseek", "openai"),
                    is_mixed_provider=True,
                ),
            )
        text: str = "\n".join(captured.output)
        self.assertIn("llm_provider=deepseek", text)
        self.assertIn("llm_primary_provider=deepseek", text)
        self.assertIn("llm_fallback_provider=openai", text)
        self.assertIn("llm_effective_primary_model=deepseek-chat", text)
        self.assertIn("llm_effective_fallback_model=gpt-5.1", text)
        self.assertIn("merge_stage_model=deepseek-chat", text)
        self.assertIn("packaging_stage_model=gpt-5.1", text)
        self.assertIn("llm_base_url=primary=https://api.deepseek.com/v1;fallback=default_openai", text)
        self.assertIn("llm_usage_reporting_mode=openai_run_local+openai_org_snapshot+provider_logs_only", text)

    def test_startup_health_logs_deepseek_usage_reporting_expectation(self) -> None:
        logger = logging.getLogger("provider-aware-health-deepseek")
        config = SimpleNamespace(
            llm_provider="deepseek",
            openai_model_primary="gpt-5.1",
            openai_model_fallback="gpt-5-mini",
            openai_timeout_sec=30.0,
            openai_max_output_tokens=1000,
            llm_source_desc_max_chars=500,
            openai_pre_delay_sec=0.0,
            deepseek_model="deepseek-chat",
            deepseek_timeout_sec=45.0,
            deepseek_max_retries=2,
            deepseek_base_url="https://api.deepseek.com/v1",
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
                    provider="deepseek",
                    primary_provider="deepseek",
                    fallback_provider="openai",
                    effective_primary_model="deepseek-chat",
                    effective_fallback_model="gpt-5.1",
                    merge_stage_model="deepseek-chat",
                    packaging_stage_model="gpt-5.1",
                    base_url="primary=https://api.deepseek.com/v1;fallback=default_openai",
                    usage_reporting_mode="provider_logs_only",
                    providers_used=("deepseek",),
                ),
                services=services,
                telegram_client=telegram_client,
                resolve_logger_name_meta=lambda: ("logger", "test", False),
                dry_run=False,
                resolved_audit_mode="merge",
                run_id="run",
            )
        text: str = "\n".join(captured.output)
        self.assertIn("LLM policy: provider=deepseek primary_model=deepseek-chat", text)
        self.assertIn("usage_reporting_mode=provider_logs_only", text)
        self.assertIn(
            f"LLM selection: audit branch execution={BRANCH_MERGE_MAIN},{BRANCH_MERGE_MAIN_FALLBACK_PACKAGING}",
            text,
        )
        self.assertIn(
            "LLM usage reporting note: provider=deepseek openai_usage_summary_expected=no",
            text,
        )


class EntrypointRegressionTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            processing_mode="audit",
            google_doc_share_mode="anyone_writer",
            timezone_kiev="Europe/Kyiv",
            timezone_cet="Europe/Berlin",
        )

    def _llm_summary(self) -> SimpleNamespace:
        return SimpleNamespace(
            provider="openai",
            primary_provider="openai",
            fallback_provider="deepseek",
            effective_primary_model="gpt-5.1",
            effective_fallback_model="deepseek-chat",
            merge_stage_model="gpt-5.1",
            packaging_stage_model="deepseek-chat",
            base_url="default_openai",
            usage_reporting_mode="openai_run_local+openai_org_snapshot+provider_logs_only",
            providers_used=("openai", "deepseek"),
        )

    def _run_main(self) -> tuple[int, MagicMock, MagicMock, MagicMock]:
        batch_runner = MagicMock()
        batch_runner.last_merge_run_summary = None
        batch_runner.log_last_merge_run_summary.return_value = None
        build_llm_summary_mock = MagicMock(return_value=self._llm_summary())
        sheets_flag_mock = MagicMock(return_value=True)
        strip_flag_mock = MagicMock(return_value=False)

        with ExitStack() as stack:
            stack.enter_context(patch.object(restreamer, "load_dotenv"))
            stack.enter_context(
                patch.object(
                    restreamer,
                    "get_project_paths",
                    return_value=SimpleNamespace(
                        project_root=Path("."),
                        entrypoint_path=Path("restreamer.py"),
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
                    restreamer,
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
            stack.enter_context(patch.object(restreamer, "setup_logging"))
            stack.enter_context(patch.object(restreamer, "setup_runtime_analytics"))
            stack.enter_context(patch.object(restreamer, "run_bootstrap_preflight"))
            stack.enter_context(
                patch.object(restreamer, "_load_config_from_env", return_value=self._config())
            )
            stack.enter_context(
                patch.object(restreamer, "build_llm_summary_snapshot", build_llm_summary_mock)
            )
            stack.enter_context(
                patch.object(restreamer, "sheets_link_writeback_enabled_from_env", sheets_flag_mock)
            )
            stack.enter_context(
                patch.object(
                    restreamer,
                    "strip_chapter_timestamps_enabled_from_env",
                    strip_flag_mock,
                )
            )
            stack.enter_context(patch.object(restreamer, "_apply_llm_usage_reset"))
            stack.enter_context(
                patch.object(restreamer, "normalize_processing_mode", return_value="audit")
            )
            stack.enter_context(
                patch.object(restreamer, "normalize_audit_mode", return_value="merge")
            )
            stack.enter_context(
                patch.object(restreamer, "build_startup_context", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch.object(restreamer, "build_run_context", return_value=SimpleNamespace())
            )
            stack.enter_context(patch.object(restreamer, "log_startup_summary"))
            stack.enter_context(patch.object(restreamer, "log_run_context"))
            stack.enter_context(patch.object(restreamer, "log_config_summary"))
            stack.enter_context(
                patch.object(restreamer, "_load_zoneinfo", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch.object(
                    restreamer,
                    "build_entrypoint_runtime_services",
                    return_value=SimpleNamespace(batch_runner=batch_runner),
                )
            )
            stack.enter_context(patch.object(restreamer, "log_run_started"))
            stack.enter_context(patch.object(restreamer, "_log_llm_usage_reports"))
            stack.enter_context(patch.object(restreamer, "record_stage_duration"))
            stack.enter_context(patch.object(restreamer, "log_stage_timing"))
            stack.enter_context(patch.object(restreamer, "emit_final_run_summary"))
            stack.enter_context(patch.object(restreamer, "_log_exit_code"))
            exit_code = restreamer.main([])

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
