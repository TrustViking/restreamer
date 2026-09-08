from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from app.observability.startup_health import (
    StartupHealthResult,
    run_startup_health_checks,
)
from app.pipeline.batch_runner import BatchRunner
from app.pipeline.operator_notifier import OperatorNotifier


class StartupHealthResultTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
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

    def _llm_summary(self) -> SimpleNamespace:
        return SimpleNamespace(
            provider="openai",
            model="gpt-5.1",
            usage_reporting_mode="openai_run_local",
        )

    def _services(self) -> SimpleNamespace:
        return SimpleNamespace(
            factory=SimpleNamespace(
                get_auth_mode=Mock(return_value="oauth"),
                get_oauth_paths=Mock(return_value=("cred", "token")),
                get_runtime_principal_email=Mock(return_value="user@example.com"),
                get_google_project_info=Mock(return_value=("id", "name")),
            ),
            drive_client=SimpleNamespace(
                get_file_owner_info=Mock(return_value="owner@example.com"),
                ping_access=Mock(return_value=("user", "user@example.com")),
            ),
            sheets_client=SimpleNamespace(
                ping_access=Mock(return_value=("sheet-id", "Sheet title"))
            ),
            docs_client=SimpleNamespace(ping_access=Mock(return_value="ok")),
        )

    def _run(
        self, *, logger: logging.Logger, telegram_client: SimpleNamespace
    ) -> StartupHealthResult:
        return run_startup_health_checks(
            logger=logger,
            config=self._config(),
            llm_summary=self._llm_summary(),
            services=self._services(),
            telegram_client=telegram_client,
            resolve_logger_name_meta=lambda: ("logger", "test", False),
            dry_run=False,
            resolved_audit_mode="nomerge",
            run_id="run",
        )

    def _build_runner(self) -> BatchRunner:
        return BatchRunner(
            logger=logging.getLogger("startup-health-result-runner"),
            config=SimpleNamespace(
                google=SimpleNamespace(enabled=True),
                templates=SimpleNamespace(),
            ),
            metadata_fetcher=MagicMock(),
            http_client=MagicMock(),
            telegram_client=MagicMock(),
            name_builder=MagicMock(),
            kiev_tz=MagicMock(),
            cet_tz=MagicMock(),
            resolve_logger_name_meta=MagicMock(return_value=("logger", "test", False)),
            notifier=OperatorNotifier(),
        )

    def test_all_checks_pass_returns_ok_result(self) -> None:
        logger = logging.getLogger("startup-health-result-ok")
        with self.assertLogs(logger, level="INFO"):
            result = self._run(
                logger=logger,
                telegram_client=SimpleNamespace(
                    get_me=Mock(return_value={"id": 1, "username": "bot"})
                ),
            )
        self.assertIs(True, result.ok)
        self.assertEqual((), result.failed_checks)

    def test_telegram_failure_is_reported_in_failed_checks(self) -> None:
        logger = logging.getLogger("startup-health-result-telegram-failure")
        with self.assertLogs(logger, level="INFO"):
            result = self._run(
                logger=logger,
                telegram_client=SimpleNamespace(
                    get_me=Mock(side_effect=RuntimeError("timeout"))
                ),
            )
        self.assertIs(False, result.ok)
        self.assertEqual(1, len(result.failed_checks))
        self.assertIn("Telegram connection failed", result.failed_checks[0])

    def test_operator_summary_line_reflects_failed_checks(self) -> None:
        runner = self._build_runner()
        failed_line = runner._startup_health_summary_line(  # type: ignore[attr-defined]
            startup_health=StartupHealthResult(
                llm_merge_enabled=False,
                failed_checks=("Telegram connection failed: timeout",),
            )
        )
        clean_line = runner._startup_health_summary_line(  # type: ignore[attr-defined]
            startup_health=StartupHealthResult(
                llm_merge_enabled=True, failed_checks=()
            )
        )
        self.assertEqual("⚠️ Проверка сервисов и конфигурации: ошибок 1", failed_line)
        self.assertEqual("✅ Проверка сервисов и конфигурации: OK", clean_line)


if __name__ == "__main__":
    unittest.main()
