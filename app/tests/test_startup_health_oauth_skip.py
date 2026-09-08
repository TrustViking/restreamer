from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from app.observability.startup_health import run_startup_health_checks


class StartupHealthOAuthSkipTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
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

    def _llm_summary(self) -> SimpleNamespace:
        return SimpleNamespace(
            provider="claude",
            model="claude-opus-4-6",
            usage_reporting_mode="openai_run_local",
        )

    def _services(self, *, factory: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            factory=factory,
            drive_client=SimpleNamespace(
                get_file_owner_info=Mock(return_value="owner@example.com"),
                ping_access=Mock(return_value=("user", "user@example.com")),
            ),
            sheets_client=SimpleNamespace(
                ping_access=Mock(return_value=("sheet-id", "Sheet title"))
            ),
            docs_client=SimpleNamespace(ping_access=Mock(return_value="ok")),
        )

    def _run(self, *, logger: logging.Logger, services: SimpleNamespace) -> None:
        run_startup_health_checks(
            logger=logger,
            config=self._config(),
            llm_summary=self._llm_summary(),
            services=services,
            telegram_client=SimpleNamespace(
                get_me=Mock(return_value={"id": 1, "username": "bot"})
            ),
            resolve_logger_name_meta=lambda: ("logger", "test", False),
            dry_run=False,
            resolved_audit_mode="nomerge",
            run_id="run",
        )

    def test_oauth_mode_skips_google_project_lookup(self) -> None:
        project_info = Mock(side_effect=AssertionError("must not be called"))
        services = self._services(
            factory=SimpleNamespace(
                get_auth_mode=Mock(return_value="oauth"),
                get_oauth_paths=Mock(return_value=("cred", "token")),
                get_runtime_principal_email=Mock(return_value="user@example.com"),
                get_google_project_info=project_info,
            )
        )
        logger = logging.getLogger("startup-health-oauth-skip")
        with self.assertLogs(logger, level="INFO") as captured:
            self._run(logger=logger, services=services)
        project_info.assert_not_called()
        self.assertIn(
            "OAuth credentials do not carry project_id",
            "\n".join(captured.output),
        )
        self.assertTrue(
            any(
                record.levelno == logging.INFO
                and "OAuth credentials do not carry project_id" in record.getMessage()
                for record in captured.records
            )
        )

    def test_service_account_success_keeps_project_lookup(self) -> None:
        project_info = Mock(return_value=("project-id-123", "Project Name"))
        services = self._services(
            factory=SimpleNamespace(
                get_auth_mode=Mock(return_value="service_account"),
                get_runtime_principal_email=Mock(return_value="sa@example.com"),
                get_google_project_info=project_info,
            )
        )
        logger = logging.getLogger("startup-health-service-account-success")
        with self.assertLogs(logger, level="INFO") as captured:
            self._run(logger=logger, services=services)
        project_info.assert_called_once_with(strict=True)
        self.assertIn(
            "Google project resolved via API: name=Project Name id=project-id-123",
            "\n".join(captured.output),
        )

    def test_service_account_failure_keeps_informational_warning(self) -> None:
        services = self._services(
            factory=SimpleNamespace(
                get_auth_mode=Mock(return_value="service_account"),
                get_runtime_principal_email=Mock(return_value="sa@example.com"),
                get_google_project_info=Mock(side_effect=RuntimeError("test error")),
            )
        )
        logger = logging.getLogger("startup-health-service-account-warning")
        with self.assertLogs(logger, level="INFO") as captured:
            self._run(logger=logger, services=services)
        self.assertTrue(
            any(
                record.levelno == logging.WARNING
                and getattr(record, "reason_code", "")
                == "google_project_id_unavailable"
                for record in captured.records
            )
        )

    def test_auth_mode_lookup_failure_uses_unknown_and_runs_project_lookup(self) -> None:
        project_info = Mock(return_value=("project-id-123", "Project Name"))
        services = self._services(
            factory=SimpleNamespace(
                get_auth_mode=Mock(side_effect=RuntimeError("auth mode failed")),
                get_runtime_principal_email=Mock(return_value="sa@example.com"),
                get_google_project_info=project_info,
            )
        )
        logger = logging.getLogger("startup-health-auth-mode-unknown")
        with self.assertLogs(logger, level="INFO") as captured:
            self._run(logger=logger, services=services)
        project_info.assert_called_once_with(strict=True)
        self.assertIn(
            "Google project resolved via API: name=Project Name id=project-id-123",
            "\n".join(captured.output),
        )


if __name__ == "__main__":
    unittest.main()
