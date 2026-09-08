"""Verify dev-mode preflight doesn't block when app_config.yaml is missing but example exists."""
from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app.paths import get_project_paths


class DevBootstrapConfigTest(unittest.TestCase):
    """In dev mode, missing app_config.yaml must not produce a blocking error."""

    def setUp(self) -> None:
        get_project_paths.cache_clear()
        self.paths = get_project_paths()
        self.config_path: Path = self.paths.runtime_config_path
        self.example_path: Path = self.paths.runtime_config_example_path
        self.backup_path: Path = self.config_path.with_suffix(".yaml.test_bak")

        if not self.example_path.exists():
            self.skipTest(f"Example config not found at {self.example_path}")

        self._had_config: bool = self.config_path.exists()
        if self._had_config:
            shutil.copy2(self.config_path, self.backup_path)
            self.config_path.unlink()

    def tearDown(self) -> None:
        if self.backup_path.exists():
            shutil.copy2(self.backup_path, self.config_path)
            self.backup_path.unlink()
        elif not self._had_config and self.config_path.exists():
            self.config_path.unlink()
        get_project_paths.cache_clear()

    def test_dev_preflight_no_blocking_error_without_app_config(self) -> None:
        from setup_deploy import STATUS_ERROR, ensure_runtime_config

        self.assertFalse(getattr(sys, "frozen", False), "Test must run in dev mode")
        self.assertFalse(self.config_path.exists())

        result = ensure_runtime_config(self.paths)
        self.assertNotEqual(
            result.status,
            STATUS_ERROR,
            f"ensure_runtime_config should bootstrap from example, got: {result.message}",
        )

    def test_ensure_user_configs_not_called_in_dev_mode(self) -> None:
        from setup_deploy import CheckResult, STATUS_PASS, run_preflight

        self.assertFalse(getattr(sys, "frozen", False), "Test must run in dev mode")

        with (
            patch(
                "setup_deploy.check_python_version",
                return_value=CheckResult("Python version", STATUS_PASS, "ok"),
            ),
            patch("setup_deploy.ensure_required_directories", return_value=[]),
            patch("setup_deploy.ensure_user_configs") as ensure_user_configs_mock,
            patch("setup_deploy.load_or_bootstrap_env", return_value=([], {})),
            patch("setup_deploy.validate_required_env_vars", return_value=[]),
            patch(
                "setup_deploy.load_google_auth_mode",
                return_value="oauth",
            ),
            patch(
                "setup_deploy.check_google_oauth_credentials",
                return_value=CheckResult("Google OAuth credentials", STATUS_PASS, "ok"),
            ),
            patch(
                "setup_deploy.ensure_runtime_config",
                return_value=CheckResult("Runtime config", STATUS_PASS, "ok"),
            ),
            patch(
                "setup_deploy.validate_core_imports",
                return_value=CheckResult("Core Python imports", STATUS_PASS, "ok"),
            ),
            patch("setup_deploy.print_summary", return_value=0),
        ):
            exit_code = run_preflight(include_network_checks=False)

        ensure_user_configs_mock.assert_not_called()
        self.assertEqual(0, exit_code)


if __name__ == "__main__":
    unittest.main()
