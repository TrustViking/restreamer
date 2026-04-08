from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config.app_config_loader import (
    load_google_oauth_credentials_path,
    load_google_oauth_token_path,
)
from app.paths import _resolve_project_root, get_project_paths


class ProjectPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        get_project_paths.cache_clear()

    def tearDown(self) -> None:
        get_project_paths.cache_clear()

    def test_project_paths_resolve_to_new_layout(self) -> None:
        paths = get_project_paths()

        self.assertEqual(paths.entrypoint_path.parent, paths.project_root)
        self.assertEqual(
            paths.runtime_config_path.relative_to(paths.project_root),
            Path("app") / "config" / "runtime" / "app_config.yaml",
        )
        self.assertEqual(
            paths.runtime_config_example_path.relative_to(paths.project_root),
            Path("app") / "config" / "runtime" / "app_config.example.yaml",
        )
        self.assertEqual(
            paths.templates_path.relative_to(paths.project_root),
            Path("app") / "llm" / "prompts" / "templates.yaml",
        )
        self.assertEqual(
            paths.secrets_env_path.relative_to(paths.project_root),
            Path("secrets") / ".env",
        )
        self.assertEqual(
            paths.oauth_token_path.relative_to(paths.project_root),
            Path("secrets") / "token.json",
        )
        self.assertEqual(
            paths.oauth_credentials_path.relative_to(paths.project_root),
            Path("secrets") / "credentials.json",
        )

        self.assertTrue(str(paths.entrypoint_path).endswith("restreamer.py"))
        self.assertTrue(
            str(paths.runtime_config_path).endswith(os.path.join("app", "config", "runtime", "app_config.yaml"))
        )
        self.assertTrue(
            str(paths.runtime_config_example_path).endswith(
                os.path.join("app", "config", "runtime", "app_config.example.yaml")
            )
        )
        self.assertTrue(
            str(paths.templates_path).endswith(os.path.join("app", "llm", "prompts", "templates.yaml"))
        )
        self.assertTrue(str(paths.secrets_env_path).endswith(os.path.join("secrets", ".env")))
        self.assertTrue(str(paths.oauth_token_path).endswith(os.path.join("secrets", "token.json")))
        self.assertTrue(
            str(paths.oauth_credentials_path).endswith(os.path.join("secrets", "credentials.json"))
        )
        self.assertEqual(paths.bundled_config_path, paths.runtime_config_path)
        self.assertEqual(paths.bundled_templates_path, paths.templates_path)

    def test_google_oauth_defaults_use_secrets_directory(self) -> None:
        self.assertEqual(get_project_paths().oauth_credentials_path, load_google_oauth_credentials_path())
        self.assertEqual(get_project_paths().oauth_token_path, load_google_oauth_token_path())


class FrozenProjectRootTests(unittest.TestCase):
    def setUp(self) -> None:
        get_project_paths.cache_clear()

    def tearDown(self) -> None:
        get_project_paths.cache_clear()

    def test_dev_mode_resolver_matches_project_paths_root(self) -> None:
        self.assertEqual(_resolve_project_root(), get_project_paths().project_root)

    def test_frozen_mode_uses_executable_parent(self) -> None:
        fake_exe: Path = (Path.cwd() / "portable" / "restreamer.exe").resolve()
        with patch("app.paths._root.sys") as mock_sys:
            mock_sys.frozen = True
            mock_sys.executable = str(fake_exe)
            resolved_root: Path = _resolve_project_root()
        self.assertEqual(fake_exe.parent, resolved_root)


class FrozenProjectPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        get_project_paths.cache_clear()

    def tearDown(self) -> None:
        get_project_paths.cache_clear()

    def test_frozen_layout_uses_external_user_configs_and_internal_defaults(self) -> None:
        fake_project_root: Path = (Path.cwd() / "portable").resolve()
        fake_internal_root: Path = (fake_project_root / "_internal").resolve()
        with (
            patch("app.paths.project_paths.PROJECT_ROOT", fake_project_root),
            patch("app.paths.project_paths.sys") as mock_sys,
        ):
            mock_sys.frozen = True
            mock_sys._MEIPASS = str(fake_internal_root)
            paths = get_project_paths()

        self.assertEqual(paths.project_root, fake_project_root)
        self.assertEqual(
            paths.runtime_config_path,
            fake_project_root / "config" / "app_config.yaml",
        )
        self.assertEqual(
            paths.templates_path,
            fake_project_root / "config" / "templates.yaml",
        )
        self.assertEqual(
            paths.runtime_config_example_path,
            fake_internal_root / "app" / "config" / "runtime" / "app_config.example.yaml",
        )
        self.assertEqual(
            paths.bundled_config_path,
            fake_internal_root / "app" / "config" / "runtime" / "app_config.yaml",
        )
        self.assertEqual(
            paths.bundled_templates_path,
            fake_internal_root / "app" / "llm" / "prompts" / "templates.yaml",
        )


if __name__ == "__main__":
    unittest.main()
