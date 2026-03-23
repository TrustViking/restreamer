from __future__ import annotations

import os
import unittest
from pathlib import Path

from app.config.app_config_loader import (
    load_google_oauth_credentials_path,
    load_google_oauth_token_path,
)
from app.paths import get_project_paths


class ProjectPathsTests(unittest.TestCase):
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

    def test_google_oauth_defaults_use_secrets_directory(self) -> None:
        self.assertEqual(get_project_paths().oauth_credentials_path, load_google_oauth_credentials_path())
        self.assertEqual(get_project_paths().oauth_token_path, load_google_oauth_token_path())


if __name__ == "__main__":
    unittest.main()
