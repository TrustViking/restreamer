from __future__ import annotations

import os
import unittest

from app.config.app_config_loader import (
    load_google_oauth_credentials_path,
    load_google_oauth_token_path,
)
from app.paths import get_project_paths


class ProjectPathsTests(unittest.TestCase):
    def test_project_paths_resolve_to_new_layout(self) -> None:
        paths = get_project_paths()
        self.assertTrue(str(paths.project_root).endswith("restreamer"))
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
