from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config.app_config_loader import load_config_from_env
from app.llm.llm_factory import get_llm_provider
from app.observability.startup_summary import build_llm_summary_snapshot
from app.paths import get_project_paths


class ActiveModelConfigTests(unittest.TestCase):
    def _logger(self) -> SimpleNamespace:
        return SimpleNamespace(
            info=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        )

    def _write_runtime_config(self) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
            handle.write(
                """
app:
  processing:
    mode: audit
  google:
    enabled: false
    drive_folder_id: "folder"
    drive_preview_folder_id: "preview"
    drive_preview_path_template: "preview/{date}/{language}"
    doc_share_mode: anyone_writer
    sheets_id: "sheet-id"
    sheets_range: "A:F"
    form_url: "https://forms.gle/example"
    contacts: "@contact"
  paths:
    local_image_dir_template: "./image/{language}/{date}"
    local_doc_dir_template: "./docs/{date}"
  timezones:
    kiev: "Europe/Kyiv"
    cet: "Europe/Berlin"
  telegram:
    enabled: false
    use_audit: false
    symbol_separator: "-"
    separator_repeat_count: 2
    symbol_separator_start: "."
    separator_start_repeat_count: 3
    symbol_broadcast: "B"
    symbol_alert: "A"
    symbol_form: "F"
    symbol_description: "D"
    symbol_pin: "P"
    symbol_done: "Y"
    flag_repeat_count: 1
    send_delay_seconds: 1.0
    max_retries: 3
  files:
    preview_filename_max_stem: 120
""".strip()
            )
            return handle.name

    def test_relative_local_paths_are_resolved_from_entrypoint_directory(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "GOOGLE_DRIVE_FOLDER_ID": "drive-folder-id",
            "GOOGLE_DRIVE_PREVIEW_FOLDER_ID": "drive-preview-folder-id",
            "GOOGLE_SHEETS_ID": "sheet-id",
            "OPENAI_MODEL": "gpt-5.1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=self._logger())
        entrypoint_dir = get_project_paths().entrypoint_path.parent.resolve()
        self.assertEqual(str(entrypoint_dir / Path("image") / "{language}" / "{date}"), config.paths.local_image_dir_template)
        self.assertEqual(str(entrypoint_dir / Path("docs") / "{date}"), config.paths.local_doc_dir_template)

    def test_active_model_is_loaded_from_environment(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "GOOGLE_DRIVE_FOLDER_ID": "drive-folder-id",
            "GOOGLE_DRIVE_PREVIEW_FOLDER_ID": "drive-preview-folder-id",
            "GOOGLE_SHEETS_ID": "sheet-id",
            "OPENAI_MODEL": "gpt-5.2",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=self._logger())
        self.assertEqual("openai", config.llm.provider)
        self.assertEqual("gpt-5.2", config.llm.model)

    def test_default_active_model_is_gpt_5_2_without_hardcoding_in_callers(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "GOOGLE_DRIVE_FOLDER_ID": "drive-folder-id",
            "GOOGLE_DRIVE_PREVIEW_FOLDER_ID": "drive-preview-folder-id",
            "GOOGLE_SHEETS_ID": "sheet-id",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=self._logger())
        self.assertEqual("gpt-5.2", config.llm.model)

    def _minimal_env(self, runtime_config_path: str) -> dict[str, str]:
        return {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "GOOGLE_DRIVE_FOLDER_ID": "drive-folder-id",
            "GOOGLE_DRIVE_PREVIEW_FOLDER_ID": "drive-preview-folder-id",
            "GOOGLE_SHEETS_ID": "sheet-id",
        }

    def test_fallback_model_and_reasoning_effort_defaults(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        with patch.dict(os.environ, self._minimal_env(runtime_config_path), clear=True):
            config = load_config_from_env(logger=self._logger())
        self.assertEqual("gpt-5.2", config.llm.fallback_model)
        self.assertEqual("medium", config.llm.reasoning_effort)

    def test_fallback_model_and_reasoning_effort_env_overrides(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env: dict[str, str] = self._minimal_env(runtime_config_path)
        env["OPENAI_MODEL"] = "gpt-5.4"
        env["OPENAI_FALLBACK_MODEL"] = "gpt-5.1"
        env["OPENAI_REASONING_EFFORT"] = "High"
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=self._logger())
        self.assertEqual("gpt-5.4", config.llm.model)
        self.assertEqual("gpt-5.1", config.llm.fallback_model)
        self.assertEqual("high", config.llm.reasoning_effort)

    def test_service_tier_default_env_override_and_validation(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env: dict[str, str] = self._minimal_env(runtime_config_path)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual("default", load_config_from_env(logger=self._logger()).llm.service_tier)
        env["OPENAI_SERVICE_TIER"] = "Flex"
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual("flex", load_config_from_env(logger=self._logger()).llm.service_tier)
        env["OPENAI_SERVICE_TIER"] = "turbo"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as raised:
                load_config_from_env(logger=self._logger())
        self.assertIn("llm.service_tier", str(raised.exception))

    def test_invalid_reasoning_effort_is_rejected(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env: dict[str, str] = self._minimal_env(runtime_config_path)
        env["OPENAI_REASONING_EFFORT"] = "turbo"
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as raised:
                load_config_from_env(logger=self._logger())
        self.assertIn("llm.reasoning_effort", str(raised.exception))

    def test_telegram_send_delay_env_override_has_priority_over_yaml(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "GOOGLE_DRIVE_FOLDER_ID": "drive-folder-id",
            "GOOGLE_DRIVE_PREVIEW_FOLDER_ID": "drive-preview-folder-id",
            "GOOGLE_SHEETS_ID": "sheet-id",
            "TELEGRAM_SEND_DELAY_SECONDS": "2.5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=self._logger())
        self.assertEqual(2.5, config.telegram.send_delay_seconds)
        self.assertEqual(3, config.telegram.max_retries)

    def test_telegram_max_retries_env_override_has_priority_over_yaml(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "GOOGLE_DRIVE_FOLDER_ID": "drive-folder-id",
            "GOOGLE_DRIVE_PREVIEW_FOLDER_ID": "drive-preview-folder-id",
            "GOOGLE_SHEETS_ID": "sheet-id",
            "TELEGRAM_MAX_RETRIES": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=self._logger())
        self.assertEqual(1.0, config.telegram.send_delay_seconds)
        self.assertEqual(5, config.telegram.max_retries)

    def test_provider_factory_returns_openai_provider_for_active_model(self) -> None:
        provider = get_llm_provider(
            config=SimpleNamespace(
                llm=SimpleNamespace(model="gpt-5.2", provider="openai")
            )
        )
        self.assertEqual("openai", provider.name)

    def test_llm_summary_snapshot_prefers_effective_runtime_model_over_stale_provider_field(self) -> None:
        config = SimpleNamespace(
            llm=SimpleNamespace(provider="openai", model="gpt-5.2"),
        )
        summary = build_llm_summary_snapshot(config)
        self.assertEqual("gpt-5.2", summary.effective_model)
        self.assertEqual("gpt-5.2", summary.configured_model)
        self.assertEqual("gpt-5.2", summary.provider_model)
        self.assertEqual("gpt-5.2", summary.model)


if __name__ == "__main__":
    unittest.main()
