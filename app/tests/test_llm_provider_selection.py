from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config.app_config_loader import load_config_from_env
from app.llm.provider_factory import get_llm_provider
from app.observability.startup_summary import build_llm_summary_snapshot
from app.paths import get_project_paths


class ActiveModelConfigTests(unittest.TestCase):
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
    drive_preview_path_template: "{streamertg}/{preview}/{language}/{date}"
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
    symbol_broadcast: "B"
    symbol_alert: "A"
    symbol_form: "F"
    symbol_description: "D"
    symbol_pin: "P"
    symbol_done: "Y"
    flag_uk: "UK"
    flag_en: "EN"
    flag_ru: "RU"
    flag_other: "OT"
    flag_repeat_count: 1
    language_name_uk: "Ukr"
    language_name_en: "Eng"
    language_name_ru: "Rus"
    language_name_other: "Other"
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
            "OPENAI_MODEL": "gpt-5.1",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        entrypoint_dir = get_project_paths().entrypoint_path.parent.resolve()
        self.assertEqual(str(entrypoint_dir / Path("image") / "{language}" / "{date}"), config.local_image_dir_template)
        self.assertEqual(str(entrypoint_dir / Path("docs") / "{date}"), config.local_doc_dir_template)

    def test_active_model_is_loaded_from_environment(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "OPENAI_MODEL": "gpt-5.2",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        self.assertEqual("openai", config.llm_provider)
        self.assertEqual("gpt-5.2", config.llm_model)
        self.assertEqual("gpt-5.2", config.openai_model)

    def test_default_active_model_is_gpt_5_1_without_hardcoding_in_callers(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        self.assertEqual("gpt-5.1", config.llm_model)

    def test_provider_factory_returns_openai_provider_for_active_model(self) -> None:
        provider = get_llm_provider(config=SimpleNamespace(llm_model="gpt-5.2"))
        self.assertEqual("openai", provider.name)

    def test_llm_summary_snapshot_prefers_effective_runtime_model_over_stale_provider_field(self) -> None:
        config = SimpleNamespace(
            llm_provider="openai",
            llm_model="gpt-5.2",
            openai_model="gpt-5.1",
        )
        summary = build_llm_summary_snapshot(config)
        self.assertEqual("gpt-5.2", summary.effective_model)
        self.assertEqual("gpt-5.2", summary.configured_model)
        self.assertEqual("gpt-5.1", summary.provider_model)
        self.assertEqual("gpt-5.2", summary.model)


if __name__ == "__main__":
    unittest.main()
