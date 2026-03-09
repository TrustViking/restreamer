from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.config.app_config_loader import load_config_from_env
from app.config.llm_routing import build_llm_routing
from app.paths import get_project_paths
from app.llm.provider_factory import get_llm_provider_for_target


class ModelResolutionTests(unittest.TestCase):
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
            "DPSK_API_KEY": "deepseek-secret",
            "OPENAI_MODEL": "gpt-5.1",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "MAIN_MODEL": "OPENAI_MODEL",
            "FALLBACK_MODEL": "DEEPSEEK_MODEL",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        entrypoint_dir = get_project_paths().entrypoint_path.parent.resolve()
        self.assertEqual(
            str(entrypoint_dir / Path("image") / "{language}" / "{date}"),
            config.local_image_dir_template,
        )
        self.assertEqual(
            str(entrypoint_dir / Path("docs") / "{date}"),
            config.local_doc_dir_template,
        )

    def test_main_and_fallback_are_resolved_only_from_aliases(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "DPSK_API_KEY": "deepseek-secret",
            "OPENAI_MODEL": "gpt-5.1",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "MAIN_MODEL": "OPENAI_MODEL",
            "FALLBACK_MODEL": "DEEPSEEK_MODEL",
            "STG_LLM_PROVIDER": "deepseek",
            "STG_OPENAI_MODEL_PRIMARY": "ignored-model",
            "STG_OPENAI_MODEL_FALLBACK": "ignored-model",
            "STG_DEEPSEEK_MODEL": "ignored-model",
            "OPENAI_MODEL_MINI": "ignored-mini",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        self.assertEqual("OPENAI_MODEL", config.configured_main_model_alias)
        self.assertEqual("DEEPSEEK_MODEL", config.configured_fallback_model_alias)
        self.assertEqual("gpt-5.1", config.llm_main_model)
        self.assertEqual("deepseek-chat", config.llm_fallback_model)
        self.assertEqual("openai", config.llm_provider)
        self.assertEqual("openai", config.llm_routing.primary.provider)
        self.assertEqual("deepseek", config.llm_routing.fallback.provider)

    def test_invalid_alias_is_rejected(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "DPSK_API_KEY": "deepseek-secret",
            "OPENAI_MODEL": "gpt-5.1",
            "DEEPSEEK_MODEL": "deepseek-chat",
            "MAIN_MODEL": "gpt-4.1",
            "FALLBACK_MODEL": "DEEPSEEK_MODEL",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as raised:
                load_config_from_env(logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        self.assertIn("Unsupported model alias", str(raised.exception))

    def test_provider_factory_uses_routing_target(self) -> None:
        provider = get_llm_provider_for_target(
            target=build_llm_routing(
                primary_input="DEEPSEEK_MODEL",
                fallback_input="OPENAI_MODEL",
                model_aliases={
                    "OPENAI_MODEL": "gpt-5.1",
                    "DEEPSEEK_MODEL": "deepseek-chat",
                },
            ).primary
        )
        self.assertEqual("deepseek", provider.name)

    def test_alias_values_can_point_to_opposite_model_families(self) -> None:
        runtime_config_path: str = self._write_runtime_config()
        self.addCleanup(lambda: os.path.exists(runtime_config_path) and os.remove(runtime_config_path))
        env = {
            "APP_CONFIG_PATH": runtime_config_path,
            "TELEGRAM_BOT_TOKEN": "token",
            "TELEGRAM_CHAT_ID": "chat",
            "GPT_API_KEY": "openai-secret",
            "DPSK_API_KEY": "deepseek-secret",
            "OPENAI_MODEL": "deepseek-chat",
            "DEEPSEEK_MODEL": "gpt-5-mini",
            "MAIN_MODEL": "DEEPSEEK_MODEL",
            "FALLBACK_MODEL": "OPENAI_MODEL",
        }
        with patch.dict(os.environ, env, clear=True):
            config = load_config_from_env(logger=SimpleNamespace(warning=lambda *args, **kwargs: None))
        self.assertEqual("gpt-5-mini", config.llm_main_model)
        self.assertEqual("deepseek-chat", config.llm_fallback_model)
        self.assertEqual("openai", config.llm_routing.primary.provider)
        self.assertEqual("deepseek", config.llm_routing.fallback.provider)


if __name__ == "__main__":
    unittest.main()
