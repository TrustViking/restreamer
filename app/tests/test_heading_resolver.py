from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.bootstrap.logging_config import resolve_base_logger_name
from app.llm.llm_client import OpenAITransportResult


def _make_transport_result(raw_text: str) -> OpenAITransportResult:
    return OpenAITransportResult(
        raw_text=raw_text,
        structured_payload=None,
        incomplete_reason="",
        output_item_types=[],
    )


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai",
            model="gpt-5.1",
            timeout_sec=30.0,
        )
    )


def _reset_heading_resolver_test_state(test_case: unittest.TestCase) -> None:
    import app.resources.heading_resolver as hr

    hr._DISK_CACHE_LOADED = False
    hr._IN_MEMORY_CACHE = {"official_links": {}, "recommended_materials": {}}
    hr._STORED_CONFIG = None

    test_case._tmp_dir = tempfile.TemporaryDirectory()
    test_case.addCleanup(test_case._tmp_dir.cleanup)
    test_case._cache_path = Path(test_case._tmp_dir.name) / "heading_cache.json"

    test_case._cache_path_patcher = patch.object(
        hr,
        "_get_cache_path",
        return_value=test_case._cache_path,
    )
    test_case._cache_path_patcher.start()
    test_case.addCleanup(test_case._cache_path_patcher.stop)


class TestSeedAndGuards(unittest.TestCase):
    def setUp(self) -> None:
        _reset_heading_resolver_test_state(self)

    def test_seeded_uk_returns_without_llm_or_disk(self) -> None:
        from app.resources.heading_resolver import resolve_official_links_heading

        with patch(
            "app.resources.heading_resolver._translate_heading_via_llm"
        ) as translate_mock:
            self.assertEqual("🌐 Офіційні ресурси:", resolve_official_links_heading("uk"))

        translate_mock.assert_not_called()
        self.assertFalse(self._cache_path.exists())

    def test_seeded_en_and_ru_return_without_llm_or_disk(self) -> None:
        from app.resources.heading_resolver import resolve_official_links_heading

        with patch(
            "app.resources.heading_resolver._translate_heading_via_llm"
        ) as translate_mock:
            self.assertEqual("🌐 Official links:", resolve_official_links_heading("en"))
            self.assertEqual(
                "🌐 Официальные ссылки:",
                resolve_official_links_heading("ru"),
            )

        translate_mock.assert_not_called()
        self.assertFalse(self._cache_path.exists())

    def test_invalid_language_inputs_bypass_llm(self) -> None:
        from app.resources.heading_resolver import resolve_official_links_heading

        invalid_inputs: list[str] = [
            "unknown",
            "other",
            "",
            "xx",
            "ru-RU",
            "english",
            "12",
            "und",
            "none",
        ]
        with patch(
            "app.resources.heading_resolver._translate_heading_via_llm"
        ) as translate_mock:
            for value in invalid_inputs:
                self.assertEqual("🌐 Official links:", resolve_official_links_heading(value))

        translate_mock.assert_not_called()
        self.assertFalse(self._cache_path.exists())


class TestCacheMechanics(unittest.TestCase):
    def setUp(self) -> None:
        _reset_heading_resolver_test_state(self)

    def test_cache_miss_llm_success_persists_and_reuses_hu(self) -> None:
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_official_links_heading,
        )

        init_heading_resolver(config=_make_config())
        with patch(
            "app.resources.heading_resolver._translate_heading_via_llm",
            return_value="Hivatalos linkek:",
        ) as translate_mock:
            self.assertEqual("🌐 Hivatalos linkek:", resolve_official_links_heading("hu"))
            self.assertEqual("🌐 Hivatalos linkek:", resolve_official_links_heading("hu"))

        self.assertEqual(1, translate_mock.call_count)
        self.assertTrue(self._cache_path.exists())
        self.assertEqual(
            {
                "official_links": {"hu": "🌐 Hivatalos linkek:"},
                "recommended_materials": {},
            },
            json.loads(self._cache_path.read_text(encoding="utf-8")),
        )

    def test_cache_miss_llm_failure_retries_without_poisoning_cache(self) -> None:
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_official_links_heading,
        )

        init_heading_resolver(config=_make_config())
        with patch(
            "app.resources.heading_resolver._translate_heading_via_llm",
            return_value=None,
        ) as translate_mock:
            self.assertEqual("🌐 Official links:", resolve_official_links_heading("hu"))
            self.assertEqual("🌐 Official links:", resolve_official_links_heading("hu"))

        self.assertEqual(2, translate_mock.call_count)
        self.assertFalse(self._cache_path.exists())

    def test_recommended_materials_cache_miss_persists_without_emoji(self) -> None:
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_recommended_materials_heading,
        )

        init_heading_resolver(config=_make_config())
        with patch(
            "app.resources.heading_resolver._translate_heading_via_llm",
            return_value="Ajánlott anyagok:",
        ) as translate_mock:
            self.assertEqual(
                "Ajánlott anyagok:",
                resolve_recommended_materials_heading("hu"),
            )

        self.assertEqual(1, translate_mock.call_count)
        self.assertEqual(
            {
                "official_links": {},
                "recommended_materials": {"hu": "Ajánlott anyagok:"},
            },
            json.loads(self._cache_path.read_text(encoding="utf-8")),
        )


class TestValidation(unittest.TestCase):
    def setUp(self) -> None:
        _reset_heading_resolver_test_state(self)

    def test_validation_rejects_multiline_response(self) -> None:
        import app.resources.heading_resolver as hr
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_official_links_heading,
        )

        init_heading_resolver(config=_make_config())
        with patch(
            "app.resources.heading_resolver.openai_request_merge",
            return_value=_make_transport_result("Foo:\nBar:"),
        ), self.assertLogs(hr.LOGGER.name, level="WARNING") as logs:
            self.assertEqual("🌐 Official links:", resolve_official_links_heading("hu"))

        log_text: str = "\n".join(logs.output)
        self.assertIn("heading_translation_rejected", log_text)
        self.assertIn("reason=multiline", log_text)
        self.assertFalse(self._cache_path.exists())

    def test_validation_rejects_missing_trailing_colon(self) -> None:
        import app.resources.heading_resolver as hr
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_official_links_heading,
        )

        init_heading_resolver(config=_make_config())
        with patch(
            "app.resources.heading_resolver.openai_request_merge",
            return_value=_make_transport_result("Hivatalos linkek"),
        ), self.assertLogs(hr.LOGGER.name, level="WARNING") as logs:
            self.assertEqual("🌐 Official links:", resolve_official_links_heading("hu"))

        log_text: str = "\n".join(logs.output)
        self.assertIn("heading_translation_rejected", log_text)
        self.assertIn("reason=missing_trailing_colon", log_text)
        self.assertFalse(self._cache_path.exists())

    def test_validation_accepts_leading_emoji_leniently(self) -> None:
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_official_links_heading,
        )

        init_heading_resolver(config=_make_config())
        with patch(
            "app.resources.heading_resolver.openai_request_merge",
            return_value=_make_transport_result("🌐 Hivatalos linkek:"),
        ):
            self.assertEqual("🌐 Hivatalos linkek:", resolve_official_links_heading("hu"))

        self.assertEqual(
            {
                "official_links": {"hu": "🌐 Hivatalos linkek:"},
                "recommended_materials": {},
            },
            json.loads(self._cache_path.read_text(encoding="utf-8")),
        )


class TestStructuralDetector(unittest.TestCase):
    def setUp(self) -> None:
        _reset_heading_resolver_test_state(self)

    def test_is_official_links_heading_accepts_multi_language_input(self) -> None:
        from app.core.official_links import is_official_links_heading

        true_values: list[str] = [
            "🌐 Hivatalos linkek:",
            "🌐 Officiële links:",
            "🌐 公式リンク:",
            "🌐 Officiel lenker:",
        ]
        false_values: list[str] = [
            "Hivatalos linkek:",
            "🌐 :",
            "random text",
            "🌐 Foo: bar",
            "",
            "🌐 Foo",
        ]

        for value in true_values:
            self.assertTrue(is_official_links_heading(value), value)
        for value in false_values:
            self.assertFalse(is_official_links_heading(value), value)


class TestInitAndAtomicity(unittest.TestCase):
    def setUp(self) -> None:
        _reset_heading_resolver_test_state(self)

    def test_resolver_not_initialized_falls_back_without_llm(self) -> None:
        import app.resources.heading_resolver as hr
        from app.resources.heading_resolver import resolve_official_links_heading

        with patch(
            "app.resources.heading_resolver._translate_heading_via_llm"
        ) as translate_mock, self.assertLogs(
            hr.LOGGER.name,
            level="WARNING",
        ) as logs:
            self.assertEqual("🌐 Official links:", resolve_official_links_heading("hu"))

        translate_mock.assert_not_called()
        self.assertIn("heading_resolver_not_initialized", "\n".join(logs.output))

    def test_idempotent_init_replaces_config_without_warning(self) -> None:
        import app.resources.heading_resolver as hr

        config_a: SimpleNamespace = _make_config()
        config_b: SimpleNamespace = _make_config()

        with self.assertRaises(AssertionError):
            with self.assertLogs(hr.LOGGER.name, level="WARNING"):
                hr.init_heading_resolver(config=config_a)
                hr.init_heading_resolver(config=config_b)

        self.assertIs(config_b, hr._STORED_CONFIG)

    def test_atomic_write_resilience_keeps_existing_disk_cache(self) -> None:
        import app.resources.heading_resolver as hr
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_official_links_heading,
        )

        original_payload: dict[str, dict[str, str]] = {
            "official_links": {"de": "🌐 Offizielle Links:"},
            "recommended_materials": {},
        }
        self._cache_path.write_text(
            json.dumps(original_payload, ensure_ascii=False),
            encoding="utf-8",
        )

        hr._DISK_CACHE_LOADED = False
        hr._IN_MEMORY_CACHE = {"official_links": {}, "recommended_materials": {}}
        hr._STORED_CONFIG = None

        init_heading_resolver(config=_make_config())
        with patch(
            "app.resources.heading_resolver.os.replace",
            side_effect=OSError("simulated"),
        ), patch(
            "app.resources.heading_resolver._translate_heading_via_llm",
            return_value="Hivatalos linkek:",
        ):
            self.assertEqual("🌐 Hivatalos linkek:", resolve_official_links_heading("hu"))

        self.assertEqual(
            original_payload,
            json.loads(self._cache_path.read_text(encoding="utf-8")),
        )
        self.assertEqual([], list(Path(self._tmp_dir.name).glob("*.tmp")))


class TestResolverObservability(unittest.TestCase):
    def setUp(self) -> None:
        _reset_heading_resolver_test_state(self)

    def test_logger_name_uses_project_base_prefix(self) -> None:
        import app.resources.heading_resolver as hr

        base_name: str = resolve_base_logger_name()
        self.assertEqual(f"{base_name}.resources.heading_resolver", hr.LOGGER.name)

    def test_cache_routing_log_lines_are_emitted(self) -> None:
        from app.resources import heading_resolver
        from app.resources.heading_resolver import (
            init_heading_resolver,
            resolve_official_links_heading,
        )

        init_heading_resolver(config=_make_config())
        with patch.object(
            heading_resolver,
            "_translate_heading_via_llm",
        ) as translation_mock:
            translation_mock.side_effect = ["Foo:", None]

            with self.assertLogs(heading_resolver.LOGGER.name, level="DEBUG") as captured:
                self.assertEqual(
                    "🌐 Офіційні ресурси:",
                    resolve_official_links_heading("uk"),
                )
                self.assertEqual("🌐 Foo:", resolve_official_links_heading("hu"))
                self.assertEqual("🌐 Foo:", resolve_official_links_heading("hu"))
                self.assertEqual("🌐 Official links:", resolve_official_links_heading("de"))

        self.assertEqual(2, translation_mock.call_count)
        joined: str = "\n".join(captured.output)
        self.assertIn("heading_cache_seed_hit kind=official_links language=uk", joined)
        self.assertIn("heading_cache_miss kind=official_links language=hu", joined)
        self.assertIn("heading_cache_populated kind=official_links language=hu", joined)
        self.assertIn("heading_cache_memory_hit kind=official_links language=hu", joined)
        self.assertIn("heading_cache_miss kind=official_links language=de", joined)
        self.assertIn(
            "heading_cache_fallback_used kind=official_links language=de reason=translation_failed",
            joined,
        )
