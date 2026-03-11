from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.core.models import LanguageMergeAttempt
from app.paths.output_naming import (
    collect_used_runtime_models,
    render_doc_title_models_segment,
)


class OutputNamingTests(unittest.TestCase):
    def test_collect_used_runtime_models_reflects_current_runtime_contract(self) -> None:
        slot_results = [
            SimpleNamespace(
                merge_audit_by_language={
                    "en": LanguageMergeAttempt(
                        language="en",
                        model_name="gpt-5.1",
                        raw_response_text="",
                        merged=None,
                        error_summary=None,
                        generator_model_name="gpt-5.1",
                        used_model_names=("gpt-5.1",),
                    )
                }
            )
        ]
        used_models = collect_used_runtime_models(
            slot_results=slot_results,
            configured_model="gpt-5.1",
        )
        self.assertEqual("gpt-5.1", used_models.configured_model)
        self.assertEqual(("gpt-5.1",), used_models.used_generation_models)
        self.assertEqual(("gpt-5.1",), used_models.all_used_models)

    def test_render_doc_title_models_segment_uses_only_seen_runtime_models(self) -> None:
        slot_results = [
            SimpleNamespace(
                merge_audit_by_language={
                    "en": LanguageMergeAttempt(
                        language="en",
                        model_name="gpt-5.1",
                        raw_response_text="",
                        merged=None,
                        error_summary=None,
                        generator_model_name="gpt-5.1",
                        used_model_names=("gpt-5.1", "gpt-5.1-mini"),
                    )
                }
            )
        ]
        used_models = collect_used_runtime_models(
            slot_results=slot_results,
            configured_model="gpt-5.1",
        )
        self.assertEqual(
            "_[gpt-5.1,gpt-5.1-mini]",
            render_doc_title_models_segment(used_runtime_models=used_models),
        )

    def test_render_doc_title_models_segment_omits_empty_runtime_models(self) -> None:
        used_models = collect_used_runtime_models(
            slot_results=[SimpleNamespace(merge_audit_by_language={})],
            configured_model="gpt-5.1",
        )
        self.assertEqual("", render_doc_title_models_segment(used_runtime_models=used_models))

    def test_collect_used_runtime_models_dedupes_blanks_and_preserves_seen_order(self) -> None:
        slot_results = [
            SimpleNamespace(
                merge_audit_by_language={
                    "en": LanguageMergeAttempt(
                        language="en",
                        model_name="gpt-5.1",
                        raw_response_text="",
                        merged=None,
                        error_summary=None,
                        generator_model_name="gpt-5.1",
                        used_model_names=("gpt-5.1", "", "gpt-5.1-mini", "gpt-5.1"),
                    ),
                    "uk": LanguageMergeAttempt(
                        language="uk",
                        model_name="gpt-5.1-mini",
                        raw_response_text="",
                        merged=None,
                        error_summary=None,
                        generator_model_name="",
                        used_model_names=(" ", "gpt-5.1-mini", "deepseek-chat"),
                    ),
                }
            )
        ]
        used_models = collect_used_runtime_models(
            slot_results=slot_results,
            configured_model="gpt-5.1",
        )
        self.assertEqual(("gpt-5.1", "gpt-5.1-mini"), used_models.used_generation_models)
        self.assertEqual(
            ("gpt-5.1", "gpt-5.1-mini", "deepseek-chat"),
            used_models.all_used_models,
        )
        self.assertEqual(
            "_[gpt-5.1,gpt-5.1-mini,deepseek-chat]",
            render_doc_title_models_segment(used_runtime_models=used_models),
        )


if __name__ == "__main__":
    unittest.main()
