from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.core.models import LanguageMergeAttempt, PackagingAudit
from app.paths.output_naming import (
    collect_used_runtime_models,
    render_doc_title_models_segment,
)


class OutputNamingTests(unittest.TestCase):
    def test_render_doc_title_models_segment_uses_only_models_seen_at_runtime(self) -> None:
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
                        polish_model_name="deepseek-chat",
                        polish_accepted=True,
                        used_model_names=("gpt-5.1", "deepseek-chat"),
                    )
                }
            )
        ]
        used_models = collect_used_runtime_models(
            slot_results=slot_results,
            configured_primary_model="gpt-5.1",
            configured_fallback_model="deepseek-chat",
        )
        self.assertEqual(("gpt-5.1",), used_models.used_generation_models)
        self.assertEqual(("deepseek-chat",), used_models.used_polish_models)
        self.assertEqual(("gpt-5.1", "deepseek-chat"), used_models.all_used_models)
        self.assertEqual(
            "_[gpt-5.1,deepseek-chat]",
            render_doc_title_models_segment(used_runtime_models=used_models),
        )

    def test_render_doc_title_models_segment_omits_unused_fallback_model(self) -> None:
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
            configured_primary_model="gpt-5.1",
            configured_fallback_model="deepseek-chat",
        )
        self.assertEqual(("gpt-5.1",), used_models.all_used_models)
        self.assertEqual(
            "_[gpt-5.1]",
            render_doc_title_models_segment(used_runtime_models=used_models),
        )

    def test_packaging_model_is_reflected_in_used_models_when_packaging_branch_runs(self) -> None:
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
                        used_model_names=("gpt-5.1", "deepseek-chat"),
                        packaging_audit=PackagingAudit(
                            packaging_model="deepseek-chat",
                            raw_response_text="{}",
                            title_text="Title",
                            hook_text="Hook",
                            hashtags_line="#one #two",
                            requested=True,
                            received=True,
                            inserted=True,
                            fallback_used=False,
                        ),
                    )
                }
            )
        ]
        used_models = collect_used_runtime_models(
            slot_results=slot_results,
            configured_primary_model="gpt-5.1",
            configured_fallback_model="deepseek-chat",
        )
        self.assertEqual(("deepseek-chat",), used_models.used_packaging_models)
        self.assertEqual(("gpt-5.1", "deepseek-chat"), used_models.all_used_models)


if __name__ == "__main__":
    unittest.main()
