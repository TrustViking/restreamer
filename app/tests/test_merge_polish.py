from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.config.llm_routing import build_llm_routing
from app.core.branching import BRANCH_MERGE_MAIN, BRANCH_MERGE_MAIN_FALLBACK_PACKAGING
from app.core.models import LanguageMergeAttempt, MergedLanguageContent
from app.llm.merge_packaging import build_packaging_overlay_attempt
from app.llm.merge_service import MergeAttemptFailure, attempt_llm_merge_with_audit
from app.llm.merge_run_summary import MergeRunSummary


class MergePackagingFlowTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            llm_provider="openai",
            llm_routing=build_llm_routing(
                primary_input="OPENAI_MODEL",
                fallback_input="DEEPSEEK_MODEL",
                model_aliases={
                    "OPENAI_MODEL": "gpt-5.1",
                    "DEEPSEEK_MODEL": "deepseek-chat",
                },
            ),
            llm_main_model="gpt-5.1",
            llm_fallback_model="deepseek-chat",
            openai_model_primary="gpt-5.1",
            openai_model_fallback="gpt-5.1",
            openai_timeout_sec=30.0,
            openai_max_output_tokens=1000,
            openai_pre_delay_sec=0.0,
            deepseek_base_url="https://api.deepseek.com/v1",
            deepseek_model="deepseek-chat",
            deepseek_reasoning_model=None,
            deepseek_timeout_sec=45.0,
            deepseek_max_retries=0,
            llm_source_desc_max_chars=500,
            templates=SimpleNamespace(
                llm_language_names_json='{"en":"English"}',
                llm_merge_title_description_prompt="unused",
            ),
        )

    def _videos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(title="Title 1", description="Paragraph one."),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(title="Title 2", description="Paragraph two."),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]

    def _merge_attempt(self, merged_content: MergedLanguageContent) -> LanguageMergeAttempt:
        return LanguageMergeAttempt(
            language="en",
            model_name="gpt-5.1",
            raw_response_text='{"title":"GPT title","description":"Paragraph one.\\n\\nParagraph two."}',
            merged=merged_content,
            error_summary=None,
            salvaged_title=merged_content.title,
            publish_source_label="primary_success",
            generator_model_name="gpt-5.1",
            used_model_names=("gpt-5.1",),
            branch_type=BRANCH_MERGE_MAIN,
            title_source="openai",
            hook_source="openai",
            hashtags_source="openai",
            body_source="main_merge",
        )

    def test_merge_gpt_attempt_does_not_apply_hidden_polish_stage(self) -> None:
        merged_content = MergedLanguageContent(
            title="GPT title",
            description="Paragraph one.\n\nParagraph two.",
        )
        with patch(
            "app.llm.merge_service.get_llm_provider_for_target",
            side_effect=lambda target: SimpleNamespace(name=target.provider, pre_delay_sec=lambda config: 0.0),
        ), patch(
            "app.llm.merge_service._attempt_merge_once",
            return_value=(merged_content, '{"title":"GPT title","description":"Paragraph one.\\n\\nParagraph two."}'),
        ), patch("app.llm.merge_service._apply_merge_polish_stage") as polish_mock:
            attempt = attempt_llm_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                merge_run_summary=MergeRunSummary(),
                branch_label="merge",
                date_key="090326",
                slot_key="1350",
            )
        self.assertIsNotNone(attempt.merged)
        self.assertEqual("GPT title", attempt.merged.title if attempt.merged else "")
        self.assertEqual(("gpt-5.1",), attempt.used_model_names)
        self.assertFalse(polish_mock.called)
        self.assertEqual(BRANCH_MERGE_MAIN, attempt.branch_type)

    def test_packaging_overlay_inserts_deepseek_fields_as_is(self) -> None:
        merged_content = MergedLanguageContent(
            title="GPT title",
            description="GPT hook paragraph.\n\nParagraph two.\n\nJoin us later.\n\n#oldtag\n\nhttps://youtube.com/watch?v=aaaaaaaaaaa",
        )
        merge_attempt = self._merge_attempt(merged_content)
        with patch(
            "app.llm.merge_packaging.get_llm_provider_for_model",
            return_value=SimpleNamespace(
                name="deepseek",
                request_merge=lambda **kwargs: SimpleNamespace(
                    raw_text='{"title":"DeepSeek title","hook":"DeepSeek hook.","hashtags":["#one","#two","#two"],"emoji_suggestions":["🔥"]}',
                    structured_payload={
                        "title": "DeepSeek title",
                        "hook": "DeepSeek hook.",
                        "hashtags": ["#one", "#two", "#two"],
                        "emoji_suggestions": ["🔥"],
                    },
                ),
            ),
        ):
            packaged_content, packaged_attempt = build_packaging_overlay_attempt(
                logger=SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
                language="en",
                videos=self._videos(),
                merge_attempt=merge_attempt,
                merged_content=merged_content,
                config=self._config(),
                branch_label=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                date_key="090326",
                slot_key="1350",
            )
        self.assertEqual("DeepSeek title", packaged_content.title)
        self.assertIn("DeepSeek hook.", packaged_content.description)
        self.assertIn("#one #two", packaged_content.description)
        self.assertEqual("deepseek", packaged_content.title_source)
        self.assertEqual("deepseek", packaged_content.hook_source)
        self.assertEqual("deepseek", packaged_content.hashtags_source)
        self.assertEqual(("gpt-5.1", "deepseek-chat"), packaged_attempt.used_model_names)
        self.assertTrue(packaged_attempt.packaging_audit.inserted if packaged_attempt.packaging_audit else False)
        self.assertIn("#one #two", packaged_content.description)
        self.assertIn("https://youtu.be/aaaaaaaaaaa", packaged_content.description)

    def test_packaging_overlay_preserves_gpt_links_and_normalizes_tail(self) -> None:
        merged_content = MergedLanguageContent(
            title="GPT title",
            description=(
                "GPT hook paragraph.\n\n"
                "Paragraph two.\n\n"
                "Join us later.\n\n"
                "#oldtag #carry\n\n"
                "https://youtube.com/watch?v=aaaaaaaaaaa\n"
                "https://youtube.com/watch?v=bbbbbbbbbbb"
            ),
        )
        merge_attempt = self._merge_attempt(merged_content)
        with patch(
            "app.llm.merge_packaging.get_llm_provider_for_model",
            return_value=SimpleNamespace(
                name="deepseek",
                request_merge=lambda **kwargs: SimpleNamespace(
                    raw_text='{"title":"DeepSeek title","hook":"DeepSeek hook.","hashtags":["#carry","#fresh","#fresh"]}',
                    structured_payload={
                        "title": "DeepSeek title",
                        "hook": "DeepSeek hook.",
                        "hashtags": ["#carry", "#fresh", "#fresh"],
                    },
                ),
            ),
        ):
            packaged_content, _ = build_packaging_overlay_attempt(
                logger=SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
                language="en",
                videos=self._videos(),
                merge_attempt=merge_attempt,
                merged_content=merged_content,
                config=self._config(),
                branch_label=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                date_key="090326",
                slot_key="1350",
            )
        self.assertIn("#carry #fresh", packaged_content.description)
        self.assertIn("https://youtu.be/aaaaaaaaaaa", packaged_content.description)
        self.assertIn("https://youtu.be/bbbbbbbbbbb", packaged_content.description)
        self.assertLess(
            packaged_content.description.index("#carry #fresh"),
            packaged_content.description.index("https://youtu.be/aaaaaaaaaaa"),
        )

    def test_packaging_overlay_falls_back_to_gpt_parts_when_payload_is_structurally_bad(self) -> None:
        merged_content = MergedLanguageContent(
            title="GPT title",
            description=(
                "GPT hook paragraph.\n\n"
                "Paragraph two.\n\n"
                "Join us later.\n\n"
                "#oldtag\n\n"
                "https://youtube.com/watch?v=aaaaaaaaaaa"
            ),
        )
        merge_attempt = self._merge_attempt(merged_content)
        with patch(
            "app.llm.merge_packaging.get_llm_provider_for_model",
            return_value=SimpleNamespace(
                name="deepseek",
                request_merge=lambda **kwargs: SimpleNamespace(
                    raw_text='{"title":"!!","hook":"#noise","hashtags":["#fresh"]}',
                    structured_payload={
                        "title": "!!",
                        "hook": "#noise",
                        "hashtags": ["#fresh"],
                    },
                ),
            ),
        ):
            packaged_content, _ = build_packaging_overlay_attempt(
                logger=SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
                language="en",
                videos=self._videos(),
                merge_attempt=merge_attempt,
                merged_content=merged_content,
                config=self._config(),
                branch_label=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                date_key="090326",
                slot_key="1350",
            )
        self.assertEqual("GPT title", packaged_content.title)
        self.assertEqual("openai", packaged_content.title_source)
        self.assertEqual("openai", packaged_content.hook_source)
        self.assertTrue(packaged_content.description.startswith("GPT hook paragraph."))
        self.assertIn("https://youtu.be/aaaaaaaaaaa", packaged_content.description)

    def test_packaging_overlay_uses_gpt_fallback_only_on_technical_failure(self) -> None:
        merged_content = MergedLanguageContent(
            title="GPT title",
            description="GPT hook paragraph.\n\nParagraph two.",
        )
        merge_attempt = self._merge_attempt(merged_content)
        with patch(
            "app.llm.merge_packaging.get_llm_provider_for_model",
            return_value=SimpleNamespace(
                name="deepseek",
                request_merge=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("transport_failed")),
            ),
        ):
            packaged_content, packaged_attempt = build_packaging_overlay_attempt(
                logger=SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
                language="en",
                videos=self._videos(),
                merge_attempt=merge_attempt,
                merged_content=merged_content,
                config=self._config(),
                branch_label=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                date_key="090326",
                slot_key="1350",
            )
        self.assertEqual("GPT title", packaged_content.title)
        self.assertEqual("openai", packaged_content.title_source)
        self.assertEqual("main_merge", packaged_content.body_source)
        self.assertTrue(packaged_attempt.packaging_audit.fallback_used if packaged_attempt.packaging_audit else False)

    def test_primary_failure_does_not_use_deepseek_as_semantic_fallback(self) -> None:
        with patch(
            "app.llm.merge_service.get_llm_provider_for_target",
            side_effect=lambda target: SimpleNamespace(name=target.provider, pre_delay_sec=lambda config: 0.0),
        ), patch(
            "app.llm.merge_service._attempt_merge_once",
            side_effect=MergeAttemptFailure(
                reason_code="invalid_description",
                reason="bad output",
                model_name="gpt-5.1",
                attempt_stage="validation",
                raw_response_text="",
            ),
        ), patch("app.llm.merge_service._apply_merge_polish_stage") as polish_mock:
            attempt = attempt_llm_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                merge_run_summary=MergeRunSummary(),
                branch_label="merge",
                date_key="090326",
                slot_key="1350",
            )
        self.assertIsNone(attempt.merged)
        self.assertEqual("merge_failed", attempt.publish_source_label)
        self.assertFalse(polish_mock.called)

    def test_fallback_openai_can_drive_packaging_stage_after_deepseek_main(self) -> None:
        config = self._config()
        config.llm_provider = "deepseek"
        config.llm_routing = build_llm_routing(
            primary_input="DEEPSEEK_MODEL",
            fallback_input="OPENAI_MODEL",
            model_aliases={
                "OPENAI_MODEL": "gpt-5.1",
                "DEEPSEEK_MODEL": "deepseek-chat",
            },
        )
        config.llm_main_model = "deepseek-chat"
        config.llm_fallback_model = "gpt-5.1"
        merged_content = MergedLanguageContent(
            title="Base title",
            description="Base hook.\n\nParagraph two.\n\nJoin us later.\n#oldtag\nhttps://allatra.org/",
        )
        merge_attempt = LanguageMergeAttempt(
            language="en",
            model_name="deepseek-chat",
            raw_response_text="{}",
            merged=merged_content,
            error_summary=None,
            generator_model_name="deepseek-chat",
            used_model_names=("deepseek-chat",),
            branch_type=BRANCH_MERGE_MAIN,
            title_source="deepseek",
            hook_source="deepseek",
            hashtags_source="deepseek",
            body_source="main_merge",
        )
        requested_models: list[str] = []

        def _provider_for_model(model_name: str) -> SimpleNamespace:
            requested_models.append(model_name)
            return SimpleNamespace(
                name="openai",
                request_merge=lambda **kwargs: SimpleNamespace(
                    raw_text='{"title":"OpenAI packaged","hook":"OpenAI hook.","hashtags":["#one","#two"]}',
                    structured_payload={
                        "title": "OpenAI packaged",
                        "hook": "OpenAI hook.",
                        "hashtags": ["#one", "#two"],
                    },
                ),
            )

        with patch(
            "app.llm.merge_packaging.get_llm_provider_for_model",
            side_effect=_provider_for_model,
        ):
            packaged_content, packaged_attempt = build_packaging_overlay_attempt(
                logger=SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
                language="en",
                videos=self._videos(),
                merge_attempt=merge_attempt,
                merged_content=merged_content,
                config=config,
                branch_label=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                date_key="090326",
                slot_key="1350",
            )
        self.assertEqual(["gpt-5.1"], requested_models)
        self.assertEqual("openai", packaged_content.title_source)
        self.assertEqual("openai", packaged_content.hook_source)
        self.assertEqual(BRANCH_MERGE_MAIN_FALLBACK_PACKAGING, packaged_attempt.branch_type)
        self.assertIn("https://allatra.org/", packaged_content.description)


if __name__ == "__main__":
    unittest.main()
