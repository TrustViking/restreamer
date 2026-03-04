from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import MergedLanguageContent
from app.llm.merge_parser import normalize_filtered_links, parse_merge_response_or_raise
from app.llm.merge_service import (
    MergeAttemptFailure,
    attempt_openai_merge_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.llm.merge_run_summary import MergeRunSummary


class MergeContractParserTests(unittest.TestCase):
    def test_valid_single_object_json_passes(self) -> None:
        merged_content, links_stats, paragraph_count = parse_merge_response_or_raise(
            provider_name="openai",
            model_name="gpt-5.1",
            raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one","#two"],"links":["https://example.com","https://example.com"]}',
        )
        self.assertEqual("Final title", merged_content.title)
        self.assertEqual(2, paragraph_count)
        self.assertEqual(("https://example.com",), merged_content.links)
        self.assertEqual(1, links_stats.duplicates_dropped)

    def test_missing_required_key_fails(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"]}',
            )
        self.assertIn("missing_keys:links", str(raised.exception))

    def test_extra_key_fails(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"],"links":[],"unexpected":"value"}',
            )
        self.assertIn("extra_keys:unexpected", str(raised.exception))

    def test_exact_exact_keys_reject_whitespace_and_case_variants(self) -> None:
        for payload_text, expected_fragment in (
            ('{"title ":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"],"links":[]}', "missing_keys:title"),
            ('{"title":"Final title"," description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"],"links":[]}', "missing_keys:description"),
            ('{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"],"Links":[]}', "missing_keys:links"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                parse_merge_response_or_raise(
                    provider_name="openai",
                    model_name="gpt-5.1",
                    raw_text=payload_text,
                )
            self.assertIn(expected_fragment, str(raised.exception))

    def test_multi_variant_response_is_invalid(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text='{"variants":[{"title":"A"}],"description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"],"links":[],"title":"A"}',
            )

    def test_link_normalization_filters_invalid_and_limits_to_three(self) -> None:
        seen_links: list[str] = []

        def normalize_link(url: str) -> str:
            seen_links.append(url)
            if "bad" in url:
                raise RuntimeError("bad url")
            if "youtube.com" in url:
                return "https://youtube.com/watch?v=normalized"
            return url.rstrip("/")

        stats = normalize_filtered_links(
            links=(
                "https://youtube.com/watch?v=one",
                "https://youtube.com/watch?v=two",
                "https://example.com/path/",
                "https://bad.example.com",
                "https://example.com/extra",
            ),
            normalize_link=normalize_link,
        )
        self.assertEqual(
            (
                "https://youtube.com/watch?v=normalized",
                "https://example.com/path",
                "https://example.com/extra",
            ),
            stats.accepted_links,
        )
        self.assertGreaterEqual(len(seen_links), 4)
        self.assertEqual(1, stats.duplicates_dropped)
        self.assertEqual(1, stats.invalid_dropped)


class MergeContractServiceTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            openai_model_primary="gpt-5.1",
            openai_model_fallback="gpt-5-mini",
            openai_timeout_sec=30.0,
            openai_max_output_tokens=1000,
            openai_pre_delay_sec=0.0,
            llm_source_desc_max_chars=500,
            templates=SimpleNamespace(
                llm_language_names_json='{"en":"English"}',
                llm_merge_title_description_prompt="Write output only in {language_name}. Return strict JSON with title, description, cta, hashtags, links.\n\n{sources_block}",
            ),
        )

    def _videos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(title="Title 1", description="Description 1"),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(title="Title 2", description="Description 2"),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]

    def test_invalid_primary_response_triggers_retry_then_fallback(self) -> None:
        responses = [
            SimpleNamespace(raw_text='{"variants":[{"title":"bad"}]}', structured_payload={"variants": [{"title": "bad"}]}),
            SimpleNamespace(raw_text='{"title":"","description":"bad","cta":"","hashtags":[],"links":[]}', structured_payload={"title": "", "description": "bad", "cta": "", "hashtags": [], "links": []}),
            SimpleNamespace(raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"],"links":["https://example.com"]}', structured_payload={"title": "Final title", "description": "Paragraph one.\n\nParagraph two.", "cta": "Watch live.", "hashtags": ["#one"], "links": ["https://example.com"]}),
        ]

        with patch("app.llm.merge_service.openai_request_merge", side_effect=responses) as request_mock:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )

        self.assertEqual(3, request_mock.call_count)
        self.assertIsNotNone(attempt.merged)
        self.assertEqual("gpt-5-mini", attempt.model_name)
        self.assertEqual("Final title", attempt.merged.title if attempt.merged else "")

    def test_merge_attempt_failure_carries_reason_code(self) -> None:
        failure = MergeAttemptFailure(
            reason_code="missing_keys",
            reason="missing_keys:links",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
        )
        self.assertEqual("missing_keys", failure.reason_code)
        self.assertEqual('{"title":"x"}', failure.raw_response_text)

    def test_merge_service_uses_passed_link_normalizer(self) -> None:
        response = SimpleNamespace(
            raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live.","hashtags":["#one"],"links":["https://youtube.com/watch?v=one","https://youtube.com/watch?v=two"]}',
            structured_payload={
                "title": "Final title",
                "description": "Paragraph one.\n\nParagraph two.",
                "cta": "Watch live.",
                "hashtags": ["#one"],
                "links": ["https://youtube.com/watch?v=one", "https://youtube.com/watch?v=two"],
            },
        )
        normalizer_calls: list[str] = []

        def normalize_link(url: str) -> str:
            normalizer_calls.append(url)
            return "https://youtube.com/watch?v=normalized"

        with patch("app.llm.merge_service.openai_request_merge", side_effect=[response]):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=normalize_link,
                no_description_text="no description",
            )

        self.assertIsNotNone(attempt.merged)
        self.assertEqual(
            ("https://youtube.com/watch?v=normalized",),
            attempt.merged.links if attempt.merged is not None else (),
        )
        self.assertEqual(2, len(normalizer_calls))

    def test_final_failure_is_honest(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"","description":"Only one paragraph.","cta":"","hashtags":[],"links":["notaurl"]}',
            structured_payload={"title": "", "description": "Only one paragraph.", "cta": "", "hashtags": [], "links": ["notaurl"]},
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        self.assertIn("merge_llm_final_failure", "\n".join(captured.output))
        self.assertIn("code=invalid_title", "\n".join(captured.output))

    def test_rejected_raw_response_is_preserved_for_final_failure(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"Bad title","description":"Only one paragraph.","cta":"Watch","hashtags":["#one"],"links":["https://youtube.com/watch?v=one"]}',
            structured_payload={
                "title": "Bad title",
                "description": "Only one paragraph.",
                "cta": "Watch",
                "hashtags": ["#one"],
                "links": ["https://youtube.com/watch?v=one"],
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        self.assertIn('"title":"Bad title"', attempt.raw_response_text)
        self.assertIn("paragraph count", str(attempt.error_summary))

    def test_merge_logs_include_stage_reason_code_and_slot_context(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"","description":"Only one paragraph.","cta":"","hashtags":[],"links":[]}',
            structured_payload={"title": "", "description": "Only one paragraph.", "cta": "", "hashtags": [], "links": []},
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("branch=merge", joined_logs)
        self.assertIn("date_key=010130", joined_logs)
        self.assertIn("slot_key=010130_1000", joined_logs)
        self.assertIn("language=en", joined_logs)
        self.assertIn("stage=primary", joined_logs)
        self.assertIn("stage=fallback", joined_logs)
        self.assertIn("code=invalid_title", joined_logs)

    def test_post_enforcement_repairs_fixable_single_paragraph_shape(self) -> None:
        summary = MergeRunSummary()
        merged_content = MergedLanguageContent(
            title="Final title",
            description=(
                "Sentence one explains the stream. Sentence two adds context. "
                "Sentence three provides supporting detail. Sentence four closes the summary."
            ),
            cta_text="Watch live.",
            hashtags_line="#one #two",
            links=("https://example.com",),
            description_selected=(
                "Sentence one explains the stream. Sentence two adds context. "
                "Sentence three provides supporting detail. Sentence four closes the summary."
            ),
            description_audit=(
                "Sentence one explains the stream. Sentence two adds context. "
                "Sentence three provides supporting detail. Sentence four closes the summary."
            ),
        )
        with self.assertLogs(level="INFO") as captured:
            result = enforce_openai_merged_paragraphs(
                language="en",
                merged_content=merged_content,
                videos=[],
                config=SimpleNamespace(),
                no_description_text="no description",
                merge_run_summary=summary,
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertNotEqual(merged_content.description, result.description)
        self.assertEqual(1, summary.paragraph_recovery_used)
        self.assertEqual(2, len([part for part in result.description.split("\n\n") if part.strip()]))
        self.assertIn("recovery_applied=yes", "\n".join(captured.output))

    def test_post_enforcement_keeps_structurally_valid_content_unchanged(self) -> None:
        summary = MergeRunSummary()
        merged_content = MergedLanguageContent(
            title="Final title",
            description="Paragraph one.\n\nParagraph two.",
            cta_text="Watch live.",
            hashtags_line="#one #two",
            links=("https://example.com",),
            description_selected="Paragraph one.\n\nParagraph two.",
            description_audit="Paragraph one.\n\nParagraph two.",
        )
        result = enforce_openai_merged_paragraphs(
            language="en",
            merged_content=merged_content,
            videos=[],
            config=SimpleNamespace(),
            no_description_text="no description",
            merge_run_summary=summary,
            branch_label="merge",
            date_key="010130",
            slot_key="010130_1000",
        )
        self.assertEqual(merged_content, result)
        self.assertEqual(0, summary.paragraph_recovery_used)


if __name__ == "__main__":
    unittest.main()
