from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import MergedLanguageContent
from app.llm.merge_quality import normalize_merge_description
from app.llm.merge_parser import parse_merge_response_or_raise, sanitize_title
from app.llm.merge_service import (
    MergeAttemptFailure,
    attempt_openai_merge_with_audit,
    attempt_openai_single_source_translate_with_audit,
    build_llm_merge_prompt_text,
    enforce_openai_merged_paragraphs,
)
from app.llm.merge_run_summary import MergeRunSummary


class MergeContractParserTests(unittest.TestCase):
    def test_valid_single_object_json_passes(self) -> None:
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name="openai",
            model_name="gpt-5.1",
            raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
        )
        self.assertEqual("Final title", merged_content.title)
        self.assertEqual(2, paragraph_count)

    def test_missing_required_key_fails(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text='{"title":"Final title"}',
            )
        self.assertIn("missing_keys:description", str(raised.exception))

    def test_extra_key_fails(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two.","cta":"Watch live."}',
            )
        self.assertIn("extra_keys:cta", str(raised.exception))

    def test_exact_exact_keys_reject_whitespace_and_case_variants(self) -> None:
        for payload_text, expected_fragment in (
            ('{"title ":"Final title","description":"Paragraph one.\\n\\nParagraph two."}', "missing_keys:title"),
            ('{"title":"Final title"," description":"Paragraph one.\\n\\nParagraph two."}', "missing_keys:description"),
            ('{"Title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}', "missing_keys:title"),
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
                raw_text='{"variants":[{"title":"A"}],"description":"Paragraph one.\\n\\nParagraph two.","title":"A"}',
            )

    def test_title_is_limited_to_99_characters(self) -> None:
        title_text: str = "A" * 120
        self.assertEqual(99, len(sanitize_title(title_text, min_chars=1, max_chars=99)))

    def test_title_with_emoji_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text='{"title":"Final title 🔥","description":"Paragraph one.\\n\\nParagraph two."}',
            )
        self.assertIn("title must not contain emoji", str(raised.exception))


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
                llm_merge_title_description_prompt="""
Write a YouTube stream title and description in {language_name}.
Generate a new final title, not a copy of any single source title.
Mentally extract key points from each source, preserve all non-trivial source-specific points,
combine overlaps, compress repetition, and produce one coherent final description.
Write a strong native YouTube title no longer than 99 characters.
Write one cohesive stream description in 2 to 4 paragraphs.
The description must cover all source inputs that were merged.
Start paragraph one with a strong factual hook grounded in the main tension, risk, or key conflict.
Keep the hook editorial and readable, but never clickbait.
Do not enumerate sources as 1) 2) 3).
Do not output generic slogans or abstract editorial phrasing.
Do not use emoji in the title.
Include one compact agenda block with 4 to 7 short bullet-like thesis lines.
Each bullet must start with one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.
Most bullets should start with 🔹.
Accent markers are rare and optional; use no more than 3 accent markers per theses block.
Avoid asserting strong person titles or role labels unless they are clearly necessary and well-supported by the sources.
Optional official links block is allowed before close line with 1 to 3 non-YouTube links from sources.
An optional one-line close should be practical CTA + 2 to 5 hashtags.
Return strict JSON with title and description only.

{sources_block}
""".strip(),
            ),
        )

    def _videos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Title 1",
                    description="Paragraph one.\n\nParagraph two.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Title 2",
                    description="Paragraph three.\n\nParagraph four.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]

    def test_prompt_targets_youtube_title_and_description_only(self) -> None:
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=self._videos(),
            config=self._config(),
            no_description_text="no description",
        )
        self.assertIn("YouTube stream title and description", prompt_text)
        self.assertIn("99 characters", prompt_text)
        self.assertIn("title and description only", prompt_text)
        self.assertIn("must cover all source inputs", prompt_text)
        self.assertIn("Do not output generic slogans", prompt_text)
        self.assertIn("compact agenda block", prompt_text)
        self.assertIn("allowed marker", prompt_text)
        self.assertIn("Most bullets should start with 🔹", prompt_text)
        self.assertIn("no more than 3 accent markers", prompt_text)
        self.assertIn("Avoid asserting strong person titles", prompt_text)
        self.assertIn("Optional official links block", prompt_text)
        self.assertIn("Do not use emoji in the title.", prompt_text)
        self.assertIn("optional one-line close", prompt_text.lower())
        self.assertNotIn("URL:", prompt_text)
        self.assertIn("Paragraph one.\n\nParagraph two.", prompt_text)

    def test_invalid_primary_response_triggers_retry_then_fallback(self) -> None:
        responses = [
            SimpleNamespace(raw_text='{"variants":[{"title":"bad"}]}', structured_payload={"variants": [{"title": "bad"}]}),
            SimpleNamespace(raw_text='{"title":"","description":"bad"}', structured_payload={"title": "", "description": "bad"}),
            SimpleNamespace(raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}', structured_payload={"title": "Final title", "description": "Paragraph one.\n\nParagraph two."}),
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
            reason="missing_keys:description",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
        )
        self.assertEqual("missing_keys", failure.reason_code)
        self.assertEqual('{"title":"x"}', failure.raw_response_text)

    def test_final_failure_is_honest(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"","description":"Only one paragraph."}',
            structured_payload={"title": "", "description": "Only one paragraph."},
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
            raw_text='{"title":"Bad title","description":"Only one paragraph."}',
            structured_payload={"title": "Bad title", "description": "Only one paragraph."},
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
            raw_text='{"title":"","description":"Only one paragraph."}',
            structured_payload={"title": "", "description": "Only one paragraph."},
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

    def test_semantic_generic_merge_is_rejected_and_fallback_can_recover(self) -> None:
        generic_response = SimpleNamespace(
            raw_text=(
                '{"title":"Important discussion","description":"A meaningful discussion about values and change.'
                '\\n\\nAn inspiring talk about the big picture."}'
            ),
            structured_payload={
                "title": "Important discussion",
                "description": "A meaningful discussion about values and change.\n\nAn inspiring talk about the big picture.",
            },
        )
        fallback_success = SimpleNamespace(
            raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
            structured_payload={
                "title": "Final title",
                "description": "Paragraph one.\n\nParagraph two.",
            },
        )
        with patch(
            "app.llm.merge_service.openai_request_merge",
            side_effect=[generic_response, generic_response, fallback_success],
        ) as request_mock:
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

    def test_per_source_dump_is_rejected(self) -> None:
        per_source_dump = SimpleNamespace(
            raw_text=(
                '{"title":"Final title","description":"SOURCE 1: Point one.\\n\\nSOURCE 2: Point two."}'
            ),
            structured_payload={
                "title": "Final title",
                "description": "SOURCE 1: Point one.\n\nSOURCE 2: Point two.",
            },
        )
        fallback_success = SimpleNamespace(
            raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
            structured_payload={
                "title": "Final title",
                "description": "Paragraph one.\n\nParagraph two.",
            },
        )
        with patch(
            "app.llm.merge_service.openai_request_merge",
            side_effect=[per_source_dump, per_source_dump, fallback_success],
        ) as request_mock:
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

    def test_style_coverage_log_is_emitted_for_successful_merge(self) -> None:
        videos = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Budget vote briefing",
                    description="John Smith explains sanctions timeline and budget vote in Brussels.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Frontline logistics briefing",
                    description="Maria Ivanova details drone pressure and aid corridor risks in Kharkiv.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]
        styled_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv: key decisions tonight","description":"Tonight we map the budget vote and frontline pressure with clear facts and timelines! 🎯\\n\\n- sanctions timeline and vote implications\\n- drone pressure and logistics bottlenecks\\n- aid corridor risks and response steps\\n\\nJohn Smith and Maria Ivanova connect political decisions with field consequences."}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: key decisions tonight",
                "description": (
                    "Tonight we map the budget vote and frontline pressure with clear facts and timelines! 🎯\n\n"
                    "- sanctions timeline and vote implications\n"
                    "- drone pressure and logistics bottlenecks\n"
                    "- aid corridor risks and response steps\n\n"
                    "John Smith and Maria Ivanova connect political decisions with field consequences."
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=videos,
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_style_coverage", joined_logs)
        self.assertIn("style_contract_version=v4_merge_quality_hardening", joined_logs)
        self.assertIn("hook_present=yes", joined_logs)
        self.assertIn("agenda_block_present=yes", joined_logs)
        self.assertIn("bullet_points_count=3", joined_logs)
        self.assertIn("semantic_bullets_count=3", joined_logs)
        self.assertIn("named_entities_preserved=2", joined_logs)
        self.assertIn("emoji_count=4", joined_logs)
        self.assertIn("source_coverage_total=2/2", joined_logs)
        self.assertIn("source_coverage_ok=yes", joined_logs)

    def test_style_coverage_counts_en_dash_and_em_dash_bullets(self) -> None:
        videos = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Budget vote briefing",
                    description="John Smith explains sanctions timeline and budget vote in Brussels.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Frontline logistics briefing",
                    description="Maria Ivanova details drone pressure and aid corridor risks in Kharkiv.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]
        styled_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv: agenda tonight","description":"Tonight we focus on the budget vote and frontline logistics without noise.\\n\\nWhat’s in this stream:\\n– sanctions timeline and vote implications\\n— drone pressure and corridor risks\\n\\nJohn Smith and Maria Ivanova connect decisions with field outcomes."}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: agenda tonight",
                "description": (
                    "Tonight we focus on the budget vote and frontline logistics without noise.\n\n"
                    "What’s in this stream:\n"
                    "– sanctions timeline and vote implications\n"
                    "— drone pressure and corridor risks\n\n"
                    "John Smith and Maria Ivanova connect decisions with field outcomes."
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=videos,
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_style_coverage", joined_logs)
        self.assertIn("agenda_block_present=yes", joined_logs)
        self.assertIn("bullet_points_count=2", joined_logs)

    def test_style_coverage_keeps_legacy_bullet_markers_and_numbering(self) -> None:
        videos = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Tech and economy briefing",
                    description="Analysts compare macro data and infrastructure bottlenecks across regions.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Operations and logistics update",
                    description="Editors review supply risks, timing gaps, and recovery scenarios.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]
        styled_response = SimpleNamespace(
            raw_text=(
                '{"title":"Tech and logistics: focused breakdown","description":"We compare the main pressure points and show where the data agrees or diverges.\\n\\n- macro data and trend shifts\\n* infrastructure bottlenecks\\n• timing gaps across regions\\n1) supply risk scenarios\\n2. recovery windows and constraints\\n\\nAnalysts and editors align the evidence into one coherent picture."}'
            ),
            structured_payload={
                "title": "Tech and logistics: focused breakdown",
                "description": (
                    "We compare the main pressure points and show where the data agrees or diverges.\n\n"
                    "- macro data and trend shifts\n"
                    "* infrastructure bottlenecks\n"
                    "• timing gaps across regions\n"
                    "1) supply risk scenarios\n"
                    "2. recovery windows and constraints\n\n"
                    "Analysts and editors align the evidence into one coherent picture."
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=videos,
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        self.assertIn("bullet_points_count=5", "\n".join(captured.output))

    def test_semantic_emoji_bullets_are_accepted_and_logged(self) -> None:
        videos = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Human rights panel",
                    description="Alice Brown discusses legal safeguards and witness testimony in Brussels.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="International conference desk",
                    description="Bob Green shares conference logistics and official initiatives updates.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]
        styled_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels panel: legal safeguards and conference agenda","description":"This stream tracks why legal safeguards and testimony matter right now.\\n\\nIn this stream you will see:\\n📌 legal safeguards and testimony timeline\\n⚖ witness rights and justice risks\\n🎤 Alice Brown key remarks\\n🌐 conference initiative milestones\\n\\nJoin the live discussion and share your view. #rights #justice"}'
            ),
            structured_payload={
                "title": "Brussels panel: legal safeguards and conference agenda",
                "description": (
                    "This stream tracks why legal safeguards and testimony matter right now.\n\n"
                    "In this stream you'll see:\n"
                    "📌 legal safeguards and testimony timeline\n"
                    "⚖ witness rights and justice risks\n"
                    "🎤 Alice Brown's key remarks\n"
                    "🌐 conference initiative milestones\n\n"
                    "Join the live discussion and share your view. #rights #justice"
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=videos,
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("semantic_bullets_count=4", joined_logs)
        self.assertIn("bullets_with_emoji_count=4", joined_logs)
        self.assertIn("bullets_with_plain_marker_count=0", joined_logs)

    def test_official_links_injected_when_missing_in_llm_output(self) -> None:
        videos = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Conference briefing",
                    description=(
                        "Main update and schedule.\n"
                        "Official website: https://interfaithconf.org/about?utm_source=yt\n"
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Initiative briefing",
                    description=(
                        "More details from initiative desk.\n"
                        "More information: https://spiritualdiplomats.org/resources?fbclid=abc\n"
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]
        response_without_links = SimpleNamespace(
            raw_text=(
                '{"title":"Conference and initiative briefing tonight","description":"Tonight we track the conference agenda and initiative updates with concrete facts.\\n\\nIn this stream you will see:\\n🔹 conference timeline and priorities\\n🎤 speaker remarks and context\\n✅ practical next steps for viewers\\n\\nJoin and follow updates. #conference #initiative"}'
            ),
            structured_payload={
                "title": "Conference and initiative briefing tonight",
                "description": (
                    "Tonight we track the conference agenda and initiative updates with concrete facts.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 conference timeline and priorities\n"
                    "🎤 speaker remarks and context\n"
                    "✅ practical next steps for viewers\n\n"
                    "Join and follow updates. #conference #initiative"
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[response_without_links]):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=videos,
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        description: str = attempt.merged.description if attempt.merged else ""
        self.assertIn("🌐 Official links:", description)
        self.assertIn("https://interfaithconf.org/about", description)
        self.assertIn("https://spiritualdiplomats.org/resources", description)
        self.assertNotIn("utm_source", description)
        self.assertNotIn("fbclid", description)

    def test_official_links_dedup_and_non_youtube_selection(self) -> None:
        videos = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Source A",
                    description=(
                        "Official page: https://interfaithconf.org/about?utm_source=yt\n"
                        "Official page mirror: https://interfaithconf.org/about?si=1\n"
                        "Video link: https://youtu.be/aaaaaaaaaaa\n"
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Source B",
                    description="Initiative: https://allatra.org/?feature=share",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]
        response_without_links = SimpleNamespace(
            raw_text=(
                '{"title":"Official resources and initiative update","description":"We summarize the key updates and practical context for tonight.\\n\\nIn this stream you will see:\\n🔹 official agenda and milestones\\n✅ what to follow next\\n\\nJoin and share. #update #resources"}'
            ),
            structured_payload={
                "title": "Official resources and initiative update",
                "description": (
                    "We summarize the key updates and practical context for tonight.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 official agenda and milestones\n"
                    "✅ what to follow next\n\n"
                    "Join and share. #update #resources"
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[response_without_links]):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=videos,
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        description: str = attempt.merged.description if attempt.merged else ""
        self.assertIn("https://interfaithconf.org/about", description)
        self.assertEqual(1, description.count("https://interfaithconf.org/about"))
        self.assertIn("https://allatra.org/", description)
        self.assertNotIn("youtu.be", description)

    def test_short_service_lines_are_normalized_to_expected_language(self) -> None:
        result = normalize_merge_description(
            description=(
                "Це короткий вступ про головну тему!\n\n"
                "In this stream you'll see:\n"
                "📌 перший акцент\n"
                "🔹 другий пункт\n\n"
                "🌐 Official links:\n"
                "https://example.org\n\n"
                "Watch the stream and share your thoughts. #подія"
            ),
            language="uk",
            source_texts=(),
        )
        self.assertIn("У цьому стрімі ви побачите:", result.description_text)
        self.assertIn("🌐 Офіційні ресурси:", result.description_text)
        self.assertIn("Дивіться ефір і діліться думками. #подія", result.description_text)
        self.assertEqual("needs_normalization", result.diagnostics.semantic_gate_status)
        self.assertTrue(result.diagnostics.wrong_language_heading_detected)

    def test_accent_marker_cap_is_enforced_and_overflow_is_logged(self) -> None:
        result = normalize_merge_description(
            description=(
                "Focused hook paragraph with enough detail to stay valid!\n\n"
                "In this stream you'll see:\n"
                "📌 first\n"
                "🎤 second\n"
                "🎥 third\n"
                "⚖ fourth\n"
                "🌐 fifth"
            ),
            language="en",
            source_texts=(),
        )
        self.assertEqual(3, result.diagnostics.accent_bullets_count)
        self.assertEqual(2, result.diagnostics.neutral_bullets_count)
        self.assertTrue(result.diagnostics.accent_overflow)
        self.assertIn("🔹 fourth", result.description_text)
        self.assertIn("🔹 fifth", result.description_text)

    def test_block_spacing_and_role_softening_are_stabilized(self) -> None:
        result = normalize_merge_description(
            description=(
                "This hook stays factual and readable with enough context!\n"
                "In this stream you'll see:\n"
                "🎤 Pastor Vitaliy Orlov comments on the community response\n"
                "🔹 relief updates continue\n"
                "🌐 Official links:\n"
                "https://example.org\n"
                "Watch the stream and share your thoughts. #update"
            ),
            language="en",
            source_texts=("Vitaliy Orlov comments on the community response.",),
        )
        self.assertIn("\n\nIn this stream you'll see:\n", result.description_text)
        self.assertIn("\n\n🌐 Official links:\nhttps://example.org\n\n", result.description_text)
        self.assertIn("🎤 Vitaliy Orlov comments on the community response", result.description_text)
        self.assertNotIn("Pastor Vitaliy Orlov", result.description_text)
        self.assertTrue(result.diagnostics.role_softening_applied)

    def test_core_wrong_language_hook_is_hard_reject(self) -> None:
        result = normalize_merge_description(
            description=(
                "This English hook is clearly not in the expected block language and stays unchanged.\n\n"
                "У цьому стрімі ви побачите:\n"
                "🔹 пункт один\n"
                "🔹 пункт два"
            ),
            language="uk",
            source_texts=(),
        )
        self.assertEqual("hard_reject", result.diagnostics.semantic_gate_status)
        self.assertIn("inconsistent_block_language", result.diagnostics.semantic_gate_reason_codes)

    def test_post_enforcement_repairs_fixable_single_paragraph_shape(self) -> None:
        summary = MergeRunSummary()
        merged_content = MergedLanguageContent(
            title="Final title",
            description=(
                "Sentence one explains the stream. Sentence two adds context. "
                "Sentence three provides supporting detail. Sentence four closes the summary."
            ),
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

    def test_single_source_translate_path_is_unchanged(self) -> None:
        one_video = [
            SimpleNamespace(
                language="en",
                metadata=SimpleNamespace(
                    title="Source title",
                    description="Original source paragraph one.\n\nOriginal source paragraph two.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            )
        ]
        plain_response = SimpleNamespace(
            raw_text="Translated paragraph one.\n\nTranslated paragraph two.",
            structured_payload=None,
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[plain_response]):
            attempt = attempt_openai_single_source_translate_with_audit(
                language="ru",
                videos=one_video,
                config=self._config(),
                attempt_label="TEST_SINGLE",
                summarize_error=lambda error: str(error),
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        self.assertEqual("single_source_plain_ok", attempt.publish_source_label)
        self.assertEqual("Source title", attempt.merged.title if attempt.merged else "")

    def test_merge_service_logs_quality_hardening_fields(self) -> None:
        videos = [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Community briefing",
                    description="Vitaliy Orlov comments on relief work and public updates.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Relief desk",
                    description="Editors track official resources and practical next steps for viewers.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]
        styled_response = SimpleNamespace(
            raw_text=(
                '{"title":"Community briefing tonight","description":"This hook stays factual and readable with enough context!\\n\\n'
                'In this stream you\\u0027ll see:\\n📌 first point\\n🎤 Pastor Vitaliy Orlov comments on relief work\\n🎥 third point\\n⚖ fourth point\\n'
                '🌐 Official links:\\nhttps://example.org\\n\\nWatch the stream and share your thoughts. #update"}'
            ),
            structured_payload={
                "title": "Community briefing tonight",
                "description": (
                    "This hook stays factual and readable with enough context!\n\n"
                    "In this stream you'll see:\n"
                    "📌 first point\n"
                    "🎤 Pastor Vitaliy Orlov comments on relief work\n"
                    "🎥 third point\n"
                    "⚖ fourth point\n"
                    "🌐 Official links:\n"
                    "https://example.org\n\n"
                    "Watch the stream and share your thoughts. #update"
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=videos,
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNotNone(attempt.merged)
        logs = "\n".join(captured.output)
        self.assertIn("neutral_bullets_count=", logs)
        self.assertIn("accent_bullets_count=", logs)
        self.assertIn("accent_overflow=yes", logs)
        self.assertIn("block_language_expected=en", logs)
        self.assertIn("person_role_claims_detected=", logs)
        self.assertIn("role_softening_applied=yes", logs)
        self.assertIn("semantic_gate_status=needs_normalization", logs)


if __name__ == "__main__":
    unittest.main()
