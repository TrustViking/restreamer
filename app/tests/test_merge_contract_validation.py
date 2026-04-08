from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import (
    BLOCK_GENERATION_MODE_REAL_MERGE,
    MergedLanguageContent,
)
from app.llm.merges.merge_quality import normalize_merge_description
from app.llm.merges.merge_service import (
    attempt_openai_merge_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.llm.merges.merge_run_summary import MergeRunSummary

from app.tests.test_merge_contract_helpers import MergeContractServiceBase


class MergeContractValidationTests(MergeContractServiceBase):
    """Tests for validation and quality concerns of merge contract."""

    def test_rejects_opening_cta_line_before_hook_even_in_same_paragraph_block(self) -> None:
        cta_first_response: SimpleNamespace = self._make_merge_response(
            title="Brussels and Kharkiv: focused operational briefing",
            description=(
                "Subscribe and write in the comments which angle we should unpack next.\n"
                "Tonight we align the Brussels vote timeline with the Kharkiv rail disruption to map concrete risks, decisions, and operational consequences.\n\n"
                "In this stream you'll see:\n"
                "🔹 Brussels sanctions vote and budget amendments after the commission session\n"
                "🔹 Anna Kovalenko tracks customs delays and coalition pressure before the chamber debate\n"
                "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                "🔹 Oleh Martynenko details evacuation routes and depot repair sequencing on the eastern line\n"
                "🔹 practical next steps for viewers following both Brussels and Kharkiv updates"
            ),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[cta_first_response, cta_first_response, cta_first_response],
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
        self.assertIsNone(attempt.merged)
        self.assertEqual(3, request_mock.call_count)
        self.assertIn("cta_as_first_paragraph", tuple(attempt.validation_reasons or ()))

    def test_rejects_adjacent_near_duplicate_opening_lines_within_paragraph(self) -> None:
        adjacent_duplicate_lines_response: SimpleNamespace = self._make_merge_response(
            title="Brussels and Kharkiv: decisions and logistics tonight",
            description=(
                "Tonight we track how the Brussels vote and Kharkiv rail disruption reshape practical decisions for the next operational window.\n\n"
                "In this stream you'll see:\n"
                "🔹 Brussels sanctions committee confirms budget amendments, customs delays, and a revised chamber timetable for Tuesday\n"
                "🔹 Brussels sanctions committee confirms budget amendments and customs delays with a revised chamber timetable\n"
                "🔹 Anna Kovalenko tracks customs delays and coalition pressure before the chamber debate\n"
                "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                "🔹 Oleh Martynenko details evacuation routes and depot repair sequencing on the eastern line\n"
                "🔹 practical next steps for viewers following both Brussels and Kharkiv updates"
            ),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[
                adjacent_duplicate_lines_response,
                adjacent_duplicate_lines_response,
                adjacent_duplicate_lines_response,
            ],
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
        self.assertIsNone(attempt.merged)
        self.assertEqual(3, request_mock.call_count)
        self.assertIn("duplicate_paragraph", tuple(attempt.validation_reasons or ()))

    def test_accepts_clean_opening_without_cta_first_or_adjacent_duplicates(self) -> None:
        clean_opening_response: SimpleNamespace = self._make_merge_response(
            title="Brussels and Kharkiv: concrete agenda tonight",
            description=(
                "Tonight we compare the Brussels budget vote with the Kharkiv rail disruption to map what changes next for operations and public messaging.\n"
                "We focus on concrete dates, responsible actors, and direct consequences instead of repeating one thesis in multiple forms.\n\n"
                "In this stream you'll see:\n"
                "🔹 Brussels sanctions vote and budget amendments after the commission session\n"
                "🔹 Anna Kovalenko tracks customs delays and coalition pressure before the chamber debate\n"
                "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                "🔹 Oleh Martynenko details evacuation routes and depot repair sequencing on the eastern line\n"
                "🔹 practical next steps for viewers following both Brussels and Kharkiv updates"
            ),
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[clean_opening_response]):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)

    def test_semantic_generic_merge_is_rejected_without_fallback(self) -> None:
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
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[generic_response, generic_response, generic_response],
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
        self.assertIsNone(attempt.merged)
        self.assertEqual("merge_failed", attempt.publish_source_label)

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
                '{"title":"Brussels and Kharkiv: key decisions tonight","description":"Tonight we map the budget vote and frontline pressure with clear facts and timelines! 🎯\\n\\n- sanctions timeline and vote implications\\n- drone pressure and logistics bottlenecks\\n- aid corridor risks and response steps\\n- John Smith connects budget decisions to field operations\\n- Maria Ivanova details evacuation routes and corridor challenges\\n\\nJohn Smith and Maria Ivanova connect political decisions with field consequences."}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: key decisions tonight",
                "description": (
                    "Tonight we map the budget vote and frontline pressure with clear facts and timelines! 🎯\n\n"
                    "- sanctions timeline and vote implications\n"
                    "- drone pressure and logistics bottlenecks\n"
                    "- aid corridor risks and response steps\n"
                    "- John Smith connects budget decisions to field operations\n"
                    "- Maria Ivanova details evacuation routes and corridor challenges\n\n"
                    "John Smith and Maria Ivanova connect political decisions with field consequences."
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
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
        self.assertIn("bullet_points_count=5", joined_logs)
        self.assertIn("named_entities_preserved=2", joined_logs)
        self.assertIn("named_entities_metric=informational", joined_logs)
        self.assertIn("emoji_count=1", joined_logs)

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
                '{"title":"Brussels and Kharkiv: agenda tonight","description":"Tonight we focus on the budget vote and frontline logistics without noise.\\n\\nIn this stream you will see:\\n\u2013 sanctions timeline and vote implications\\n\u2014 drone pressure and corridor risks\\n\u2013 John Smith tracks the vote timeline and budget amendments\\n\u2014 Maria Ivanova details evacuation routes and depot damage\\n\u2013 practical next steps for viewers following both source lines\\n\\nJohn Smith and Maria Ivanova connect decisions with field outcomes."}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: agenda tonight",
                "description": (
                    "Tonight we focus on the budget vote and frontline logistics without noise.\n\n"
                    "In this stream you will see:\n"
                    "\u2013 sanctions timeline and vote implications\n"
                    "\u2014 drone pressure and corridor risks\n"
                    "\u2013 John Smith tracks the vote timeline and budget amendments\n"
                    "\u2014 Maria Ivanova details evacuation routes and depot damage\n"
                    "\u2013 practical next steps for viewers following both source lines\n\n"
                    "John Smith and Maria Ivanova connect decisions with field outcomes."
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
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
        self.assertIn("bullet_points_count=5", joined_logs)

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
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
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
                '{"title":"Brussels panel: legal safeguards and conference agenda","description":"This stream tracks why legal safeguards and testimony matter right now.\\n\\nIn this stream you will see:\\n📌 legal safeguards and testimony timeline\\n⚖ witness rights and justice risks\\n🎤 Alice Brown key remarks\\n🌐 conference initiative milestones\\n🔹 Bob Green details practical next steps from the initiative desk\\n\\nJoin the live discussion and share your view. #rights #justice"}'
            ),
            structured_payload={
                "title": "Brussels panel: legal safeguards and conference agenda",
                "description": (
                    "This stream tracks why legal safeguards and testimony matter right now.\n\n"
                    "In this stream you'll see:\n"
                    "📌 legal safeguards and testimony timeline\n"
                    "⚖ witness rights and justice risks\n"
                    "🎤 Alice Brown's key remarks\n"
                    "🌐 conference initiative milestones\n"
                    "🔹 Bob Green details practical next steps from the initiative desk\n\n"
                    "Join the live discussion and share your view. #rights #justice"
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
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
        self.assertIn("semantic_bullets_count=5", joined_logs)
        self.assertIn("bullets_with_emoji_count=5", joined_logs)
        self.assertIn("bullets_with_plain_marker_count=0", joined_logs)

    def test_merge_stage_does_not_inject_official_links_when_missing_in_llm_output(self) -> None:
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
                '{"title":"Conference and initiative briefing tonight","description":"Tonight we track the conference agenda and initiative updates with concrete facts.\\n\\nIn this stream you will see:\\n🔹 conference timeline and priorities\\n🎤 speaker remarks and context\\n✅ practical next steps for viewers\\n🔹 initiative milestones and official site updates\\n🔹 how viewers can follow the next session and registration steps\\n\\nJoin and follow updates. #conference #initiative"}'
            ),
            structured_payload={
                "title": "Conference and initiative briefing tonight",
                "description": (
                    "Tonight we track the conference agenda and initiative updates with concrete facts.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 conference timeline and priorities\n"
                    "🎤 speaker remarks and context\n"
                    "✅ practical next steps for viewers\n"
                    "🔹 initiative milestones and official site updates\n"
                    "🔹 how viewers can follow the next session and registration steps\n\n"
                    "Join and follow updates. #conference #initiative"
                ),
            },
            )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[response_without_links]):
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
        self.assertNotIn("🌐 Official links:", description)
        self.assertNotIn("https://interfaithconf.org/about", description)
        self.assertNotIn("https://spiritualdiplomats.org/resources", description)

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
                '{"title":"Official resources and initiative update","description":"We summarize the key updates and practical context for tonight.\\n\\nIn this stream you will see:\\n🔹 official agenda and milestones from Source A\\n🔹 initiative highlights and updates from Source B\\n✅ what to follow next for viewers\\n🔹 deduplication of key links and official resources\\n🔹 practical viewer steps and next session details\\n\\nJoin and share. #update #resources"}'
            ),
            structured_payload={
                "title": "Official resources and initiative update",
                "description": (
                    "We summarize the key updates and practical context for tonight.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 official agenda and milestones from Source A\n"
                    "🔹 initiative highlights and updates from Source B\n"
                    "✅ what to follow next for viewers\n"
                    "🔹 deduplication of key links and official resources\n"
                    "🔹 practical viewer steps and next session details\n\n"
                    "Join and share. #update #resources"
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[response_without_links]):
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
        self.assertNotIn("https://interfaithconf.org/about", description)
        self.assertNotIn("https://allatra.org/", description)
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

    def test_script_mix_guard_rejects_cyrillic_contamination_inside_english_body(self) -> None:
        result = normalize_merge_description(
            description=(
                "This hook stays factual and readable with enough context about the main topic!\n\n"
                "In this stream you'll see:\n"
                "🔹 budget timeline and mиксed contamination inside the main bullet\n"
                "🔹 verified operational follow-up for the next debate window"
            ),
            language="en",
            source_texts=(),
        )
        self.assertEqual("hard_reject", result.diagnostics.semantic_gate_status)
        self.assertIn("script_mix_contamination", result.diagnostics.semantic_gate_reason_codes)
        self.assertIn("mиксed", result.diagnostics.script_mix_suspects)

    def test_script_mix_guard_ignores_allowed_brands_and_urls_in_cyrillic_text(self) -> None:
        result = normalize_merge_description(
            description=(
                "Цей вступ лишається фактичним і зрозумілим для глядачів.\n\n"
                "У цьому стрімі ви побачите:\n"
                "🔹 OpenAI та YouTube згадуються як бренди без зайвої мовної кари\n"
                "🔹 NASA і AI залишаються допустимими абревіатурами\n\n"
                "🌐 Офіційні ресурси:\n"
                "https://example.org/openai\n\n"
                "Дивіться ефір і діліться думками. #update"
            ),
            language="uk",
            source_texts=(),
        )
        self.assertFalse(result.diagnostics.script_mix_detected)
        self.assertNotIn("script_mix_contamination", result.diagnostics.semantic_gate_reason_codes)

    def test_script_mix_guard_ignores_bare_domain_in_cyrillic_text(self) -> None:
        """lstv.co.uk inside RU text must NOT trigger script_mix_contamination."""
        result = normalize_merge_description(
            description=(
                "После решения украинского суда антикультист дал ссылку на lstv.co.uk как ключевой эпизод.\n\n"
                "⚖ Конфликт фактов и манипуляций: что стоит за антикультовой риторикой.\n"
                "🔹 Луиджи Корвальо, член правления ФЕКРИС: реакция на реабилитацию.\n"
                "🔹 Техническая верификация lstv.co.uk: инфраструктура хостинга.\n"
                "🔹 Следы в открытых источниках и вывод расследования.\n\n"
                "Оставляйте комментарии по фактам. #расследование"
            ),
            language="ru",
            source_texts=(),
        )
        self.assertFalse(result.diagnostics.script_mix_detected)
        self.assertNotIn("script_mix_contamination", result.diagnostics.semantic_gate_reason_codes)

    def test_script_mix_guard_ignores_multi_level_bare_domain(self) -> None:
        """Multi-level domains like news.bbc.co.uk must not be flagged."""
        result = normalize_merge_description(
            description=(
                "Ця новина була опублікована на news.bbc.co.uk та підтверджена.\n\n"
                "У цьому стрімі ви побачите:\n"
                "🔹 пункт один\n"
                "🔹 пункт два\n\n"
                "Дивіться ефір. #новини"
            ),
            language="uk",
            source_texts=(),
        )
        self.assertFalse(result.diagnostics.script_mix_detected)

    def test_script_mix_guard_catches_mixed_script_token_in_title(self) -> None:
        """A token like FЕКРИС (Latin F + Cyrillic ЕКРИС) in title must be caught."""
        result = normalize_merge_description(
            description=(
                "Антикульт под лупой: что стоит за риторикой.\n\n"
                "🔹 пункт один\n"
                "🔹 пункт два\n\n"
                "Оставляйте комментарии. #тест"
            ),
            language="ru",
            source_texts=(),
            title="Антикульт под лупой: сайт lstv.co.uk, FЕКРИС и тени",
        )
        self.assertTrue(result.diagnostics.script_mix_detected)
        self.assertIn("script_mix_contamination", result.diagnostics.semantic_gate_reason_codes)
        self.assertIn("FЕКРИС", result.diagnostics.script_mix_suspects)
        self.assertNotIn("lstv", result.diagnostics.script_mix_suspects)

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
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[styled_response]), self.assertLogs(
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

    def test_normalize_merge_description_does_not_double_prefix_bullet_as_lead_in(self) -> None:
        description_with_bullet_as_lead_in: str = (
            "Hook paragraph with a question?\n\n"
            "🔹 First bullet as lead-in\n"
            "🔹 Second bullet\n"
            "🔹 Third bullet"
        )
        result = normalize_merge_description(
            description=description_with_bullet_as_lead_in,
            language="en",
            source_texts=(),
        )
        for line in result.description_text.splitlines():
            stripped_line: str = line.strip()
            if not stripped_line:
                continue
            self.assertFalse(
                stripped_line.startswith("🔹 🔹"),
                msg=f"Double bullet marker found in line: {stripped_line!r}",
            )


if __name__ == "__main__":
    unittest.main()
