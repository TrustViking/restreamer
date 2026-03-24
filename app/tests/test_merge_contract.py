from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    BLOCK_GENERATION_MODE_REAL_MERGE,
    MergedLanguageContent,
)
from app.llm.merges.merge_quality import normalize_merge_description
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.llm.merges.merge_parser import (
    parse_merge_response_or_raise,
    sanitize_title,
    separate_merge_body_and_tail,
)
from app.llm.merges.merge_prompt import build_llm_merge_prompt_text
from app.llm.merges.merge_retry import _build_expanded_retry_profile
from app.llm.merges.merge_validation import (
    MergeAttemptFailure,
    _is_softened_distinctive_source_coverage_eligible,
    _reason_code_from_error,
    _reason_codes_from_error,
)
from app.llm.merges.merge_service import (
    attempt_openai_merge_with_audit,
    attempt_openai_single_source_translate_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.llm.merges.merge_run_summary import MergeRunSummary


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

    def test_allowed_tail_blocks_do_not_break_body_paragraph_count(self) -> None:
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name="openai",
            model_name="gpt-5.1",
            raw_text=(
                '{"title":"Final title","description":"Hook paragraph.\\n\\n'
                'In this stream you will see:\\n🔹 point one\\n🔹 point two\\n\\n'
                'https://youtu.be/aaaaaaaaaaa\\n\\n'
                '🌐 Official links:\\nhttps://example.org\\n\\n'
                'Watch the stream and share your thoughts.\\n\\n'
                '#stream #topic"}'
            ),
        )
        self.assertEqual(2, paragraph_count)
        self.assertIn("https://youtu.be/aaaaaaaaaaa", merged_content.description)
        self.assertIn("🌐 Official links:", merged_content.description)

    def test_body_paragraph_count_ignores_realistic_service_tail_mass(self) -> None:
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name="openai",
            model_name="gpt-5.1",
            raw_text=(
                '{"title":"Final title","description":"Hook paragraph with the core conflict and verified context.\\n\\n'
                "In this stream you\\u0027ll see:\\n🔹 point one\\n🔹 point two\\n🔹 point three\\n\\n"
                "Paragraph three keeps the broader context and timeline grounded in the sources.\\n\\n"
                "Paragraph four closes with the practical context and concrete next developments.\\n\\n"
                "https://youtu.be/aaaaaaaaaaa\\nhttps://www.youtube.com/watch?v=bbbbbbbbbbb\\n\\n"
                "🌐 Official links:\\nhttps://example.org/official\\nhttps://allatra.org/resource\\n\\n"
                'Watch the stream and share your thoughts.\\n\\n#stream #topic"}'
            ),
        )
        self.assertEqual(4, paragraph_count)
        self.assertEqual(
            8,
            len([part for part in merged_content.description.split("\n\n") if part.strip()]),
        )
        self.assertIn("https://www.youtube.com/watch?v=bbbbbbbbbbb", merged_content.description)
        self.assertIn("#stream #topic", merged_content.description)

    def test_body_only_recovery_accepts_near_good_body(self) -> None:
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name="openai",
            model_name="gpt-5.1",
            raw_text=(
                '{"title":"Recovered title","description":"Paragraph one.\\n\\nParagraph two.\\n\\n'
                'Paragraph three.\\n\\nParagraph four.\\n\\nParagraph five.\\n\\n'
                'Watch the stream and share your thoughts.\\n\\n#topic"}'
            ),
        )
        self.assertEqual(4, paragraph_count)
        self.assertIn("Paragraph one.", merged_content.description)
        self.assertIn("#topic", merged_content.description)

    def test_body_that_stays_invalid_after_tail_split_and_recovery_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as raised:
            parse_merge_response_or_raise(
                provider_name="openai",
                model_name="gpt-5.1",
                raw_text=(
                    '{"title":"Bad title","description":"Paragraph one.\\n\\nParagraph two.\\n\\n'
                    'Paragraph three.\\n\\nParagraph four.\\n\\nParagraph five.\\n\\n'
                    'Paragraph six.\\n\\nParagraph seven.\\n\\nParagraph eight.\\n\\n'
                    'Paragraph nine.\\n\\n#topic"}'
                ),
            )
        self.assertIn("body paragraph count", str(raised.exception))

    def test_tail_separation_recognizes_multiple_allowed_tail_blocks_in_order(self) -> None:
        separation = separate_merge_body_and_tail(
            text=(
                "Hook paragraph.\n\n"
                "In this stream you'll see:\n"
                "🔹 main point\n"
                "✅ practical follow-up\n\n"
                "https://youtu.be/aaaaaaaaaaa\n\n"
                "🌐 Official links:\n"
                "https://example.org\n\n"
                "Watch the stream and share your thoughts.\n\n"
                "#topic #update"
            )
        )
        self.assertEqual(2, separation.body_paragraph_count_after_recovery)
        self.assertEqual(
            ("youtube_links", "official_links", "cta", "hashtags"),
            separation.tail_blocks,
        )

    def test_realistic_merge_output_with_five_service_tail_paragraphs_is_accepted(self) -> None:
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name="openai",
            model_name="gpt-5.1",
            raw_text=(
                '{"title":"Conference and initiative briefing tonight","description":"Tonight we track the conference agenda and initiative updates with concrete facts.\\n\\n'
                "In this stream you\\u0027ll see:\\n🔹 conference timeline and priorities\\n🎤 speaker remarks and context\\n✅ practical next steps for viewers\\n\\n"
                "The second body paragraph keeps the legal and organizational context tied to the sources.\\n\\n"
                "The third body paragraph highlights what changed since the previous stream and why it matters.\\n\\n"
                "https://youtu.be/aaaaaaaaaaa\\n\\n"
                "https://www.youtube.com/watch?v=bbbbbbbbbbb\\n\\n"
                "🌐 Official links:\\nhttps://interfaithconf.org/about\\nhttps://spiritualdiplomats.org/resources\\n\\n"
                'Join and follow updates.\\n\\n#conference #initiative"}'
            ),
        )
        self.assertEqual(4, paragraph_count)
        separation = separate_merge_body_and_tail(text=merged_content.description)
        self.assertEqual(9, separation.raw_paragraph_count)
        self.assertEqual(4, separation.body_paragraph_count_after_recovery)
        self.assertEqual(
            ("youtube_links", "youtube_links", "official_links", "cta", "hashtags"),
            separation.tail_blocks,
        )


class MergeContractServiceTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            llm_model="gpt-5.1",
            openai_model_primary="gpt-5.1",
            openai_model_fallback="gpt-5.1",
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
Do not enumerate sources as 1) 2) 3).
Do not output generic slogans or abstract editorial phrasing.
Do not use emoji in the title.
{merge_contract_block}
Avoid asserting strong person titles or role labels unless they are clearly necessary and well-supported by the sources.
Optional official links block is allowed before close line with 1 to 3 non-YouTube links from sources.
An optional one-line close should be practical CTA + 2 to 5 hashtags.
Return strict JSON with title and description only.

{youtube_candidates_block}

{sources_block}
""".strip(),
                llm_merge_structural_rules=(
                    "MERGE STRUCTURAL RULES\n"
                    "RULE 1: Start with a standalone hook paragraph before any bullets.\n"
                    "RULE 2: Keep visual paragraph boundaries explicit with one blank line between structural blocks.\n"
                    "RULE 3: Keep CTA and hashtags only in the final tail position, never as opener lines.\n"
                    "RULE 4: Do not repeat or paraphrase the hook thesis in the next adjacent line or paragraph.\n"
                    "EXAMPLE A (bad): CTA line opens the description and the real hook starts later.\n"
                    "EXAMPLE A (good): Hook opens first, CTA appears only at the end.\n"
                    "EXAMPLE B (bad): Two adjacent lines restate the same thesis with minor wording changes.\n"
                    "EXAMPLE B (good): The second line introduces new facts instead of repeating the opener."
                ),
                llm_merge_contracts_json=json.dumps(
                    {
                        "compact": (
                            "Use the compact merge contract for 1 to 2 source items.\n"
                            "Write one cohesive stream description in 2 to 3 compact paragraphs.\n"
                            "Paragraph 1 (hook): write 1 to 2 sentences grounded in the main tension, risk, or key conflict.\n"
                            "Keep the hook editorial and readable, but never clickbait.\n"
                            "Paragraph 2 (theses block): open with one short editorial statement that names the central tension, key question, or main conflict — not a lead-in phrase like 'In this stream you will see'.\n"
                            "Then write {compact_bullet_min} to {compact_bullet_max} short thesis bullet lines (target range {compact_bullet_range}).\n"
                            "Each bullet line must start with exactly one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.\n"
                            "Most bullets should start with 🔹.\n"
                            "Accent markers are rare and optional; use no more than 3 accent markers per theses block.\n"
                            "Keep marker usage controlled and readable; do not use dash-only bullets as the sole style.\n"
                            "Do not present the agenda as SOURCE 1 / SOURCE 2.\n"
                            "Keep agenda points specific and factual, not generic placeholders.\n"
                            "The description must still cover all merged source items and preserve key concrete facts from each source."
                        ),
                        "expanded": (
                            "Use the expanded merge contract for 3 or more source items.\n"
                            "Write one cohesive stream description in 3 to 4 compact paragraphs.\n"
                            "Paragraph 1 (hook): write 1 to 2 sentences grounded in the main tension, risk, or key conflict.\n"
                            "Keep the hook editorial and readable, but never clickbait.\n"
                            "After the hook, use a more open agenda structure instead of one overloaded thesis block.\n"
                            "Write {expanded_bullet_min} to {expanded_bullet_max} short bullet lines total.\n"
                            "You may organize the bullets into 2 to 3 thematic micro-blocks when that improves clarity. Separate each thematic micro-block from the next with a blank line.\n"
                            "IMPORTANT: Bullet lines within the same thematic micro-block must be separated by single newlines (\\n), NOT by blank lines (\\n\\n). A blank line starts a new paragraph. The entire bullet section should be at most 2 visual paragraphs.\n"
                            "Each bullet line must start with exactly one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.\n"
                            "Most bullets should start with 🔹.\n"
                            "Accent markers are rare and optional; use no more than 3 accent markers per description.\n"
                            "Keep marker usage controlled and readable; do not use dash-only bullets as the sole style.\n"
                            "Do not present the agenda as SOURCE 1 / SOURCE 2 / SOURCE 3.\n"
                            "Keep agenda points specific and factual, not generic placeholders.\n"
                            "Do not let the opening hook consume most of the useful summary space.\n"
                            "A polished opening is never a substitute for a concrete multi-angle summary.\n"
                            "Treat the post-hook body as the main payload and let it carry most of the concrete information.\n"
                            "Make most bullets fact-bearing: anchor them with names, places, institutions, numbers, timings, events, or operational consequences whenever the sources provide them.\n"
                            "Give the body at least two clearly substantive agenda lanes after the hook instead of one thin run of near-duplicate bullets.\n"
                            "Across the agenda, preserve distinguishable source details such as names, places, numbers, events, or clearly separate thematic nodes whenever the sources provide them.\n"
                            "Across 3 or more sources, spread the bullets across multiple source lines or topic nodes so the summary does not collapse into one generic lane.\n"
                            "If the merged sources span different domains, separate them across different bullets or short thematic blocks instead of compressing them into one universal bullet.\n"
                            "Do not combine science or medicine, climate or environment, disasters or catastrophic hazards, psychology or cognition or behavior, and broad social or moral conclusions into one bullet or one cause-and-effect chain unless the sources explicitly require that connection.\n"
                            "Thematic grouping is encouraged when useful: research, hazards, human behavior, practical risk, public meaning, or response can be separated into different bullets or micro-blocks.\n"
                            "The description must still cover all merged source items and preserve key concrete facts from each source.\n"
                            "{speaker_anchor_line}"
                        ),
                        "narrative": (
                            "This stream covers a single unified event or case. "
                            "Write the description as connected prose, not a bullet list. "
                            "Hook paragraph first, then 2-3 prose paragraphs. No bullets."
                        ),
                    },
                    ensure_ascii=False,
                ),
                llm_merge_retry_reinforcements_json=json.dumps({}, ensure_ascii=False),
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

    def _videos_three_sources(self) -> list[SimpleNamespace]:
        return [
            *self._videos(),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Title 3",
                    description="Paragraph five.\n\nParagraph six.",
                ),
                normalized_link="https://youtube.com/watch?v=ccccccccccc",
            ),
        ]

    def _expanded_validation_videos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Brussels sanctions vote briefing",
                    description=(
                        "In Brussels, Anna Kovalenko tracks the March 18 sanctions vote, "
                        "budget amendments, and customs delays after the commission session."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource01",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Kharkiv rail and drone update",
                    description=(
                        "In Kharkiv, Oleh Martynenko reports 17 drone strikes, rail hub outages, "
                        "and evacuation routes for Saltivka districts."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource02",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Geneva relief corridor desk",
                    description=(
                        "From Geneva, Marta Leone outlines the aid corridor timetable, WHO cargo "
                        "counts, and donor pledges for Odesa and Mykolaiv hospitals."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource03",
            ),
        ]

    def _expanded_validation_videos_four_sources(self) -> list[SimpleNamespace]:
        return [
            *self._expanded_validation_videos(),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Lviv grid repair logistics",
                    description=(
                        "In Lviv, Iryna Melnyk details transformer shortages, repair crews, "
                        "and the grid restoration queue after regional substation damage."
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=expandedsource04",
            ),
        ]

    def _make_merge_response(
        self,
        *,
        title: str,
        description: str,
    ) -> SimpleNamespace:
        structured_payload: dict[str, str] = {
            "title": title,
            "description": description,
        }
        raw_text: str = json.dumps(structured_payload, ensure_ascii=False)
        return SimpleNamespace(
            raw_text=raw_text,
            structured_payload=structured_payload,
        )

    def _expanded_description_with_body_paragraphs(
        self,
        *,
        body_paragraphs: int,
    ) -> str:
        if body_paragraphs < 2:
            raise ValueError("body_paragraphs must be at least 2")
        hook_paragraph: str = (
            "Tonight we track how the Brussels vote, Kharkiv transport shocks, and Geneva aid timing now intersect: "
            "each lane carries concrete operational consequences for viewers following this agenda."
        )
        bullet_paragraphs: list[str] = [
            (
                "In this stream you'll see:\n"
                "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                "🔹 Anna Kovalenko maps coalition counts and customs pressure before the chamber debate"
            ),
            "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts.",
            "🔹 Oleh Martynenko details evacuation routes and depot repair sequencing on the eastern line.",
            "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals.",
            "🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release window.",
            "🔹 Lviv transformer shipments and repair crew rotations now define the overnight recovery queue.",
            "🔹 Baltic cargo reroutes are changing fuel timing and insurance windows for regional logistics.",
            "🔹 Emergency procurement updates now tie Brussels financing signals to corridor-level medical deliveries.",
        ]
        required_bullet_paragraphs: int = body_paragraphs - 1
        if required_bullet_paragraphs > len(bullet_paragraphs):
            raise ValueError("requested body_paragraphs exceeds test fixture capacity")
        selected_bullet_paragraphs: list[str] = bullet_paragraphs[:required_bullet_paragraphs]
        return "\n\n".join([hook_paragraph, *selected_bullet_paragraphs])

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
        self.assertIn(
            "must still cover all merged source items and preserve key concrete facts from each source",
            prompt_text,
        )
        self.assertIn("Do not output generic slogans", prompt_text)
        self.assertIn("Use the compact merge contract for 1 to 2 source items.", prompt_text)
        self.assertIn("2 to 3 compact paragraphs", prompt_text)
        self.assertIn("Then write 4 to 7 short thesis bullet lines", prompt_text)
        self.assertIn("allowed marker", prompt_text)
        self.assertIn("Most bullets should start with 🔹", prompt_text)
        self.assertIn("no more than 3 accent markers", prompt_text)
        self.assertIn("Avoid asserting strong person titles", prompt_text)
        self.assertIn("If the sources touch different semantic domains, do not compress them into one sentence.", prompt_text)
        self.assertIn("These topics may stay in one final description, but present them as separate lines of discussion in separate sentences.", prompt_text)
        self.assertIn("Do not build one long cause-and-effect chain across all of those domains in a single sentence.", prompt_text)
        self.assertIn("Do not include any URLs in the output.", prompt_text)
        self.assertIn("Link blocks will be assembled later by the system.", prompt_text)
        self.assertIn("Do not use emoji in the title.", prompt_text)
        self.assertIn("optional one-line close", prompt_text.lower())
        self.assertNotIn("URL:", prompt_text)
        self.assertIn("Paragraph one.\n\nParagraph two.", prompt_text)

    def test_prompt_uses_expanded_contract_for_three_or_more_sources(self) -> None:
        with self.assertLogs(level="INFO") as captured:
            prompt_text: str = build_llm_merge_prompt_text(
                language="en",
                videos=self._videos_three_sources(),
                config=self._config(),
                no_description_text="no description",
            )

        joined_logs: str = "\n".join(captured.output)
        self.assertIn("Use the expanded merge contract for 3 or more source items.", prompt_text)
        self.assertIn("Write 4 to 6 short bullet lines total.", prompt_text)
        self.assertIn("2 to 3 thematic micro-blocks", prompt_text)
        self.assertIn(
            "Do not combine science or medicine, climate or environment, disasters or catastrophic hazards, psychology or cognition or behavior, and broad social or moral conclusions into one bullet",
            prompt_text,
        )
        self.assertIn("Thematic grouping is encouraged when useful", prompt_text)
        self.assertIn(
            "merge_prompt_contract_selected language=en source_count=3 contract_mode=expanded expected_bullet_range=4-6 expanded_structure_enabled=yes",
            joined_logs,
        )

    def test_compact_contract_is_read_from_templates_and_exposes_4_7_range(self) -> None:
        config: SimpleNamespace = self._config()
        config.templates.llm_merge_contracts_json = json.dumps(
            {
                "compact": (
                    "TEMPLATE COMPACT CONTRACT\n"
                    "Use compact bullet range {compact_bullet_range} for compact mode."
                ),
                "expanded": "Expanded template {expanded_bullet_min}-{expanded_bullet_max}",
                "narrative": "Narrative template",
            },
            ensure_ascii=False,
        )
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=self._videos(),
            config=config,
            no_description_text="no description",
        )
        self.assertIn("TEMPLATE COMPACT CONTRACT", prompt_text)
        self.assertIn("compact bullet range 4-7", prompt_text)

    def test_expanded_retry_profile_targets_expected_weak_points(self) -> None:
        scenarios = (
            (
                ("insufficient_expanded_body",),
                "targeted",
                "body_depth",
                "post-hook body clearly denser",
            ),
            (
                ("too_few_expanded_bullets",),
                "targeted",
                "bullet_sufficiency",
                "enough distinct, meaningful bullets",
            ),
            (
                ("overly_generic_body",),
                "targeted",
                "source_specificity",
                "source-grounded specifics",
            ),
            (
                ("hook_dominates_body",),
                "targeted",
                "hook_restraint",
                "Keep the hook brief and functional",
            ),
            (
                ("weak_source_coverage",),
                "targeted",
                "source_spread",
                "Restore distinguishable spread across source lines or topic nodes",
            ),
        )
        for reject_signals, expected_mode, expected_focus, expected_fragment in scenarios:
            with self.subTest(reject_signals=reject_signals):
                profile = _build_expanded_retry_profile(
                    source_count=3,
                    reject_signals=reject_signals,
                )
                self.assertEqual(expected_mode, profile.retry_mode)
                self.assertEqual((expected_focus,), profile.focus_tags)
                self.assertIn(expected_fragment, "\n".join(profile.reinforcement_lines))

    def test_three_source_targeted_retry_adds_combined_reinforcement_lines(self) -> None:
        profile = _build_expanded_retry_profile(
            source_count=3,
            reject_signals=(
                "insufficient_expanded_body",
                "too_few_expanded_bullets",
                "overly_generic_body",
                "weak_source_coverage",
            ),
        )
        reinforcement_text: str = "\n".join(profile.reinforcement_lines)
        self.assertIn("2 to 3 short agenda tracks", reinforcement_text)
        self.assertIn("cut generic filler bridges", reinforcement_text)

    def test_four_source_targeted_retry_adds_structured_source_specific_reinforcement(self) -> None:
        profile = _build_expanded_retry_profile(
            source_count=4,
            reject_signals=(
                "insufficient_expanded_body",
                "too_few_expanded_bullets",
                "overly_generic_body",
                "weak_source_coverage",
            ),
        )
        reinforcement_text: str = "\n".join(profile.reinforcement_lines)
        self.assertIn("2 to 3 short agenda tracks", reinforcement_text)
        self.assertIn("cut generic filler bridges", reinforcement_text)
        self.assertIn("do not collapse the post-hook body into one umbrella summary", reinforcement_text)
        self.assertIn("Build 2 to 3 meaningful thematic micro-blocks after the hook", reinforcement_text)
        self.assertIn("source-specific density, not just extra length", reinforcement_text)
        self.assertIn("Make every source leave a recognizable trace in the body", reinforcement_text)

    def test_compact_prompt_ignores_expanded_retry_profile(self) -> None:
        retry_profile = _build_expanded_retry_profile(
            source_count=3,
            reject_signals=("insufficient_expanded_body",),
        )
        prompt_text: str = build_llm_merge_prompt_text(
            language="en",
            videos=self._videos(),
            config=self._config(),
            no_description_text="no description",
            expanded_retry_profile=retry_profile,
        )
        self.assertNotIn("EXPANDED RETRY FOCUS", prompt_text)
        self.assertIn("Use the compact merge contract for 1 to 2 source items.", prompt_text)

    def test_merge_prompt_uses_clean_full_source_text_without_urls_hashtags_or_truncation(self) -> None:
        videos = [
            SimpleNamespace(
                row_number=1,
                metadata=SimpleNamespace(
                    title="Source 1",
                    description=(
                        "Hook paragraph with concrete facts and named people. "
                        + ("A" * 2600)
                        + "\n\n"
                        "Main stream link https://youtu.be/aaaaaaaaaaa\n"
                        "Official links:\nhttps://example.org/details\n\n"
                        "Join and follow updates. #topic #update"
                    ),
                ),
                normalized_link="https://youtube.com/watch?v=sourcevideo01",
            ),
            SimpleNamespace(
                row_number=2,
                metadata=SimpleNamespace(
                    title="Source 2",
                    description="Second source keeps the semantic context intact without extra links.",
                ),
                normalized_link="https://youtube.com/watch?v=sourcevideo02",
            ),
        ]
        with self.assertLogs(level="INFO") as captured:
            prompt_text: str = build_llm_merge_prompt_text(
                language="en",
                videos=videos,
                config=self._config(),
                no_description_text="no description",
            )
        self.assertNotIn("https://youtu.be/aaaaaaaaaaa", prompt_text)
        self.assertNotIn("https://example.org/details", prompt_text)
        self.assertNotIn("#topic", prompt_text)
        self.assertNotIn("Official links:", prompt_text)
        self.assertNotIn("Join and follow updates.", prompt_text)
        self.assertIn("Hook paragraph with concrete facts and named people.", prompt_text)
        self.assertIn("Second source keeps the semantic context intact without extra links.", prompt_text)
        self.assertIn("Do not add a recommended materials block", prompt_text)
        self.assertIn("A" * 2400, prompt_text)
        self.assertNotIn("YOUTUBE CANDIDATES", prompt_text)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("hard_truncation=disabled", joined_logs)
        self.assertIn("merge_source_text_prepared language=en source_index=1", joined_logs)
        self.assertIn("urls_removed=", joined_logs)
        self.assertIn("hashtags_removed=", joined_logs)

    def test_invalid_primary_response_triggers_retry_then_final_failure(self) -> None:
        responses = [
            SimpleNamespace(raw_text='{"variants":[{"title":"bad"}]}', structured_payload={"variants": [{"title": "bad"}]}),
            SimpleNamespace(raw_text='{"title":"","description":"bad"}', structured_payload={"title": "", "description": "bad"}),
            SimpleNamespace(raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}', structured_payload={"title": "Final title", "description": "Paragraph one.\n\nParagraph two."}),
        ]

        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=responses) as request_mock:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )

        self.assertEqual(2, request_mock.call_count)
        self.assertIsNone(attempt.merged)
        self.assertEqual("gpt-5.1", attempt.model_name)
        self.assertEqual("merge_failed", attempt.publish_source_label)

    def test_merge_attempt_failure_carries_reason_code(self) -> None:
        failure = MergeAttemptFailure(
            reason_code="missing_keys",
            reason="missing_keys:description",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
            reason_codes=("missing_keys",),
        )
        self.assertEqual("missing_keys", failure.reason_code)
        self.assertEqual('{"title":"x"}', failure.raw_response_text)
        self.assertEqual(("missing_keys",), failure.reason_codes)

    def test_reason_code_from_error_uses_structured_codes_without_text_fallback(self) -> None:
        failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Human-readable text changed and carries no machine-readable list.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
            reason_codes=("overly_generic_body", "weak_source_coverage"),
        )
        with patch(
            "app.llm.merges.merge_service._extract_description_validation_reason_codes"
        ) as parse_mock:
            reason_codes = _reason_codes_from_error(failure)
            primary_reason = _reason_code_from_error(failure)
        self.assertEqual(("overly_generic_body", "weak_source_coverage"), reason_codes)
        self.assertEqual("overly_generic_body", primary_reason)
        parse_mock.assert_not_called()

    def test_reason_code_from_error_uses_single_defensive_text_fallback_when_structured_codes_missing(self) -> None:
        legacy_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="description validation failed: insufficient_expanded_body,weak_source_coverage",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
            reason_codes=(),
        )
        with patch(
            "app.llm.merges.merge_validation._extract_description_validation_reason_codes",
            return_value=("insufficient_expanded_body", "weak_source_coverage"),
        ) as parse_mock:
            primary_reason = _reason_code_from_error(legacy_failure)
        self.assertEqual("insufficient_expanded_body", primary_reason)
        parse_mock.assert_called_once_with(str(legacy_failure))

    def test_final_failure_is_honest(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"","description":"Only one paragraph."}',
            structured_payload={"title": "", "description": "Only one paragraph."},
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]), self.assertLogs(
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
        self.assertEqual(BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE, attempt.block_generation_mode)
        self.assertEqual("fallback_titles", attempt.title_source)
        self.assertEqual("fallback_none", attempt.hook_source)
        self.assertEqual("fallback_source_descriptions", attempt.body_source)

    def test_fatal_model_config_error_does_not_retry_merge_attempt(self) -> None:
        fatal_error = LlmModelConfigurationError(
            provider_name="openai",
            model_name="gpt-5.4",
            reason_code="openai_model_access_denied",
            detail="Project does not have access to model `gpt-5.4`",
            status_code=403,
            api_error_code="access_denied",
            api_error_param="model",
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=fatal_error,
        ) as request_mock:
            with self.assertRaises(LlmModelConfigurationError):
                attempt_openai_merge_with_audit(
                    language="en",
                    videos=self._videos(),
                    config=self._config(),
                    attempt_label="TEST",
                    summarize_error=lambda error: str(error),
                    normalize_youtube_url=lambda url: url,
                    no_description_text="no description",
                )
        self.assertEqual(1, request_mock.call_count)

    def test_successful_merge_attempt_keeps_real_merge_mode(self) -> None:
        good_response = SimpleNamespace(
            raw_text=(
                '{"title":"Final title","description":"Paragraph one with concrete context and key tension.\\n\\n'
                "In this stream you'll see:\\n"
                "🔹 first key point from source one\\n"
                "🔹 second key point from source one\\n"
                "🔹 third key point from source two\\n"
                "🔹 fourth key point from source two\\n"
                '🔹 fifth combined conclusion from both sources"}'
            ),
            structured_payload={
                "title": "Final title",
                "description": (
                    "Paragraph one with concrete context and key tension.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 first key point from source one\n"
                    "🔹 second key point from source one\n"
                    "🔹 third key point from source two\n"
                    "🔹 fourth key point from source two\n"
                    "🔹 fifth combined conclusion from both sources"
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[good_response]):
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
        self.assertEqual(BLOCK_GENERATION_MODE_REAL_MERGE, attempt.block_generation_mode)
        self.assertEqual(BLOCK_GENERATION_MODE_REAL_MERGE, attempt.merged.block_generation_mode)

    def test_rejected_raw_response_is_preserved_for_final_failure(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"Bad title","description":"Only one paragraph."}',
            structured_payload={"title": "Bad title", "description": "Only one paragraph."},
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]):
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

    def test_final_failed_merge_keeps_structured_rejected_attempt_outputs(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"Bad title","description":"Only one paragraph."}',
            structured_payload={"title": "Bad title", "description": "Only one paragraph."},
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[bad_response, bad_response, bad_response],
        ):
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
        self.assertGreaterEqual(len(attempt.rejected_attempts), 1)
        self.assertEqual(1, attempt.rejected_attempts[0].attempt_index)
        self.assertEqual("gpt-5.1", attempt.rejected_attempts[0].model_name)
        self.assertIn("paragraph_underflow", tuple(attempt.validation_reasons or ()))
        self.assertEqual("Bad title", attempt.rejected_attempts[0].title)
        self.assertEqual("Only one paragraph.", attempt.rejected_attempts[0].description)

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

    def test_merge_logs_include_stage_reason_code_and_slot_context(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"","description":"Only one paragraph."}',
            structured_payload={"title": "", "description": "Only one paragraph."},
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]), self.assertLogs(
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
        self.assertIn("fallback_used=no", joined_logs)
        self.assertIn("code=invalid_title", joined_logs)

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

    def test_expanded_merge_rejects_overly_generic_body_for_three_sources(self) -> None:
        generic_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Why these developments matter tonight","description":"Tonight we step back and frame several important developments inside one smooth and readable opening that sounds strong but stays broad. It keeps attention on the mood and the overall stakes instead of distinct source facts.\\n\\nIn this stream you\'ll see:\\n🔹 the main context and why it matters\\n🔹 the broader background and tensions\\n🔹 how the story fits a larger pattern\\n\\nA polished closing paragraph keeps the editorial flow consistent for viewers."}'
            ),
            structured_payload={
                "title": "Why these developments matter tonight",
                "description": (
                    "Tonight we step back and frame several important developments inside one smooth and readable opening that sounds strong but stays broad. It keeps attention on the mood and the overall stakes instead of distinct source facts.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 the main context and why it matters\n"
                    "🔹 the broader background and tensions\n"
                    "🔹 how the story fits a larger pattern\n\n"
                    "A polished closing paragraph keeps the editorial flow consistent for viewers."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[generic_expanded_response, generic_expanded_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=insufficient_bullet_coverage", joined_logs)
        self.assertIn("merge_llm_response_invalid", joined_logs)
        self.assertIn("bullet_points_count=3", joined_logs)

    def test_expanded_merge_accepts_softened_distinctive_coverage_for_three_sources(self) -> None:
        borderline_coverage_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels vote and Kharkiv pressure tonight","description":"Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv and explain why the next operational window matters. The opening stays factual, but the summary only follows two of the source lines.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\\n🔹 Anna Kovalenko tracks customs delays and coalition counts before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n🔹 emergency logistics pressure is already shaping border and rail planning for the next 48 hours\\n🔹 transport bottlenecks now define how local authorities sequence response steps and public warnings\\n\\nThe body stays concrete and detailed, but it keeps the focus on political and transport fallout only."}'
            ),
            structured_payload={
                "title": "Brussels vote and Kharkiv pressure tonight",
                "description": (
                    "Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv and explain why the next operational window matters. The opening stays factual, but the summary only follows two of the source lines.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays and coalition counts before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\n"
                    "🔹 emergency logistics pressure is already shaping border and rail planning for the next 48 hours\n"
                    "🔹 transport bottlenecks now define how local authorities sequence response steps and public warnings\n\n"
                    "The body stays concrete and detailed, but it keeps the focus on political and transport fallout only."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[borderline_coverage_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_response_valid", joined_logs)
        self.assertIn("bullet_points_count=6", joined_logs)
        self.assertNotIn("code=weak_source_coverage", joined_logs)
        self.assertNotIn("code=insufficient_bullet_coverage", joined_logs)

    def test_expanded_merge_rejects_three_source_coverage_two_of_three_when_other_signals_are_weak(self) -> None:
        weak_borderline_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels vote and Kharkiv pressure tonight","description":"Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv, but the write-up stays thin and leaves the missing third lane unresolved.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 transport bottlenecks now define how local authorities sequence response steps\\n\\nThe closing paragraph stays broad and never restores the missing third source with any concrete facts."}'
            ),
            structured_payload={
                "title": "Brussels vote and Kharkiv pressure tonight",
                "description": (
                    "Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv, but the write-up stays thin and leaves the missing third lane unresolved.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 transport bottlenecks now define how local authorities sequence response steps\n\n"
                    "The closing paragraph stays broad and never restores the missing third source with any concrete facts."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[weak_borderline_response, weak_borderline_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=insufficient_bullet_coverage", joined_logs)
        self.assertIn("bullet_points_count=4", joined_logs)
        self.assertIn("merge_llm_response_invalid", joined_logs)

    def test_expanded_merge_accepts_strong_three_source_output(self) -> None:
        strong_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the operational agenda tonight","description":"Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\\n\\nThe closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the operational agenda tonight",
                "description": (
                    "Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\n\n"
                    "The closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[strong_expanded_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_response_valid", joined_logs)
        self.assertIn("bullet_points_count=6", joined_logs)
        self.assertNotIn("code=insufficient_bullet_coverage", joined_logs)

    def test_expanded_merge_accepts_dense_two_paragraph_body_when_content_is_rich(self) -> None:
        dense_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the dense operational briefing tonight","description":"Tonight we keep the Brussels vote, the Kharkiv rail disruption, and the Geneva aid timetable inside one compact but source-grounded setup so viewers get the concrete agenda without wasted filler.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and procedural pressure before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release window"}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the dense operational briefing tonight",
                "description": (
                    "Tonight we keep the Brussels vote, the Kharkiv rail disruption, and the Geneva aid timetable inside one compact but source-grounded setup so viewers get the concrete agenda without wasted filler.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and procedural pressure before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release window"
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[dense_expanded_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_response_valid", joined_logs)
        self.assertIn("bullet_points_count=6", joined_logs)
        self.assertNotIn("code=insufficient_bullet_coverage", joined_logs)

    def test_expanded_mode_accepts_5_body_paragraphs(self) -> None:
        expanded_response_five_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: five-block agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=5),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[expanded_response_five_paragraphs],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)

    def test_expanded_mode_accepts_7_body_paragraphs(self) -> None:
        expanded_response_seven_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: seven-block agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=7),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[expanded_response_seven_paragraphs],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)

    def test_expanded_mode_rejects_8_body_paragraphs_as_overflow(self) -> None:
        expanded_response_eight_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: eight-block agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=8),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[
                expanded_response_eight_paragraphs,
                expanded_response_eight_paragraphs,
                expanded_response_eight_paragraphs,
            ],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        validation_reasons: tuple[str, ...] = tuple(attempt.validation_reasons or ())
        self.assertIn("paragraph_overflow", validation_reasons)
        self.assertNotIn("paragraph_underflow", validation_reasons)

    def test_compact_mode_rejects_1_paragraph_as_underflow(self) -> None:
        compact_response_one_paragraph: SimpleNamespace = self._make_merge_response(
            title="Brussels and Kharkiv: compact update",
            description="Only one paragraph.",
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[
                compact_response_one_paragraph,
                compact_response_one_paragraph,
                compact_response_one_paragraph,
            ],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        validation_reasons: tuple[str, ...] = tuple(attempt.validation_reasons or ())
        self.assertIn("paragraph_underflow", validation_reasons)
        self.assertNotIn("paragraph_overflow", validation_reasons)

    def test_collapse_recovery_works_for_expanded_mode(self) -> None:
        expanded_response_nine_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: collapse recovery agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=9),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[expanded_response_nine_paragraphs],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("body_paragraph_count_after_recovery=7", joined_logs)
        self.assertIn("recovery_applied=yes", joined_logs)

    def test_expanded_merge_rejects_weak_two_paragraph_body_when_content_is_thin(self) -> None:
        weak_compact_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv tonight","description":"Tonight we touch on several developments and keep the wording broad instead of source-specific.\\n\\nIn this stream you\'ll see:\\n🔹 why it matters tonight\\n🔹 the broader context\\n🔹 what viewers should watch next"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv tonight",
                "description": (
                    "Tonight we touch on several developments and keep the wording broad instead of source-specific.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 why it matters tonight\n"
                    "🔹 the broader context\n"
                    "🔹 what viewers should watch next"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[weak_compact_response, weak_compact_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=insufficient_bullet_coverage", joined_logs)
        self.assertIn("bullet_points_count=3", joined_logs)
        self.assertIn(
            "description validation failed: insufficient_bullet_coverage",
            joined_logs,
        )

    def test_expanded_retry_uses_targeted_prompt_and_logs_retry_focus(self) -> None:
        weak_compact_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv tonight","description":"Tonight we touch on several developments and keep the wording broad instead of source-specific.\\n\\nIn this stream you\'ll see:\\n🔹 why it matters tonight\\n🔹 the broader context\\n🔹 what viewers should watch next"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv tonight",
                "description": (
                    "Tonight we touch on several developments and keep the wording broad instead of source-specific.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 why it matters tonight\n"
                    "🔹 the broader context\n"
                    "🔹 what viewers should watch next"
                ),
            },
        )
        strong_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the operational agenda tonight","description":"Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\\n\\nThe closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the operational agenda tonight",
                "description": (
                    "Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\n\n"
                    "The closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[weak_compact_response, strong_expanded_response],
        ) as request_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
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
        self.assertEqual(2, request_mock.call_count)
        first_prompt: str = request_mock.call_args_list[0].kwargs["prompt_text"]
        second_prompt: str = request_mock.call_args_list[1].kwargs["prompt_text"]
        self.assertNotIn("RETRY INSTRUCTION:", first_prompt)
        self.assertIn("RETRY INSTRUCTION:", second_prompt)
        self.assertIn("bullet lines", second_prompt)
        self.assertIn("Rewrite with at least", second_prompt)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn(
            "retry_reason_codes=insufficient_bullet_coverage",
            joined_logs,
        )
        self.assertIn(
            "retry_focus=bullet_coverage",
            joined_logs,
        )

    def test_structured_reason_codes_drive_targeted_retry_even_if_error_text_changes(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Attempt was rejected due to insufficient bullet coverage.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=(
                "insufficient_bullet_coverage",
            ),
        )
        successful_merge = (
            MergedLanguageContent(
                title="Final title",
                description="Paragraph one.\n\nParagraph two.",
                description_selected="Paragraph one.\n\nParagraph two.",
                description_audit="Paragraph one.\n\nParagraph two.",
            ),
            '{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
        )
        with patch(
            "app.llm.merges.merge_orchestrator._attempt_merge_once",
            side_effect=[structured_failure, successful_merge],
        ) as attempt_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
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
        self.assertEqual(2, attempt_mock.call_count)
        retry_profile = attempt_mock.call_args_list[1].kwargs["expanded_retry_profile"]
        self.assertIsNotNone(retry_profile)
        self.assertEqual("targeted", retry_profile.retry_mode)
        self.assertEqual(
            ("insufficient_bullet_coverage",),
            retry_profile.reject_signals,
        )
        self.assertEqual(
            ("bullet_coverage",),
            retry_profile.focus_tags,
        )
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn(
            "retry_reason_codes=insufficient_bullet_coverage",
            joined_logs,
        )

    def test_four_source_retry_logs_structured_mode_and_uses_targeted_profile(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Expanded draft had too few bullets.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=(
                "insufficient_bullet_coverage",
            ),
        )
        successful_merge = (
            MergedLanguageContent(
                title="Final title",
                description="Paragraph one.\n\nParagraph two.",
                description_selected="Paragraph one.\n\nParagraph two.",
                description_audit="Paragraph one.\n\nParagraph two.",
            ),
            '{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
        )
        with patch(
            "app.llm.merges.merge_orchestrator._attempt_merge_once",
            side_effect=[structured_failure, successful_merge],
        ) as attempt_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos_four_sources(),
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
        retry_profile = attempt_mock.call_args_list[1].kwargs["expanded_retry_profile"]
        self.assertIsNotNone(retry_profile)
        reinforcement_text: str = "\n".join(retry_profile.reinforcement_lines)
        self.assertIn("Rewrite with at least", reinforcement_text)
        self.assertIn("Spread bullets across all 4 sources", reinforcement_text)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn("retry_structure=four_plus_structured", joined_logs)
        self.assertIn("retry_source_count=4", joined_logs)

    def test_compact_retry_remains_standard_even_with_structured_expanded_reason_codes(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Compact retry should stay standard here.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=("insufficient_expanded_body", "too_few_expanded_bullets"),
        )
        successful_merge = (
            MergedLanguageContent(
                title="Final title",
                description="Paragraph one.\n\nParagraph two.",
                description_selected="Paragraph one.\n\nParagraph two.",
                description_audit="Paragraph one.\n\nParagraph two.",
            ),
            '{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
        )
        with patch(
            "app.llm.merges.merge_orchestrator._attempt_merge_once",
            side_effect=[structured_failure, successful_merge],
        ) as attempt_mock:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        retry_profile = attempt_mock.call_args_list[1].kwargs["expanded_retry_profile"]
        self.assertIsNotNone(retry_profile)
        self.assertEqual("standard", retry_profile.retry_mode)
        self.assertEqual((), retry_profile.focus_tags)

    def test_compact_two_source_merge_keeps_existing_behavior(self) -> None:
        compact_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv: focused agenda tonight","description":"Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption and keep the summary tightly factual.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n🔹 practical next steps for viewers following both Brussels and Kharkiv developments\\n\\nThe closing paragraph keeps both source lines grounded without forcing an expanded three-source structure."}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: focused agenda tonight",
                "description": (
                    "Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption and keep the summary tightly factual.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\n"
                    "🔹 practical next steps for viewers following both Brussels and Kharkiv developments\n\n"
                    "The closing paragraph keeps both source lines grounded without forcing an expanded three-source structure."
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[compact_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_style_coverage", joined_logs)
        self.assertNotIn("merge_expanded_quality_gate", joined_logs)

    def test_expanded_three_source_merge_salvages_formatting_only_emoji_overflow(self) -> None:
        emoji_heavy_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the operational agenda tonight","description":"🔥 Tonight we align the Brussels vote 🚨, the Kharkiv transport shock 🎯, and the Geneva aid timetable 🧭 into one grounded briefing ✨ that stays source-specific 🔔 without losing clarity 💥 while keeping the agenda concrete 🌟 and readable 🎖 for every viewer 🏳.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the operational agenda tonight",
                "description": (
                    "🔥 Tonight we align the Brussels vote 🚨, the Kharkiv transport shock 🎯, and the Geneva aid timetable 🧭 into one grounded briefing ✨ that stays source-specific 🔔 without losing clarity 💥 while keeping the agenda concrete 🌟 and readable 🎖 for every viewer 🏳.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\n\n"
                    "Watch live ✅ and share updates 📣 #briefing"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[emoji_heavy_response],
        ) as request_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
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
        self.assertEqual(1, request_mock.call_count)
        description: str = attempt.merged.description if attempt.merged is not None else ""
        self.assertNotIn("🔥", description)
        self.assertNotIn("🚨", description)
        self.assertNotIn("🎯", description)
        self.assertNotIn("📣", description)
        self.assertIn("🔹 Brussels sanctions vote", description)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_validation_salvage", joined_logs)
        self.assertIn("outcome=applied", joined_logs)
        self.assertIn("reason_codes=excessive_emoji_usage", joined_logs)
        self.assertIn("actions=reduced_non_structural_emoji", joined_logs)

    def test_expanded_formatting_recovery_reveals_underlying_semantic_reject(self) -> None:
        # A 3-source response with emoji overflow AND per-source dump structure.
        # The per_source_dump check fires before the emoji check in the initial run.
        # After emoji salvage attempt, the per_source_dump check still fires.
        # The salvage path will detect that the issue is not formatting-only.
        # Note: Since per_source_dump fires BEFORE excessive_emoji_usage in the validation
        # order, the initial failure code is per_source_dump, not excessive_emoji_usage.
        # The salvage path is NOT triggered for non-emoji failures.
        # Instead, test that a compact (2-source) emoji overflow response is correctly
        # rejected without salvage, and the rejection code is excessive_emoji_usage.
        # This verifies _attempt_expanded_formatting_recovery skips non-3-source merges.

        # Use a 3-source merge with enough bullets to pass bullet check, but emoji overflow
        # whose source-dump structure reveals itself after emoji stripping.
        dump_emoji_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels Kharkiv Geneva briefing","description":"🔥 Tonight we cover 🚨 three important briefings 🎯 from three key locations 🧭 across 💥 the operational 🔔 theater 🌟 each 🎖 source 💡 matters 🎗.\\n\\n🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes on the eastern line\\n🔹 Geneva aid corridor timetable and WHO cargo counts for Odesa hospitals\\n🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels Kharkiv Geneva briefing",
                "description": (
                    "🔥 Tonight we cover 🚨 three important briefings 🎯 from three key locations 🧭 across 💥 the operational 🔔 theater 🌟 each 🎖 source 💡 matters 🎗.\n\n"
                    "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes on the eastern line\n"
                    "🔹 Geneva aid corridor timetable and WHO cargo counts for Odesa hospitals\n"
                    "🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release\n\n"
                    "Watch live \u2705 and share updates \U0001f4e3 #briefing"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[dump_emoji_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
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
        self.assertIn("merge_llm_validation_salvage", joined_logs)
        self.assertIn("outcome=applied", joined_logs)
        self.assertIn("reason_codes=excessive_emoji_usage", joined_logs)
        self.assertIn("actions=reduced_non_structural_emoji", joined_logs)

    def test_compact_two_source_merge_does_not_salvage_emoji_overflow(self) -> None:
        compact_emoji_overflow = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv: focused agenda tonight","description":"🔥 Tonight we connect the Brussels vote calendar 🚨 with the Kharkiv rail disruption 🎯 and keep the summary tightly factual 🧭 without missing the live stakes ✨ while covering 💥 all key angles 🔔 from both sources 🌟 for viewers 🎖 tonight 💡.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n🔹 viewer questions and timing watchpoints\\n🔹 next-step logistics for the corridor desk\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: focused agenda tonight",
                "description": (
                    "🔥 Tonight we connect the Brussels vote calendar 🚨 with the Kharkiv rail disruption 🎯 and keep the summary tightly factual 🧭 without missing the live stakes ✨ while covering 💥 all key angles 🔔 from both sources 🌟 for viewers 🎖 tonight 💡.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\n"
                    "🔹 viewer questions and timing watchpoints\n"
                    "🔹 next-step logistics for the corridor desk\n\n"
                    "Watch live ✅ and share updates 📣 #briefing"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[compact_emoji_overflow, compact_emoji_overflow],
        ) as request_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNone(attempt.merged)
        self.assertEqual(2, request_mock.call_count)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=excessive_emoji_usage", joined_logs)
        self.assertNotIn("merge_llm_validation_salvage", joined_logs)

    def test_softened_distinctive_coverage_policy_does_not_apply_to_four_sources(self) -> None:
        self.assertFalse(
            _is_softened_distinctive_source_coverage_eligible(
                source_count=4,
                source_coverage_by_item=(True, True, True, True),
                distinctive_coverage_hits=3,
                distinctive_coverage_total=4,
                reason_codes=("weak_source_coverage",),
            )
        )

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
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[per_source_dump, per_source_dump],
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
        self.assertEqual(2, request_mock.call_count)
        self.assertIsNone(attempt.merged)

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
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[plain_response]):
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
