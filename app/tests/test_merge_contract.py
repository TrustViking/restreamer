from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    BLOCK_GENERATION_MODE_REAL_MERGE,
    MergedLanguageContent,
)
from app.llm.merge_quality import normalize_merge_description
from app.llm.model_compatibility import LlmModelConfigurationError
from app.llm.merge_parser import (
    parse_merge_response_or_raise,
    sanitize_title,
    separate_merge_body_and_tail,
)
from app.llm.merge_service import (
    MergeAttemptFailure,
    _build_expanded_retry_profile,
    _is_softened_distinctive_source_coverage_eligible,
    _reason_code_from_error,
    _reason_codes_from_error,
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
                    'Paragraph six.\\n\\nParagraph seven.\\n\\n#topic"}'
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
        self.assertIn("Then write 4 to 7 short thesis bullet lines.", prompt_text)
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
        self.assertIn("Write 6 to 9 short bullet lines total.", prompt_text)
        self.assertIn("2 to 3 thematic micro-blocks", prompt_text)
        self.assertIn(
            "Do not combine science or medicine, climate or environment, disasters or catastrophic hazards, psychology or cognition or behavior, and broad social or moral conclusions into one bullet",
            prompt_text,
        )
        self.assertIn("Thematic grouping is encouraged when useful", prompt_text)
        self.assertIn(
            "merge_prompt_contract_selected language=en source_count=3 contract_mode=expanded expected_bullet_range=6-9 expanded_structure_enabled=yes",
            joined_logs,
        )

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
            "app.llm.merge_service._extract_description_validation_reason_codes"
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
            "app.llm.merge_service._extract_description_validation_reason_codes",
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
            "app.llm.merge_service.openai_request_merge",
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
            raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
            structured_payload={
                "title": "Final title",
                "description": "Paragraph one.\n\nParagraph two.",
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[good_response]):
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

    def test_final_failed_merge_keeps_structured_rejected_attempt_outputs(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"Bad title","description":"Only one paragraph."}',
            structured_payload={"title": "Bad title", "description": "Only one paragraph."},
        )
        with patch(
            "app.llm.merge_service.openai_request_merge",
            side_effect=[bad_response, bad_response],
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
        self.assertEqual(2, len(attempt.rejected_attempts))
        self.assertEqual(1, attempt.rejected_attempts[0].attempt_index)
        self.assertEqual("gpt-5.1", attempt.rejected_attempts[0].model_name)
        self.assertEqual(("invalid_description",), tuple(attempt.validation_reasons or ()))
        self.assertEqual("Bad title", attempt.rejected_attempts[0].title)
        self.assertEqual("Only one paragraph.", attempt.rejected_attempts[0].description)

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
            "app.llm.merge_service.openai_request_merge",
            side_effect=[generic_response, generic_response],
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
            "app.llm.merge_service.openai_request_merge",
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
        self.assertIn("code=too_few_expanded_bullets", joined_logs)
        self.assertIn("merge_expanded_quality_gate", joined_logs)
        self.assertIn("quality_gate_reason_codes=too_few_expanded_bullets", joined_logs)

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
            "app.llm.merge_service.openai_request_merge",
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
        self.assertIn("distinctive_source_coverage=2/3", joined_logs)
        self.assertIn("softened_distinctive_source_coverage_applied=yes", joined_logs)
        self.assertIn("quality_gate_status=pass", joined_logs)
        self.assertIn("quality_gate_reason_codes=none", joined_logs)
        self.assertNotIn("code=weak_source_coverage", joined_logs)

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
            "app.llm.merge_service.openai_request_merge",
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
        self.assertIn("distinctive_source_coverage=2/3", joined_logs)
        self.assertIn("softened_distinctive_source_coverage_applied=no", joined_logs)
        self.assertIn("code=too_few_expanded_bullets", joined_logs)
        self.assertIn(
            "quality_gate_reason_codes=too_few_expanded_bullets,weak_source_coverage",
            joined_logs,
        )

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
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[strong_expanded_response]), self.assertLogs(
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
        self.assertIn("merge_expanded_quality_gate", joined_logs)
        self.assertIn("quality_gate_status=pass", joined_logs)
        self.assertIn("distinctive_source_coverage=3/3", joined_logs)

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
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[dense_expanded_response]), self.assertLogs(
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
        self.assertIn("body_paragraph_count=2", joined_logs)
        self.assertIn("compact_body_relaxed=yes", joined_logs)
        self.assertIn("quality_gate_status=pass", joined_logs)

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
            "app.llm.merge_service.openai_request_merge",
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
        self.assertIn("body_paragraph_count=2", joined_logs)
        self.assertIn("compact_body_relaxed=no", joined_logs)
        self.assertIn(
            "description validation failed: too_few_expanded_bullets,insufficient_expanded_body,overly_generic_body,weak_source_coverage",
            joined_logs,
        )
        self.assertIn(
            "quality_gate_reason_codes=too_few_expanded_bullets,insufficient_expanded_body,overly_generic_body,weak_source_coverage",
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
            "app.llm.merge_service.openai_request_merge",
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
        self.assertNotIn("EXPANDED RETRY FOCUS", first_prompt)
        self.assertIn("EXPANDED RETRY FOCUS", second_prompt)
        self.assertIn("post-hook body clearly denser", second_prompt)
        self.assertIn("enough distinct, meaningful bullets", second_prompt)
        self.assertIn("source-grounded specifics", second_prompt)
        self.assertIn("Restore distinguishable spread across source lines or topic nodes", second_prompt)
        self.assertIn("2 to 3 short agenda tracks", second_prompt)
        self.assertIn("cut generic filler bridges", second_prompt)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn(
            "retry_reason_codes=too_few_expanded_bullets,insufficient_expanded_body,overly_generic_body,weak_source_coverage",
            joined_logs,
        )
        self.assertIn(
            "retry_focus=body_depth,bullet_sufficiency,source_specificity,source_spread",
            joined_logs,
        )

    def test_structured_reason_codes_drive_targeted_retry_even_if_error_text_changes(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Expanded draft stayed readable but needs another pass.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=(
                "insufficient_expanded_body",
                "too_few_expanded_bullets",
                "overly_generic_body",
                "weak_source_coverage",
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
            "app.llm.merge_service._attempt_merge_once",
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
            (
                "insufficient_expanded_body",
                "too_few_expanded_bullets",
                "overly_generic_body",
                "weak_source_coverage",
            ),
            retry_profile.reject_signals,
        )
        self.assertEqual(
            (
                "body_depth",
                "bullet_sufficiency",
                "source_specificity",
                "source_spread",
            ),
            retry_profile.focus_tags,
        )
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn(
            "retry_reason_codes=insufficient_expanded_body,too_few_expanded_bullets,overly_generic_body,weak_source_coverage",
            joined_logs,
        )

    def test_four_source_retry_logs_structured_mode_and_uses_targeted_profile(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Expanded draft stayed generic and thin.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=(
                "insufficient_expanded_body",
                "too_few_expanded_bullets",
                "overly_generic_body",
                "weak_source_coverage",
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
            "app.llm.merge_service._attempt_merge_once",
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
        self.assertIn("meaningful thematic micro-blocks", reinforcement_text)
        self.assertIn("recognizable trace in the body", reinforcement_text)
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
            "app.llm.merge_service._attempt_merge_once",
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
                '{"title":"Brussels and Kharkiv: focused agenda tonight","description":"Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption and keep the summary tightly factual.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n\\nThe closing paragraph keeps both source lines grounded without forcing an expanded three-source structure."}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: focused agenda tonight",
                "description": (
                    "Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption and keep the summary tightly factual.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\n\n"
                    "The closing paragraph keeps both source lines grounded without forcing an expanded three-source structure."
                ),
            },
        )
        with patch("app.llm.merge_service.openai_request_merge", side_effect=[compact_response]), self.assertLogs(
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
                '{"title":"Brussels, Kharkiv, Geneva: the operational agenda tonight","description":"🔥 Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing 🚨 that stays source-specific 🎯 without losing clarity.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the operational agenda tonight",
                "description": (
                    "🔥 Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing 🚨 that stays source-specific 🎯 without losing clarity.\n\n"
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
            "app.llm.merge_service.openai_request_merge",
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
        weak_emoji_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv tonight","description":"🔥 Tonight we touch on several developments and keep the wording broad instead of source-specific 🚨 while the editorial frame stays polished 🎯 but generic 🧭 ✨.\\n\\nIn this stream you\'ll see:\\n🔹 why it matters tonight\\n🔹 the broader context\\n🔹 what viewers should watch next\\n\\nWatch live ✅ and share updates 📣 🔔 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv tonight",
                "description": (
                    "🔥 Tonight we touch on several developments and keep the wording broad instead of source-specific 🚨 while the editorial frame stays polished 🎯 but generic 🧭 ✨.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 why it matters tonight\n"
                    "🔹 the broader context\n"
                    "🔹 what viewers should watch next\n\n"
                    "Watch live ✅ and share updates 📣 🔔 #briefing"
                ),
            },
        )
        with patch(
            "app.llm.merge_service.openai_request_merge",
            side_effect=[weak_emoji_response, weak_emoji_response],
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
        self.assertIsNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_validation_salvage", joined_logs)
        self.assertIn("outcome=revealed_non_formatting_issue", joined_logs)
        self.assertIn(
            "replacement_reason_codes=too_few_expanded_bullets,insufficient_expanded_body,overly_generic_body,weak_source_coverage,hook_dominates_body",
            joined_logs,
        )
        self.assertIn("code=too_few_expanded_bullets", joined_logs)

    def test_compact_two_source_merge_does_not_salvage_emoji_overflow(self) -> None:
        compact_emoji_overflow = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv: focused agenda tonight","description":"🔥 Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption 🚨 and keep the summary tightly factual 🎯 without missing the live stakes 🧭.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n🔹 viewer questions and timing watchpoints\\n🔹 next-step logistics for the corridor desk\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: focused agenda tonight",
                "description": (
                    "🔥 Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption 🚨 and keep the summary tightly factual 🎯 without missing the live stakes 🧭.\n\n"
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
            "app.llm.merge_service.openai_request_merge",
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
            "app.llm.merge_service.openai_request_merge",
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
        self.assertIn("named_entities_metric=informational", joined_logs)
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
