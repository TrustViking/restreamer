from __future__ import annotations

import unittest

from app.llm.merges.merge_parser import (
    parse_merge_response_or_raise,
    sanitize_title,
    separate_merge_body_and_tail,
)


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
                '#stream #topic"}'
            ),
        )
        self.assertEqual(4, paragraph_count)
        self.assertEqual(
            7,
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
                '#topic"}'
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
                "#topic #update"
            )
        )
        self.assertEqual(2, separation.body_paragraph_count_after_recovery)
        self.assertEqual(
            ("youtube_links", "official_links", "hashtags"),
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
                '#conference #initiative"}'
            ),
        )
        self.assertEqual(4, paragraph_count)
        separation = separate_merge_body_and_tail(text=merged_content.description)
        self.assertEqual(8, separation.raw_paragraph_count)
        self.assertEqual(4, separation.body_paragraph_count_after_recovery)
        self.assertEqual(
            ("youtube_links", "youtube_links", "official_links", "hashtags"),
            separation.tail_blocks,
        )


if __name__ == "__main__":
    unittest.main()
