from __future__ import annotations

import unittest

from app.publish.post_llm_sanitation import sanitize_post_llm_text


class PostLlmSanitationTailTests(unittest.TestCase):
    def test_embedded_hashtags_are_split_from_cta_and_separated_by_blank_line(self) -> None:
        text: str = (
            "Scientists compare nanoplastics data across multiple studies.\n\n"
            "Join us tonight and share if you find these scientific findings important. "
            "#nanoplastics #microplastics"
        )
        with self.assertLogs(level="INFO") as captured:
            result = sanitize_post_llm_text(
                text,
                language="en",
                source_label="merge",
            )
        self.assertEqual(
            "Scientists compare nanoplastics data across multiple studies.\n\n"
            "#nanoplastics #microplastics",
            result.full_text,
        )
        self.assertEqual(
            "Join us tonight and share if you find these scientific findings important.",
            result.cta_text,
        )
        self.assertEqual("#nanoplastics #microplastics", result.hashtags_line)
        self.assertTrue(result.hashtags_split_from_cta)
        logs: str = "\n".join(captured.output)
        self.assertIn(
            "tail_parse lang=en source=merge cta_found=yes hashtags_found=yes hashtags_split_from_cta=yes",
            logs,
        )
        self.assertIn(
            "tail_layout lang=en source=merge layout=body_blank_cta_blank_hashtags",
            logs,
        )

    def test_separate_hashtags_line_stays_stable(self) -> None:
        text: str = (
            "Scientists compare nanoplastics data across multiple studies.\n\n"
            "Join us tonight for the full discussion.\n\n"
            "#nanoplastics #microplastics"
        )
        result = sanitize_post_llm_text(text, language="en", source_label="merge")
        self.assertEqual(
            "Scientists compare nanoplastics data across multiple studies.\n\n"
            "#nanoplastics #microplastics",
            result.full_text,
        )
        self.assertFalse(result.hashtags_split_from_cta)
        self.assertEqual("body_blank_cta_blank_hashtags", result.tail_layout)

    def test_hashtags_without_cta_keep_single_blank_line_from_body(self) -> None:
        text: str = (
            "Scientists compare nanoplastics data across multiple studies.\n\n"
            "#nanoplastics #microplastics"
        )
        result = sanitize_post_llm_text(text, language="en", source_label="merge")
        self.assertEqual(text, result.full_text)
        self.assertEqual("", result.cta_text)
        self.assertEqual("body_blank_hashtags", result.tail_layout)

    def test_text_without_hashtags_does_not_gain_extra_blank_lines(self) -> None:
        text: str = (
            "Scientists compare nanoplastics data across multiple studies.\n\n"
            "Join us tonight for the full discussion."
        )
        result = sanitize_post_llm_text(text, language="en", source_label="merge")
        self.assertEqual(
            "Scientists compare nanoplastics data across multiple studies.",
            result.full_text,
        )
        self.assertEqual("", result.hashtags_line)
        self.assertEqual("body_blank_cta", result.tail_layout)

    def test_comment_cta_dropped_from_final_output(self) -> None:
        """Comment-CTA paragraph must be stripped from final compose output."""
        text: str = (
            "Свидетельства жертв звучат на фоне следствия.\n\n"
            "⚖ Дело открыто и находится на стадии следствия.\n"
            "🔹 Встреча с послом: правительство ознакомлено.\n\n"
            "Напишите в комментариях, какие вопросы вы считаете ключевыми.\n\n"
            "#Танзания #ЗащитаДетей"
        )
        result = sanitize_post_llm_text(text, language="ru", source_label="merge")
        self.assertNotIn("Напишите в комментариях", result.full_text)
        self.assertIn("#Танзания", result.full_text)
        self.assertIn("Свидетельства жертв", result.full_text)


if __name__ == "__main__":
    unittest.main()
