"""Regression tests for closing CTA removal from final published text."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.publish.post_llm_sanitation import sanitize_post_llm_text


class NoCtaInPublishedOutputTests(unittest.TestCase):
    def test_english_watch_cta_dropped(self) -> None:
        text: str = (
            "A judge ruled that expert opinions are inadmissible evidence.\n\n"
            "⚖ The court rejected the witness statement.\n"
            "🔹 Defendant has appealed the verdict.\n\n"
            "Watch the full stream and share your thoughts.\n\n"
            "#Justice #Court"
        )
        result = sanitize_post_llm_text(text, language="en", source_label="merge")
        self.assertNotIn("Watch the full", result.full_text)
        self.assertIn("#Justice", result.full_text)
        self.assertIn("court rejected", result.full_text.lower())

    def test_ukrainian_cta_dropped_via_body_stripper(self) -> None:
        text: str = (
            "Суд відкинув «експертні висновки» як неналежні докази.\n\n"
            "⚖ Резонансна судова справа.\n"
            "🔹 Аналіз: чому «експертні висновки» не відповідають вимогам.\n\n"
            "Дивіться ефір і діліться думками.\n\n"
            "#АЛЛАТРА #Суд"
        )
        with (
            patch(
                "app.publish.sanitizers.tail_parser._looks_like_cta_line_shared",
                return_value=False,
            ),
            patch("app.publish.sanitizers.tail_parser.CTA_HINTS", ()),
        ):
            result = sanitize_post_llm_text(text, language="uk", source_label="merge")
        self.assertEqual(result.cta_text, "")
        self.assertNotIn("Дивіться ефір", result.full_text)
        self.assertIn("#АЛЛАТРА", result.full_text)
        self.assertIn("експертні висновки", result.full_text)

    def test_bullet_block_with_cta_hint_word_is_preserved(self) -> None:
        text: str = (
            "A judge ruled that expert opinions are inadmissible evidence.\n\n"
            "🔹 Watch how the court rejected the witnesses statement.\n"
            "🔹 Legal timeline and appeal context.\n\n"
            "#Justice #Court"
        )
        result = sanitize_post_llm_text(text, language="en", source_label="merge")
        self.assertIn("Watch how the court", result.full_text)
        self.assertIn("Legal timeline", result.full_text)
        self.assertIn("#Justice", result.full_text)


if __name__ == "__main__":
    unittest.main()
