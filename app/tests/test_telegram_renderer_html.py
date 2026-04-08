from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
import unittest

from app.publish.telegram_renderer import (
    build_telegram_header_text,
    build_telegram_key_form_reminder,
    build_telegram_language_block,
    telegram_safe_html,
)


class TelegramRendererHtmlTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            telegram=SimpleNamespace(
                symbol_broadcast="❇️",
                symbol_alert="🚨",
                symbol_form="❇️",
                symbol_description="❇️",
                symbol_pin="📌",
                flag_repeat_count=1,
            ),
            templates=SimpleNamespace(
                telegram_header="{form_url}|{contacts}|{generated_doc_url}",
                telegram_key_form_reminder="{form_url}",
                telegram_language_block="{title}\n{description}",
            ),
        )

    def test_telegram_safe_html_escapes_and_converts_bold_markers(self) -> None:
        self.assertEqual(
            "A &amp; B &lt; C &gt; D <b>bold</b>",
            telegram_safe_html("A & B < C > D *bold*"),
        )

    def test_language_block_escapes_html_and_applies_bold_markers(self) -> None:
        config = self._config()
        video = SimpleNamespace(
            language="en",
            forced_block_language=None,
            date_display="13.04.2026",
            scheduled_at_kiev=datetime(2026, 4, 13, 14, 0, 0),
            metadata=SimpleNamespace(
                title="Title <x> *Bold*",
                description="Desc > one & two *strong*",
            ),
        )
        rendered: str = build_telegram_language_block(
            video=video,
            config=config,
            templates=None,
        )
        self.assertIn("Title &lt;x&gt; <b>Bold</b>", rendered)
        self.assertIn("Desc &gt; one &amp; two <b>strong</b>", rendered)

    def test_header_and_key_form_escape_urls_and_contacts(self) -> None:
        config = self._config()
        context = {
            "time_cet": "13:00",
            "time_kiev": "14:00",
            "time_gmt": "11:00",
            "date": "13.04.2026",
            "form_url": "https://example.com/form?a=1&b=2",
            "contacts": "@a & @b <team>",
        }
        header: str = build_telegram_header_text(
            context=context,
            generated_doc_url="https://example.com/doc?a=1&b=2",
            config=config,
        )
        key_form: str = build_telegram_key_form_reminder(context=context, config=config)
        self.assertIn("https://example.com/form?a=1&amp;b=2", header)
        self.assertIn("@a &amp; @b &lt;team&gt;", header)
        self.assertIn("https://example.com/doc?a=1&amp;b=2", header)
        self.assertEqual("https://example.com/form?a=1&amp;b=2", key_form)


if __name__ == "__main__":
    unittest.main()

