from __future__ import annotations

import unittest

from app.core.text_utils import utf16_len
from app.publish.doc_header import DailyDocHeader, HeaderLine, TextStyleSpan


class DailyDocHeaderTests(unittest.TestCase):
    def test_render_text_preserves_order_and_blank_lines(self) -> None:
        header: DailyDocHeader = DailyDocHeader(
            lines=(
                HeaderLine(text="Ежедневные стримы / Everyday streams", is_bold=True),
                HeaderLine(text="10:00 CET/CEST (11:00 Kiev, 09:00 GMT)", is_bold=False),
                HeaderLine(text="", is_bold=False),
                HeaderLine(
                    text="DE - 20:00 ⚠ [merge failed — source list]",
                    is_bold=True,
                ),
                HeaderLine(text="Merged title line", is_bold=False),
            )
        )

        expected_text: str = (
            "Ежедневные стримы / Everyday streams\n"
            "10:00 CET/CEST (11:00 Kiev, 09:00 GMT)\n"
            "\n"
            "DE - 20:00 ⚠ [merge failed — source list]\n"
            "Merged title line"
        )
        self.assertEqual(expected_text, header.render_text())

    def test_build_relative_style_spans_marks_only_bold_non_empty_lines(self) -> None:
        warning_heading: str = "DE - 20:00 ⚠ [merge failed — source list]"
        header: DailyDocHeader = DailyDocHeader(
            lines=(
                HeaderLine(text="Main title", is_bold=True),
                HeaderLine(text="Regular subtitle", is_bold=False),
                HeaderLine(text="", is_bold=True),
                HeaderLine(text=warning_heading, is_bold=True),
                HeaderLine(text="Body line", is_bold=False),
            )
        )

        main_title_span: TextStyleSpan = TextStyleSpan(
            start=0,
            end=utf16_len("Main title"),
            is_bold=True,
        )
        warning_start: int = utf16_len("Main title\nRegular subtitle\n\n")
        warning_span: TextStyleSpan = TextStyleSpan(
            start=warning_start,
            end=warning_start + utf16_len(warning_heading),
            is_bold=True,
        )

        self.assertEqual(
            (main_title_span, warning_span),
            header.build_relative_style_spans(),
        )

    def test_build_relative_style_spans_uses_utf16_for_emoji(self) -> None:
        bold_emoji_line: str = "🚀 Launch 😀"
        header: DailyDocHeader = DailyDocHeader(
            lines=(
                HeaderLine(text=bold_emoji_line, is_bold=True),
                HeaderLine(text="plain", is_bold=False),
            )
        )

        spans: tuple[TextStyleSpan, ...] = header.build_relative_style_spans()
        self.assertEqual(1, len(spans))
        self.assertEqual(0, spans[0].start)
        self.assertEqual(utf16_len(bold_emoji_line), spans[0].end)
        self.assertGreater(utf16_len(bold_emoji_line), len(bold_emoji_line))


if __name__ == "__main__":
    unittest.main()
