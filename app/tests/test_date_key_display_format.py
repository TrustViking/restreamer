from __future__ import annotations

import unittest

from app.core.text_utils import format_date_key_for_display


class TestFormatDateKeyForDisplay(unittest.TestCase):

    def test_standard_date_key(self) -> None:
        self.assertEqual(format_date_key_for_display("270426"), "27.04.2026")

    def test_single_digit_day(self) -> None:
        self.assertEqual(format_date_key_for_display("010125"), "01.01.2025")

    def test_end_of_year(self) -> None:
        self.assertEqual(format_date_key_for_display("311226"), "31.12.2026")

    def test_empty_string(self) -> None:
        self.assertEqual(format_date_key_for_display(""), "")

    def test_invalid_format_returns_as_is(self) -> None:
        self.assertEqual(format_date_key_for_display("not-a-date"), "not-a-date")

    def test_already_formatted_returns_as_is(self) -> None:
        self.assertEqual(format_date_key_for_display("27.04.2026"), "27.04.2026")

    def test_whitespace_stripped(self) -> None:
        self.assertEqual(format_date_key_for_display("  270426  "), "27.04.2026")


if __name__ == "__main__":
    unittest.main()
