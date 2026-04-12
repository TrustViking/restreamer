from __future__ import annotations

import unittest

from app.publish.inline_formatting import (
    has_bold_markers,
    parse_inline_bold,
    strip_bold_markers,
)


class ParseInlineBoldTests(unittest.TestCase):
    def test_full_line_bold(self) -> None:
        result = parse_inline_bold("*СПІКЕРИ КОНФЕРЕНЦІЇ:*")
        self.assertEqual([("СПІКЕРИ КОНФЕРЕНЦІЇ:", True)], result)

    def test_inline_bold_in_middle(self) -> None:
        result = parse_inline_bold(
            "Производство: *ALLATRA Global Research Center (ALLATRA GRC)* — глобальный"
        )
        self.assertEqual(
            [
                ("Производство: ", False),
                ("ALLATRA Global Research Center (ALLATRA GRC)", True),
                (" — глобальный", False),
            ],
            result,
        )

    def test_no_markers(self) -> None:
        result = parse_inline_bold("Plain text without markers")
        self.assertEqual([("Plain text without markers", False)], result)

    def test_multiple_bold_segments(self) -> None:
        result = parse_inline_bold("A *B* C *D* E")
        self.assertEqual(
            [("A ", False), ("B", True), (" C ", False), ("D", True), (" E", False)],
            result,
        )

    def test_empty_string(self) -> None:
        result = parse_inline_bold("")
        self.assertEqual([("", False)], result)


class StripBoldMarkersTests(unittest.TestCase):
    def test_strip_heading(self) -> None:
        self.assertEqual(
            "СПІКЕРИ КОНФЕРЕНЦІЇ:",
            strip_bold_markers("*СПІКЕРИ КОНФЕРЕНЦІЇ:*"),
        )

    def test_strip_inline(self) -> None:
        self.assertEqual(
            "Производство: ALLATRA GRC — глобальный",
            strip_bold_markers("Производство: *ALLATRA GRC* — глобальный"),
        )

    def test_no_markers_unchanged(self) -> None:
        self.assertEqual("Plain text", strip_bold_markers("Plain text"))


class HasBoldMarkersTests(unittest.TestCase):
    def test_has_markers(self) -> None:
        self.assertTrue(has_bold_markers("*bold*"))

    def test_no_markers(self) -> None:
        self.assertFalse(has_bold_markers("no bold"))


if __name__ == "__main__":
    unittest.main()
