from __future__ import annotations

import unittest

from app.publish.post_llm_sanitation import _clean_double_bullet_markers


class DoubleBulletCleanupTests(unittest.TestCase):
    def test_double_bullet_collapsed(self) -> None:
        result: str = _clean_double_bullet_markers("🔹 🔹 Some text")
        self.assertEqual(result, "🔹 Some text")

    def test_mixed_markers_collapsed(self) -> None:
        result: str = _clean_double_bullet_markers("🔹 📌 Some text")
        self.assertEqual(result, "🔹 Some text")

    def test_triple_marker_collapsed(self) -> None:
        result: str = _clean_double_bullet_markers("📌 🔹 🔹 Some text")
        self.assertEqual(result, "📌 Some text")

    def test_single_marker_unchanged(self) -> None:
        result: str = _clean_double_bullet_markers("🔹 Normal bullet")
        self.assertEqual(result, "🔹 Normal bullet")

    def test_non_marker_line_unchanged(self) -> None:
        result: str = _clean_double_bullet_markers("Normal paragraph text")
        self.assertEqual(result, "Normal paragraph text")

    def test_multiline_mixed(self) -> None:
        input_text: str = (
            "🔹 Clean bullet line\n"
            "🔹 🔹 Doubled bullet line\n"
            "Regular prose line\n"
            "📌 🎤 Mixed markers line"
        )
        expected: str = (
            "🔹 Clean bullet line\n"
            "🔹 Doubled bullet line\n"
            "Regular prose line\n"
            "📌 Mixed markers line"
        )
        result: str = _clean_double_bullet_markers(input_text)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
