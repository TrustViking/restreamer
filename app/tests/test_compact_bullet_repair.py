from __future__ import annotations

import unittest

from app.llm.merges.merge_quality import normalize_merge_description
from app.llm.merges.quality_service_lines import is_bullet_line


class CompactBulletRepairTests(unittest.TestCase):
    def _description(self, bullet_count: int, *, lead_in: bool = False) -> tuple[str, list[str]]:
        bullet_texts: list[str] = [
            f"Point {index} stays in original order with concrete detail."
            for index in range(1, bullet_count + 1)
        ]
        lines: list[str] = [
            "Tonight we map concrete outcomes from linked source agendas.",
            "",
        ]
        if lead_in:
            lines.append("In this stream you'll see:")
        lines.extend(f"🔹 {text}" for text in bullet_texts)
        return ("\n".join(lines), bullet_texts)

    def _output_bullet_lines(self, description: str) -> list[str]:
        return [
            line.strip()
            for line in str(description or "").splitlines()
            if is_bullet_line(line)
        ]

    def test_eight_bullets_two_sources_trimmed_to_seven(self) -> None:
        description, bullet_texts = self._description(8)
        result = normalize_merge_description(
            description=description,
            language="en",
            source_texts=(),
            title="",
            source_count=2,
        )

        self.assertEqual(7, len(self._output_bullet_lines(result.description_text)))
        for text in bullet_texts[:7]:
            self.assertIn(text, result.description_text)
        self.assertNotIn(bullet_texts[7], result.description_text)
        self.assertTrue(result.normalization_applied)

    def test_seven_bullets_two_sources_unchanged(self) -> None:
        description, bullet_texts = self._description(7)
        result = normalize_merge_description(
            description=description,
            language="en",
            source_texts=(),
            title="",
            source_count=2,
        )

        self.assertEqual(7, len(self._output_bullet_lines(result.description_text)))
        for text in bullet_texts:
            self.assertIn(text, result.description_text)
        self.assertFalse(result.normalization_applied)

    def test_eight_bullets_three_sources_unchanged(self) -> None:
        description, bullet_texts = self._description(8)
        result = normalize_merge_description(
            description=description,
            language="en",
            source_texts=(),
            title="",
            source_count=3,
        )

        self.assertEqual(8, len(self._output_bullet_lines(result.description_text)))
        for text in bullet_texts:
            self.assertIn(text, result.description_text)

    def test_source_count_zero_does_not_trim(self) -> None:
        description, bullet_texts = self._description(8)
        result = normalize_merge_description(
            description=description,
            language="en",
            source_texts=(),
            title="",
        )

        self.assertEqual(8, len(self._output_bullet_lines(result.description_text)))
        for text in bullet_texts:
            self.assertIn(text, result.description_text)

    def test_trim_preserves_lead_in_line(self) -> None:
        description, bullet_texts = self._description(8, lead_in=True)
        result = normalize_merge_description(
            description=description,
            language="en",
            source_texts=(),
            title="",
            source_count=2,
        )

        self.assertIn("In this stream you'll see:", result.description_text)
        self.assertEqual(7, len(self._output_bullet_lines(result.description_text)))
        for text in bullet_texts[:7]:
            self.assertIn(text, result.description_text)
        self.assertNotIn(bullet_texts[7], result.description_text)


if __name__ == "__main__":
    unittest.main()
