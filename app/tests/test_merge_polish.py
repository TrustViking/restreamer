from __future__ import annotations

import unittest

from app.llm.merge_polish import (
    MergePolishResult,
    build_merge_polish_prompt,
    bullet_marker_count,
    infer_polish_reject_reason,
    is_too_aggressive_rewrite,
    paragraph_count,
)


class MergePolishTests(unittest.TestCase):
    def test_build_merge_polish_prompt_preserves_current_single_stage_contract(self) -> None:
        prompt_text: str = build_merge_polish_prompt(
            language="en",
            title_text="Final title",
            description_text="Paragraph one.\n\nParagraph two.",
        )
        self.assertIn("lightly polish phrasing and flow", prompt_text)
        self.assertIn("Keep the same number of paragraphs", prompt_text)
        self.assertIn('TITLE: "Final title"', prompt_text)
        self.assertIn("DESCRIPTION:\nParagraph one.\n\nParagraph two.", prompt_text)

    def test_paragraph_count_counts_blocks_only(self) -> None:
        self.assertEqual(3, paragraph_count("One.\n\nTwo.\n\nThree."))

    def test_bullet_marker_count_counts_allowed_markers(self) -> None:
        self.assertEqual(
            3,
            bullet_marker_count("Lead.\n🔹 first\n📌 second\nnot a bullet\n✅ third"),
        )

    def test_infer_polish_reject_reason_maps_current_validation_failures(self) -> None:
        self.assertEqual(
            "formatting_degraded",
            infer_polish_reject_reason("description body paragraph count must be between 2 and 4"),
        )
        self.assertEqual(
            "facts_lost",
            infer_polish_reject_reason("description validation failed: semantic_source_grounding_too_low"),
        )
        self.assertEqual(
            "structure_changed",
            infer_polish_reject_reason("structure_changed"),
        )

    def test_is_too_aggressive_rewrite_is_false_for_close_rewording(self) -> None:
        self.assertFalse(
            is_too_aggressive_rewrite(
                original_text="Title\nParagraph one.\n\nParagraph two.",
                polished_text="Title\nParagraph one refined.\n\nParagraph two.",
            )
        )

    def test_merge_polish_result_dataclass_keeps_explicit_fields(self) -> None:
        result = MergePolishResult(
            source_model="gpt-5.1",
            polish_model="gpt-5.1",
            original_text="Original",
            polished_text="Polished",
            accepted=True,
            reject_reason=None,
            validation_passed=True,
        )
        self.assertTrue(result.accepted)
        self.assertTrue(result.validation_passed)
        self.assertIsNone(result.reject_reason)


if __name__ == "__main__":
    unittest.main()
