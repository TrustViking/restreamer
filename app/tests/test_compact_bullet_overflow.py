from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.core.models import MergedLanguageContent
from app.llm.merges.merge_links import OfficialLinksFillResult, OfficialLinksSelection
from app.llm.merges.merge_quality import count_overloaded_bullets
from app.llm.merges.merge_quality import normalize_merge_description
from app.llm.merges.merge_validation import _validate_coverage_preserving_merge_or_raise


class CompactBulletOverflowTests(unittest.TestCase):
    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="gpt-5.1",
                timeout_sec=30.0,
                max_output_tokens=1000,
                pre_delay_sec=0.0,
                source_desc_max_chars=500,
                run_if_single_source=False,
            ),
            templates=SimpleNamespace(
                llm_language_names_json='{"en":"English"}',
                llm_language_names={"en": "English"},
                llm_merge_title_description_prompt=(
                    "Write in {language_name}.\n"
                    "{merge_contract_block}\n\n"
                    "{youtube_candidates_block}\n\n"
                    "{sources_block}"
                ),
                llm_merge_structural_rules="RULE 1: Hook first.\nRULE 2: Keep structure.",
                llm_merge_contracts_json=json.dumps({}, ensure_ascii=False),
                llm_merge_retry_reinforcements_json=json.dumps({}, ensure_ascii=False),
            ),
        )

    def _videos(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Source one",
                    description="Source one paragraph with concrete facts.",
                ),
                normalized_link="https://youtube.com/watch?v=aaaaaaaaaaa",
            ),
            SimpleNamespace(
                metadata=SimpleNamespace(
                    title="Source two",
                    description="Source two paragraph with concrete facts.",
                ),
                normalized_link="https://youtube.com/watch?v=bbbbbbbbbbb",
            ),
        ]

    def test_eight_bullets_with_two_sources_raises_compact_bullet_overflow(self) -> None:
        description_text: str = (
            "Tonight we map concrete outcomes from two linked source agendas and keep each detail grounded.\n\n"
            "In this stream you'll see:\n"
            "🔹 First concrete point from source one with timing and consequence.\n"
            "🔹 Second concrete point from source one with named actors.\n"
            "🔹 Third concrete point from source one with institution context.\n"
            "🔹 Fourth concrete point from source two with operational detail.\n"
            "🔹 Fifth concrete point from source two with location and result.\n"
            "🔹 Sixth concrete point from source two with practical implication.\n"
            "🔹 Seventh concrete point linking both sources with explicit facts.\n"
            "🔹 Eighth concrete point extending the same compact agenda."
        )
        merge_quality = normalize_merge_description(
            description=description_text,
            language="en",
            source_texts=(),
            title="Two-source agenda with too many bullets",
        ).diagnostics
        merged_content = MergedLanguageContent(
            title="Two-source agenda with too many bullets",
            description=description_text,
            description_selected=description_text,
            description_audit=description_text,
        )
        with self.assertRaises(Exception) as captured:
            _validate_coverage_preserving_merge_or_raise(
                merged_content=merged_content,
                videos=self._videos(),
                official_links_selection=OfficialLinksSelection(
                    found_in_sources=0,
                    kept_links=(),
                ),
                official_links_fill=OfficialLinksFillResult(
                    description=description_text,
                    links_in_output=0,
                    fill_applied=False,
                ),
                merge_quality=merge_quality,
            )
        self.assertIn("compact_bullet_overflow", tuple(captured.exception.reason_codes))


    def test_very_long_bullet_is_overloaded_without_names(self) -> None:
        """Bullet > 500 chars is overloaded regardless of named entities."""
        long_bullet: str = "🔹 " + "слово " * 85  # ~510 chars
        description: str = f"Hook paragraph.\n\n{long_bullet}\n🔹 Short bullet."
        self.assertGreaterEqual(count_overloaded_bullets(description), 1)

    def test_medium_bullet_without_names_is_not_overloaded(self) -> None:
        """Bullet 281-500 chars without named entities is NOT overloaded."""
        medium_bullet: str = "🔹 " + "слово " * 50  # ~300 chars
        description: str = f"Hook paragraph.\n\n{medium_bullet}\n🔹 Short."
        self.assertEqual(count_overloaded_bullets(description), 0)


if __name__ == "__main__":
    unittest.main()
