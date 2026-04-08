from __future__ import annotations

from datetime import datetime
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import MergedLanguageContent
from app.publish import doc_helpers, telegram_renderer


class PublishGateBlockingTests(unittest.TestCase):
    def _video(self, *, title: str, description: str) -> SimpleNamespace:
        return SimpleNamespace(
            language="en",
            forced_block_language=None,
            date_display="20.03.2026",
            scheduled_at_kiev=datetime(2026, 3, 20, 18, 0, 0),
            metadata=SimpleNamespace(
                title=title,
                description=description,
                url="https://youtube.com/watch?v=abcdefghijk",
            ),
            normalized_link="https://youtube.com/watch?v=abcdefghijk",
            original_link="https://youtube.com/watch?v=abcdefghijk",
        )

    def _merged_content(self) -> MergedLanguageContent:
        return MergedLanguageContent(
            title="Merged title",
            description="Merged description",
            title_selected="Merged title",
            description_selected="Merged description",
            title_audit="Merged title",
            description_audit="Merged description",
        )

    def test_doc_falls_back_when_publish_stage_duplicate_is_true(self) -> None:
        videos: list[SimpleNamespace] = [
            self._video(title="Source title", description="Source description one."),
        ]
        blocked_payload: SimpleNamespace = SimpleNamespace(
            title_text="Merged title",
            description_text="Merged description",
            block_generation_mode="real_merge",
            has_publish_stage_duplicate=True,
            has_publish_stage_opener_cta=False,
        )
        with patch(
            "app.publish.doc_helpers.build_sanitized_merged_publication_payload",
            return_value=blocked_payload,
        ), self.assertLogs(level="WARNING") as captured:
            description_text: str = doc_helpers._build_descriptions_summary(
                videos=videos,
                templates=None,
                merged_content=self._merged_content(),
                merge_attempt=None,
            )
        self.assertEqual("Source description one.", description_text)
        self.assertNotIn("Merged description", description_text)
        self.assertIn("merge_publish_gate_blocked target=doc", "\n".join(captured.output))

    def test_telegram_falls_back_to_nomerge_when_publish_stage_opener_cta_is_true(self) -> None:
        videos: list[SimpleNamespace] = [
            self._video(title="Source title", description="Source description one."),
        ]
        blocked_payload: SimpleNamespace = SimpleNamespace(
            title_text="Merged title",
            description_text="Merged description",
            block_generation_mode="real_merge",
            has_publish_stage_duplicate=False,
            has_publish_stage_opener_cta=True,
        )
        config: SimpleNamespace = SimpleNamespace(
            telegram=SimpleNamespace(
                use_audit=True,
                symbol_pin="📌",
                flag_repeat_count=1,
            ),
            templates=SimpleNamespace(
                telegram_language_merged_block="{title}\n{description}",
            ),
        )
        with patch(
            "app.publish.telegram_renderer.build_sanitized_merged_publication_payload",
            return_value=blocked_payload,
        ), self.assertLogs(level="WARNING") as captured:
            rendered_block: str = telegram_renderer.build_telegram_language_merged_block(
                language="en",
                videos=videos,
                merged_content=self._merged_content(),
                merge_attempt=None,
                config=config,
                templates=None,
            )
        self.assertIn("Source title", rendered_block)
        self.assertIn("Source description one.", rendered_block)
        self.assertNotIn("Merged title", rendered_block)
        self.assertNotIn("Merged description", rendered_block)
        self.assertIn("merge_publish_gate_blocked target=telegram", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
