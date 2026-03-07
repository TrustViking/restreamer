from __future__ import annotations

import unittest

from app.core.language import detect_language, detect_language_decision, normalize_language
from app.core.models import VideoMetadata


class LanguagePolicyTests(unittest.TestCase):
    def _metadata(
        self,
        *,
        title: str,
        description: str,
        youtube_language: str | None = None,
        channel_language: str | None = None,
    ) -> VideoMetadata:
        return VideoMetadata(
            url="https://youtu.be/aaaaaaaaaaa",
            title=title,
            description=description,
            thumbnail_url="https://i.ytimg.com/vi/aaaaaaaaaaa/hqdefault.jpg",
            youtube_language=youtube_language,
            channel_language=channel_language,
        )

    def test_text_probe_wins_on_metadata_conflict(self) -> None:
        metadata = self._metadata(
            title="Live briefing on humanitarian policy and legal updates",
            description="In this stream we discuss concrete evidence, interview highlights, and action points.",
            youtube_language="uk",
            channel_language="uk",
        )
        decision = detect_language_decision(metadata)
        self.assertEqual("en", decision.text_probe_language)
        self.assertEqual("en", decision.final_language)
        self.assertEqual("text_probe_override", decision.language_decision_source)
        self.assertTrue(decision.language_conflict)

    def test_metadata_only_fallback_still_works(self) -> None:
        metadata = self._metadata(
            title="",
            description="",
            youtube_language="ru",
            channel_language=None,
        )
        decision = detect_language_decision(metadata)
        self.assertEqual("metadata_only", decision.language_decision_source)
        self.assertEqual("ru", decision.final_language)

    def test_sheet_override_has_highest_priority(self) -> None:
        metadata = self._metadata(
            title="English title",
            description="English description",
            youtube_language="ru",
            channel_language="ru",
        )
        decision = detect_language_decision(
            metadata,
            sheet_override_language="ukr",
        )
        self.assertEqual("sheet_override", decision.language_decision_source)
        self.assertEqual("uk", decision.final_language)

    def test_language_aliases_are_normalized(self) -> None:
        self.assertEqual("uk", normalize_language("ua"))
        self.assertEqual("uk", normalize_language("ukr"))
        self.assertEqual("en", normalize_language("eng"))
        self.assertEqual("ru", normalize_language("rus"))

    def test_detect_language_uses_new_policy(self) -> None:
        metadata = self._metadata(
            title="English evidence review",
            description="We compare legal timelines and humanitarian indicators in detail.",
            youtube_language="ru",
            channel_language="ru",
        )
        self.assertEqual("en", detect_language(metadata))


if __name__ == "__main__":
    unittest.main()
