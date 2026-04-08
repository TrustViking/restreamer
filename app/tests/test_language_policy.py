from __future__ import annotations

import unittest
from unittest.mock import patch

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
        audio_languages: tuple[str, ...] = (),
        subtitle_languages: tuple[str, ...] = (),
        auto_caption_languages: tuple[str, ...] = (),
    ) -> VideoMetadata:
        return VideoMetadata(
            url="https://youtu.be/aaaaaaaaaaa",
            title=title,
            description=description,
            thumbnail_url="https://i.ytimg.com/vi/aaaaaaaaaaa/hqdefault.jpg",
            youtube_language=youtube_language,
            channel_language=channel_language,
            audio_languages=audio_languages,
            subtitle_languages=subtitle_languages,
            auto_caption_languages=auto_caption_languages,
        )

    def test_langdetect_metadata_agreement(self) -> None:
        metadata = self._metadata(
            title="English title",
            description="English description",
            youtube_language="en",
            channel_language="en",
        )
        with patch("app.core.language.detect", return_value="en"):
            decision = detect_language_decision(metadata)
        self.assertEqual("en", decision.final_language)
        self.assertEqual("langdetect_metadata_agreement", decision.language_decision_source)
        self.assertFalse(decision.language_conflict)

    def test_conflict_no_arbiters_falls_back_to_metadata(self) -> None:
        """When langdetect and metadata conflict and no arbiter signals exist,
        fallback to video_language (metadata_arbitration_fallback)."""
        metadata = self._metadata(
            title="English title",
            description="English description",
            youtube_language="en",
            channel_language=None,
        )
        with patch("app.core.language.detect", return_value="uk"):
            decision = detect_language_decision(metadata)
        self.assertEqual("en", decision.final_language)
        self.assertEqual("metadata_arbitration_fallback", decision.language_decision_source)
        self.assertTrue(decision.language_conflict)

    def test_metadata_fallback_still_works(self) -> None:
        metadata = self._metadata(
            title="",
            description="",
            youtube_language="ru",
            channel_language=None,
        )
        decision = detect_language_decision(metadata)
        self.assertEqual("metadata_fallback", decision.language_decision_source)
        self.assertEqual("ru", decision.final_language)

    def test_language_aliases_are_normalized(self) -> None:
        self.assertEqual("uk", normalize_language("ua"))
        self.assertEqual("uk", normalize_language("ukr"))
        self.assertEqual("en", normalize_language("eng"))
        self.assertEqual("ru", normalize_language("rus"))

    def test_dynamic_language_from_langdetect_is_supported(self) -> None:
        metadata = self._metadata(
            title="Deutscher Titel",
            description="Deutscher Beschreibungstext fuer die Sendung",
            youtube_language=None,
            channel_language=None,
        )
        with patch("app.core.language.detect", return_value="de"):
            decision = detect_language_decision(metadata)
        self.assertEqual("de", decision.final_language)
        self.assertEqual("langdetect", decision.language_decision_source)

    def test_undetected_when_metadata_and_langdetect_missing(self) -> None:
        metadata = self._metadata(
            title="",
            description="",
            youtube_language=None,
            channel_language=None,
        )
        decision = detect_language_decision(metadata)
        self.assertEqual("unknown", decision.final_language)
        self.assertEqual("undetected", decision.language_decision_source)

    def test_conflict_without_arbiters_returns_metadata(self) -> None:
        """detect_language() returns metadata language when no arbiters break the tie."""
        metadata = self._metadata(
            title="English evidence review",
            description="",
            youtube_language="ru",
            channel_language="ru",
        )
        with patch("app.core.language.detect", return_value="en"):
            self.assertEqual("ru", detect_language(metadata))

    def test_russian_text_with_ukraine_word_is_detected_as_russian(self) -> None:
        metadata = self._metadata(
            title="Обсуждение новостей",
            description=(
                "Сегодня обсуждаем экономику, политику и ситуацию вокруг слова Україна "
                "в контексте русскоязычного обсуждения в одном выпуске."
            ),
            youtube_language=None,
            channel_language=None,
        )
        with patch("app.core.language.detect", return_value="ru"):
            decision = detect_language_decision(metadata)
        self.assertEqual("ru", decision.final_language)

    def test_consensus_three_signals(self) -> None:
        """When ≥3 independent signals agree, consensus wins.
        auto_caption_languages includes fr-orig which extracts as 'fr'."""
        metadata = self._metadata(
            title="Cours de français pour débutants",
            description="Apprenez le français facilement avec cette leçon.",
            youtube_language="fr",
            audio_languages=("fr",),
            auto_caption_languages=("ab", "en", "fr-orig", "fr", "de"),
        )
        with patch("app.core.language.detect", return_value="fr"):
            decision = detect_language_decision(metadata)
        self.assertEqual("fr", decision.final_language)
        self.assertEqual("consensus", decision.language_decision_source)
        self.assertFalse(decision.language_conflict)

    def test_metadata_arbitration_with_audio_and_orig(self) -> None:
        """Audio + auto_caption_orig confirm metadata over langdetect.
        May resolve via consensus when 3 votes align."""
        metadata = self._metadata(
            title="START TO UNDERSTAND FRENCH with a Simple Vlog",
            description="",
            youtube_language="fr",
            audio_languages=("fr",),
            auto_caption_languages=("ab", "en", "fr-orig", "fr"),
        )
        with patch("app.core.language.detect", return_value="en"):
            decision = detect_language_decision(metadata)
        self.assertEqual("fr", decision.final_language)
        self.assertIn(
            decision.language_decision_source,
            ("metadata_arbitration", "consensus"),
        )
        if decision.language_decision_source == "consensus":
            self.assertFalse(decision.language_conflict)
        else:
            self.assertTrue(decision.language_conflict)

    def test_langdetect_arbitration_wins(self) -> None:
        """When arbiter votes confirm langdetect, langdetect wins.
        Here audio and orig both say 'en', overriding video_language='de'."""
        metadata = self._metadata(
            title="English title about everything",
            description="",
            youtube_language="de",
            audio_languages=("en",),
            auto_caption_languages=("ab", "en-orig", "en", "de"),
        )
        with patch("app.core.language.detect", return_value="en"):
            decision = detect_language_decision(metadata)
        self.assertEqual("en", decision.final_language)
        self.assertIn(
            decision.language_decision_source,
            ("langdetect_arbitration", "consensus"),
        )
        if decision.language_decision_source == "consensus":
            self.assertFalse(decision.language_conflict)
        else:
            self.assertTrue(decision.language_conflict)

    def test_empty_new_fields_behaves_like_metadata_fallback(self) -> None:
        """With empty audio/subtitle/auto_caption fields and no langdetect,
        metadata_fallback still works as before."""
        metadata = self._metadata(
            title="",
            description="",
            youtube_language="uk",
            channel_language=None,
        )
        decision = detect_language_decision(metadata)
        self.assertEqual("metadata_fallback", decision.language_decision_source)
        self.assertEqual("uk", decision.final_language)

    def test_decision_has_new_fields(self) -> None:
        """LanguageDecision includes the new diagnostic fields."""
        metadata = self._metadata(
            title="Test title",
            description="Test description text that is long enough for detection.",
            youtube_language="en",
            audio_languages=("en", "fr"),
            auto_caption_languages=("ab", "en-orig", "en"),
        )
        with patch("app.core.language.detect", return_value="en"):
            decision = detect_language_decision(metadata)
        self.assertEqual("en", decision.audio_language)
        self.assertEqual("en", decision.auto_caption_language)
        # title_language and description_language are set (may vary by langdetect)
        self.assertIsNotNone(decision.langdetect_language)

    def test_german_video_with_english_subtitles_detected_as_german(self) -> None:
        """Regression: German educational video (audio=de, video_language=de)
        with English title/description and subtitle_languages=('en','de')
        must be detected as 'de', not 'en'.
        subtitle_first is NOT used for voting — it would wrongly vote 'en'."""
        metadata = self._metadata(
            title="Learn German with this easy lesson for beginners",
            description="In this video we practice everyday German conversations.",
            youtube_language="de",
            audio_languages=("de",),
            subtitle_languages=("en", "de"),
            auto_caption_languages=("ab", "aa", "de-orig", "de", "en"),
        )
        with patch("app.core.language.detect", return_value="en"):
            decision = detect_language_decision(metadata)
        self.assertEqual("de", decision.final_language)
        self.assertIn(
            decision.language_decision_source,
            ("metadata_arbitration", "consensus"),
        )
        if decision.language_decision_source == "consensus":
            self.assertFalse(decision.language_conflict)
        else:
            self.assertTrue(decision.language_conflict)


if __name__ == "__main__":
    unittest.main()
