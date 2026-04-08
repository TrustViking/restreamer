from __future__ import annotations

import unittest
from unittest.mock import patch

from app.core.language_profile import VideoLanguageProfile
from app.core.models import VideoMetadata


class VideoLanguageProfileTests(unittest.TestCase):
    def _metadata(
        self,
        *,
        title: str = "Test title",
        description: str = "Test description",
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

    def test_full_profile_from_metadata(self) -> None:
        metadata = self._metadata(
            title="French lesson",
            description="Apprenez le français",
            youtube_language="fr",
            channel_language="fr",
            audio_languages=("fr", "en"),
            subtitle_languages=("fr", "en", "de"),
            auto_caption_languages=("fr", "en"),
        )
        with patch("app.core.language.detect", return_value="fr"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertEqual("fr", profile.video_language)
        self.assertEqual("fr", profile.channel_language)
        self.assertEqual(("fr", "en"), profile.audio_languages)
        self.assertIn("fr", profile.subtitle_languages)
        self.assertEqual("fr", profile.auto_caption_languages[0])

    def test_empty_metadata_produces_empty_profile(self) -> None:
        metadata = self._metadata(
            title="",
            description="",
            youtube_language=None,
            channel_language=None,
        )
        profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertIsNone(profile.video_language)
        self.assertIsNone(profile.channel_language)
        self.assertEqual((), profile.audio_languages)
        self.assertEqual((), profile.subtitle_languages)
        self.assertEqual((), profile.auto_caption_languages)
        self.assertIsNone(profile.title_language)
        self.assertIsNone(profile.description_language)
        self.assertIsNone(profile.langdetect_text_language)

    def test_language_normalization_in_profile(self) -> None:
        metadata = self._metadata(
            youtube_language="fr-FR",
            channel_language="en-US",
            audio_languages=("fr-FR", "en-US"),
            subtitle_languages=("de-DE",),
            auto_caption_languages=("fr-FR",),
        )
        with patch("app.core.language.detect", return_value="fr"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertEqual("fr", profile.video_language)
        self.assertEqual("en", profile.channel_language)
        self.assertEqual(("fr", "en"), profile.audio_languages)
        self.assertEqual(("de",), profile.subtitle_languages)
        self.assertEqual(("fr",), profile.auto_caption_languages)

    def test_duplicate_audio_languages_are_deduplicated(self) -> None:
        metadata = self._metadata(
            audio_languages=("fr", "fr", "en", "fr"),
        )
        with patch("app.core.language.detect", return_value="en"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertEqual(("fr", "en"), profile.audio_languages)

    def test_empty_string_languages_are_filtered(self) -> None:
        metadata = self._metadata(
            audio_languages=("", "fr", ""),
            subtitle_languages=("", ""),
            auto_caption_languages=("en", ""),
        )
        with patch("app.core.language.detect", return_value="en"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertEqual(("fr",), profile.audio_languages)
        self.assertEqual((), profile.subtitle_languages)
        self.assertEqual(("en",), profile.auto_caption_languages)

    def test_auto_caption_orig_extraction(self) -> None:
        """The *-orig key in auto_caption_languages is extracted correctly."""
        metadata = self._metadata(
            auto_caption_languages=("ab", "aa", "en", "fr-orig", "fr", "de"),
        )
        with patch("app.core.language.detect", return_value="en"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertEqual("fr", profile.auto_caption_orig_language)

    def test_auto_caption_orig_none_when_no_orig_key(self) -> None:
        """When no *-orig key exists, auto_caption_orig_language is None."""
        metadata = self._metadata(
            auto_caption_languages=("ab", "en", "fr", "de"),
        )
        with patch("app.core.language.detect", return_value="en"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertIsNone(profile.auto_caption_orig_language)

    def test_auto_caption_orig_empty_list(self) -> None:
        """When auto_caption_languages is empty, auto_caption_orig_language is None."""
        metadata = self._metadata(
            auto_caption_languages=(),
        )
        with patch("app.core.language.detect", return_value="en"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertIsNone(profile.auto_caption_orig_language)

    def test_auto_caption_orig_uk_orig(self) -> None:
        """uk-orig key extracts as 'uk'."""
        metadata = self._metadata(
            auto_caption_languages=("ab", "aa", "uk-orig", "uk", "en"),
        )
        with patch("app.core.language.detect", return_value="en"):
            profile = VideoLanguageProfile.from_metadata(metadata)
        self.assertEqual("uk", profile.auto_caption_orig_language)


if __name__ == "__main__":
    unittest.main()
