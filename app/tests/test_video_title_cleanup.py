from __future__ import annotations

import unittest

from app.core.video_title_cleanup import sanitize_source_video_title


class VideoTitleCleanupTests(unittest.TestCase):
    def test_trailing_hashtags_removed(self) -> None:
        source: str = "33 серия: Стивен Хассен. Проверка на вшивость  #эксперимент #ии #антикульт"
        expected: str = "33 серия: Стивен Хассен. Проверка на вшивость"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_protected_issue_number_preserved(self) -> None:
        source: str = "The Most Terrible Cult | #12"
        expected: str = "The Most Terrible Cult | #12"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_mixed_trailing_removable_stripped_protected_kept(self) -> None:
        source: str = "Название | #14 #эксперимент #ии"
        expected: str = "Название | #14"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_mid_title_issue_number_untouched_without_trailing_zone(self) -> None:
        source: str = "Серия #15. FECRIS. Узаконенный экстремизм"
        expected: str = "Серия #15. FECRIS. Узаконенный экстремизм"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_numeric_only_trailing_hashtag_preserved(self) -> None:
        source: str = "Название #2026"
        expected: str = "Название #2026"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_single_letter_hashtag_removed(self) -> None:
        source: str = "Название #AI"
        expected: str = "Название"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_trailing_separator_cleanup_after_removal(self) -> None:
        source: str = "Название | #эксперимент"
        expected: str = "Название"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_trailing_dash_separator_cleanup_after_removal(self) -> None:
        source: str = "Название - #cult #AI"
        expected: str = "Название"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_empty_and_whitespace_only_input_returns_empty(self) -> None:
        self.assertEqual(sanitize_source_video_title(""), "")
        self.assertEqual(sanitize_source_video_title("   \t  "), "")

    def test_title_without_hashtags_returns_unchanged(self) -> None:
        source: str = "A clean title without tags"
        expected: str = "A clean title without tags"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_title_entirely_hashtags_returns_empty(self) -> None:
        source: str = "#cult #AI #ии"
        expected: str = ""
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_protected_only_trailing_zone_stays_intact(self) -> None:
        source: str = "Series | #1 #2 #3"
        expected: str = "Series | #1 #2 #3"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_mixed_alphanumeric_hashtag_is_removable(self) -> None:
        source: str = "Title #abc123"
        expected: str = "Title"
        self.assertEqual(sanitize_source_video_title(source), expected)

    def test_numeric_hashtag_is_protected(self) -> None:
        source: str = "Title #123"
        expected: str = "Title #123"
        self.assertEqual(sanitize_source_video_title(source), expected)
