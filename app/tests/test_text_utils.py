from __future__ import annotations

import pytest
from app.core.text_utils import (
    split_paragraphs,
    is_youtube_url,
    is_youtube_host,
    has_duplicate_paragraphs,
    normalize_whitespace,
    utf16_len,
)


class TestSplitParagraphs:
    def test_basic(self) -> None:
        assert split_paragraphs("a\n\nb") == ["a", "b"]

    def test_empty(self) -> None:
        assert split_paragraphs("") == []

    def test_strips(self) -> None:
        assert split_paragraphs("  a  \n\n  b  ") == ["a", "b"]

    def test_crlf(self) -> None:
        assert split_paragraphs("a\r\n\r\nb") == ["a", "b"]


class TestIsYoutubeUrl:
    def test_youtu_be(self) -> None:
        assert is_youtube_url("https://youtu.be/abc") is True

    def test_www(self) -> None:
        assert is_youtube_url("https://www.youtube.com/watch?v=x") is True

    def test_non_youtube(self) -> None:
        assert is_youtube_url("https://example.com") is False

    def test_empty(self) -> None:
        assert is_youtube_url("") is False


class TestIsYoutubeHost:
    def test_match(self) -> None:
        assert is_youtube_host("youtube.com") is True

    def test_no_match(self) -> None:
        assert is_youtube_host("example.com") is False


class TestHasDuplicateParagraphs:
    def test_no_dupes(self) -> None:
        assert has_duplicate_paragraphs("First para.\n\nSecond para.") is False

    def test_exact_dupe(self) -> None:
        assert has_duplicate_paragraphs(
            "Same long text here enough tokens.\n\nSame long text here enough tokens."
        ) is True

    def test_short_ignored(self) -> None:
        assert has_duplicate_paragraphs("Hi.\n\nHi.") is False


class TestNormalizeWhitespace:
    def test_basic(self) -> None:
        assert normalize_whitespace("  a   b  ") == "a b"

    def test_newlines(self) -> None:
        assert normalize_whitespace("a\n\nb") == "a b"


class TestUtf16Len:
    def test_ascii(self) -> None:
        assert utf16_len("hello") == 5

    def test_cyrillic(self) -> None:
        # Cyrillic is BMP - same length as Python len()
        assert utf16_len("Привет") == 6

    def test_empty(self) -> None:
        assert utf16_len("") == 0

    def test_flag_emoji(self) -> None:
        # 🇫🇷 = U+1F1EB U+1F1F7, each is a surrogate pair in UTF-16
        assert utf16_len("🇫🇷") == 4
        assert len("🇫🇷") == 2  # Python code points for comparison

    def test_flag_emoji_ua(self) -> None:
        assert utf16_len("🇺🇦") == 4

    def test_mixed_text_with_flag(self) -> None:
        text = "OTHER - 20:00"
        assert utf16_len(text) == 13  # all BMP, same as len()
        text_with_flag = "🇫🇷 French"
        assert utf16_len(text_with_flag) == 11  # 4 + 1 + 6

    def test_regular_emoji(self) -> None:
        # ✅ is U+2705, BMP, single UTF-16 unit
        assert utf16_len("✅") == 1
        # 🔹 is U+1F539, non-BMP, surrogate pair
        assert utf16_len("🔹") == 2

    def test_newline_preserved(self) -> None:
        assert utf16_len("abc\n") == 4

    def test_header_line_index_calculation(self) -> None:
        """Simulate the exact scenario from the bug: header text with flag emoji.

        After inserting "OTHER - 20:00\n" and "🇫🇷 French title\n",
        the cursor for the next line "RU - 20:00\n" must account for
        UTF-16 lengths, not Python len().
        """
        lines = [
            "OTHER - 20:00\n",
            "🇫🇷 French title\n",
            "RU - 20:00\n",
        ]
        # Simulate cursor logic from _insert_header_text
        cursor_utf16: int = 0
        cursor_python: int = 0
        ranges_utf16: list[tuple[int, int]] = []
        ranges_python: list[tuple[int, int]] = []
        for line in lines:
            start_utf16 = 100 + cursor_utf16  # 100 = hypothetical start_index
            end_utf16 = start_utf16 + utf16_len(line)
            ranges_utf16.append((start_utf16, end_utf16))
            cursor_utf16 += utf16_len(line)

            start_python = 100 + cursor_python
            end_python = start_python + len(line)
            ranges_python.append((start_python, end_python))
            cursor_python += len(line)

        # UTF-16 range for "RU - 20:00\n" should be 11 units long
        ru_start_utf16, ru_end_utf16 = ranges_utf16[2]
        assert ru_end_utf16 - ru_start_utf16 == utf16_len("RU - 20:00\n") == 11

        # Python-len range would be wrong - it would be 2 units short
        # because 🇫🇷 is 2 in Python but 4 in UTF-16
        ru_start_python, ru_end_python = ranges_python[2]
        assert ru_start_python < ru_start_utf16  # Python cursor is behind
        assert ru_start_utf16 - ru_start_python == 2  # exactly the flag emoji difference
