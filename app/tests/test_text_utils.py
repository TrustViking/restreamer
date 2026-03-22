from __future__ import annotations

import pytest
from app.core.text_utils import (
    split_paragraphs,
    is_youtube_url,
    is_youtube_host,
    has_duplicate_paragraphs,
    normalize_whitespace,
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
