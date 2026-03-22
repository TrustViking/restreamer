from __future__ import annotations

from app.publish.sanitizers.tail_parser import EmbeddedTailParts, TailParser, TailParts


class TestTailParserIsSourceUrlLine:
    def test_valid_url(self) -> None:
        assert TailParser.is_source_url_line("https://example.com") is True

    def test_invalid_text(self) -> None:
        assert TailParser.is_source_url_line("not a url") is False

    def test_empty(self) -> None:
        assert TailParser.is_source_url_line("") is False


class TestTailParserIsHashtagsLine:
    def test_valid_hashtags(self) -> None:
        assert TailParser.is_hashtags_line("#AI #Climate #Ukraine") is True

    def test_not_hashtags(self) -> None:
        assert TailParser.is_hashtags_line("This is regular text") is False

    def test_mixed_content(self) -> None:
        assert TailParser.is_hashtags_line("#AI and some text") is False


class TestTailParserLooksCta:
    def test_watch_cta(self) -> None:
        assert TailParser.looks_like_cta_line("Watch the full stream here") is True

    def test_subscribe_cta(self) -> None:
        assert TailParser.looks_like_cta_line("Subscribe to our channel") is True

    def test_regular_text(self) -> None:
        assert TailParser.looks_like_cta_line("The conference discussed climate change") is False

    def test_ukrainian_cta(self) -> None:
        assert TailParser.looks_like_cta_line("Дивіться ефір до кінця") is True


class TestTailParserCleanDoubleBullets:
    def test_double_marker(self) -> None:
        result: str = TailParser.clean_double_bullet_markers("🔹 🔹 Some text")
        assert result.startswith("🔹 ")
        assert "🔹 🔹" not in result

    def test_single_marker_unchanged(self) -> None:
        result: str = TailParser.clean_double_bullet_markers("🔹 Normal bullet")
        assert result == "🔹 Normal bullet"


class TestTailParserMergeHashtags:
    def test_dedup(self) -> None:
        result: str = TailParser.merge_hashtag_lines(["#AI #Climate", "#AI #Ukraine"])
        assert "#AI" in result
        assert "#Climate" in result
        assert "#Ukraine" in result
        assert result.count("#AI") == 1

    def test_empty(self) -> None:
        assert TailParser.merge_hashtag_lines([]) == ""


class TestTailParserSplitTail:
    def test_simple_tail_with_url(self) -> None:
        lines: list[str] = [
            "First paragraph text here.",
            "",
            "https://example.com",
        ]
        result: TailParts = TailParser.split_tail(lines)
        assert len(result.source_urls) == 1
        assert result.body_end_index < len(lines)

    def test_no_tail(self) -> None:
        lines: list[str] = ["Just a paragraph.", "With two lines."]
        result: TailParts = TailParser.split_tail(lines)
        assert result.source_urls == []
        assert result.body_end_index == len(lines)


class TestTailParserExtractEmbedded:
    def test_paragraph_with_trailing_url(self) -> None:
        text: str = "Some description text.\n\nhttps://example.com"
        result: EmbeddedTailParts = TailParser.extract_embedded(text)
        assert "Some description" in result.body_text
