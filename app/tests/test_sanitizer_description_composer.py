from __future__ import annotations

from app.publish.sanitizers.description_composer import DescriptionComposer


def _no_title(_url: str) -> None:
    return None


class TestDescriptionComposerCompose:
    def test_body_only(self) -> None:
        composer: DescriptionComposer = DescriptionComposer()
        result: str = composer.compose(
            language="en",
            body_text="Some body text here.",
            cta_text="",
            hashtags_line="",
            recommended_youtube_urls=[],
            source_urls=[],
            fetch_title_fn=_no_title,
        )
        assert result == "Some body text here."

    def test_body_with_hashtags(self) -> None:
        composer: DescriptionComposer = DescriptionComposer()
        result: str = composer.compose(
            language="en",
            body_text="Body text.",
            cta_text="",
            hashtags_line="#AI #Climate",
            recommended_youtube_urls=[],
            source_urls=[],
            fetch_title_fn=_no_title,
        )
        assert "Body text." in result
        assert "#AI #Climate" in result
        assert "\n\n" in result

    def test_body_with_official_links(self) -> None:
        composer: DescriptionComposer = DescriptionComposer()
        result: str = composer.compose(
            language="en",
            body_text="Body text.",
            cta_text="",
            hashtags_line="",
            recommended_youtube_urls=[],
            source_urls=["https://example.com"],
            fetch_title_fn=_no_title,
        )
        assert "Official links" in result
        assert "https://example.com" in result

    def test_body_with_cta(self) -> None:
        composer: DescriptionComposer = DescriptionComposer()
        result: str = composer.compose(
            language="uk",
            body_text="Текст опису.",
            cta_text="Дивіться до кінця!",
            hashtags_line="",
            recommended_youtube_urls=[],
            source_urls=[],
            fetch_title_fn=_no_title,
        )
        assert "Дивіться до кінця!" in result


class TestDescriptionComposerLayout:
    def test_empty_layout(self) -> None:
        result: str = DescriptionComposer.resolve_layout(
            body_text="",
            cta_text="",
            hashtags_line="",
            recommended_youtube_urls=[],
            source_urls=[],
        )
        assert result == "empty"

    def test_body_only_layout(self) -> None:
        result: str = DescriptionComposer.resolve_layout(
            body_text="Some text",
            cta_text="",
            hashtags_line="",
            recommended_youtube_urls=[],
            source_urls=[],
        )
        assert "body" in result

    def test_full_layout(self) -> None:
        result: str = DescriptionComposer.resolve_layout(
            body_text="Text",
            cta_text="Watch",
            hashtags_line="#tag",
            recommended_youtube_urls=["https://youtu.be/x"],
            source_urls=["https://example.com"],
        )
        assert "body" in result
        assert "cta" in result
        assert "hashtags" in result
