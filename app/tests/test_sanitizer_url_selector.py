from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

from app.core.models import VideoMetadata
from app.core.url_utils import (
    is_social_platform_host,
    normalize_official_link_display,
    strip_tracking_params,
)
from app.publish.sanitizers.url_selector import (
    AuthoritativeUrlSelector,
    _determine_recommended_video_language,
    _is_complete_source_url,
    _sanitize_source_url,
    _sanitize_url,
)


class TestSanitizeUrl:
    def test_strips_utm_params(self) -> None:
        result: str = _sanitize_url("https://example.com/page?utm_source=twitter&id=123")
        assert "utm_source" not in result
        assert "id=123" in result

    def test_strips_tracking_params(self) -> None:
        result: str = _sanitize_url("https://example.com/?fbclid=abc123")
        assert "fbclid" not in result

    def test_preserves_clean_url(self) -> None:
        assert _sanitize_url("https://example.com/about") == "https://example.com/about"

    def test_strips_trailing_slash(self) -> None:
        result: str = _sanitize_url("https://example.com/")
        assert not result.endswith("/")

    def test_empty(self) -> None:
        assert _sanitize_url("") == ""


class TestStripTrackingParams:
    def test_preserves_fragment_and_param_order(self) -> None:
        result: str = strip_tracking_params(
            "https://example.com/path?a=1&utm_source=twitter&a=2&fbclid=abc&b=3#part"
        )
        assert result == "https://example.com/path?a=1&a=2&b=3#part"

    def test_invalid_url_returns_as_is(self) -> None:
        value: str = "not-a-url?utm_source=twitter"
        assert strip_tracking_params(value) == value


class TestNormalizeOfficialLinkDisplay:
    def test_social_x_keeps_path(self) -> None:
        assert (
            normalize_official_link_display("https://x.com/pastormarkburns/status/123456")
            == "https://x.com/pastormarkburns/status/123456"
        )

    def test_social_facebook_keeps_path_and_www(self) -> None:
        assert (
            normalize_official_link_display("https://www.facebook.com/SomePage")
            == "https://www.facebook.com/SomePage"
        )

    def test_social_telegram_keeps_path(self) -> None:
        assert (
            normalize_official_link_display("https://t.me/somechannel/12345")
            == "https://t.me/somechannel/12345"
        )

    def test_social_instagram_strips_tracking_but_keeps_path(self) -> None:
        assert (
            normalize_official_link_display("https://www.instagram.com/user/?utm_source=ig")
            == "https://www.instagram.com/user/"
        )

    def test_social_mobile_facebook_subdomain_keeps_path(self) -> None:
        assert (
            normalize_official_link_display("https://m.facebook.com/page/123")
            == "https://m.facebook.com/page/123"
        )

    def test_regular_site_strips_www_and_path(self) -> None:
        assert (
            normalize_official_link_display("https://www.spiritualdiplomats.org/ukraine")
            == "https://spiritualdiplomats.org"
        )

    def test_regular_site_strips_language_path(self) -> None:
        assert normalize_official_link_display("https://allatra.org/uk") == "https://allatra.org"

    def test_social_without_path_stays_same(self) -> None:
        assert normalize_official_link_display("https://x.com") == "https://x.com"

    def test_regular_site_keeps_bare_domain(self) -> None:
        assert normalize_official_link_display("https://example.com") == "https://example.com"

    def test_regular_site_strips_trailing_slash(self) -> None:
        assert normalize_official_link_display("https://example.com/") == "https://example.com"

    def test_regular_site_strips_query(self) -> None:
        assert normalize_official_link_display("https://www.example.org?ref=123") == "https://example.org"

    def test_invalid_url_returns_as_is(self) -> None:
        assert normalize_official_link_display("not-a-url") == "not-a-url"

    def test_empty(self) -> None:
        assert normalize_official_link_display("") == ""


class TestIsSocialPlatformHost:
    def test_mobile_facebook_subdomain(self) -> None:
        assert is_social_platform_host("m.facebook.com") is True

    def test_mobile_twitter_subdomain(self) -> None:
        assert is_social_platform_host("mobile.twitter.com") is True

    def test_instagram_link_subdomain(self) -> None:
        assert is_social_platform_host("l.instagram.com") is True

    def test_non_social_subdomain(self) -> None:
        assert is_social_platform_host("api.example.com") is False

    def test_base_social_domain(self) -> None:
        assert is_social_platform_host("facebook.com") is True


class TestIsCompleteSourceUrl:
    def test_valid_https(self) -> None:
        assert _is_complete_source_url("https://example.com") is True

    def test_valid_http(self) -> None:
        assert _is_complete_source_url("http://example.com/page") is True

    def test_invalid_no_scheme(self) -> None:
        assert _is_complete_source_url("example.com") is False

    def test_invalid_empty(self) -> None:
        assert _is_complete_source_url("") is False


class TestSanitizeSourceUrl:
    def test_valid_url(self) -> None:
        result: Optional[str] = _sanitize_source_url("https://example.com/page?utm_medium=email")
        assert result is not None
        assert "utm_medium" not in result

    def test_invalid_url(self) -> None:
        assert _sanitize_source_url("not-a-url") is None

    def test_youtube_url_normalized(self) -> None:
        result: Optional[str] = _sanitize_source_url(
            "https://www.youtube.com/watch?v=abc123def45&si=tracking"
        )
        assert result is not None
        assert "si=" not in result


class TestExtractSemanticTokens:
    def test_basic_extraction(self) -> None:
        tokens: set[str] = AuthoritativeUrlSelector._extract_semantic_tokens(
            "Climate change affects global economies"
        )
        assert "climate" in tokens
        assert "change" in tokens
        assert "global" in tokens
        assert "economies" in tokens

    def test_stopwords_excluded(self) -> None:
        tokens: set[str] = AuthoritativeUrlSelector._extract_semantic_tokens(
            "This is about the topic today"
        )
        assert "about" not in tokens
        assert "this" not in tokens
        assert "today" not in tokens


@patch("app.publish.sanitizers.url_selector.YtDlpYouTubeMetadataFetcher")
def test_determine_language_all_signals_agree_uk(mock_fetcher_cls: MagicMock) -> None:
    mock_metadata = VideoMetadata(
        url="https://youtu.be/test123test",
        title="Український огляд нанопластику",
        description="Це відео про нанопластик і його вплив на здоров'я людей",
        thumbnail_url="https://i.ytimg.com/vi/test123test/hqdefault.jpg",
        youtube_language="uk",
        channel_language="uk",
    )
    mock_fetcher_cls.return_value.fetch.return_value = mock_metadata
    result = _determine_recommended_video_language("https://youtu.be/test123test")
    assert result == "uk"


@patch("app.publish.sanitizers.url_selector.YtDlpYouTubeMetadataFetcher")
def test_determine_language_signals_disagree_reconciles_to_en(mock_fetcher_cls: MagicMock) -> None:
    mock_metadata = VideoMetadata(
        url="https://youtu.be/test123test",
        title="Nanoplastic overview and policy impacts",
        description="This is about nanoplastic and long-term health effects.",
        thumbnail_url="https://i.ytimg.com/vi/test123test/hqdefault.jpg",
        youtube_language="en",
        channel_language="uk",
    )
    mock_fetcher_cls.return_value.fetch.return_value = mock_metadata
    result = _determine_recommended_video_language("https://youtu.be/test123test")
    assert result == "en"


@patch("app.publish.sanitizers.url_selector.YtDlpYouTubeMetadataFetcher")
def test_determine_language_en_video_discarded_for_uk_target(mock_fetcher_cls: MagicMock) -> None:
    mock_metadata = VideoMetadata(
        url="https://youtu.be/SOWIeKU-90Y",
        title="The invisible threat of nanoplastic",
        description="This video explains the invisible threat of nanoplastic.",
        thumbnail_url="https://i.ytimg.com/vi/SOWIeKU-90Y/hqdefault.jpg",
        youtube_language="en",
        channel_language="en",
    )
    mock_fetcher_cls.return_value.fetch.return_value = mock_metadata
    result = _determine_recommended_video_language("https://youtu.be/SOWIeKU-90Y")
    assert result == "en"


@patch("app.publish.sanitizers.url_selector.YtDlpYouTubeMetadataFetcher")
def test_determine_language_fetch_failure_returns_none(mock_fetcher_cls: MagicMock) -> None:
    mock_fetcher_cls.return_value.fetch.side_effect = Exception("Network error")
    result = _determine_recommended_video_language("https://youtu.be/test123test")
    assert result is None


@patch("app.publish.sanitizers.url_selector.YtDlpYouTubeMetadataFetcher")
def test_determine_language_no_metadata_returns_none(mock_fetcher_cls: MagicMock) -> None:
    mock_metadata = VideoMetadata(
        url="https://youtu.be/test123test",
        title="",
        description="",
        thumbnail_url="https://i.ytimg.com/vi/test123test/hqdefault.jpg",
        youtube_language=None,
        channel_language=None,
    )
    mock_fetcher_cls.return_value.fetch.return_value = mock_metadata
    result = _determine_recommended_video_language("https://youtu.be/test123test")
    assert result is None


class TestRecommendedMaterialsFallback:
    """Verify that when zero candidates pass the main threshold,
    a candidate with source_hits >= 2 is selected via fallback."""

    def test_fallback_selects_multi_source_candidate(self) -> None:
        """RU-like scenario: 3 candidates, all semantic_overlap=0,
        one with source_hits=2 should be selected via fallback."""

        def _make_video(desc: str, lang: str = "ru") -> MagicMock:
            video: MagicMock = MagicMock()
            video.metadata = MagicMock()
            video.metadata.description = desc
            video.metadata.title = "Title"
            video.metadata.language = lang
            video.normalized_link = "https://youtu.be/XXXXXXXXXXX"
            return video

        video1: MagicMock = _make_video(
            "Content alpha https://youtu.be/AAAAAAAAAAA some text https://youtu.be/CCCCCCCCCCC"
        )
        video2: MagicMock = _make_video(
            "Content beta https://youtu.be/AAAAAAAAAAA other text https://youtu.be/DDDDDDDDDDD"
        )

        with patch(
            "app.publish.sanitizers.url_selector._determine_recommended_video_language",
            return_value="ru",
        ):
            selected, raw_found, deduped, repeated = (
                AuthoritativeUrlSelector.select_recommended_youtube(
                    source_videos=[video1, video2],
                    summary_text="unrelated summary without matching tokens",
                    target_language="ru",
                    source_count=2,
                )
            )

        assert raw_found >= 3
        assert deduped >= 3
        assert repeated >= 1
        assert len(selected) == 1, f"Expected 1 fallback selection, got {len(selected)}"
        assert "AAAAAAAAAAA" in selected[0], "Should select the multi-source candidate"

    def test_no_fallback_when_main_threshold_passes(self) -> None:
        """EN-like scenario: candidates have semantic overlap, main threshold works.
        Fallback should NOT override the main selection."""

        def _make_video(desc: str) -> MagicMock:
            video: MagicMock = MagicMock()
            video.metadata = MagicMock()
            video.metadata.description = desc
            video.metadata.title = "Storms weather cyclone"
            video.metadata.language = "en"
            video.normalized_link = "https://youtu.be/XXXXXXXXXXX"
            return video

        video1: MagicMock = _make_video(
            "Storms weather cyclone https://youtu.be/EEEEEEEEEEE discussion"
        )
        video2: MagicMock = _make_video(
            "Storms weather cyclone https://youtu.be/FFFFFFFFFFF analysis"
        )

        with patch(
            "app.publish.sanitizers.url_selector._determine_recommended_video_language",
            return_value="en",
        ):
            selected, _, _, _ = AuthoritativeUrlSelector.select_recommended_youtube(
                source_videos=[video1, video2],
                summary_text="Storms weather cyclone severe impact",
                target_language="en",
                source_count=2,
            )

        assert len(selected) >= 1
