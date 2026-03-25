from __future__ import annotations

from typing import Optional

from app.core.url_utils import normalize_official_link_display, strip_tracking_params
from app.publish.sanitizers.url_selector import (
    AuthoritativeUrlSelector,
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
