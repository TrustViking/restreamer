import pytest

from app.core.url_normalizer import normalize_display_url


class TestNormalizeDisplayUrl:
    """Tests for trailing slash normalization on display URLs."""

    def test_root_domain_with_trailing_slash(self) -> None:
        assert normalize_display_url("https://gpandreoli.com/") == "https://gpandreoli.com"

    def test_root_domain_without_trailing_slash(self) -> None:
        assert normalize_display_url("https://gpandreoli.com") == "https://gpandreoli.com"

    def test_www_root_domain_with_trailing_slash(self) -> None:
        assert normalize_display_url("https://www.spiritualdiplomats.org/") == "https://www.spiritualdiplomats.org"

    def test_real_path_preserved(self) -> None:
        assert normalize_display_url("https://allatra.org/uk") == "https://allatra.org/uk"

    def test_deep_path_preserved(self) -> None:
        assert normalize_display_url("https://harvard.academia.edu/EgonCholakian") == "https://harvard.academia.edu/EgonCholakian"

    def test_path_with_trailing_slash_preserved(self) -> None:
        assert normalize_display_url("https://example.com/about/") == "https://example.com/about/"

    def test_root_with_query_preserved(self) -> None:
        assert normalize_display_url("https://example.com/?q=1") == "https://example.com/?q=1"

    def test_root_with_fragment_preserved(self) -> None:
        assert normalize_display_url("https://example.com/#section") == "https://example.com/#section"

    def test_http_scheme(self) -> None:
        assert normalize_display_url("http://interfaithconf.org/") == "http://interfaithconf.org"

    def test_empty_string(self) -> None:
        assert normalize_display_url("") == ""

    def test_non_http_scheme(self) -> None:
        assert normalize_display_url("ftp://files.example.com/") == "ftp://files.example.com/"

    def test_youtube_url_unchanged(self) -> None:
        assert normalize_display_url("https://youtu.be/abc123") == "https://youtu.be/abc123"
