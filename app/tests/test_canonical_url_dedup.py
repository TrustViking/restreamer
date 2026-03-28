from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.models import NormalizedImage, PlannedVideo, VideoMetadata
from app.core.url_utils import _canonical_domain_key
from app.publish.sanitizers.url_selector import _select_authoritative_non_youtube_urls


def _make_video(*, description: str, url: str = "https://youtu.be/aaaaaaaaaaa") -> PlannedVideo:
    scheduled_at: datetime = datetime(2026, 3, 20, 9, 0, tzinfo=ZoneInfo("Europe/Bucharest"))
    return PlannedVideo(
        row_number=1,
        original_link=url,
        normalized_link=url,
        scheduled_at_kiev=scheduled_at,
        date_key="2026-03-20",
        date_display="2026-03-20",
        language="uk",
        metadata=VideoMetadata(
            url=url,
            title="Test video",
            description=description,
            thumbnail_url="https://example.com/thumb.jpg",
            youtube_language="uk",
        ),
        thumbnail=NormalizedImage(
            bytes_data=b"",
            extension="jpg",
            mime_type="image/jpeg",
        ),
        local_thumbnail_path=None,
    )


class CanonicalDomainKeyTests(unittest.TestCase):
    def test_root_and_lang_suffix_same_key(self) -> None:
        root_key: str = _canonical_domain_key("https://allatra.org/")
        lang_key: str = _canonical_domain_key("https://allatra.org/uk")
        self.assertEqual(root_key, lang_key)

    def test_www_stripped(self) -> None:
        www_key: str = _canonical_domain_key("https://www.example.com/page")
        no_www_key: str = _canonical_domain_key("https://example.com/page")
        self.assertEqual(www_key, no_www_key)

    def test_real_path_preserved(self) -> None:
        about_key: str = _canonical_domain_key("https://example.com/about")
        root_key: str = _canonical_domain_key("https://example.com/")
        self.assertNotEqual(about_key, root_key)

    def test_lang_code_variants_collapse_to_domain(self) -> None:
        base_key: str = _canonical_domain_key("https://example.com/")
        self.assertEqual(_canonical_domain_key("https://example.com/en"), base_key)
        self.assertEqual(_canonical_domain_key("https://example.com/ru"), base_key)
        self.assertEqual(_canonical_domain_key("https://example.com/fr-fr"), base_key)

    def test_case_insensitive_domain(self) -> None:
        upper_key: str = _canonical_domain_key("https://EXAMPLE.COM/page")
        lower_key: str = _canonical_domain_key("https://example.com/page")
        self.assertEqual(upper_key, lower_key)

    def test_query_and_fragment_stripped(self) -> None:
        full_key: str = _canonical_domain_key("https://example.com/page?ref=1#top")
        plain_key: str = _canonical_domain_key("https://example.com/page")
        self.assertEqual(full_key, plain_key)

    def test_official_links_selection_deduplicates_root_and_lang_url(self) -> None:
        # Both URLs appear in descriptions; only one should survive selection
        description_with_both: str = (
            "Stream content here.\n"
            "https://allatra.org/\n"
            "https://allatra.org/uk"
        )
        video: PlannedVideo = _make_video(description=description_with_both)
        selected_urls, _, duplicate_count = _select_authoritative_non_youtube_urls(
            source_videos=[video],
            extracted_tail_urls=[],
        )
        allatra_urls: list[str] = [url for url in selected_urls if "allatra.org" in url]
        self.assertEqual(len(allatra_urls), 1, f"Expected 1 allatra URL, got: {allatra_urls}")
        self.assertGreater(duplicate_count, 0)


if __name__ == "__main__":
    unittest.main()
