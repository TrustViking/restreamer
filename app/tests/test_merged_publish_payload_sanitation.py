from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.core.models import MergedLanguageContent
from app.publish.post_llm_sanitation import build_sanitized_merged_publication_payload


class MergedPublishPayloadSanitationTests(unittest.TestCase):
    def _video(self, url: str = "") -> SimpleNamespace:
        return SimpleNamespace(
            language="en",
            normalized_link="",
            original_link="",
            metadata=SimpleNamespace(url=url, description="", title="Source title"),
        )

    def _payload(self, description: str) -> MergedLanguageContent:
        return MergedLanguageContent(
            title="Merged title",
            description=description,
            description_selected=description,
            description_audit=description,
        )

    def test_embedded_hashtags_are_split_in_real_merged_publish_path(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\nJoin us tonight and share your thoughts. #nanoplastics #microplastics"
        )
        with self.assertLogs(level="INFO") as captured:
            payload = build_sanitized_merged_publication_payload(
                language="en",
                merged_content=merged_content,
                merge_attempt=SimpleNamespace(
                    publish_source_label="merge_success",
                    plain_repair_used=False,
                ),
                use_audit_text=False,
                source_videos=[self._video()],
            )
        self.assertEqual(
            "Body paragraph.\n\n"
            "Join us tonight and share your thoughts.\n\n"
            "#nanoplastics #microplastics",
            payload.description_text,
        )
        logs: str = "\n".join(captured.output)
        self.assertIn("merged_publish_sanitation_applied=yes", logs)
        self.assertIn("hashtags_split_from_cta=yes", logs)
        self.assertIn("tail_layout=body_blank_cta_blank_hashtags", logs)

    def test_already_separate_hashtags_stay_stable(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\nJoin us tonight and share your thoughts.\n\n#nanoplastics #microplastics"
        )
        payload = build_sanitized_merged_publication_payload(
            language="en",
            merged_content=merged_content,
            merge_attempt=None,
            use_audit_text=False,
            source_videos=[self._video()],
        )
        self.assertEqual(
            "Body paragraph.\n\nJoin us tonight and share your thoughts.\n\n#nanoplastics #microplastics",
            payload.description_text,
        )

    def test_hashtags_without_cta_keep_single_blank_line(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n#nanoplastics #microplastics"
        )
        payload = build_sanitized_merged_publication_payload(
            language="en",
            merged_content=merged_content,
            merge_attempt=None,
            use_audit_text=False,
            source_videos=[self._video()],
        )
        self.assertEqual("Body paragraph.\n\n#nanoplastics #microplastics", payload.description_text)

    def test_no_hashtags_do_not_add_extra_blank_lines(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\nJoin us tonight and share your thoughts."
        )
        payload = build_sanitized_merged_publication_payload(
            language="en",
            merged_content=merged_content,
            merge_attempt=None,
            use_audit_text=False,
            source_videos=[self._video()],
        )
        self.assertEqual(
            "Body paragraph.\n\nJoin us tonight and share your thoughts.",
            payload.description_text,
        )

    def test_empty_official_links_heading_is_suppressed(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n🌐 Official links:\n\nJoin us tonight and share your thoughts."
        )
        with self.assertLogs(level="INFO") as captured:
            payload = build_sanitized_merged_publication_payload(
                language="en",
                merged_content=merged_content,
                merge_attempt=None,
                use_audit_text=False,
                source_videos=[self._video()],
            )
        self.assertNotIn("🌐 Official links:", payload.description_text)
        logs: str = "\n".join(captured.output)
        self.assertIn("official_links_block=suppressed", logs)
        self.assertIn("official_links_final_count=0", logs)

    def test_non_empty_official_links_block_stays_cohesive(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n"
            "🌐 Official links:\n\n"
            "https://example.org/official\n"
            "https://allatra.org/resource\n\n"
            "Join us tonight and share your thoughts. #nanoplastics #microplastics"
        )
        with self.assertLogs(level="INFO") as captured:
            payload = build_sanitized_merged_publication_payload(
                language="en",
                merged_content=merged_content,
                merge_attempt=None,
                use_audit_text=False,
                source_videos=[self._video()],
            )
        self.assertEqual(
            "Body paragraph.\n\n"
            "🌐 Official links:\n"
            "https://example.org/official\n"
            "https://allatra.org/resource\n\n"
            "Join us tonight and share your thoughts.\n\n"
            "#nanoplastics #microplastics",
            payload.description_text,
        )
        self.assertEqual(1, payload.description_text.count("🌐 Official links:"))
        logs: str = "\n".join(captured.output)
        self.assertIn("official_links_heading_found=yes", logs)
        self.assertIn("official_links_text_links=2", logs)
        self.assertIn("official_links_final_count=2", logs)
        self.assertIn("official_links_block=emitted", logs)
        self.assertIn(
            "tail_layout=body_blank_official_links_blank_cta_blank_hashtags",
            logs,
        )

    def test_official_links_from_text_and_source_videos_are_deduped_into_one_block(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n"
            "🌐 Official links:\n\n"
            "https://example.org/official?utm_source=yt\n\n"
            "Join us tonight and share your thoughts.\n\n"
            "#nanoplastics #microplastics"
        )
        with self.assertLogs(level="INFO") as captured:
            payload = build_sanitized_merged_publication_payload(
                language="en",
                merged_content=merged_content,
                merge_attempt=None,
                use_audit_text=False,
                source_videos=[
                    self._video("https://example.org/official"),
                    self._video("https://example.org/second-source"),
                ],
            )
        self.assertEqual(
            "Body paragraph.\n\n"
            "🌐 Official links:\n"
            "https://example.org/official\n"
            "https://example.org/second-source\n\n"
            "Join us tonight and share your thoughts.\n\n"
            "#nanoplastics #microplastics",
            payload.description_text,
        )
        self.assertEqual(1, payload.description_text.count("🌐 Official links:"))
        self.assertEqual(1, payload.description_text.count("https://example.org/official"))
        logs: str = "\n".join(captured.output)
        self.assertIn("official_links_text_links=1", logs)
        self.assertIn("official_links_source_links=2", logs)
        self.assertIn("official_links_final_count=2", logs)
        self.assertIn("official_links_dedup_applied=yes", logs)

    def test_source_youtube_urls_are_not_auto_injected(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\nJoin us tonight and share your thoughts."
        )
        payload = build_sanitized_merged_publication_payload(
            language="en",
            merged_content=merged_content,
            merge_attempt=None,
            use_audit_text=False,
            source_videos=[
                self._video("https://youtu.be/aaaaaaaaaaa"),
                self._video("https://www.youtube.com/watch?v=bbbbbbbbbbb"),
            ],
        )
        self.assertNotIn("youtu", payload.description_text)

    def test_explicit_selected_youtube_urls_are_preserved_without_source_auto_fill(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n"
            "https://youtu.be/ccccccccccc\n\n"
            "Join us tonight and share your thoughts."
        )
        payload = build_sanitized_merged_publication_payload(
            language="en",
            merged_content=merged_content,
            merge_attempt=None,
            use_audit_text=False,
            source_videos=[
                self._video("https://youtu.be/aaaaaaaaaaa"),
                self._video("https://youtu.be/bbbbbbbbbbb"),
            ],
        )
        self.assertIn("https://youtu.be/ccccccccccc", payload.description_text)
        self.assertNotIn("https://youtu.be/aaaaaaaaaaa", payload.description_text)
        self.assertNotIn("https://youtu.be/bbbbbbbbbbb", payload.description_text)


if __name__ == "__main__":
    unittest.main()
