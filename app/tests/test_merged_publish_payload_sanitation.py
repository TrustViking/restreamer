from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import BLOCK_GENERATION_MODE_REAL_MERGE, MergedLanguageContent
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
        self.assertEqual(BLOCK_GENERATION_MODE_REAL_MERGE, payload.block_generation_mode)

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

    @patch("app.publish.post_llm_sanitation._fetch_recommended_youtube_title", return_value=None)
    def test_official_links_remnants_are_collapsed_into_one_final_block_without_youtube_mix(self, _title_mock) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n"
            "🌐 Official links:\n\n"
            "https://example.org/official?utm_source=yt\n\n"
            "Official links:\n"
            "https://example.org/official\n"
            "https://example.org/second\n\n"
            "Official links:\n\n"
            "https://youtu.be/ccccccccccc\n\n"
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
                    self._video("https://example.org/second"),
                    self._video(),
                ],
            )
        self.assertEqual(1, payload.description_text.count("🌐 Official links:"))
        self.assertEqual(1, payload.description_text.count("https://example.org/official"))
        self.assertEqual(1, payload.description_text.count("https://example.org/second"))
        self.assertNotIn("Recommended materials:", payload.description_text)
        self.assertNotIn("https://youtu.be/ccccccccccc", payload.description_text)
        self.assertNotIn("Official links:\n\nOfficial links:", payload.description_text)
        logs: str = "\n".join(captured.output)
        self.assertIn("official_links_text_links=2", logs)
        self.assertIn("official_links_final_count=2", logs)
        self.assertIn("recommended_materials_final_count=0", logs)
        self.assertIn("recommended_materials_block=skipped", logs)
        self.assertIn("ignored_llm_youtube_urls=1", logs)
        self.assertIn("official_links_dedup_applied=yes", logs)

    def test_heading_and_paragraph_remnants_do_not_leave_mixed_old_and_new_official_links_chunks(self) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n"
            "🌐 Official links:\n"
            "https://example.org/official\n"
            "https://example.org/second\n\n"
            "Official links:\n\n"
            "Join us tonight and share your thoughts.\n\n"
            "#nanoplastics #microplastics"
        )
        payload = build_sanitized_merged_publication_payload(
            language="en",
            merged_content=merged_content,
            merge_attempt=None,
            use_audit_text=False,
            source_videos=[],
        )
        self.assertEqual(
            "Body paragraph.\n\n"
            "🌐 Official links:\n"
            "https://example.org/official\n"
            "https://example.org/second\n\n"
            "Join us tonight and share your thoughts.\n\n"
            "#nanoplastics #microplastics",
            payload.description_text,
        )
        self.assertEqual(1, payload.description_text.count("🌐 Official links:"))
        self.assertNotIn("Official links:\n\nJoin us", payload.description_text)

    @patch("app.publish.post_llm_sanitation._fetch_recommended_youtube_title", return_value=None)
    def test_source_youtube_urls_are_not_auto_injected(self, _title_mock) -> None:
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

    @patch(
        "app.publish.post_llm_sanitation._fetch_recommended_youtube_title",
        side_effect=[
            "Title ✅ Real | #Hash & % $",
            "Second title 😎 / unchanged",
        ],
    )
    def test_recommended_materials_are_rendered_as_title_and_url_lines(self, title_mock) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph about the conference agenda legal overview and practical details.\n\n"
            "Join us tonight and share your thoughts."
        )
        source_videos = [
            SimpleNamespace(
                language="en",
                normalized_link="",
                original_link="",
                metadata=SimpleNamespace(
                    url="",
                    title="Conference source",
                    description=(
                        "Conference agenda and legal overview.\n"
                        "https://youtu.be/aaaaaaaaaaa\n"
                        "Official page https://example.org/official"
                    ),
                ),
            ),
            SimpleNamespace(
                language="en",
                normalized_link="",
                original_link="",
                metadata=SimpleNamespace(
                    url="",
                    title="Second conference source",
                    description=(
                        "Legal overview and conference agenda continue.\n"
                        "https://www.youtube.com/watch?v=aaaaaaaaaaa&feature=share\n"
                        "https://youtu.be/bbbbbbbbbbb"
                    ),
                ),
            ),
            SimpleNamespace(
                language="en",
                normalized_link="",
                original_link="",
                metadata=SimpleNamespace(
                    url="",
                    title="Third conference source",
                    description=(
                        "Conference agenda legal overview practical details https://youtu.be/ccccccccccc\n"
                        "Misc clip https://youtu.be/ddddddddddd"
                    ),
                ),
            ),
        ]
        with self.assertLogs(level="INFO") as captured:
            payload = build_sanitized_merged_publication_payload(
                language="en",
                merged_content=merged_content,
                merge_attempt=None,
                use_audit_text=False,
                source_videos=source_videos,
            )
        self.assertIn("Recommended materials:", payload.description_text)
        self.assertIn("✅ Title ✅ Real | #Hash & % $", payload.description_text)
        self.assertIn("👉 https://youtu.be/aaaaaaaaaaa", payload.description_text)
        self.assertIn("✅ Second title 😎 / unchanged", payload.description_text)
        self.assertIn("👉 https://youtu.be/ccccccccccc", payload.description_text)
        self.assertNotIn("https://youtu.be/ddddddddddd", payload.description_text)
        self.assertNotIn("✅ Title Real", payload.description_text)
        self.assertLess(
            payload.description_text.index("Recommended materials:"),
            payload.description_text.index("Join us tonight and share your thoughts."),
        )
        self.assertEqual(
            [
                ("https://youtu.be/aaaaaaaaaaa",),
                ("https://youtu.be/ccccccccccc",),
            ],
            [call.args for call in title_mock.call_args_list],
        )
        logs: str = "\n".join(captured.output)
        self.assertIn("raw_youtube_urls_found=5", logs)
        self.assertIn("deduped_youtube_candidates=4", logs)
        self.assertIn("repeated_youtube_candidates=1", logs)
        self.assertIn("recommended_materials_final_count=2", logs)
        self.assertIn("recommended_block_rendered_with_titles urls=2 titles_rendered=2 title_fetch_failures=0", logs)

    @patch(
        "app.publish.post_llm_sanitation._fetch_recommended_youtube_title",
        side_effect=[RuntimeError("yt-dlp unavailable"), None],
    )
    def test_recommended_materials_keep_url_when_title_fetch_fails(self, _title_mock) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph about the conference agenda legal overview and practical details.\n\n"
            "Join us tonight and share your thoughts."
        )
        source_videos = [
            SimpleNamespace(
                language="en",
                normalized_link="",
                original_link="",
                metadata=SimpleNamespace(
                    url="",
                    title="Conference source",
                    description=(
                        "Conference agenda and legal overview.\n"
                        "https://youtu.be/aaaaaaaaaaa\n"
                        "https://youtu.be/ccccccccccc"
                    ),
                ),
            ),
            SimpleNamespace(
                language="en",
                normalized_link="",
                original_link="",
                metadata=SimpleNamespace(
                    url="",
                    title="Third conference source",
                    description=(
                        "Conference agenda legal overview practical details https://youtu.be/aaaaaaaaaaa\n"
                        "Conference agenda legal overview practical details https://youtu.be/ccccccccccc"
                    ),
                ),
            ),
        ]
        with self.assertLogs(level="INFO") as captured:
            payload = build_sanitized_merged_publication_payload(
                language="en",
                merged_content=merged_content,
                merge_attempt=None,
                use_audit_text=False,
                source_videos=source_videos,
            )
        self.assertIn("Recommended materials:", payload.description_text)
        self.assertIn("👉 https://youtu.be/aaaaaaaaaaa", payload.description_text)
        self.assertIn("👉 https://youtu.be/ccccccccccc", payload.description_text)
        self.assertNotIn("✅", payload.description_text)
        logs: str = "\n".join(captured.output)
        self.assertIn("recommended_block_rendered_with_titles urls=2 titles_rendered=0 title_fetch_failures=2", logs)

    @patch("app.publish.post_llm_sanitation._fetch_recommended_youtube_title", return_value=None)
    def test_llm_youtube_tail_is_ignored_when_raw_sources_have_no_worthy_candidates(self, _title_mock) -> None:
        merged_content: MergedLanguageContent = self._payload(
            "Body paragraph.\n\n"
            "https://youtu.be/ccccccccccc\n\n"
            "Join us tonight and share your thoughts."
        )
        with self.assertLogs(level="INFO") as captured:
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
        self.assertNotIn("https://youtu.be/ccccccccccc", payload.description_text)
        self.assertNotIn("Recommended materials:", payload.description_text)
        logs: str = "\n".join(captured.output)
        self.assertIn("recommended_materials_final_count=0", logs)
        self.assertIn("recommended_materials_block=skipped", logs)
        self.assertIn("ignored_llm_youtube_urls=1", logs)


if __name__ == "__main__":
    unittest.main()
