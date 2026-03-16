from __future__ import annotations

import logging
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    NormalizedImage,
    PlannedVideo,
    VideoMetadata,
)
from app.pipeline.daily_doc_publish import publish_daily_document
from app.pipeline.slot_processing import SlotProcessResult


class DailyDocPublishTests(unittest.TestCase):
    def _video(self) -> PlannedVideo:
        scheduled_at = datetime(2026, 3, 11, 9, 0, tzinfo=ZoneInfo("Europe/Bucharest"))
        return PlannedVideo(
            row_number=1,
            original_link="https://youtu.be/abc",
            normalized_link="https://youtu.be/abc",
            scheduled_at_kiev=scheduled_at,
            date_key="2026-03-11",
            date_display="2026-03-11",
            language="uk",
            metadata=VideoMetadata(
                url="https://youtu.be/abc",
                title="Title",
                description="Description",
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

    def test_first_table_uses_regular_writer_path_without_special_insert_index(self) -> None:
        docs_client = MagicMock()
        docs_client.create_document.return_value = "doc-123"
        docs_client.get_document_end_index.return_value = 10

        drive_client = MagicMock()
        report_writer = MagicMock()
        name_builder = MagicMock()
        name_builder.build_doc_title.return_value = "Daily doc"
        name_builder.build_docx_path.return_value = None

        config = SimpleNamespace(
            templates=SimpleNamespace(
                google_doc_language_headings_json='{"uk":"UK"}',
                google_doc_header="HEADER",
            ),
            llm_model="gpt-test",
            google_doc_share_mode="private",
            google_drive_folder_id="",
        )

        video = self._video()
        slot_result = SlotProcessResult(
            slot_key="2026-03-11_09-00",
            slot_time_key="09:00",
            header_context={},
            day_videos=[video],
            language_groups={"uk": [video], "en": [], "ru": [], "other": []},
            merged_content_by_language={},
            merge_audit_by_language={},
            real_merge_blocks=0,
        )

        publish_daily_document(
            logger=logging.getLogger("test.daily_doc_publish"),
            config=config,
            docs_client=docs_client,
            drive_client=drive_client,
            report_writer=report_writer,
            name_builder=name_builder,
            slot_results=[slot_result],
            date_key="2026-03-11",
            dry_run=False,
            processing_mode="nomerge",
            kiev_tz=ZoneInfo("Europe/Bucharest"),
            branch_label="nomerge",
        )

        self.assertEqual(1, docs_client.get_document_end_index.call_count)
        first_call = report_writer.write_language_table.call_args_list[0]
        self.assertNotIn("table_insert_index", first_call.kwargs)

    def test_partial_fallback_block_is_marked_in_header_and_publish_logs(self) -> None:
        docs_client = MagicMock()
        docs_client.create_document.return_value = "doc-123"
        docs_client.get_document_end_index.return_value = 10

        drive_client = MagicMock()
        report_writer = MagicMock()
        name_builder = MagicMock()
        name_builder.build_doc_title.return_value = "Daily doc"
        name_builder.build_docx_path.return_value = None

        config = SimpleNamespace(
            templates=SimpleNamespace(
                google_doc_language_headings_json='{"uk":"UK"}',
                google_doc_header="{language_time_titles}",
            ),
            llm_model="gpt-test",
            google_doc_share_mode="private",
            google_drive_folder_id="",
        )
        video = self._video()
        slot_result = SlotProcessResult(
            slot_key="2026-03-11_09-00",
            slot_time_key="09:00",
            header_context={},
            day_videos=[video],
            language_groups={"uk": [video], "en": [], "ru": [], "other": []},
            merged_content_by_language={},
            merge_audit_by_language={
                "uk": SimpleNamespace(
                    language="uk",
                    model_name="gpt-test",
                    generator_model_name="gpt-test",
                    used_model_names=("gpt-test",),
                    title_source="fallback_titles",
                    hook_source="fallback_none",
                    hashtags_source="fallback_none",
                    body_source="fallback_source_descriptions",
                    block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
                    salvaged_title=None,
                )
            },
            real_merge_blocks=1,
            merge_candidate_blocks=1,
            fallback_merge_blocks=1,
            merge_artifact_status="partial",
            fallback_merge_targets=("2026-03-11_09-00:uk",),
        )

        with self.assertLogs(level="INFO") as captured:
            publish_daily_document(
                logger=logging.getLogger("test.daily_doc_publish.partial"),
                config=config,
                docs_client=docs_client,
                drive_client=drive_client,
                report_writer=report_writer,
                name_builder=name_builder,
                slot_results=[slot_result],
                date_key="2026-03-11",
                dry_run=False,
                processing_mode="merge",
                kiev_tz=ZoneInfo("Europe/Bucharest"),
                branch_label="merge",
            )

        header_text = report_writer.write_header_only.call_args.kwargs["header_text"]
        self.assertIn("UK - 09:00 (Partial fallback)", header_text)
        logs: str = "\n".join(captured.output)
        self.assertIn("merge_block_publish_truth", logs)
        self.assertIn("block_generation_mode=fallback_after_merge_failure", logs)
        self.assertIn("artifact_marker_applied=yes", logs)
        self.assertIn("title_source=fallback_titles", logs)
        self.assertIn("body_source=fallback_source_descriptions", logs)

    def test_fallback_only_block_is_marked_as_rejected_fallback_in_header_and_logs(self) -> None:
        docs_client = MagicMock()
        docs_client.create_document.return_value = "doc-123"
        docs_client.get_document_end_index.return_value = 10

        drive_client = MagicMock()
        report_writer = MagicMock()
        name_builder = MagicMock()
        name_builder.build_doc_title.return_value = "Daily doc"
        name_builder.build_docx_path.return_value = None

        config = SimpleNamespace(
            templates=SimpleNamespace(
                google_doc_language_headings_json='{"uk":"UK"}',
                google_doc_header="{language_time_titles}",
            ),
            llm_model="gpt-test",
            google_doc_share_mode="private",
            google_drive_folder_id="",
        )
        video = self._video()
        slot_result = SlotProcessResult(
            slot_key="2026-03-11_09-00",
            slot_time_key="09:00",
            header_context={},
            day_videos=[video],
            language_groups={"uk": [video], "en": [], "ru": [], "other": []},
            merged_content_by_language={},
            merge_audit_by_language={
                "uk": SimpleNamespace(
                    language="uk",
                    model_name="gpt-test",
                    generator_model_name="gpt-test",
                    used_model_names=("gpt-test",),
                    title_source="fallback_titles",
                    hook_source="fallback_none",
                    hashtags_source="fallback_none",
                    body_source="fallback_source_descriptions",
                    block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
                    salvaged_title=None,
                )
            },
            real_merge_blocks=0,
            merge_candidate_blocks=1,
            fallback_merge_blocks=1,
            merge_artifact_status="fallback_only",
            fallback_merge_targets=("2026-03-11_09-00:uk",),
        )

        with self.assertLogs(level="INFO") as captured:
            publish_daily_document(
                logger=logging.getLogger("test.daily_doc_publish.fallback_only"),
                config=config,
                docs_client=docs_client,
                drive_client=drive_client,
                report_writer=report_writer,
                name_builder=name_builder,
                slot_results=[slot_result],
                date_key="2026-03-11",
                dry_run=False,
                processing_mode="merge",
                kiev_tz=ZoneInfo("Europe/Bucharest"),
                branch_label="merge",
            )

        header_text = report_writer.write_header_only.call_args.kwargs["header_text"]
        self.assertIn(
            "UK - 09:00 (Merge rejected fallback artifact)",
            header_text,
        )
        logs: str = "\n".join(captured.output)
        self.assertIn("merge_artifact_status=fallback_only", logs)
        self.assertIn("fallback_targets=2026-03-11_09-00:uk", logs)
        self.assertIn("artifact_marker_applied=yes", logs)


if __name__ == "__main__":
    unittest.main()
