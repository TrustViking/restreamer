from __future__ import annotations

import logging
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from app.core.models import NormalizedImage, PreparedVideo, VideoMetadata
from app.google.drive_client import GoogleDriveClient, LOGGER as DRIVE_LOGGER
from app.planning.batch_planner import materialize_prepared_previews


class PreviewDedupeTests(unittest.TestCase):
    class _FakeDriveFilesResource:
        def __init__(self, files_payload: list[dict[str, object]]) -> None:
            self._files_payload = files_payload

        def list(self, **kwargs: object) -> "PreviewDedupeTests._FakeDriveFilesResource":
            return self

        def execute(self) -> dict[str, object]:
            return {"files": list(self._files_payload)}

    class _FakeDriveService:
        def __init__(self, files_payload: list[dict[str, object]]) -> None:
            self._files_resource = PreviewDedupeTests._FakeDriveFilesResource(files_payload)

        def files(self) -> "PreviewDedupeTests._FakeDriveFilesResource":
            return self._files_resource

    def _prepared_video(self) -> PreparedVideo:
        return PreparedVideo(
            row_number=1,
            original_link="https://youtu.be/abc",
            normalized_link="https://youtu.be/abc",
            date_raw="09.03.2026",
            time_raw="13:50",
            scheduled_at_kiev=datetime(2026, 3, 9, 13, 50, tzinfo=timezone.utc),
            date_key="090326",
            date_display="09.03.2026",
            language="en",
            metadata=VideoMetadata(
                url="https://youtu.be/abc",
                title="Example title",
                description="Example description",
                thumbnail_url="https://example.com/thumb.jpg",
                youtube_language="en",
            ),
            thumbnail=NormalizedImage(
                bytes_data=b"preview-bytes",
                extension=".jpg",
                mime_type="image/jpeg",
            ),
            local_thumbnail_path=None,
            merge_raw="",
            merge_languages=[],
        )

    def test_local_preview_duplicate_is_skipped_when_name_and_size_match(self) -> None:
        prepared_video = self._prepared_video()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "090326" / "en" / "1_en_example.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(prepared_video.thumbnail.bytes_data)
            with self.assertLogs("app.planning.batch_planner", level="INFO") as captured:
                materialize_prepared_previews(
                    logger=logging.getLogger("app.planning.batch_planner"),
                    config=SimpleNamespace(
                        google=SimpleNamespace(
                            drive_preview_folder_id=None,
                            drive_folder_id=None,
                            drive_preview_path_template="{language}/{date}",
                        ),
                    ),
                    drive_client=SimpleNamespace(),
                    name_builder=SimpleNamespace(build_image_path=lambda **kwargs: image_path),
                    prepared_videos=[prepared_video],
                    dry_run=True,
                )
            self.assertIn("preview_save_skipped_duplicate_local", "\n".join(captured.output))

    def test_drive_preview_duplicate_is_skipped_when_name_and_size_match(self) -> None:
        prepared_video = self._prepared_video()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "090326" / "en" / "1_en_example.jpg"
            drive_client = SimpleNamespace(
                ensure_folder_path=lambda **kwargs: "folder-1",
                find_file_by_name_and_size=lambda **kwargs: ("file-1", len(prepared_video.thumbnail.bytes_data)),
                upload_image_and_make_public=lambda **kwargs: (_ for _ in ()).throw(AssertionError("upload must be skipped")),
            )
            with self.assertLogs("app.planning.batch_planner", level="INFO") as captured:
                materialize_prepared_previews(
                    logger=logging.getLogger("app.planning.batch_planner"),
                    config=SimpleNamespace(
                        google=SimpleNamespace(
                            drive_preview_folder_id="preview-root",
                            drive_folder_id="preview-root",
                            drive_preview_path_template="{language}/{date}",
                        ),
                    ),
                    drive_client=drive_client,
                    name_builder=SimpleNamespace(build_image_path=lambda **kwargs: image_path),
                    prepared_videos=[prepared_video],
                    dry_run=False,
                )
            self.assertIn("preview_save_skipped_duplicate_drive", "\n".join(captured.output))

    def test_drive_duplicate_lookup_logs_for_duplicate_found(self) -> None:
        drive_client = GoogleDriveClient(
            drive_service=self._FakeDriveService(
                [{"id": "file-1", "name": "preview.jpg", "size": "123"}]
            )
        )
        with self.assertLogs(DRIVE_LOGGER.name, level="INFO") as captured:
            result = drive_client.find_file_by_name_and_size(
                folder_id="folder-1",
                file_name="preview.jpg",
                expected_size=123,
            )
        self.assertEqual(("file-1", 123), result)
        text = "\n".join(captured.output)
        self.assertIn("drive_preview_duplicate_lookup_started", text)
        self.assertIn("candidate_count=1", text)
        self.assertIn("status=duplicate_found", text)

    def test_drive_duplicate_lookup_logs_for_duplicate_not_found(self) -> None:
        drive_client = GoogleDriveClient(
            drive_service=self._FakeDriveService(
                [{"id": "file-1", "name": "preview.jpg", "size": "999"}]
            )
        )
        with self.assertLogs(DRIVE_LOGGER.name, level="INFO") as captured:
            result = drive_client.find_file_by_name_and_size(
                folder_id="folder-1",
                file_name="preview.jpg",
                expected_size=123,
            )
        self.assertIsNone(result)
        text = "\n".join(captured.output)
        self.assertIn("drive_preview_duplicate_lookup_started", text)
        self.assertIn("candidate_count=1", text)
        self.assertIn("status=duplicate_not_found", text)
        self.assertIn("candidate_sizes=999", text)


if __name__ == "__main__":
    unittest.main()
