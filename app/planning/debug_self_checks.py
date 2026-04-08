from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from app.core.models import NormalizedImage, PlannedVideo, VideoMetadata
from app.planning import (
    deduplicate_planned_videos_within_date_language,
    planned_video_block_language,
)


def run_debug_self_checks() -> None:
    _run_block_language_self_test()
    _run_dedup_self_test()


def _make_test_video(
    *,
    row_number: int = 2,
    language: str = "en",
    forced_block_language: str | None = None,
) -> PlannedVideo:
    scheduled_at: datetime = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    return PlannedVideo(
        row_number=row_number,
        original_link="https://youtu.be/AAAAAAAAAAA",
        normalized_link="https://youtu.be/AAAAAAAAAAA",
        scheduled_at_kiev=scheduled_at,
        date_key="220226",
        date_display="22.02.2026",
        language=language,
        metadata=VideoMetadata(
            url="https://youtu.be/AAAAAAAAAAA",
            title="Base title",
            description="Base description",
            thumbnail_url="https://i.ytimg.com/vi/AAAAAAAAAAA/hqdefault.jpg",
            youtube_language="en",
        ),
        thumbnail=NormalizedImage(
            bytes_data=b"x",
            extension="jpg",
            mime_type="image/jpeg",
        ),
        local_thumbnail_path=None,
        forced_block_language=forced_block_language,
    )


def _run_block_language_self_test() -> None:
    video_en: PlannedVideo = _make_test_video(language="en")
    assert planned_video_block_language(video_en) == "en"

    video_forced_uk: PlannedVideo = _make_test_video(
        language="en",
        forced_block_language="uk",
    )
    assert planned_video_block_language(video_forced_uk) == "uk"


def _run_dedup_self_test() -> None:
    video_a: PlannedVideo = _make_test_video(row_number=2)
    video_b: PlannedVideo = _make_test_video(row_number=8)
    deduped: List[PlannedVideo] = deduplicate_planned_videos_within_date_language(
        [video_a, video_b]
    )
    assert len(deduped) == 1
    assert deduped[0].row_number == 2
