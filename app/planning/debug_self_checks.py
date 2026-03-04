from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from typing import List

from app.core.models import NormalizedImage, PlannedVideo, VideoMetadata
from app.planning import (
    deduplicate_planned_videos_within_date_language,
    expand_sheet_range_to_af,
    parse_merge_languages,
    planned_video_block_language,
    sheet_range_includes_merge_column,
)


def run_debug_self_checks() -> None:
    _run_merge_column_self_test()
    _run_merge_range_semantics_and_dedup_self_test()


def _run_merge_column_self_test() -> None:
    assert parse_merge_languages("") == []
    assert parse_merge_languages("  ") == []
    assert parse_merge_languages("uk") == ["uk"]
    assert parse_merge_languages("ua,ru") == ["uk", "ru"]
    assert parse_merge_languages("ua|ru") == ["uk", "ru"]
    assert parse_merge_languages("ua;ru") == ["uk", "ru"]
    assert parse_merge_languages("RU,ua,en,ru") == ["uk", "en", "ru"]

    scheduled_at: datetime = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    base_video: PlannedVideo = PlannedVideo(
        row_number=10,
        original_link="https://youtu.be/AAAAAAAAAAA",
        normalized_link="https://youtu.be/AAAAAAAAAAA",
        scheduled_at_kiev=scheduled_at,
        date_key="220226",
        date_display="22.02.2026",
        language="en",
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
    )

    unchanged: List[PlannedVideo] = [base_video] + [
        dataclasses.replace(base_video, forced_block_language=language_code)
        for language_code in parse_merge_languages("")
    ]
    assert len(unchanged) == 1
    assert [planned_video_block_language(item) for item in unchanged] == ["en"]

    cloned: List[PlannedVideo] = [base_video] + [
        dataclasses.replace(base_video, forced_block_language=language_code)
        for language_code in parse_merge_languages("ua,ru")
    ]
    assert len(cloned) == 3
    assert [planned_video_block_language(item) for item in cloned] == [
        "en",
        "uk",
        "ru",
    ]

    base_block_lang: str = planned_video_block_language(base_video)
    clones_with_skip: List[PlannedVideo] = []
    for merge_language in parse_merge_languages("uk"):
        if merge_language == base_block_lang:
            continue
        clones_with_skip.append(
            dataclasses.replace(base_video, forced_block_language=merge_language)
        )
    assert len(clones_with_skip) == 1
    assert [planned_video_block_language(item) for item in clones_with_skip] == ["uk"]

    uk_base_video: PlannedVideo = dataclasses.replace(base_video, language="uk")
    uk_base_block_lang: str = planned_video_block_language(uk_base_video)
    uk_clones_with_skip: List[PlannedVideo] = []
    for merge_language in parse_merge_languages("uk"):
        if merge_language == uk_base_block_lang:
            continue
        uk_clones_with_skip.append(
            dataclasses.replace(uk_base_video, forced_block_language=merge_language)
        )
    assert len(uk_clones_with_skip) == 0


def _run_merge_range_semantics_and_dedup_self_test() -> None:
    assert sheet_range_includes_merge_column("A:F") is True
    assert sheet_range_includes_merge_column("A:D") is False
    assert sheet_range_includes_merge_column("Sheet1!A:D") is False
    assert expand_sheet_range_to_af("A:D") == "A:F"
    assert expand_sheet_range_to_af("Sheet1!A:D") == "Sheet1!A:F"

    scheduled_at: datetime = datetime(2026, 2, 24, 14, 0, tzinfo=timezone.utc)
    base_video: PlannedVideo = PlannedVideo(
        row_number=2,
        original_link="https://youtu.be/xg7W6wrOYBA",
        normalized_link="https://youtu.be/xg7W6wrOYBA",
        scheduled_at_kiev=scheduled_at,
        date_key="240226",
        date_display="24.02.2026",
        language="en",
        metadata=VideoMetadata(
            url="https://youtu.be/xg7W6wrOYBA",
            title="Base EN",
            description="Desc EN",
            thumbnail_url="https://i.ytimg.com/vi/xg7W6wrOYBA/hqdefault.jpg",
            youtube_language="en",
        ),
        thumbnail=NormalizedImage(
            bytes_data=b"x",
            extension="jpg",
            mime_type="image/jpeg",
        ),
        local_thumbnail_path=None,
    )
    merge_languages: List[str] = parse_merge_languages("ua,ru")

    override_items: List[PlannedVideo] = [base_video]
    base_block_lang: str = planned_video_block_language(base_video)
    for merge_language in merge_languages:
        if merge_language == base_block_lang:
            continue
        override_items.append(
            dataclasses.replace(base_video, forced_block_language=merge_language)
        )
    if (
        merge_languages
        and "override" == "override"
        and base_block_lang not in merge_languages
    ):
        override_items.pop(0)
    assert [planned_video_block_language(item) for item in override_items] == [
        "uk",
        "ru",
    ]

    add_items: List[PlannedVideo] = [base_video]
    for merge_language in merge_languages:
        if merge_language == base_block_lang:
            continue
        add_items.append(
            dataclasses.replace(base_video, forced_block_language=merge_language)
        )
    assert [planned_video_block_language(item) for item in add_items] == [
        "en",
        "uk",
        "ru",
    ]

    duplicate_row_video: PlannedVideo = dataclasses.replace(base_video, row_number=8)
    deduped_items: List[PlannedVideo] = deduplicate_planned_videos_within_date_language(
        [base_video, duplicate_row_video]
    )
    assert len(deduped_items) == 1
    assert deduped_items[0].row_number == 2
