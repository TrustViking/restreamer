from __future__ import annotations

import re
from typing import Tuple

from app.core.models import PlannedVideo
from app.planning.merge_rules import planned_video_block_language


def planned_video_time_key(video: PlannedVideo) -> str:
    return video.scheduled_at_kiev.strftime("%H%M")


def planned_video_slot_key(video: PlannedVideo) -> str:
    return f"{video.date_key}_{planned_video_time_key(video)}"


def format_time_key_for_display(time_key: str) -> str:
    stripped: str = time_key.strip()
    if re.fullmatch(r"\d{4}", stripped):
        return f"{stripped[:2]}:{stripped[2:]}"
    return stripped


def language_index(language: str) -> int:
    order: Tuple[str, ...] = ("uk", "en", "ru", "other")
    try:
        return order.index(language)
    except ValueError:
        return len(order)


__all__ = [
    "format_time_key_for_display",
    "language_index",
    "planned_video_block_language",
    "planned_video_slot_key",
    "planned_video_time_key",
]
