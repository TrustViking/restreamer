from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List
from zoneinfo import ZoneInfo

from app.core.models import PlannedVideo
from app.planning import planned_video_block_language


def build_header_context(
    videos: List[PlannedVideo],
    form_url: str,
    contacts: str,
    cet_tz: ZoneInfo,
) -> Dict[str, str]:
    if not videos:
        raise ValueError("videos must not be empty")

    first_stream: PlannedVideo = min(
        videos,
        key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
    )
    stream_dt_kiev: datetime = first_stream.scheduled_at_kiev
    stream_dt_cet: datetime = stream_dt_kiev.astimezone(cet_tz)
    stream_dt_gmt: datetime = stream_dt_kiev.astimezone(timezone.utc)
    kiev_minus_1: datetime = stream_dt_kiev - timedelta(hours=1)
    gmt_minus_1: datetime = stream_dt_gmt - timedelta(hours=1)

    times_by_language: Dict[str, str] = {"uk": "", "en": "", "ru": ""}
    for language in ("uk", "en", "ru"):
        same_language: List[PlannedVideo] = [
            item for item in videos if planned_video_block_language(item) == language
        ]
        if not same_language:
            continue
        unique_times: List[str] = sorted(
            {item.scheduled_at_kiev.strftime("%H:%M") for item in same_language}
        )
        times_by_language[language] = ", ".join(unique_times)

    return {
        "date": stream_dt_kiev.strftime("%d.%m.%Y"),
        "time_cet": stream_dt_cet.strftime("%H:%M"),
        "time_kiev": stream_dt_kiev.strftime("%H:%M"),
        "time_gmt": stream_dt_gmt.strftime("%H:%M"),
        "time_kiev_minus_1": kiev_minus_1.strftime("%H:%M"),
        "time_gmt_minus_1": gmt_minus_1.strftime("%H:%M"),
        "time_ukr": times_by_language["uk"],
        "time_eng": times_by_language["en"],
        "time_ru": times_by_language["ru"],
        "form_url": form_url,
        "contacts": contacts,
    }
