from __future__ import annotations

from app.core.models import PlannedVideo


def planned_video_block_language(video: PlannedVideo) -> str:
    forced_raw = getattr(video, "forced_block_language", None)
    forced: str = str(forced_raw or "").strip().lower()
    if forced:
        return forced
    language_raw = getattr(video, "language", None)
    language: str = str(language_raw or "").strip().lower()
    return language or "unknown"
