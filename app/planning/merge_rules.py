from __future__ import annotations

import os
from typing import List, Optional

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import PlannedVideo


LOGGER = _get_logger_impl(__name__)


def parse_merge_languages(value: str) -> List[str]:
    stable_order: List[str] = ["uk", "en", "ru"]
    token_map: dict[str, str] = {
        "uk": "uk",
        "ua": "uk",
        "ukr": "uk",
        "en": "en",
        "eng": "en",
        "ru": "ru",
        "rus": "ru",
    }
    selected: set[str] = set()
    raw_value: str = str(value or "").strip()
    if not raw_value:
        return []
    for raw_token in raw_value.replace("|", ",").replace(";", ",").split(","):
        token: str = raw_token.strip().lower()
        if not token:
            continue
        mapped: Optional[str] = token_map.get(token)
        if mapped:
            selected.add(mapped)
    return [language for language in stable_order if language in selected]


def merge_semantics_from_env() -> str:
    raw_value: str = str(os.getenv("STG_MERGE_SEMANTICS", "override") or "").strip()
    normalized_value: str = raw_value.lower()
    if normalized_value in {"override", "add"}:
        return normalized_value
    LOGGER.warning(
        "Unknown STG_MERGE_SEMANTICS=%r; falling back to override.",
        raw_value,
    )
    return "override"


def planned_video_block_language(video: PlannedVideo) -> str:
    if video.forced_block_language in {"uk", "en", "ru", "other"}:
        return str(video.forced_block_language)
    return video.language
