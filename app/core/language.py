from __future__ import annotations

import re
from typing import Optional

from app.core.models import VideoMetadata


def normalize_language(raw_language: Optional[str]) -> Optional[str]:
    if not raw_language:
        return None
    normalized: str = raw_language.strip().lower()
    if normalized.startswith(("uk", "ua")):
        return "uk"
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("ru"):
        return "ru"
    return None


def detect_language_from_text(text: str) -> str:
    low: str = text.lower()
    if not low:
        return "other"
    if any(ch in low for ch in "іїєґ"):
        return "uk"

    cyrillic_count: int = len(re.findall(r"[а-яё]", low))
    latin_count: int = len(re.findall(r"[a-z]", low))
    if cyrillic_count > 0 and latin_count == 0:
        return "ru"
    if latin_count >= cyrillic_count and latin_count > 0:
        return "en"
    if cyrillic_count > latin_count:
        return "ru"
    return "other"


def detect_language(metadata: VideoMetadata) -> str:
    from_youtube: Optional[str] = normalize_language(metadata.youtube_language)
    if from_youtube:
        return from_youtube
    return detect_language_from_text(
        f"{metadata.title}\n{metadata.description}".strip()
    )
