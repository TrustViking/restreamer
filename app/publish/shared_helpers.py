"""Shared helpers used by both doc_helpers and telegram_renderer."""
from __future__ import annotations

import re
from typing import List, Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppTemplates
from app.core.constants import URL_PATTERN
from app.core.cta_detection import starts_with_cta_prefix
from app.core.env_flags import (
    strip_chapter_timestamps,
    strip_chapter_timestamps_enabled_from_env,
)
from app.core.models import MergedPublicationPayload, PlannedVideo
from app.core.description_cleaner import _looks_like_service_tail_paragraph
from app.core.text_utils import is_youtube_url as _is_youtube_url
from app.core.text_utils import split_paragraphs
from app.core.url_normalizer import normalize_display_url
from app.ingest.youtube_metadata import normalize_youtube_video_url
from app.resources import promotional_opener_phrases


LOGGER = _get_logger_impl(__name__)


def numbered_lines(values: Sequence[str]) -> str:
    """Format a sequence of strings as numbered lines or single value."""
    cleaned_values: List[str] = [
        str(item or "").strip() for item in values if str(item or "").strip()
    ]
    if not cleaned_values:
        return "1) ..."
    if len(cleaned_values) == 1:
        return cleaned_values[0]
    return "\n".join(
        [f"{index}) {item}" for index, item in enumerate(cleaned_values, start=1)]
    )


def numbered_original_titles(videos: List[PlannedVideo]) -> str:
    """Format source video titles as numbered lines."""
    source_titles: List[str] = [video.metadata.title for video in videos]
    return numbered_lines(source_titles)


def is_merge_payload_blocked(payload: MergedPublicationPayload) -> bool:
    """Check if a merged payload is blocked by publish-stage quality gates."""
    has_publish_stage_duplicate: bool = bool(
        getattr(payload, "has_publish_stage_duplicate", False)
    )
    has_publish_stage_opener_cta: bool = bool(
        getattr(payload, "has_publish_stage_opener_cta", False)
    )
    return has_publish_stage_duplicate or has_publish_stage_opener_cta


def no_description_text(templates: Optional[AppTemplates] = None) -> str:
    """Return the 'no description' placeholder text from templates."""
    if templates is None:
        return "no description"
    value: str = str(templates.common_no_description_text or "").strip()
    return value or "no description"


def _looks_like_promotional_opener(text: str) -> bool:
    normalized: str = re.sub(r"\s+", " ", str(text or "").strip()).casefold()
    if not normalized:
        return False
    return any(phrase in normalized for phrase in promotional_opener_phrases())


def normalize_urls_in_text(text: str) -> str:
    """Normalize all URLs in text."""

    def _replace_url(match: re.Match[str]) -> str:
        matched_url: str = str(match.group(0) or "")
        if _is_youtube_url(matched_url):
            try:
                return normalize_youtube_video_url(matched_url)
            except (ValueError, Exception):
                return matched_url
        return normalize_display_url(matched_url)

    return URL_PATTERN.sub(_replace_url, text)


def light_polish_single_source_description(text: str) -> str:
    """Apply light polish to a single-source description."""
    original_text: str = str(text or "")
    paragraphs: List[str] = split_paragraphs(original_text)
    if not paragraphs:
        return original_text

    cleaned_paragraphs: List[str] = list(paragraphs)
    if cleaned_paragraphs and (
        starts_with_cta_prefix(cleaned_paragraphs[0])
        or _looks_like_promotional_opener(cleaned_paragraphs[0])
    ):
        cleaned_paragraphs = cleaned_paragraphs[1:]
    while cleaned_paragraphs and _looks_like_service_tail_paragraph(cleaned_paragraphs[-1]):
        cleaned_paragraphs = cleaned_paragraphs[:-1]

    polished_text: str = "\n\n".join(cleaned_paragraphs).strip()
    if not polished_text:
        return original_text
    polished_text = normalize_urls_in_text(polished_text)
    return polished_text


def fallback_source_description_text(
    video: PlannedVideo,
    templates: Optional[AppTemplates],
) -> str:
    """Build fallback source description for a single video."""
    description_text: str = video.metadata.description.strip() or no_description_text(
        templates
    )
    if strip_chapter_timestamps_enabled_from_env():
        description_text = strip_chapter_timestamps(description_text)
    description_text = light_polish_single_source_description(description_text)
    return description_text
