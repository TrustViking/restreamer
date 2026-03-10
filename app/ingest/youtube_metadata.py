from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger
from app.core.models import VideoMetadata


LOGGER = get_logger(__name__)


class YouTubeMetadataFetcher:
    def fetch(self, video_url: str) -> VideoMetadata:
        raise NotImplementedError


class YtDlpYouTubeMetadataFetcher(YouTubeMetadataFetcher):
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._timeout_seconds: float = timeout_seconds

    def fetch(self, video_url: str) -> VideoMetadata:
        from yt_dlp import YoutubeDL

        ydl_options: Dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "socket_timeout": self._timeout_seconds,
            "extract_flat": False,
            "noplaylist": True,
        }
        with YoutubeDL(cast(Any, ydl_options)) as ydl:
            ydl_any: Any = ydl
            info: Dict[str, Any] = cast(
                Dict[str, Any],
                ydl_any.extract_info(video_url, download=False),
            )
        LOGGER.debug(
            "yt-dlp diagnostics: _type=%s id=%s webpage_url=%s",
            info.get("_type"),
            info.get("id"),
            info.get("webpage_url"),
        )
        title: str = str(info.get("title") or "").strip()
        description: str = str(info.get("description") or "").strip()
        thumbnail_url: str = str(info.get("thumbnail") or "").strip()
        video_id: str = str(info.get("id") or "").strip()
        youtube_language: Optional[str] = str(info.get("language") or "").strip() or None
        channel_language: Optional[str] = (
            str(info.get("channel_language") or "").strip() or None
        )
        duration_value: Any = info.get("duration")
        duration_seconds: Optional[int] = None
        if isinstance(duration_value, (int, float)) and duration_value > 0:
            duration_seconds = int(duration_value)
        canonical_url: str = (
            str(info.get("webpage_url") or info.get("original_url") or video_url).strip()
            or video_url
        )

        if not title:
            raise ValueError("Не удалось получить title (yt-dlp вернул пусто).")
        if not thumbnail_url:
            fallback_id: Optional[str] = None
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                fallback_id = video_id
            else:
                fallback_match: Optional[re.Match[str]] = re.search(
                    r"(?:youtu\.be/|v=)([A-Za-z0-9_-]{11})",
                    video_url,
                )
                if fallback_match:
                    fallback_id = fallback_match.group(1)

            if fallback_id:
                thumbnail_url = f"https://i.ytimg.com/vi/{fallback_id}/hqdefault.jpg"
                LOGGER.warning(
                    "yt-dlp returned empty thumbnail_url; using fallback video id %s: %s",
                    fallback_id,
                    thumbnail_url,
                )
            else:
                raise ValueError(
                    "Не удалось получить thumbnail_url (yt-dlp вернул пусто)."
                )
        if not description:
            LOGGER.warning(
                "yt-dlp returned empty description for url=%s (id=%s).",
                video_url,
                video_id or "unknown",
            )

        return VideoMetadata(
            url=video_url,
            title=title,
            description=description,
            thumbnail_url=thumbnail_url,
            youtube_language=youtube_language,
            channel_language=channel_language,
            duration_seconds=duration_seconds,
            canonical_url=canonical_url,
        )


def extract_youtube_video_id(raw_value: str) -> Optional[str]:
    text: str = str(raw_value or "").strip()
    if not text:
        return None
    context_patterns: Tuple[Tuple[str, str], ...] = (
        (
            "context",
            r"(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/watch\?[^#\s]*?(?:[?&]v=|&v=)([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
    )
    first_context_match: Optional[Tuple[int, str, str]] = None
    for method_name, pattern in context_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate: str = str(match.group(1) or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                continue
            candidate_pos: int = match.start(1)
            if first_context_match is None or candidate_pos < first_context_match[0]:
                first_context_match = (candidate_pos, method_name, candidate)
    if first_context_match is not None:
        _, method_name, candidate = first_context_match
        LOGGER.debug("youtube_id_extracted method=%s id=%s", method_name, candidate)
        return candidate

    for match in re.finditer(r"([A-Za-z0-9_-]{11})", text):
        candidate = str(match.group(1) or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            continue
        start_index: int = match.start(1)
        end_index: int = match.end(1)
        char_before: str = text[start_index - 1] if start_index > 0 else ""
        char_after: str = text[end_index] if end_index < len(text) else ""
        if char_before and re.fullmatch(r"[A-Za-z0-9_-]", char_before):
            continue
        if char_after and re.fullmatch(r"[A-Za-z0-9_-]", char_after):
            continue
        LOGGER.debug("youtube_id_extracted method=raw id=%s", candidate)
        return candidate
    return None


def normalize_youtube_link(raw: str) -> Optional[str]:
    video_id: Optional[str] = extract_youtube_video_id(raw)
    if not video_id:
        return None
    return f"https://youtu.be/{video_id}"


def normalize_youtube_video_url(video_url: str) -> str:
    normalized: Optional[str] = normalize_youtube_link(video_url)
    if not normalized:
        raise ValueError(f"Не удалось извлечь YouTube video id из URL: {video_url}")
    return normalized
