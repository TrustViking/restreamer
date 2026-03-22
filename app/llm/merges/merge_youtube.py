from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence
from urllib.parse import urlsplit

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import PlannedVideo, VideoMetadata
from app.ingest.youtube_metadata import YtDlpYouTubeMetadataFetcher, normalize_youtube_link
from app.llm.merges.merge_constants import URL_PATTERN

LOGGER = _get_logger_impl(__name__)

@dataclass(frozen=True)
class MergeYouTubeCandidate:
    url: str
    title: str
    duration_seconds: Optional[int]
    duration_label: str
    metadata_resolved: bool

@dataclass(frozen=True)
class MergeYouTubeCandidatesResult:
    raw_youtube_urls_found: int
    invalid_youtube_urls_skipped: int
    deduped_candidates: tuple[MergeYouTubeCandidate, ...]
    metadata_resolved_count: int

def _is_youtube_host(host: str) -> bool:
    normalized_host: str = str(host or "").strip().lower()
    if not normalized_host:
        return False
    return (
        normalized_host.endswith("youtube.com")
        or normalized_host.endswith("youtu.be")
        or normalized_host.endswith("youtube-nocookie.com")
    )

def _format_duration_label(duration_seconds: Optional[int]) -> str:
    if duration_seconds is None or duration_seconds <= 0:
        return "unknown"
    total_seconds: int = int(duration_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"

def _fetch_merge_youtube_candidate_metadata(video_url: str) -> VideoMetadata:
    fetcher = YtDlpYouTubeMetadataFetcher()
    return fetcher.fetch(video_url)

def _extract_merge_youtube_candidates(
    videos: Sequence[PlannedVideo],
) -> MergeYouTubeCandidatesResult:
    raw_youtube_urls_found: int = 0
    invalid_youtube_urls_skipped: int = 0
    candidate_urls_by_key: dict[str, str] = {}

    for video in videos:
        description_text: str = str(video.metadata.description or "")
        for match in URL_PATTERN.finditer(description_text):
            raw_url: str = str(match.group(0) or "").strip()
            if not raw_url:
                continue
            try:
                netloc: str = urlsplit(raw_url).netloc
            except Exception:
                netloc = ""
            if not _is_youtube_host(netloc):
                continue
            raw_youtube_urls_found += 1
            normalized_url: Optional[str] = normalize_youtube_link(raw_url)
            if normalized_url is None:
                invalid_youtube_urls_skipped += 1
                continue
            candidate_urls_by_key.setdefault(normalized_url, normalized_url)

    candidates: List[MergeYouTubeCandidate] = []
    metadata_resolved_count: int = 0
    for normalized_url in candidate_urls_by_key.values():
        candidate_title: str = ""
        candidate_duration_seconds: Optional[int] = None
        candidate_url: str = normalized_url
        metadata_resolved: bool = False
        try:
            metadata: VideoMetadata = _fetch_merge_youtube_candidate_metadata(normalized_url)
            candidate_title = str(metadata.title or "").strip()
            candidate_duration_seconds = metadata.duration_seconds
            candidate_url = (
                normalize_youtube_link(
                    str(metadata.canonical_url or metadata.url or normalized_url).strip()
                )
                or normalized_url
            )
            metadata_resolved = True
            metadata_resolved_count += 1
        except Exception as error:
            LOGGER.info(
                "merge_youtube_candidate_metadata_failed url=%s reason=%s",
                normalized_url,
                error,
            )
        candidates.append(
            MergeYouTubeCandidate(
                url=candidate_url,
                title=candidate_title or "Unknown YouTube stream",
                duration_seconds=candidate_duration_seconds,
                duration_label=_format_duration_label(candidate_duration_seconds),
                metadata_resolved=metadata_resolved,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.duration_seconds is not None,
            candidate.duration_seconds or -1,
            candidate.title.lower(),
            candidate.url,
        ),
        reverse=True,
    )
    return MergeYouTubeCandidatesResult(
        raw_youtube_urls_found=raw_youtube_urls_found,
        invalid_youtube_urls_skipped=invalid_youtube_urls_skipped,
        deduped_candidates=tuple(candidates),
        metadata_resolved_count=metadata_resolved_count,
    )

def _build_youtube_candidates_block(
    candidates_result: MergeYouTubeCandidatesResult,
) -> str:
    if not candidates_result.deduped_candidates:
        return (
            "YOUTUBE CANDIDATES (optional)\n"
            "No valid YouTube candidates were found in merged source descriptions."
        )
    block_lines: List[str] = [
        "YOUTUBE CANDIDATES (optional)",
        "You may include 0, 1, or 2 YouTube URLs from this list only.",
        "Choosing none is valid.",
        "Pick only the most relevant main streams for the final summary.",
        "Prefer longer broadcasts. Do not pick short promo, teaser, clip, or secondary videos.",
        "If you include selected YouTube URLs, place each selected URL on its own line near the end of the description before any official links block or CTA.",
    ]
    for index, candidate in enumerate(candidates_result.deduped_candidates, start=1):
        block_lines.extend(
            (
                f"CANDIDATE {index}",
                f"TITLE: {candidate.title}",
                f"DURATION_SECONDS: {candidate.duration_seconds if candidate.duration_seconds is not None else 'unknown'}",
                f"DURATION_TEXT: {candidate.duration_label}",
                f"URL: {candidate.url}",
            )
        )
    return "\n".join(block_lines).strip()

def _count_youtube_urls_in_text(text: str) -> int:
    seen_urls: set[str] = set()
    for match in URL_PATTERN.finditer(str(text or "")):
        raw_url: str = str(match.group(0) or "").strip()
        if not raw_url:
            continue
        try:
            netloc: str = urlsplit(raw_url).netloc
        except Exception:
            netloc = ""
        if not _is_youtube_host(netloc):
            continue
        normalized_url: Optional[str] = normalize_youtube_link(raw_url)
        if normalized_url is not None:
            seen_urls.add(normalized_url)
    return len(seen_urls)
