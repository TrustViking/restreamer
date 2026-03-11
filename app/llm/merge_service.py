from __future__ import annotations

import dataclasses
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE
from app.core.models import LanguageMergeAttempt, MergedLanguageContent, PlannedVideo, VideoMetadata
from app.ingest.youtube_metadata import YtDlpYouTubeMetadataFetcher, normalize_youtube_link
from app.llm.merge_polish import (
    MergePolishResult,
    build_merge_polish_prompt,
    bullet_marker_count,
    infer_polish_reject_reason,
    is_too_aggressive_rewrite,
)
from app.llm.merge_quality import (
    MergeQualityDiagnostics,
    MergeQualityNormalizationResult,
    normalize_merge_description,
)
from app.llm.merge_parser import (
    build_plain_merged_content_or_raise,
    clean_and_validate_llm_description,
    parse_merge_response_or_raise,
    separate_merge_body_and_tail,
)
from app.llm.merge_run_summary import MergeRunSummary
from app.llm.openai_client import LlmTraceContext, OpenAITransportResult, openai_request_merge
from app.llm.provider_factory import get_llm_provider
from app.llm.providers.base import LlmProvider

LOGGER = _get_logger_impl(__name__)
PRIMARY_ATTEMPTS: int = 2
STYLE_CONTRACT_VERSION: str = "v4_merge_quality_hardening"
_SEMANTIC_BULLET_MARKERS: tuple[str, ...] = ("🔹", "📌", "🎤", "🎥", "⚖", "🌐", "✅")
_SEMANTIC_BULLET_MARKER_SET: set[str] = set(_SEMANTIC_BULLET_MARKERS)
_BULLET_PLAIN_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
)
_URL_PATTERN: re.Pattern[str] = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_TRACKING_QUERY_KEYS: tuple[str, ...] = (
    "si",
    "feature",
    "pp",
    "fbclid",
    "gclid",
    "igsh",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref_src",
    "ref_url",
    "spm",
)
_OFFICIAL_LINK_CONTEXT_HINTS: tuple[str, ...] = (
    "official",
    "website",
    "initiative",
    "resource",
    "resources",
    "conference",
    "more information",
    "details",
    "site",
    "official links",
    "офіцій",
    "ініціатив",
    "ресурс",
    "сайт",
    "официал",
)
_SEMANTIC_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]{3,}",
    flags=re.UNICODE,
)
_SEMANTIC_STOPWORDS: set[str] = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "among",
    "and",
    "around",
    "because",
    "before",
    "between",
    "brief",
    "call",
    "conversation",
    "cover",
    "discussion",
    "during",
    "each",
    "from",
    "into",
    "more",
    "most",
    "other",
    "over",
    "stream",
    "talk",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "today",
    "topic",
    "topics",
    "update",
    "updates",
    "with",
    "будет",
    "более",
    "важный",
    "вместе",
    "всем",
    "всех",
    "главном",
    "диалог",
    "для",
    "день",
    "его",
    "или",
    "как",
    "который",
    "людей",
    "материал",
    "между",
    "миру",
    "наша",
    "наши",
    "нем",
    "них",
    "новый",
    "новости",
    "обзор",
    "общем",
    "почему",
    "разговор",
    "сегодня",
    "событие",
    "среди",
    "тема",
    "темы",
    "эфир",
    "этот",
    "важлива",
    "всіх",
    "головне",
    "діалог",
    "людей",
    "матеріал",
    "наші",
    "новий",
    "новини",
    "огляд",
    "подія",
    "потік",
    "розмова",
    "сьогодні",
    "теми",
    "цей",
    "ефір",
}


@dataclass(frozen=True)
class MergeAttemptFailure(RuntimeError):
    reason_code: str
    reason: str
    model_name: str
    attempt_stage: str
    raw_response_text: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True)
class ParagraphEnforcementResult:
    description_text: str
    mutated: bool
    recovery_applied: bool
    note: str
    body_paragraphs_before: int
    body_paragraphs_after: int


@dataclass(frozen=True)
class MergeSemanticDiagnostics:
    hook_present: bool
    agenda_block_present: bool
    bullet_points_count: int
    semantic_bullets_count: int
    bullets_with_emoji_count: int
    bullets_with_plain_marker_count: int
    bullet_marker_types: tuple[str, ...]
    named_entities_preserved: int
    source_named_entities_total: int
    emoji_count: int
    source_coverage_hits: int
    source_coverage_total: int
    source_coverage_by_item: tuple[bool, ...]
    official_links_found_in_sources: int
    official_links_kept: int
    official_links_in_output: int
    official_links_fill_applied: bool
    merge_quality: MergeQualityDiagnostics


@dataclass(frozen=True)
class OfficialLinksSelection:
    found_in_sources: int
    kept_links: tuple[str, ...]


@dataclass(frozen=True)
class OfficialLinksFillResult:
    description: str
    links_in_output: int
    fill_applied: bool


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


@dataclass(frozen=True)
class PreparedMergeSourceDescription:
    text: str
    raw_chars: int
    cleaned_chars: int
    urls_removed: int
    hashtags_removed: int
    service_paragraphs_dropped: int


def _reason_code_from_error(error: Exception) -> str:
    error_text: str = str(error or "")
    if "not a valid single JSON object" in error_text:
        return "not_json_object"
    if error_text.startswith("missing_keys:"):
        return "missing_keys"
    if error_text.startswith("extra_keys:"):
        return "extra_keys"
    if "forbidden_key:" in error_text:
        return "forbidden_variants"
    if (
        "title must be a non-empty string" in error_text
        or "title is invalid" in error_text
        or "title must not contain emoji" in error_text
    ):
        return "invalid_title"
    if (
        "description must be a non-empty string" in error_text
        or "description paragraph count" in error_text
        or "description body paragraph count" in error_text
        or "description validation failed" in error_text
    ):
        return "invalid_description"
    return "unexpected_error"


def _language_name_for_merge_prompt(language: str, llm_language_names_json: str) -> str:
    default_names: dict[str, str] = {
        "uk": "Ukrainian",
        "en": "English",
        "ru": "Russian",
        "other": "the original language of sources",
    }
    try:
        import json

        payload = json.loads(llm_language_names_json)
        if isinstance(payload, dict):
            return str(payload.get(language, payload.get("other", default_names["other"])))
    except Exception:
        pass
    return default_names.get(language, default_names["other"])


def _normalize_source_description_text(text: str) -> str:
    normalized: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _strip_source_urls_from_text(text: str) -> tuple[str, int]:
    removed_urls: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed_urls
        removed_urls += 1
        return ""

    cleaned_text: str = _URL_PATTERN.sub(_replace, str(text or ""))
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return (cleaned_text.strip(" ,;:-"), removed_urls)


def _strip_source_hashtags_from_text(text: str) -> tuple[str, int]:
    hashtag_pattern: re.Pattern[str] = re.compile(r"(?<!\w)#[^\s#]+", flags=re.UNICODE)
    removed_hashtags: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed_hashtags
        removed_hashtags += 1
        return ""

    cleaned_text: str = hashtag_pattern.sub(_replace, str(text or ""))
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return (cleaned_text.strip(" ,;:-"), removed_hashtags)


def _is_official_links_heading_line(text: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?im)(?:🌐\s*)?(?:official links|офіційні ресурси|официальные ссылки)\s*:",
            str(text or "").strip(),
        )
    )


def _looks_like_service_tail_paragraph(text: str) -> bool:
    normalized_text: str = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not normalized_text:
        return True
    if _is_official_links_heading_line(normalized_text):
        return True
    service_hints: tuple[str, ...] = (
        "watch",
        "join",
        "share",
        "follow",
        "subscribe",
        "learn more",
        "links below",
        "details below",
        "диві",
        "долуч",
        "підпис",
        "смотрите",
        "подпис",
        "подробности",
    )
    semantic_tokens: List[str] = _SEMANTIC_TOKEN_PATTERN.findall(normalized_text)
    return (
        len(normalized_text) <= 220
        and len(semantic_tokens) <= 12
        and any(hint in normalized_text for hint in service_hints)
    )


def _clean_source_description_for_llm(text: str) -> PreparedMergeSourceDescription:
    normalized_text: str = _normalize_source_description_text(text)
    raw_chars: int = len(normalized_text)
    if not normalized_text:
        return PreparedMergeSourceDescription(
            text="",
            raw_chars=0,
            cleaned_chars=0,
            urls_removed=0,
            hashtags_removed=0,
            service_paragraphs_dropped=0,
        )

    cleaned_paragraphs: List[str] = []
    urls_removed: int = 0
    hashtags_removed: int = 0
    service_paragraphs_dropped: int = 0

    for paragraph in _extract_description_paragraphs_raw(normalized_text):
        cleaned_lines: List[str] = []
        for raw_line in str(paragraph or "").split("\n"):
            line: str = str(raw_line or "").strip()
            if not line:
                continue
            cleaned_line, line_urls_removed = _strip_source_urls_from_text(line)
            cleaned_line, line_hashtags_removed = _strip_source_hashtags_from_text(
                cleaned_line
            )
            urls_removed += line_urls_removed
            hashtags_removed += line_hashtags_removed
            cleaned_line = re.sub(r"\s{2,}", " ", cleaned_line).strip(" ,;:-")
            if not cleaned_line or _is_official_links_heading_line(cleaned_line):
                continue
            cleaned_lines.append(cleaned_line)

        cleaned_paragraph: str = "\n".join(cleaned_lines).strip()
        if not cleaned_paragraph:
            service_paragraphs_dropped += 1
            continue
        cleaned_paragraphs.append(cleaned_paragraph)

    while cleaned_paragraphs and _looks_like_service_tail_paragraph(cleaned_paragraphs[-1]):
        cleaned_paragraphs.pop()
        service_paragraphs_dropped += 1

    cleaned_text: str = "\n\n".join(
        paragraph for paragraph in cleaned_paragraphs if paragraph.strip()
    ).strip()
    return PreparedMergeSourceDescription(
        text=cleaned_text,
        raw_chars=raw_chars,
        cleaned_chars=len(cleaned_text),
        urls_removed=urls_removed,
        hashtags_removed=hashtags_removed,
        service_paragraphs_dropped=service_paragraphs_dropped,
    )


def _extract_description_paragraphs_raw(text: str) -> List[str]:
    normalized_text: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized_text) if part.strip()]


def _count_emoji(text: str) -> int:
    emoji_pattern: re.Pattern[str] = re.compile(
        r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(str(text or "")))


def _bullet_marker_for_line(line: str) -> str:
    stripped: str = str(line or "").strip()
    if not stripped:
        return ""
    for marker in _SEMANTIC_BULLET_MARKERS:
        if stripped.startswith(f"{marker} "):
            return marker
    if _BULLET_PLAIN_PATTERN.match(stripped):
        first_token: str = stripped.split(maxsplit=1)[0]
        return first_token
    return ""


def _is_youtube_host(host: str) -> bool:
    normalized_host: str = str(host or "").strip().lower()
    if not normalized_host:
        return False
    return (
        normalized_host.endswith("youtube.com")
        or normalized_host.endswith("youtu.be")
        or normalized_host.endswith("youtube-nocookie.com")
    )


def _normalize_link_candidate(url: str) -> Optional[str]:
    raw_url: str = str(url or "").strip().strip("<>()[]{}").rstrip(".,;")
    if not raw_url:
        return None
    parts = urlsplit(raw_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    filtered_query_items: List[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key: str = key.lower().strip()
        if normalized_key.startswith("utm_") or normalized_key in _TRACKING_QUERY_KEYS:
            continue
        filtered_query_items.append((key, value))
    sanitized_query: str = urlencode(filtered_query_items, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, sanitized_query, ""))


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
        for match in _URL_PATTERN.finditer(description_text):
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
    for match in _URL_PATTERN.finditer(str(text or "")):
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


def _official_links_heading(language: str) -> str:
    if language == "uk":
        return "🌐 Офіційні ресурси:"
    if language == "ru":
        return "🌐 Официальные ссылки:"
    return "🌐 Official links:"


def _canonical_link_key(url: str) -> str:
    parts = urlsplit(url)
    host: str = parts.netloc.lower().strip()
    path: str = (parts.path or "/").rstrip("/")
    return f"{host}{path}"


def _line_has_official_context(line_text: str) -> bool:
    normalized_line: str = str(line_text or "").strip().lower()
    if not normalized_line:
        return False
    return any(hint in normalized_line for hint in _OFFICIAL_LINK_CONTEXT_HINTS)


def _extract_official_links_from_sources(videos: Sequence[PlannedVideo]) -> OfficialLinksSelection:
    raw_candidates: List[tuple[str, str]] = []
    domain_counts: dict[str, int] = {}
    for video in videos:
        for line in str(video.metadata.description or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            urls: List[str] = [match.group(0) for match in _URL_PATTERN.finditer(line)]
            if not urls:
                continue
            for url in urls:
                normalized_url: Optional[str] = _normalize_link_candidate(url)
                if normalized_url is None:
                    continue
                parts = urlsplit(normalized_url)
                if _is_youtube_host(parts.netloc):
                    continue
                raw_candidates.append((normalized_url, line))
                domain: str = parts.netloc.lower().strip()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if not raw_candidates:
        return OfficialLinksSelection(found_in_sources=0, kept_links=())

    scored_candidates: List[tuple[int, int, str]] = []
    for index, (url, line_text) in enumerate(raw_candidates):
        parts = urlsplit(url)
        domain: str = parts.netloc.lower().strip()
        score: int = 0
        if parts.scheme == "https":
            score += 20
        if _line_has_official_context(line_text):
            score += 20
        score += min(20, domain_counts.get(domain, 1) * 5)
        if parts.query:
            score -= 5
        score += max(0, 15 - (len(url) // 15))
        scored_candidates.append((score, -index, url))

    scored_candidates.sort(reverse=True)
    selected_links: List[str] = []
    seen_keys: set[str] = set()
    for _, _, url in scored_candidates:
        canonical_key: str = _canonical_link_key(url)
        if canonical_key in seen_keys:
            continue
        seen_keys.add(canonical_key)
        selected_links.append(url)
        if len(selected_links) >= 3:
            break
    return OfficialLinksSelection(
        found_in_sources=len(raw_candidates),
        kept_links=tuple(selected_links),
    )


def _description_has_official_links_block(description: str) -> bool:
    normalized: str = str(description or "")
    if not normalized:
        return False
    heading_present: bool = bool(
        re.search(
            r"(?im)^\s*(?:🌐\s*)?(?:official links|офіційні ресурси|официальные ссылки)\s*:\s*$",
            normalized,
        )
    )
    if not heading_present:
        return False
    return any(
        not _is_youtube_host(urlsplit(match.group(0)).netloc)
        for match in _URL_PATTERN.finditer(normalized)
    )


def _count_output_official_links(description: str) -> int:
    seen_keys: set[str] = set()
    for match in _URL_PATTERN.finditer(str(description or "")):
        normalized_url: Optional[str] = _normalize_link_candidate(match.group(0))
        if normalized_url is None:
            continue
        if _is_youtube_host(urlsplit(normalized_url).netloc):
            continue
        seen_keys.add(_canonical_link_key(normalized_url))
    return len(seen_keys)


def _looks_like_close_paragraph(text: str) -> bool:
    normalized: str = str(text or "").strip().lower()
    if not normalized:
        return False
    if "#" in normalized:
        return True
    return any(hint in normalized for hint in ("subscribe", "join", "watch", "follow", "диві", "долуч", "смотрите", "подпис"))


def _inject_official_links_block_if_missing(
    *,
    description: str,
    language: str,
    official_links: Sequence[str],
) -> OfficialLinksFillResult:
    if not official_links:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    if _description_has_official_links_block(description):
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    paragraphs: List[str] = _extract_description_paragraphs_raw(description)
    if not paragraphs:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=0,
            fill_applied=False,
        )
    links_block: str = "\n".join([_official_links_heading(language), *official_links]).strip()
    updated_paragraphs: List[str] = list(paragraphs)
    if len(updated_paragraphs) <= 3:
        if _looks_like_close_paragraph(updated_paragraphs[-1]):
            updated_paragraphs.insert(-1, links_block)
        else:
            updated_paragraphs.append(links_block)
    elif _looks_like_close_paragraph(updated_paragraphs[-1]):
        updated_paragraphs[-1] = f"{links_block}\n{updated_paragraphs[-1].strip()}".strip()
    else:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    updated_description: str = "\n\n".join(
        paragraph for paragraph in updated_paragraphs if paragraph.strip()
    ).strip()
    return OfficialLinksFillResult(
        description=updated_description,
        links_in_output=_count_output_official_links(updated_description),
        fill_applied=True,
    )


def _contains_agenda_heading(text: str) -> bool:
    agenda_headings: tuple[str, ...] = (
        "что в этом стриме",
        "в этом выпуске",
        "о чем поговорим",
        "що в цьому стрімі",
        "про що поговоримо",
        "what's in this stream",
        "what’s in this stream",
        "in this stream",
    )
    lines: List[str] = [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    for line in lines:
        normalized_line: str = line.strip(" -–—:;.!?")
        for heading in agenda_headings:
            if (
                normalized_line == heading
                or normalized_line.startswith(f"{heading}:")
                or normalized_line.startswith(f"{heading} -")
                or normalized_line.startswith(f"{heading} –")
                or normalized_line.startswith(f"{heading} —")
            ):
                return True
    return False


def _looks_like_per_source_dump(text: str) -> bool:
    source_line_pattern: re.Pattern[str] = re.compile(
        r"^\s*(?:source|video)\s*\d+[:.)-]?",
        flags=re.IGNORECASE,
    )
    lines: List[str] = [line.strip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    source_line_hits: int = sum(1 for line in lines if source_line_pattern.match(line))
    if source_line_hits >= 2:
        return True
    lowered_text: str = str(text or "").lower()
    return ("source 1" in lowered_text and "source 2" in lowered_text) or (
        "video 1" in lowered_text and "video 2" in lowered_text
    )


def _extract_named_entities(text: str) -> set[str]:
    entity_pattern: re.Pattern[str] = re.compile(
        r"\b(?:[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'-]{2,})(?:\s+[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'-]{2,})+\b",
        flags=re.UNICODE,
    )
    return {match.group(0).strip().lower() for match in entity_pattern.finditer(str(text or ""))}


def _build_source_coverage_flags(
    *,
    merged_anchors: set[str],
    source_anchors: Sequence[set[str]],
) -> tuple[bool, ...]:
    coverage_flags: List[bool] = []
    for source_index, anchors in enumerate(source_anchors, start=1):
        if len(anchors) < 3:
            coverage_flags.append(True)
            continue
        others: set[str] = set()
        for other_index, other_anchors in enumerate(source_anchors, start=1):
            if other_index == source_index:
                continue
            others.update(other_anchors)
        source_specific: set[str] = anchors - others
        probe: set[str] = source_specific if len(source_specific) >= 3 else anchors
        coverage_flags.append(bool(merged_anchors & probe))
    return tuple(coverage_flags)


def _build_merge_semantic_diagnostics(
    *,
    merged_content: MergedLanguageContent,
    videos: Sequence[PlannedVideo],
    official_links_selection: OfficialLinksSelection,
    official_links_fill: OfficialLinksFillResult,
    merge_quality: MergeQualityDiagnostics,
) -> MergeSemanticDiagnostics:
    description_text: str = str(merged_content.description or "").strip()
    merged_text_for_anchors: str = f"{merged_content.title.strip()}\n{description_text}"
    merged_anchors: set[str] = _extract_semantic_anchors(merged_text_for_anchors)
    source_anchors: List[set[str]] = [
        _extract_semantic_anchors(
            f"{video.metadata.title.strip()}\n{video.metadata.description.strip()}"
        )
        for video in videos
    ]
    source_coverage_by_item: tuple[bool, ...] = _build_source_coverage_flags(
        merged_anchors=merged_anchors,
        source_anchors=source_anchors,
    )
    source_entity_candidates: set[str] = set()
    for video in videos:
        source_entity_candidates.update(
            _extract_named_entities(
                f"{video.metadata.title.strip()}\n{video.metadata.description.strip()}"
            )
        )
    merged_entities: set[str] = _extract_named_entities(
        f"{merged_content.title.strip()}\n{description_text}"
    )
    named_entities_preserved: int = len(source_entity_candidates & merged_entities)
    marker_types: List[str] = []
    semantic_bullets_count: int = 0
    bullets_with_plain_marker_count: int = 0
    for line in description_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        marker: str = _bullet_marker_for_line(line)
        if not marker:
            continue
        if marker in _SEMANTIC_BULLET_MARKER_SET:
            semantic_bullets_count += 1
        else:
            bullets_with_plain_marker_count += 1
        if marker not in marker_types:
            marker_types.append(marker)
    bullet_points_count: int = semantic_bullets_count + bullets_with_plain_marker_count
    agenda_heading_present: bool = _contains_agenda_heading(description_text)
    paragraphs: List[str] = _extract_description_paragraphs_raw(description_text)
    first_paragraph: str = paragraphs[0] if paragraphs else ""
    hook_present: bool = len(first_paragraph) >= 60 and (
        "!" in first_paragraph or "?" in first_paragraph or ":" in first_paragraph
    )
    return MergeSemanticDiagnostics(
        hook_present=hook_present,
        agenda_block_present=(bullet_points_count >= 3) or agenda_heading_present,
        bullet_points_count=bullet_points_count,
        semantic_bullets_count=semantic_bullets_count,
        bullets_with_emoji_count=semantic_bullets_count,
        bullets_with_plain_marker_count=bullets_with_plain_marker_count,
        bullet_marker_types=tuple(marker_types),
        named_entities_preserved=named_entities_preserved,
        source_named_entities_total=len(source_entity_candidates),
        emoji_count=_count_emoji(description_text),
        source_coverage_hits=sum(1 for value in source_coverage_by_item if value),
        source_coverage_total=len(source_coverage_by_item),
        source_coverage_by_item=source_coverage_by_item,
        official_links_found_in_sources=official_links_selection.found_in_sources,
        official_links_kept=len(official_links_selection.kept_links),
        official_links_in_output=official_links_fill.links_in_output,
        official_links_fill_applied=official_links_fill.fill_applied,
        merge_quality=merge_quality,
    )


def _log_merge_style_diagnostics(
    *,
    diagnostics: MergeSemanticDiagnostics,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    model_name: str,
    attempt_index: int,
) -> None:
    coverage_by_item_text: str = ",".join(
        f"{index}:{'yes' if covered else 'no'}"
        for index, covered in enumerate(diagnostics.source_coverage_by_item, start=1)
    )
    marker_types_text: str = ",".join(diagnostics.bullet_marker_types) or "none"
    LOGGER.info(
        "merge_style_coverage branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d style_contract_version=%s hook_present=%s agenda_block_present=%s bullet_points_count=%d semantic_bullets_count=%d bullets_with_emoji_count=%d bullets_with_plain_marker_count=%d bullet_marker_types=%s neutral_bullets_count=%d accent_bullets_count=%d accent_marker_types=%s accent_overflow=%s block_spacing_ok=%s named_entities_preserved=%d source_named_entities_total=%d named_entities_metric=%s emoji_count=%d source_coverage_total=%d/%d source_coverage_ok=%s source_coverage_by_item=%s official_links_found_in_sources=%d official_links_kept=%d official_links_in_output=%d official_links_fill_applied=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        model_name,
        attempt_index,
        STYLE_CONTRACT_VERSION,
        "yes" if diagnostics.hook_present else "no",
        "yes" if diagnostics.agenda_block_present else "no",
        diagnostics.bullet_points_count,
        diagnostics.semantic_bullets_count,
        diagnostics.bullets_with_emoji_count,
        diagnostics.bullets_with_plain_marker_count,
        marker_types_text,
        diagnostics.merge_quality.neutral_bullets_count,
        diagnostics.merge_quality.accent_bullets_count,
        ",".join(diagnostics.merge_quality.accent_marker_types) or "none",
        "yes" if diagnostics.merge_quality.accent_overflow else "no",
        "yes" if diagnostics.merge_quality.block_spacing_ok else "no",
        diagnostics.named_entities_preserved,
        diagnostics.source_named_entities_total,
        "informational",
        diagnostics.emoji_count,
        diagnostics.source_coverage_hits,
        diagnostics.source_coverage_total,
        (
            "yes"
            if diagnostics.source_coverage_hits == diagnostics.source_coverage_total
            else "no"
        ),
        coverage_by_item_text or "none",
        diagnostics.official_links_found_in_sources,
        diagnostics.official_links_kept,
        diagnostics.official_links_in_output,
        "yes" if diagnostics.official_links_fill_applied else "no",
    )
    LOGGER.info(
        "merge_semantic_gate branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d block_language_expected=%s hook_language_detected=%s lead_in_language_detected=%s links_heading_language_detected=%s cta_language_detected=%s language_consistency_ok=%s wrong_language_heading_detected=%s person_role_claims_detected=%d suspicious_role_labels_detected=%s role_softening_applied=%s semantic_gate_status=%s semantic_gate_reason_codes=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        model_name,
        attempt_index,
        diagnostics.merge_quality.block_language_expected,
        diagnostics.merge_quality.hook_language_detected,
        diagnostics.merge_quality.lead_in_language_detected,
        diagnostics.merge_quality.links_heading_language_detected,
        diagnostics.merge_quality.cta_language_detected,
        "yes" if diagnostics.merge_quality.language_consistency_ok else "no",
        "yes" if diagnostics.merge_quality.wrong_language_heading_detected else "no",
        diagnostics.merge_quality.person_role_claims_detected,
        ",".join(diagnostics.merge_quality.suspicious_role_labels_detected) or "none",
        "yes" if diagnostics.merge_quality.role_softening_applied else "no",
        diagnostics.merge_quality.semantic_gate_status,
        ",".join(diagnostics.merge_quality.semantic_gate_reason_codes) or "none",
    )


def _token_to_semantic_anchor(token: str) -> str:
    token_lower: str = str(token or "").lower().strip()
    if not token_lower:
        return ""
    if token_lower in _SEMANTIC_STOPWORDS:
        return ""
    has_digit: bool = any(char.isdigit() for char in token_lower)
    min_length: int = 2 if has_digit else 4
    if len(token_lower) < min_length:
        return ""
    if has_digit:
        return token_lower
    return token_lower[:6] if len(token_lower) > 6 else token_lower


def _extract_semantic_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    for token in _SEMANTIC_TOKEN_PATTERN.findall(str(text or "")):
        anchor: str = _token_to_semantic_anchor(token)
        if anchor:
            anchors.add(anchor)
    return anchors


def _validate_coverage_preserving_merge_or_raise(
    *,
    merged_content: MergedLanguageContent,
    videos: Sequence[PlannedVideo],
    official_links_selection: OfficialLinksSelection,
    official_links_fill: OfficialLinksFillResult,
    merge_quality: MergeQualityDiagnostics,
) -> MergeSemanticDiagnostics:
    diagnostics: MergeSemanticDiagnostics = _build_merge_semantic_diagnostics(
        merged_content=merged_content,
        videos=videos,
        official_links_selection=official_links_selection,
        official_links_fill=official_links_fill,
        merge_quality=merge_quality,
    )
    merged_anchors: set[str] = _extract_semantic_anchors(
        f"{merged_content.title.strip()}\n{merged_content.description.strip()}"
    )
    if _looks_like_per_source_dump(merged_content.description):
        raise RuntimeError("description validation failed: per_source_enumeration")
    if len(merged_anchors) < 3:
        raise RuntimeError("description validation failed: semantic_too_generic")
    if diagnostics.emoji_count > 10:
        raise RuntimeError("description validation failed: excessive_emoji_usage")
    if merge_quality.semantic_gate_status == "hard_reject":
        reason_codes: str = ",".join(merge_quality.semantic_gate_reason_codes) or "semantic_gate"
        raise RuntimeError(f"description validation failed: {reason_codes}")

    source_anchors: List[set[str]] = [
        _extract_semantic_anchors(
            f"{video.metadata.title.strip()}\n{video.metadata.description.strip()}"
        )
        for video in videos
    ]
    all_source_anchors: set[str] = set().union(*source_anchors) if source_anchors else set()
    if all_source_anchors:
        overlap_count: int = len(merged_anchors & all_source_anchors)
        source_anchor_count: int = len(all_source_anchors)
        if source_anchor_count <= 6:
            min_overlap = 1
        elif source_anchor_count <= 12:
            min_overlap = 2
        else:
            min_overlap = max(3, min(12, source_anchor_count // 6))
        if overlap_count < min_overlap:
            raise RuntimeError("description validation failed: semantic_source_grounding_too_low")

    for source_index, covered in enumerate(diagnostics.source_coverage_by_item, start=1):
        if not covered:
            raise RuntimeError(
                f"description validation failed: semantic_source_{source_index}_coverage_missing"
            )
    return diagnostics


def build_llm_merge_prompt_text(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    no_description_text: str,
) -> str:
    if len(videos) < 2:
        raise ValueError("Expected at least 2 videos for merged generation.")
    language_name: str = _language_name_for_merge_prompt(
        language,
        config.templates.llm_language_names_json,
    )
    source_blocks: List[str] = []
    raw_source_chars_total: int = 0
    cleaned_source_chars_total: int = 0
    for index, video in enumerate(videos, start=1):
        prepared_description: PreparedMergeSourceDescription = _clean_source_description_for_llm(
            video.metadata.description.strip() or no_description_text
        )
        description_for_prompt: str = prepared_description.text or no_description_text
        raw_source_chars_total += prepared_description.raw_chars
        cleaned_source_chars_total += len(description_for_prompt)
        LOGGER.info(
            "merge_source_text_prepared language=%s source_index=%d row=%s raw_chars=%d cleaned_chars=%d urls_removed=%d hashtags_removed=%d service_paragraphs_dropped=%d hard_truncation=disabled",
            language,
            index,
            getattr(video, "row_number", "unknown"),
            prepared_description.raw_chars,
            len(description_for_prompt),
            prepared_description.urls_removed,
            prepared_description.hashtags_removed,
            prepared_description.service_paragraphs_dropped,
        )
        source_blocks.append(
            "\n".join(
                [
                    f"SOURCE {index}",
                    f"TITLE: {video.metadata.title.strip()}",
                    f"DESCRIPTION: {description_for_prompt}",
                ]
            )
        )
    LOGGER.info(
        "merge_prompt_sources_ready language=%s source_count=%d raw_source_chars_total=%d cleaned_source_chars_total=%d hard_truncation=disabled",
        language,
        len(videos),
        raw_source_chars_total,
        cleaned_source_chars_total,
    )
    link_policy_block: str = (
        "SYSTEM LINK POLICY\n"
        "Do not include any URLs in the output.\n"
        "Do not add a recommended materials block or an official links block.\n"
        "Link blocks will be assembled later by the system."
    )
    cross_domain_sentence_policy_block: str = (
        "CROSS-DOMAIN SENTENCE POLICY\n"
        "If the sources touch different semantic domains, do not compress them into one sentence.\n"
        "Especially do not merge medicine or biology or neurobiology, climate or weather or ecology, disasters or geophysics or natural hazards, psychology or thinking or behavior, and social or moral or civilizational conclusions into one sentence.\n"
        "These topics may stay in one final description, but present them as separate lines of discussion in separate sentences.\n"
        "Do not build one long cause-and-effect chain across all of those domains in a single sentence.\n"
        "Several calm sentences are better than one overloaded super-sentence."
    )
    template_prompt: str = str(config.templates.llm_merge_title_description_prompt or "").strip()
    if template_prompt:
        formatted_template: str = template_prompt.format(
            language_name=language_name,
            sources_block="\n\n".join(source_blocks),
            youtube_candidates_block="",
        ).strip()
        return f"{formatted_template}\n\n{cross_domain_sentence_policy_block}\n\n{link_policy_block}".strip()
    return (
        "You are writing a YouTube stream title and description.\n"
        f"Write output only in {language_name}.\n"
        "Use only facts explicitly present in the source descriptions.\n"
        "Treat the sources as one complete stream, not as a list of separate videos.\n"
        "Generate a new final title, not a copy of any single source title.\n"
        "Do not use emoji in the title.\n"
        "Mentally extract key points from each source, preserve all non-trivial source-specific points,\n"
        "combine overlaps, compress repetition, and produce one coherent final description.\n"
        "Write a strong native YouTube title no longer than 99 characters.\n"
        "Write one cohesive stream description in 2 to 4 compact paragraphs.\n"
        "The description must cover all source inputs that were merged.\n"
        "Do not drop a source-specific fact, event, or angle without clear overlap-based reason.\n"
        "Start paragraph one with a strong factual hook grounded in the main tension, risk, or key conflict.\n"
        "Keep the hook editorial and readable, but never clickbait.\n"
        "Include one compact 'what is in this stream' agenda block using 4 to 7 short thesis bullet lines.\n"
        "Each thesis line must start with one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.\n"
        "Most thesis bullets should start with 🔹.\n"
        "Accent markers are rare and optional; use no more than 3 accent markers per theses block.\n"
        "Do not present the agenda as SOURCE 1 / SOURCE 2 / SOURCE 3.\n"
        "Keep agenda points specific and factual, not generic placeholders.\n"
        "Preserve important recognizable names from sources when relevant; never invent names.\n"
        "Avoid asserting strong person titles or role labels unless they are clearly necessary and well-supported by the sources.\n"
        "Keep marker usage controlled and readable; do not use dash-only bullets as the sole style.\n"
        f"{cross_domain_sentence_policy_block}\n"
        f"{link_policy_block}\n"
        "An optional one-line closing sentence should be a light practical CTA with 2 to 5 hashtags.\n"
        "Do not enumerate sources as 1) 2) 3).\n"
        "Do not write a dry digest, protocol, or generic CTA block.\n"
        "Do not output generic slogans, abstract editorial text, or propagandistic phrasing.\n"
        "Do not replace concrete facts with broad statements like 'an important conversation about everything'.\n"
        'Output only one strict JSON object with exactly these keys: title, description.\n\n'
        f"{'\n\n'.join(source_blocks)}"
    ).strip()


def _structured_merge_schema() -> dict[str, object]:
    return {
        "name": "restreamer_merge_summary_v2",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "description"],
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 99},
                "description": {"type": "string", "minLength": 1},
            },
        },
    }


def _single_source_translate_prompt(
    *,
    source_language: str,
    target_language: str,
    source_description: str,
) -> str:
    return (
        "You are a precise editor and translator.\n"
        f"Source language: {source_language}.\n"
        f"Target language: {target_language}.\n"
        "Task: translate and lightly rewrite for readability while preserving facts.\n"
        "Return only plain paragraph text.\n"
        "No headings, JSON, markdown, CTA, hashtags, or links.\n\n"
        f"{source_description.strip()}"
    ).strip()


def _request_plain_text(
    *,
    prompt_text: str,
    config: AppConfig,
    provider: LlmProvider,
    model_name: str,
    attempt_label: str,
    trace_context: LlmTraceContext,
) -> str:
    if provider.pre_delay_sec(config=config) > 0:
        time.sleep(max(0.0, float(provider.pre_delay_sec(config=config))))
    response: OpenAITransportResult = provider.request_merge(
        prompt_text=prompt_text,
        config=config,
        model_name=model_name,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=None,
        temperature=0.0,
        trace_context=trace_context,
    )
    return response.raw_text


def _request_structured_merge_payload(
    *,
    prompt_text: str,
    config: AppConfig,
    provider: LlmProvider,
    model_name: str,
    attempt_label: str,
    trace_context: LlmTraceContext,
) -> OpenAITransportResult:
    if provider.pre_delay_sec(config=config) > 0:
        time.sleep(max(0.0, float(provider.pre_delay_sec(config=config))))
    return provider.request_merge(
        prompt_text=prompt_text,
        config=config,
        model_name=model_name,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=_structured_merge_schema(),
        temperature=0.0,
        trace_context=trace_context,
    )


def _apply_merge_polish_stage(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    source_model: str,
    polish_provider: LlmProvider,
    polish_model: str,
    merged_content: MergedLanguageContent,
    attempt_label: str,
    branch_label: str,
    date_key: str,
    slot_key: str,
    merge_run_summary: Optional[MergeRunSummary],
) -> tuple[MergedLanguageContent, MergePolishResult]:
    if merge_run_summary is not None:
        merge_run_summary.record_polish_requested()
    LOGGER.info(
        "merge_polish_requested branch=%s date_key=%s slot_key=%s language=%s source_model=%s polish_model=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        source_model,
        polish_model,
    )
    original_title: str = str(merged_content.title or "").strip()
    original_description: str = str(merged_content.description or "").strip()
    try:
        response: OpenAITransportResult = _request_structured_merge_payload(
            prompt_text=build_merge_polish_prompt(
                language=language,
                title_text=original_title,
                description_text=original_description,
            ),
            config=config,
            provider=polish_provider,
            model_name=polish_model,
            attempt_label=attempt_label,
            trace_context=LlmTraceContext(
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                provider=polish_provider.name,
                model_name=polish_model,
                attempt_index=1,
                request_kind="merge_polish",
                source_count=len(videos),
            ),
        )
        polished_content, _ = parse_merge_response_or_raise(
            provider_name=polish_provider.name,
            model_name=polish_model,
            raw_text=response.raw_text,
            structured_payload=response.structured_payload,
        )
        original_body = separate_merge_body_and_tail(text=original_description)
        polished_body = separate_merge_body_and_tail(text=polished_content.description)
        if original_body.body_paragraph_count_after_recovery != polished_body.body_paragraph_count_after_recovery:
            raise RuntimeError("structure_changed")
        if bullet_marker_count(original_description) != bullet_marker_count(polished_content.description):
            raise RuntimeError("structure_changed")
        if is_too_aggressive_rewrite(
            original_text=f"{original_title}\n{original_description}",
            polished_text=f"{polished_content.title}\n{polished_content.description}",
        ):
            raise RuntimeError("too_aggressive_rewrite")
        quality_result: MergeQualityNormalizationResult = normalize_merge_description(
            description=polished_content.description,
            language=language,
            source_texts=tuple(
                f"{video.metadata.title.strip()}\n{video.metadata.description.strip()}"
                for video in videos
            ),
        )
        if quality_result.description_text != polished_content.description:
            polished_content = dataclasses.replace(
                polished_content,
                description=quality_result.description_text,
                description_selected=quality_result.description_text,
                description_audit=quality_result.description_text,
            )
        official_links_selection: OfficialLinksSelection = _extract_official_links_from_sources(
            videos
        )
        official_links_fill: OfficialLinksFillResult = _inject_official_links_block_if_missing(
            description=polished_content.description,
            language=language,
            official_links=official_links_selection.kept_links,
        )
        if official_links_fill.fill_applied:
            polished_content = dataclasses.replace(
                polished_content,
                description=official_links_fill.description,
                description_selected=official_links_fill.description,
                description_audit=official_links_fill.description,
            )
        _validate_coverage_preserving_merge_or_raise(
            merged_content=polished_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=quality_result.diagnostics,
        )
        if merge_run_summary is not None:
            merge_run_summary.record_polish_accepted()
        LOGGER.info(
            "merge_polish_accepted branch=%s date_key=%s slot_key=%s language=%s source_model=%s polish_model=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            source_model,
            polish_model,
        )
        return (
            polished_content,
            MergePolishResult(
                source_model=source_model,
                polish_model=polish_model,
                original_text=f"{original_title}\n{original_description}",
                polished_text=f"{polished_content.title}\n{polished_content.description}",
                accepted=True,
                reject_reason=None,
                validation_passed=True,
            ),
        )
    except Exception as error:
        reject_reason: str = (
            "too_aggressive_rewrite"
            if str(error) == "too_aggressive_rewrite"
            else infer_polish_reject_reason(str(error))
        )
        if merge_run_summary is not None:
            merge_run_summary.record_polish_rejected()
        LOGGER.warning(
            "merge_polish_discarded branch=%s date_key=%s slot_key=%s language=%s source_model=%s polish_model=%s reason=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            source_model,
            polish_model,
            reject_reason,
        )
        return (
            merged_content,
            MergePolishResult(
                source_model=source_model,
                polish_model=polish_model,
                original_text=f"{original_title}\n{original_description}",
                polished_text="",
                accepted=False,
                reject_reason=reject_reason,
                validation_passed=False,
            ),
        )


def attempt_openai_single_source_translate_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    return attempt_llm_single_source_translate_with_audit(
        language=language,
        videos=videos,
        config=config,
        attempt_label=attempt_label,
        summarize_error=summarize_error,
        no_description_text=no_description_text,
        merge_run_summary=merge_run_summary,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )


def attempt_llm_single_source_translate_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    del merge_run_summary
    if len(videos) != 1:
        raise RuntimeError("single-source translate expects exactly one video")
    source_video: PlannedVideo = videos[0]
    model_name: str = config.llm_model
    provider: LlmProvider = get_llm_provider(config=config)
    main_source_label: str = _main_stage_field_source(provider_name=provider.name)
    try:
        raw_text: str = _request_plain_text(
            prompt_text=_single_source_translate_prompt(
                source_language=source_video.language,
                target_language=language,
                source_description=source_video.metadata.description.strip() or no_description_text,
            ),
            config=config,
            provider=provider,
            model_name=model_name,
            attempt_label=attempt_label,
            trace_context=LlmTraceContext(
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                provider=provider.name,
                model_name=model_name,
                attempt_index=1,
                request_kind="single_source_plain",
                source_count=1,
            ),
        )
        cleaned_text, is_valid, reasons = clean_and_validate_llm_description(text=raw_text)
        if not is_valid:
            raise RuntimeError(
                "single-source plain validation failed: " + ("; ".join(reasons) or "unknown")
            )
        merged_content: MergedLanguageContent = build_plain_merged_content_or_raise(
            model_name=model_name,
            title_text=source_video.metadata.title.strip() or "Untitled",
            description_text=cleaned_text,
        )
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text=raw_text,
            merged=dataclasses.replace(
                merged_content,
                branch_type=BRANCH_MERGE,
                title_source=main_source_label,
                hook_source=main_source_label,
                hashtags_source=main_source_label,
                body_source="main_merge",
            ),
            error_summary=None,
            salvaged_title=merged_content.title,
            publish_source_label="single_source_plain_ok",
            generator_model_name=model_name,
            used_model_names=(model_name,),
            branch_type=BRANCH_MERGE,
            title_source=main_source_label,
            hook_source=main_source_label,
            hashtags_source=main_source_label,
            body_source="main_merge",
        )
    except Exception as error:
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text="",
            merged=None,
            error_summary=summarize_error(error),
            salvaged_title=source_video.metadata.title.strip() or "Untitled",
            publish_source_label="single_source_plain_failed",
            generator_model_name=model_name,
            used_model_names=(model_name,),
            branch_type=BRANCH_MERGE,
            title_source=main_source_label,
            hook_source=main_source_label,
            hashtags_source="fallback_none",
            body_source="main_merge",
        )


def _log_merge_attempt_start(
    *,
    provider_name: str,
    model_name: str,
    attempt_index: int,
    is_fallback: bool,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
) -> None:
    stage_name: str = "fallback" if is_fallback else "primary"
    if is_fallback:
        LOGGER.info(
            "merge_llm_fallback_start branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=%s attempt=%d model=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            provider_name,
            stage_name,
            attempt_index,
            model_name,
        )
        return
    LOGGER.info(
        "merge_llm_primary_attempt branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=%s attempt=%d model=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        provider_name,
        stage_name,
        attempt_index,
        model_name,
    )


def _log_merge_attempt_invalid(
    *,
    provider_name: str,
    model_name: str,
    attempt_index: int,
    is_fallback: bool,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    reason_code: str,
    reason: str,
    raw_response_text: str,
) -> None:
    stage_name: str = "fallback" if is_fallback else "primary"
    LOGGER.info(
        "merge_llm_response_invalid branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=%s model=%s attempt=%d code=%s raw_response_received=%s reason=%s raw_chars=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        provider_name,
        stage_name,
        model_name,
        attempt_index,
        reason_code,
        "yes" if bool(str(raw_response_text or "").strip()) else "no",
        reason,
        len(str(raw_response_text or "")),
    )


def _enforce_merged_description_structure(
    *,
    merged_content: MergedLanguageContent,
) -> ParagraphEnforcementResult:
    original_description: str = str(merged_content.description or "").strip()
    tail_separation = separate_merge_body_and_tail(text=original_description)
    body_paragraphs_before: int = tail_separation.body_paragraph_count
    body_paragraphs_after: int = tail_separation.body_paragraph_count_after_recovery
    if not tail_separation.body_text.strip():
        return ParagraphEnforcementResult(
            description_text=original_description,
            mutated=False,
            recovery_applied=False,
            note="empty_description",
            body_paragraphs_before=0,
            body_paragraphs_after=0,
        )
    normalized_description: str = tail_separation.full_text or original_description
    mutated: bool = normalized_description != original_description
    return ParagraphEnforcementResult(
        description_text=normalized_description,
        mutated=mutated,
        recovery_applied=tail_separation.recovery_applied and mutated,
        note=tail_separation.recovery_note,
        body_paragraphs_before=body_paragraphs_before,
        body_paragraphs_after=body_paragraphs_after,
    )


def _attempt_merge_once(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    provider: LlmProvider,
    model_name: str,
    attempt_index: int,
    attempt_label: str,
    branch_label: str,
    date_key: str,
    slot_key: str,
    no_description_text: str,
) -> tuple[MergedLanguageContent, str]:
    prompt_text: str = build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
        no_description_text=no_description_text,
    )
    if provider.pre_delay_sec(config=config) > 0:
        time.sleep(max(0.0, float(provider.pre_delay_sec(config=config))))
    response: OpenAITransportResult = provider.request_merge(
        prompt_text=prompt_text,
        config=config,
        model_name=model_name,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=_structured_merge_schema(),
        temperature=0.0,
        trace_context=LlmTraceContext(
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
            provider=provider.name,
            model_name=model_name,
            attempt_index=attempt_index,
            request_kind="structured",
            source_count=len(videos),
        ),
    )
    raw_response_text: str = response.raw_text
    try:
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name=provider.name,
            model_name=model_name,
            raw_text=raw_response_text,
            structured_payload=response.structured_payload,
        )
    except Exception as error:
        raise MergeAttemptFailure(
            reason_code=_reason_code_from_error(error),
            reason=str(error),
            model_name=model_name,
            attempt_stage="validation",
            raw_response_text=raw_response_text,
        ) from error
    official_links_selection: OfficialLinksSelection = _extract_official_links_from_sources(
        videos
    )
    official_links_fill: OfficialLinksFillResult = OfficialLinksFillResult(
        description=merged_content.description,
        links_in_output=_count_output_official_links(merged_content.description),
        fill_applied=False,
    )
    quality_result: MergeQualityNormalizationResult = normalize_merge_description(
        description=merged_content.description,
        language=language,
        source_texts=tuple(
            (
                f"{video.metadata.title.strip()}\n"
                f"{_clean_source_description_for_llm(video.metadata.description.strip()).text}"
            ).strip()
            for video in videos
        ),
    )
    if quality_result.description_text != merged_content.description:
        merged_content = dataclasses.replace(
            merged_content,
            description=quality_result.description_text,
            description_selected=quality_result.description_text,
            description_audit=quality_result.description_text,
        )
    try:
        diagnostics: MergeSemanticDiagnostics = _validate_coverage_preserving_merge_or_raise(
            merged_content=merged_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=quality_result.diagnostics,
        )
    except Exception as error:
        raise MergeAttemptFailure(
            reason_code=_reason_code_from_error(error),
            reason=str(error),
            model_name=model_name,
            attempt_stage="validation",
            raw_response_text=raw_response_text,
        ) from error
    LOGGER.info(
        "merge_llm_response_valid model=%s attempt=%d title_length=%d description_length=%d paragraph_count=%d youtube_links_in_llm_output=%d",
        model_name,
        attempt_index,
        len(merged_content.title),
        len(merged_content.description),
        paragraph_count,
        _count_youtube_urls_in_text(merged_content.description),
    )
    _log_merge_style_diagnostics(
        diagnostics=diagnostics,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
        language=language,
        model_name=model_name,
        attempt_index=attempt_index,
    )
    return (merged_content, raw_response_text)


def attempt_openai_merge_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    normalize_youtube_url: Callable[[str], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    return attempt_llm_merge_with_audit(
        language=language,
        videos=videos,
        config=config,
        attempt_label=attempt_label,
        summarize_error=summarize_error,
        normalize_youtube_url=normalize_youtube_url,
        no_description_text=no_description_text,
        merge_run_summary=merge_run_summary,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )


def attempt_llm_merge_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    normalize_youtube_url: Callable[[str], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    del normalize_youtube_url
    primary_model: str = config.llm_model
    primary_provider: LlmProvider = get_llm_provider(config=config)
    main_source_label: str = _main_stage_field_source(provider_name=primary_provider.name)
    last_raw_response: str = ""
    last_error_summary: str = "unknown error"

    for attempt_index in range(1, PRIMARY_ATTEMPTS + 1):
        _log_merge_attempt_start(
            provider_name=primary_provider.name,
            model_name=primary_model,
            attempt_index=attempt_index,
            is_fallback=False,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
        )
        if attempt_index > 1:
            LOGGER.info(
                "merge_llm_retry provider=%s attempt=%d model=%s",
                primary_provider.name,
                attempt_index,
                primary_model,
            )
            if merge_run_summary is not None:
                merge_run_summary.record_retry_used()
        try:
            merged_content, raw_response_text = _attempt_merge_once(
                language=language,
                videos=videos,
                config=config,
                provider=primary_provider,
                model_name=primary_model,
                attempt_index=attempt_index,
                attempt_label=f"{attempt_label}_PRIMARY_{attempt_index}",
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                no_description_text=no_description_text,
            )
            last_raw_response = raw_response_text
            if merge_run_summary is not None:
                merge_run_summary.record_merge_success()
            LOGGER.info(
                "merge_branch_ready branch=%s date_key=%s slot_key=%s language=%s generator_model=%s",
                branch_label,
                date_key,
                slot_key,
                language,
                primary_model,
            )
            LOGGER.info(
                "merge_provider_summary provider=%s model=%s structured_ok=yes fallback_used=no parse_repair_used=no final_status=success",
                primary_provider.name,
                primary_model,
            )
            return LanguageMergeAttempt(
                language=language,
                model_name=primary_model,
                raw_response_text=raw_response_text,
                merged=dataclasses.replace(
                    merged_content,
                    branch_type=BRANCH_MERGE,
                    title_source=main_source_label,
                    hook_source=main_source_label,
                    hashtags_source=main_source_label,
                    body_source="main_merge",
                ),
                error_summary=None,
                salvaged_title=merged_content.title,
                publish_source_label="primary_success",
                generator_model_name=primary_model,
                used_model_names=(primary_model,),
                branch_type=BRANCH_MERGE,
                title_source=main_source_label,
                hook_source=main_source_label,
                hashtags_source=main_source_label,
                body_source="main_merge",
            )
        except MergeAttemptFailure as error:
            last_raw_response = error.raw_response_text
            last_error_summary = error.reason
            _log_merge_attempt_invalid(
                provider_name=primary_provider.name,
                model_name=primary_model,
                attempt_index=attempt_index,
                is_fallback=False,
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                reason_code=error.reason_code,
                reason=error.reason,
                raw_response_text=error.raw_response_text,
            )
            if merge_run_summary is not None:
                merge_run_summary.record_validation_rejected()
        except Exception as error:
            last_error_summary = summarize_error(error)
            _log_merge_attempt_invalid(
                provider_name=primary_provider.name,
                model_name=primary_model,
                attempt_index=attempt_index,
                is_fallback=False,
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                reason_code="unexpected_error",
                reason=last_error_summary,
                raw_response_text="",
            )

    LOGGER.error(
        "merge_llm_final_failure branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=primary code=%s fallback_used=no raw_response_received=%s reason=%s raw_chars=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        primary_provider.name,
        _reason_code_from_error(RuntimeError(last_error_summary)),
        "yes" if bool(last_raw_response.strip()) else "no",
        last_error_summary,
        len(last_raw_response),
    )
    LOGGER.info(
        "merge_provider_summary provider=%s model=%s structured_ok=no fallback_used=no parse_repair_used=no final_status=failed",
        primary_provider.name,
        primary_model,
    )
    if merge_run_summary is not None:
        merge_run_summary.record_final_failure()
    return LanguageMergeAttempt(
        language=language,
        model_name=primary_model,
        raw_response_text=last_raw_response,
        merged=None,
        error_summary=last_error_summary,
        salvaged_title=None,
        publish_source_label="merge_failed",
        generator_model_name=primary_model,
        used_model_names=(primary_model,),
        branch_type=BRANCH_MERGE,
        title_source=main_source_label,
        hook_source=main_source_label,
        hashtags_source="fallback_none",
        body_source="main_merge",
    )


def enforce_openai_merged_paragraphs(
    *,
    language: str,
    merged_content: MergedLanguageContent,
    videos: List[PlannedVideo],
    config: AppConfig,
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> MergedLanguageContent:
    del videos, config, no_description_text
    enforcement_result: ParagraphEnforcementResult = _enforce_merged_description_structure(
        merged_content=merged_content
    )
    LOGGER.info(
        "merge_post_enforcement branch=%s date_key=%s slot_key=%s language=%s body_paragraphs_before=%d body_paragraphs_after=%d mutated=%s recovery_applied=%s reason=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        enforcement_result.body_paragraphs_before,
        enforcement_result.body_paragraphs_after,
        "yes" if enforcement_result.mutated else "no",
        "yes" if enforcement_result.recovery_applied else "no",
        enforcement_result.note,
    )
    if not enforcement_result.mutated:
        return merged_content
    if enforcement_result.recovery_applied and merge_run_summary is not None:
        merge_run_summary.record_paragraph_recovery_used()
    updated_content: MergedLanguageContent = dataclasses.replace(
        merged_content,
        description=enforcement_result.description_text,
        description_selected=enforcement_result.description_text,
        description_audit=enforcement_result.description_text,
    )
    return updated_content
def _main_stage_field_source(*, provider_name: str) -> str:
    normalized_provider_name: str = str(provider_name or "").strip().lower()
    return normalized_provider_name or "main_model"
