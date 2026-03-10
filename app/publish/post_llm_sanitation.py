from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    MergedPublicationPayload,
    PlannedVideo,
)
from app.ingest.youtube_metadata import normalize_youtube_video_url
from app.llm.merge_quality import normalize_merge_description
from app.observability.runtime_analytics import record_malformed_tail_url_cleanup


LOGGER = _get_logger_impl(__name__)

_HASHTAG_TOKEN_RE: re.Pattern[str] = re.compile(r"^#[^\s#]+$")
_URL_LINE_RE: re.Pattern[str] = re.compile(r"^https?://\S+$", re.IGNORECASE)
_URL_RE: re.Pattern[str] = re.compile(r"https?://\S+", re.IGNORECASE)
_OFFICIAL_LINKS_HEADING_RE: re.Pattern[str] = re.compile(
    r"(?im)^\s*(?:🌐\s*)?(?:official links|офіційні ресурси|официальные ссылки)\s*:\s*$"
)
_CTA_HINTS: tuple[str, ...] = (
    "watch",
    "learn more",
    "join",
    "subscribe",
    "follow",
    "read more",
    "links below",
    "details below",
    "дивіться",
    "долуч",
    "підпис",
    "узнать больше",
    "смотрите",
    "подпис",
    "подробности",
)
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


@dataclass(frozen=True)
class PostLlmSanitizationResult:
    body_text: str
    cta_text: str
    hashtags_line: str
    source_urls: List[str]
    full_text: str
    normalized_url_count: int
    tail_was_separated: bool
    cta_found: bool
    hashtags_found: bool
    hashtags_split_from_cta: bool
    tail_layout: str
    source_urls_found: int
    malformed_source_urls_dropped: int


@dataclass(frozen=True)
class AuthoritativeSourceUrlsResult:
    source_urls: List[str]
    selected_youtube_urls: List[str]
    inspected_source_videos: int
    emitted_source_urls: int
    emitted_source_video_urls: int
    emitted_selected_youtube_urls: int
    preserved_non_youtube_tail_urls: int
    duplicate_urls_removed: int
    malformed_tail_urls_dropped: int


@dataclass(frozen=True)
class _OfficialLinksExtractionResult:
    cleaned_text: str
    heading_found: bool
    source_urls: List[str]
    empty_blocks_suppressed: int


def sanitize_post_llm_title(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def resolve_post_llm_source_label(
    merge_attempt: Optional[LanguageMergeAttempt],
    *,
    default_label: str = "unknown",
) -> str:
    if merge_attempt is None:
        return default_label
    explicit_label: str = str(merge_attempt.publish_source_label or "").strip()
    if explicit_label:
        return explicit_label
    if merge_attempt.plain_repair_used:
        return "repair_used"
    return default_label


def build_sanitized_merged_publication_payload(
    *,
    language: str,
    merged_content: MergedLanguageContent,
    merge_attempt: Optional[LanguageMergeAttempt],
    use_audit_text: bool,
    source_videos: Optional[Sequence[PlannedVideo]] = None,
) -> MergedPublicationPayload:
    raw_title: str = (
        str(merged_content.title_audit or merged_content.title or "").strip()
        if use_audit_text
        else str(merged_content.title_selected or merged_content.title or "").strip()
    )
    raw_description: str = (
        str(merged_content.description_audit or merged_content.description or "").strip()
        if use_audit_text
        else str(
            merged_content.description_selected or merged_content.description or ""
        ).strip()
    )
    source_label: str = resolve_post_llm_source_label(
        merge_attempt,
        default_label="merged_publish",
    )
    source_videos_sequence: Sequence[PlannedVideo] = tuple(source_videos or ())
    official_links_extraction: _OfficialLinksExtractionResult = _extract_official_links_blocks(
        raw_description
    )
    cleanup_event_key: str = (
        f"{language}:{source_label}:{hash(_normalize_text(official_links_extraction.cleaned_text))}"
    )
    sanitization_result: PostLlmSanitizationResult = sanitize_post_llm_text(
        official_links_extraction.cleaned_text,
        language=language,
        source_label=source_label,
    )
    extracted_text_source_urls: List[str] = _dedupe_nonempty(
        official_links_extraction.source_urls + sanitization_result.source_urls
    )
    selected_youtube_urls: List[str] = [
        url for url in extracted_text_source_urls if _is_youtube_url(url)
    ]
    authoritative_source_urls_result: AuthoritativeSourceUrlsResult = (
        build_authoritative_merged_source_urls(
            language=language,
            source_videos=source_videos_sequence,
            extracted_tail_urls=extracted_text_source_urls,
            malformed_tail_urls_dropped=sanitization_result.malformed_source_urls_dropped,
            cleanup_event_key=cleanup_event_key,
        )
    )
    final_official_links_urls: List[str] = authoritative_source_urls_result.source_urls
    final_selected_youtube_urls: List[str] = (
        authoritative_source_urls_result.selected_youtube_urls
    )
    if source_videos_sequence:
        normalized_body_text: str = normalize_merge_description(
            description=sanitization_result.body_text,
            language=language,
            source_texts=(),
        ).description_text
    else:
        normalized_body_text = sanitization_result.body_text
    final_description: str = _compose_full_text(
        language=language,
        body_text=normalized_body_text,
        cta_text=sanitization_result.cta_text,
        hashtags_line=sanitization_result.hashtags_line,
        youtube_urls=final_selected_youtube_urls,
        source_urls=final_official_links_urls,
    )
    official_links_count: int = len(final_official_links_urls)
    official_links_block_status: str = (
        "emitted"
        if official_links_count > 0
        else (
            "suppressed"
            if official_links_extraction.empty_blocks_suppressed > 0
            else "absent"
        )
    )
    final_layout: str = _resolve_tail_layout(
        body_text=normalized_body_text,
        cta_text=sanitization_result.cta_text,
        hashtags_line=sanitization_result.hashtags_line,
        youtube_urls=final_selected_youtube_urls,
        source_urls=final_official_links_urls,
    )
    LOGGER.info(
        "merged_publish_sanitation_applied=yes lang=%s source=%s cta_found=%s hashtags_found=%s hashtags_split_from_cta=%s tail_layout=%s youtube_links_text_candidates=%d youtube_links_final_count=%d official_links_heading_found=%s official_links_text_links=%d official_links_source_links=%d official_links_final_count=%d official_links_block=%s official_links_dedup_applied=%s empty_official_links_suppressed=%d",
        language,
        source_label,
        "yes" if sanitization_result.cta_found else "no",
        "yes" if sanitization_result.hashtags_found else "no",
        "yes" if sanitization_result.hashtags_split_from_cta else "no",
        final_layout,
        len(selected_youtube_urls),
        len(final_selected_youtube_urls),
        "yes" if official_links_extraction.heading_found else "no",
        len(extracted_text_source_urls),
        authoritative_source_urls_result.emitted_source_video_urls,
        official_links_count,
        official_links_block_status,
        "yes" if authoritative_source_urls_result.duplicate_urls_removed > 0 else "no",
        official_links_extraction.empty_blocks_suppressed,
    )
    return MergedPublicationPayload(
        title_text=sanitize_post_llm_title(raw_title),
        description_text=final_description,
    )


def sanitize_post_llm_text_for_merged_publish(
    *,
    text: str,
    language: str,
    source_label: str,
    source_videos: Sequence[PlannedVideo],
) -> str:
    official_links_extraction: _OfficialLinksExtractionResult = _extract_official_links_blocks(
        text
    )
    cleanup_event_key: str = (
        f"{language}:{source_label}:{hash(_normalize_text(official_links_extraction.cleaned_text))}"
    )
    sanitization_result: PostLlmSanitizationResult = sanitize_post_llm_text(
        official_links_extraction.cleaned_text,
        language=language,
        source_label=source_label,
    )
    extracted_text_source_urls: List[str] = _dedupe_nonempty(
        official_links_extraction.source_urls + sanitization_result.source_urls
    )
    authoritative_source_urls: AuthoritativeSourceUrlsResult = (
        build_authoritative_merged_source_urls(
            language=language,
            source_videos=source_videos,
            extracted_tail_urls=extracted_text_source_urls,
            malformed_tail_urls_dropped=sanitization_result.malformed_source_urls_dropped,
            cleanup_event_key=cleanup_event_key,
        )
    )
    normalized_body_text: str = normalize_merge_description(
        description=sanitization_result.body_text,
        language=language,
        source_texts=(),
    ).description_text
    return _compose_full_text(
        language=language,
        body_text=normalized_body_text,
        cta_text=sanitization_result.cta_text,
        hashtags_line=sanitization_result.hashtags_line,
        youtube_urls=authoritative_source_urls.selected_youtube_urls,
        source_urls=authoritative_source_urls.source_urls,
    )


def sanitize_post_llm_text(
    text: str,
    *,
    language: str = "unknown",
    source_label: str = "unknown",
    log_summary: bool = True,
) -> PostLlmSanitizationResult:
    normalized_text: str = _normalize_text(text)
    if not normalized_text:
        result: PostLlmSanitizationResult = PostLlmSanitizationResult(
            body_text="",
            cta_text="",
            hashtags_line="",
            source_urls=[],
            full_text="",
            normalized_url_count=0,
            tail_was_separated=False,
            cta_found=False,
            hashtags_found=False,
            hashtags_split_from_cta=False,
            tail_layout="empty",
            source_urls_found=0,
            malformed_source_urls_dropped=0,
        )
        if log_summary:
            _log_sanitation_summary(
                language=language,
                source_label=source_label,
                result=result,
            )
        return result

    lines: List[str] = normalized_text.split("\n")
    extracted_tail: _TailParts = _split_tail_parts(lines)
    initial_body_text: str = _join_lines(lines[: extracted_tail.body_end_index])
    embedded_tail: _EmbeddedTailParts = _extract_embedded_tail_fragments(
        initial_body_text
    )
    body_text, body_url_changes = _sanitize_urls_in_text(embedded_tail.body_text)
    cta_lines: List[str] = _dedupe_cta_lines(
        embedded_tail.cta_lines + extracted_tail.cta_lines
    )
    cta_text, cta_url_changes = _sanitize_urls_in_text("\n".join(cta_lines).strip())
    hashtags_line: str = _merge_hashtag_lines(
        embedded_tail.hashtag_lines + extracted_tail.hashtag_lines
    )
    source_urls: List[str] = _dedupe_nonempty(
        embedded_tail.source_urls + extracted_tail.source_urls
    )
    youtube_urls: List[str] = [url for url in source_urls if _is_youtube_url(url)]
    non_youtube_source_urls: List[str] = [
        url for url in source_urls if not _is_youtube_url(url)
    ]
    normalized_url_count: int = (
        extracted_tail.source_url_change_count
        + embedded_tail.source_url_change_count
        + body_url_changes
        + cta_url_changes
    )
    tail_layout: str = _resolve_tail_layout(
        body_text=body_text,
        cta_text=cta_text,
        hashtags_line=hashtags_line,
        youtube_urls=youtube_urls,
        source_urls=non_youtube_source_urls,
    )
    LOGGER.info(
        "tail_parse lang=%s source=%s cta_found=%s hashtags_found=%s hashtags_split_from_cta=%s",
        language,
        source_label,
        "yes" if bool(cta_text) else "no",
        "yes" if bool(hashtags_line) else "no",
        "yes"
        if (embedded_tail.hashtags_split_from_cta or extracted_tail.hashtags_split_from_cta)
        else "no",
    )
    LOGGER.info(
        "tail_layout lang=%s source=%s layout=%s",
        language,
        source_label,
        tail_layout,
    )
    full_text: str = _compose_full_text(
        language=language,
        body_text=body_text,
        cta_text=cta_text,
        hashtags_line=hashtags_line,
        youtube_urls=youtube_urls,
        source_urls=non_youtube_source_urls,
    )
    result = PostLlmSanitizationResult(
        body_text=body_text,
        cta_text=cta_text,
        hashtags_line=hashtags_line,
        source_urls=source_urls,
        full_text=full_text,
        normalized_url_count=normalized_url_count,
        tail_was_separated=bool(cta_text or hashtags_line or source_urls),
        cta_found=bool(cta_text),
        hashtags_found=bool(hashtags_line),
        hashtags_split_from_cta=(
            embedded_tail.hashtags_split_from_cta or extracted_tail.hashtags_split_from_cta
        ),
        tail_layout=tail_layout,
        source_urls_found=len(source_urls),
        malformed_source_urls_dropped=(
            extracted_tail.malformed_source_urls_dropped
            + embedded_tail.malformed_source_urls_dropped
        ),
    )
    if log_summary:
        _log_sanitation_summary(
            language=language,
            source_label=source_label,
            result=result,
        )
    return result


@dataclass(frozen=True)
class _TailParts:
    body_end_index: int
    cta_lines: List[str]
    hashtag_lines: List[str]
    hashtags_split_from_cta: bool
    source_urls: List[str]
    source_url_change_count: int
    malformed_source_urls_dropped: int


@dataclass(frozen=True)
class _EmbeddedTailParts:
    body_text: str
    cta_lines: List[str]
    hashtag_lines: List[str]
    hashtags_split_from_cta: bool
    source_urls: List[str]
    source_url_change_count: int
    malformed_source_urls_dropped: int


def build_authoritative_merged_source_urls(
    *,
    language: str,
    source_videos: Sequence[PlannedVideo],
    extracted_tail_urls: Sequence[str],
    malformed_tail_urls_dropped: int,
    cleanup_event_key: Optional[str] = None,
) -> AuthoritativeSourceUrlsResult:
    authoritative_urls: List[str] = []
    selected_youtube_urls: List[str] = []
    seen_canonical_urls: set[str] = set()
    duplicate_urls_removed: int = 0
    emitted_source_video_urls: int = 0

    for video in source_videos:
        normalized_source_url: Optional[str] = _normalize_authoritative_video_url(video)
        if not normalized_source_url:
            continue
        if _is_youtube_url(normalized_source_url):
            continue
        canonical_key: str = normalized_source_url.rstrip("/")
        if canonical_key in seen_canonical_urls:
            duplicate_urls_removed += 1
            continue
        seen_canonical_urls.add(canonical_key)
        authoritative_urls.append(normalized_source_url)
        emitted_source_video_urls += 1

    preserved_non_youtube_tail_urls: List[str] = []
    for extracted_tail_url in extracted_tail_urls:
        cleaned_tail_url: str = str(extracted_tail_url or "").strip()
        if not cleaned_tail_url or not _is_complete_source_url(cleaned_tail_url):
            continue
        if _is_youtube_url(cleaned_tail_url):
            canonical_key = cleaned_tail_url.rstrip("/")
            if canonical_key in seen_canonical_urls:
                duplicate_urls_removed += 1
                continue
            seen_canonical_urls.add(canonical_key)
            selected_youtube_urls.append(cleaned_tail_url)
            continue
        canonical_key = cleaned_tail_url.rstrip("/")
        if canonical_key in seen_canonical_urls:
            duplicate_urls_removed += 1
            continue
        seen_canonical_urls.add(canonical_key)
        authoritative_urls.append(cleaned_tail_url)
        preserved_non_youtube_tail_urls.append(cleaned_tail_url)

    extracted_tail_dropped_count: int = malformed_tail_urls_dropped + sum(
        1 for item in extracted_tail_urls if not _is_complete_source_url(item)
    )
    LOGGER.info(
        "merged_source_urls_built lang=%s inspected=%d emitted=%d selected_youtube_urls=%d deduped=%d preserved_non_youtube_tail_urls=%d source_urls_mode=authoritative_non_youtube_from_inputs_plus_explicit_tail_selection",
        language,
        len(source_videos),
        len(authoritative_urls),
        len(selected_youtube_urls),
        duplicate_urls_removed,
        len(preserved_non_youtube_tail_urls),
    )
    if extracted_tail_dropped_count > 0:
        LOGGER.info(
            "merged_source_urls_cleaned lang=%s malformed_tail_urls_dropped=%d",
            language,
            extracted_tail_dropped_count,
        )
        record_malformed_tail_url_cleanup(
            extracted_tail_dropped_count,
            event_key=cleanup_event_key,
        )
    return AuthoritativeSourceUrlsResult(
        source_urls=authoritative_urls,
        selected_youtube_urls=selected_youtube_urls,
        inspected_source_videos=len(source_videos),
        emitted_source_urls=len(authoritative_urls),
        emitted_source_video_urls=emitted_source_video_urls,
        emitted_selected_youtube_urls=len(selected_youtube_urls),
        preserved_non_youtube_tail_urls=len(preserved_non_youtube_tail_urls),
        duplicate_urls_removed=duplicate_urls_removed,
        malformed_tail_urls_dropped=extracted_tail_dropped_count,
    )


def _normalize_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_tail_parts(lines: Sequence[str]) -> _TailParts:
    body_end_index: int = len(lines)
    source_urls: List[str] = []
    source_url_change_count: int = 0
    malformed_source_urls_dropped: int = 0
    while body_end_index > 0:
        candidate: str = lines[body_end_index - 1].strip()
        if not candidate:
            body_end_index -= 1
            continue
        if not _is_source_url_line(candidate):
            break
        sanitized_url: Optional[str] = _sanitize_source_url(candidate)
        if sanitized_url is None:
            malformed_source_urls_dropped += 1
            body_end_index -= 1
            continue
        if sanitized_url != candidate:
            source_url_change_count += 1
        source_urls.insert(0, sanitized_url)
        body_end_index -= 1

    hashtag_lines: List[str] = []
    hashtags_split_from_cta: bool = False
    while body_end_index > 0:
        candidate = lines[body_end_index - 1].strip()
        if not candidate:
            body_end_index -= 1
            continue
        if not _is_hashtags_line(candidate):
            break
        hashtag_lines.insert(0, candidate)
        body_end_index -= 1

    cta_lines: List[str] = []
    while body_end_index > 0:
        candidate = lines[body_end_index - 1].strip()
        if not candidate:
            body_end_index -= 1
            continue
        candidate_without_hashtags: str
        extracted_hashtags: str
        candidate_without_hashtags, extracted_hashtags = _extract_hashtag_tail_from_paragraph(
            candidate
        )
        if extracted_hashtags and _is_standalone_cta_line(candidate_without_hashtags):
            hashtag_lines.insert(0, extracted_hashtags)
            hashtags_split_from_cta = True
            candidate = candidate_without_hashtags
        if not candidate:
            body_end_index -= 1
            continue
        if not _is_standalone_cta_line(candidate):
            break
        cta_lines.insert(0, candidate)
        body_end_index -= 1

    return _TailParts(
        body_end_index=body_end_index,
        cta_lines=cta_lines,
        hashtag_lines=hashtag_lines,
        hashtags_split_from_cta=hashtags_split_from_cta,
        source_urls=source_urls,
        source_url_change_count=source_url_change_count,
        malformed_source_urls_dropped=malformed_source_urls_dropped,
    )


def _extract_embedded_tail_fragments(text: str) -> _EmbeddedTailParts:
    paragraphs: List[str] = _split_paragraphs(text)
    cleaned_paragraphs: List[str] = []
    cta_lines: List[str] = []
    hashtag_lines: List[str] = []
    hashtags_split_from_cta: bool = False
    source_urls: List[str] = []
    source_url_change_count: int = 0
    malformed_source_urls_dropped: int = 0

    for paragraph in paragraphs:
        cleaned_paragraph: str = paragraph.strip()
        cleaned_paragraph, extracted_urls, url_change_count, malformed_url_count = (
            _extract_url_tail_from_paragraph(cleaned_paragraph)
        )
        cleaned_paragraph, extracted_hashtags = _extract_hashtag_tail_from_paragraph(
            cleaned_paragraph
        )
        cleaned_paragraph, extracted_cta = _extract_cta_tail_from_paragraph(
            cleaned_paragraph
        )
        if extracted_hashtags and extracted_cta:
            hashtags_split_from_cta = True
        if cleaned_paragraph:
            cleaned_paragraphs.append(cleaned_paragraph)
        source_urls.extend(extracted_urls)
        source_url_change_count += url_change_count
        malformed_source_urls_dropped += malformed_url_count
        if extracted_hashtags:
            hashtag_lines.append(extracted_hashtags)
        if extracted_cta:
            cta_lines.append(extracted_cta)

    return _EmbeddedTailParts(
        body_text="\n\n".join(cleaned_paragraphs).strip(),
        cta_lines=cta_lines,
        hashtag_lines=hashtag_lines,
        hashtags_split_from_cta=hashtags_split_from_cta,
        source_urls=_dedupe_nonempty(source_urls),
        source_url_change_count=source_url_change_count,
        malformed_source_urls_dropped=malformed_source_urls_dropped,
    )


def _extract_url_tail_from_paragraph(
    paragraph: str,
) -> tuple[str, List[str], int, int]:
    normalized_paragraph: str = str(paragraph or "").strip()
    if not normalized_paragraph:
        return ("", [], 0, 0)
    match: Optional[re.Match[str]] = re.search(
        r"(?is)(?:^|[\s\(\[])(https?://\S+(?:\s+https?://\S+)*)\s*$",
        normalized_paragraph,
    )
    if match is None:
        return (normalized_paragraph, [], 0, 0)
    prefix_text: str = normalized_paragraph[: match.start(1)].rstrip()
    raw_urls: List[str] = [item for item in match.group(1).split() if item]
    sanitized_urls: List[str] = []
    change_count: int = 0
    malformed_url_count: int = 0
    for raw_url in raw_urls:
        sanitized_url: Optional[str] = _sanitize_source_url(raw_url)
        if sanitized_url is None:
            malformed_url_count += 1
            continue
        if sanitized_url != raw_url:
            change_count += 1
        sanitized_urls.append(sanitized_url)
    return (
        prefix_text,
        _dedupe_nonempty(sanitized_urls),
        change_count,
        malformed_url_count,
    )


def _extract_hashtag_tail_from_paragraph(paragraph: str) -> tuple[str, str]:
    normalized_paragraph: str = str(paragraph or "").strip()
    if not normalized_paragraph:
        return ("", "")
    match: Optional[re.Match[str]] = re.search(
        r"(?is)(#[^\s#]+(?:\s+#[^\s#]+)*)\s*$",
        normalized_paragraph,
    )
    if match is None:
        return (normalized_paragraph, "")
    hashtag_candidate: str = match.group(1).strip()
    if not _is_hashtags_line(hashtag_candidate):
        return (normalized_paragraph, "")
    prefix_text: str = normalized_paragraph[: match.start(1)].rstrip(" ,;")
    return (prefix_text, hashtag_candidate)


def _extract_cta_tail_from_paragraph(paragraph: str) -> tuple[str, str]:
    normalized_paragraph: str = str(paragraph or "").strip()
    if not normalized_paragraph:
        return ("", "")
    sentence_parts: List[str] = re.split(r"(?<=[.!?…])\s+", normalized_paragraph)
    if len(sentence_parts) < 2:
        if _looks_like_cta_line(normalized_paragraph):
            return ("", normalized_paragraph)
        return (normalized_paragraph, "")
    cta_candidate: str = sentence_parts[-1].strip()
    if not _looks_like_cta_line(cta_candidate):
        return (normalized_paragraph, "")
    body_candidate: str = " ".join(sentence_parts[:-1]).strip()
    if not body_candidate:
        return (normalized_paragraph, "")
    return (body_candidate, cta_candidate)


def _is_source_url_line(line: str) -> bool:
    candidate: str = str(line or "").strip().strip("<>()[]{}").rstrip(".,;")
    return bool(candidate and _URL_LINE_RE.fullmatch(candidate))


def _is_hashtags_line(line: str) -> bool:
    tokens: List[str] = [item for item in str(line or "").split() if item]
    return bool(tokens) and all(_HASHTAG_TOKEN_RE.fullmatch(item) for item in tokens)


def _looks_like_cta_line(line: str) -> bool:
    normalized_line: str = re.sub(r"\s+", " ", str(line or "")).strip().lower()
    if not normalized_line:
        return False
    return any(hint in normalized_line for hint in _CTA_HINTS)


def _is_standalone_cta_line(line: str) -> bool:
    normalized_line: str = re.sub(r"\s+", " ", str(line or "")).strip().lower()
    if not normalized_line:
        return False
    normalized_line = normalized_line.lstrip("-*•> ")
    return any(normalized_line.startswith(hint) for hint in _CTA_HINTS)


def _sanitize_urls_in_text(text: str) -> tuple[str, int]:
    change_count: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal change_count
        raw_url: str = match.group(0)
        sanitized_url: str = _sanitize_url(raw_url)
        if sanitized_url != raw_url:
            change_count += 1
        return sanitized_url

    return (_URL_RE.sub(_replace, str(text or "")), change_count)


def _sanitize_url(url: str) -> str:
    raw_url: str = str(url or "").strip()
    if not raw_url:
        return ""
    prefix: str = ""
    suffix: str = ""
    while raw_url and raw_url[0] in "<([":
        prefix += raw_url[0]
        raw_url = raw_url[1:]
    while raw_url and raw_url[-1] in ">)].,;":
        suffix = raw_url[-1] + suffix
        raw_url = raw_url[:-1]
    if not raw_url:
        return prefix + suffix
    parts = urlsplit(raw_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return prefix + raw_url + suffix
    filtered_query_items: List[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key: str = key.lower().strip()
        if normalized_key.startswith("utm_") or normalized_key in _TRACKING_QUERY_KEYS:
            continue
        filtered_query_items.append((key, value))
    sanitized_query: str = urlencode(filtered_query_items, doseq=True)
    sanitized_url: str = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, sanitized_query, parts.fragment)
    )
    return prefix + sanitized_url + suffix


def _sanitize_source_url(url: str) -> Optional[str]:
    sanitized_url: str = _sanitize_url(url).strip()
    if not sanitized_url:
        return None
    if not _is_complete_source_url(sanitized_url):
        return None
    if not _is_youtube_url(sanitized_url):
        return sanitized_url
    try:
        return normalize_youtube_video_url(sanitized_url)
    except Exception:
        LOGGER.info(
            "non_authoritative_youtube_tail_url_dropped reason=youtube_normalization_failed raw=%r",
            sanitized_url,
        )
        return None


def _normalize_authoritative_video_url(video: PlannedVideo) -> Optional[str]:
    candidates: List[str] = [
        str(video.normalized_link or "").strip(),
        str(video.metadata.url or "").strip(),
        str(video.original_link or "").strip(),
    ]
    for candidate in candidates:
        if not candidate or not _is_complete_source_url(candidate):
            continue
        if not _is_youtube_url(candidate):
            return _sanitize_url(candidate)
        try:
            return normalize_youtube_video_url(candidate)
        except Exception:
            continue
    return None


def _is_youtube_url(url: str) -> bool:
    cleaned_url: str = str(url or "").strip().strip("<>()[]{}").rstrip(".,;")
    if not cleaned_url:
        return False
    try:
        parts = urlsplit(cleaned_url)
    except Exception:
        return False
    host: str = str(parts.netloc or "").strip().lower()
    return host in {"youtu.be", "www.youtu.be", "youtube.com", "www.youtube.com", "m.youtube.com"}


def _is_complete_source_url(url: str) -> bool:
    cleaned_url: str = str(url or "").strip().strip("<>()[]{}").rstrip(".,;")
    if not cleaned_url or not _URL_LINE_RE.fullmatch(cleaned_url):
        return False
    parts = urlsplit(cleaned_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return False
    host: str = parts.netloc.strip().lower()
    if "." not in host or host.endswith(".") or ".." in host:
        return False
    if re.search(r"\s", cleaned_url):
        return False
    return True


def _merge_hashtag_lines(lines: Sequence[str]) -> str:
    deduped_tokens: List[str] = []
    seen_tokens: set[str] = set()
    for line in lines:
        for token in str(line or "").split():
            cleaned_token: str = token.strip()
            if not cleaned_token or not _HASHTAG_TOKEN_RE.fullmatch(cleaned_token):
                continue
            normalized_key: str = cleaned_token.lower()
            if normalized_key in seen_tokens:
                continue
            seen_tokens.add(normalized_key)
            deduped_tokens.append(cleaned_token)
    return " ".join(deduped_tokens)


def _extract_official_links_blocks(text: str) -> _OfficialLinksExtractionResult:
    paragraphs: List[str] = _split_paragraphs(text)
    if not paragraphs:
        return _OfficialLinksExtractionResult(
            cleaned_text="",
            heading_found=False,
            source_urls=[],
            empty_blocks_suppressed=0,
        )
    kept_paragraphs: List[str] = []
    extracted_source_urls: List[str] = []
    suppressed_count: int = 0
    heading_found: bool = False
    paragraph_index: int = 0
    while paragraph_index < len(paragraphs):
        paragraph: str = paragraphs[paragraph_index]
        normalized_paragraph: str = str(paragraph or "").strip()
        paragraph_heading_found, paragraph_source_urls = _extract_official_links_from_heading_paragraph(
            normalized_paragraph
        )
        if not paragraph_heading_found:
            kept_paragraphs.append(normalized_paragraph)
            paragraph_index += 1
            continue
        heading_found = True
        if paragraph_source_urls:
            extracted_source_urls.extend(paragraph_source_urls)
            paragraph_index += 1
            continue
        next_paragraph: str = (
            str(paragraphs[paragraph_index + 1] or "").strip()
            if paragraph_index + 1 < len(paragraphs)
            else ""
        )
        next_paragraph_source_urls: List[str] = _extract_official_links_url_lines(
            next_paragraph.splitlines()
        )
        if next_paragraph_source_urls:
            extracted_source_urls.extend(next_paragraph_source_urls)
            paragraph_index += 2
            continue
        suppressed_count += 1
        paragraph_index += 1
    return _OfficialLinksExtractionResult(
        cleaned_text="\n\n".join(paragraph for paragraph in kept_paragraphs if paragraph).strip(),
        heading_found=heading_found,
        source_urls=_dedupe_nonempty(extracted_source_urls),
        empty_blocks_suppressed=suppressed_count,
    )


def _extract_official_links_from_heading_paragraph(
    paragraph: str,
) -> tuple[bool, List[str]]:
    lines: List[str] = [str(line or "").strip() for line in str(paragraph or "").splitlines()]
    nonempty_lines: List[str] = [line for line in lines if line]
    if not nonempty_lines:
        return (False, [])
    if not _OFFICIAL_LINKS_HEADING_RE.fullmatch(nonempty_lines[0]):
        return (False, [])
    return (True, _extract_official_links_url_lines(nonempty_lines[1:]))


def _extract_official_links_url_lines(lines: Sequence[str]) -> List[str]:
    source_urls: List[str] = []
    for line in lines:
        normalized_line: str = str(line or "").strip()
        if not normalized_line:
            continue
        if not _is_source_url_line(normalized_line):
            return []
        sanitized_url: Optional[str] = _sanitize_source_url(normalized_line)
        if sanitized_url is None or _is_youtube_url(sanitized_url):
            continue
        source_urls.append(sanitized_url)
    return _dedupe_nonempty(source_urls)


def _resolve_official_links_heading(language: str) -> str:
    normalized_language: str = str(language or "").strip().lower()
    if normalized_language == "ru":
        return "🌐 Официальные ссылки:"
    if normalized_language == "uk":
        return "🌐 Офіційні ресурси:"
    return "🌐 Official links:"


def _compose_full_text(
    *,
    language: str,
    body_text: str,
    cta_text: str,
    hashtags_line: str,
    youtube_urls: Sequence[str],
    source_urls: Sequence[str],
) -> str:
    parts: List[str] = []
    if body_text:
        parts.append(body_text.strip())
    if youtube_urls:
        parts.append(
            "\n".join(
                str(youtube_url or "").strip()
                for youtube_url in youtube_urls
                if str(youtube_url or "").strip()
            ).strip()
        )
    if source_urls:
        parts.append(
            "\n".join(
                [_resolve_official_links_heading(language)]
                + [
                    str(source_url or "").strip()
                    for source_url in source_urls
                    if str(source_url or "").strip()
                ]
            ).strip()
        )
    if cta_text:
        parts.append(cta_text.strip())
    if hashtags_line:
        parts.append(hashtags_line.strip())
    return "\n\n".join(part for part in parts if part).strip()


def _resolve_tail_layout(
    *,
    body_text: str,
    cta_text: str,
    hashtags_line: str,
    youtube_urls: Sequence[str],
    source_urls: Sequence[str],
) -> str:
    layout_parts: List[str] = []
    if body_text:
        layout_parts.append("body")
    if youtube_urls:
        layout_parts.extend(("blank", "youtube_links"))
    if source_urls:
        layout_parts.extend(("blank", "official_links"))
    if cta_text:
        layout_parts.extend(("blank", "cta"))
    if hashtags_line:
        layout_parts.extend(("blank", "hashtags"))
    return "_".join(layout_parts) if layout_parts else "empty"


def _split_paragraphs(text: str) -> List[str]:
    normalized_text: str = _normalize_text(text)
    if not normalized_text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized_text) if part.strip()]


def _join_lines(lines: Sequence[str]) -> str:
    return "\n".join(lines).strip()


def _dedupe_nonempty(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned: str = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _dedupe_cta_lines(lines: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned_line: str = re.sub(r"\s+", " ", str(line or "")).strip()
        if not cleaned_line:
            continue
        normalized_key: str = cleaned_line.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        result.append(cleaned_line)
    return result


def _log_sanitation_summary(
    *,
    language: str,
    source_label: str,
    result: PostLlmSanitizationResult,
) -> None:
    LOGGER.info(
        "post_llm_sanitation lang=%s source=%s urls_normalized=%d tail_separated=%s tail_cta_found=%s hashtags_found=%s source_urls_found=%d malformed_source_urls_dropped=%d",
        language,
        source_label,
        result.normalized_url_count,
        "yes" if result.tail_was_separated else "no",
        "yes" if result.cta_found else "no",
        "yes" if result.hashtags_found else "no",
        result.source_urls_found,
        result.malformed_source_urls_dropped,
    )
