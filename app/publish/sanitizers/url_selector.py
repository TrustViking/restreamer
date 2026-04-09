from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence
from urllib.parse import urlsplit

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.language import detect_language_decision, normalize_language
from app.core.models import PlannedVideo
from app.core.official_links import is_official_links_heading
from app.core.text_utils import is_youtube_url as _is_youtube_url_canonical
from app.core.url_normalizer import normalize_display_url
from app.core.url_utils import _canonical_domain_key, normalize_official_link_display, strip_tracking_params
from app.ingest.youtube_metadata import YtDlpYouTubeMetadataFetcher, normalize_youtube_video_url
from app.core.constants import SEMANTIC_STOPWORDS, SEMANTIC_TOKEN_PATTERN, URL_PATTERN, URL_LINE_PATTERN
from app.observability.runtime_analytics import record_malformed_tail_url_cleanup


LOGGER = _get_logger_impl(__name__)

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
    raw_youtube_urls_found: int
    deduped_youtube_candidates: int
    repeated_youtube_candidates: int
    ignored_llm_youtube_urls: int


@dataclass(frozen=True)
class _RecommendedYouTubeCandidate:
    url: str
    source_hits: int
    total_occurrences: int
    semantic_overlap_count: int
    first_seen_order: int


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


def _is_youtube_url(url: str) -> bool:
    return _is_youtube_url_canonical(url)


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
    sanitized_url: str = strip_tracking_params(raw_url)
    sanitized_url = normalize_display_url(sanitized_url)
    return prefix + sanitized_url + suffix


def _is_complete_source_url(url: str) -> bool:
    cleaned_url: str = str(url or "").strip().strip("<>()[]{}").rstrip(".,;")
    if not cleaned_url or not URL_LINE_PATTERN.fullmatch(cleaned_url):
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


def _sanitize_urls_in_text(text: str) -> tuple[str, int]:
    change_count: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal change_count
        raw_url: str = match.group(0)
        sanitized_url: str = _sanitize_url(raw_url)
        if sanitized_url != raw_url:
            change_count += 1
        return sanitized_url

    return (URL_PATTERN.sub(_replace, str(text or "")), change_count)


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


def _determine_recommended_video_language(url: str) -> Optional[str]:
    """Determine unambiguous recommended-video language using metadata and description."""
    cleaned_url: str = str(url or "").strip()
    if not cleaned_url:
        return None
    try:
        metadata = YtDlpYouTubeMetadataFetcher().fetch(cleaned_url)
    except Exception:
        return None

    yt_lang: Optional[str] = normalize_language(metadata.youtube_language)
    ch_lang: Optional[str] = normalize_language(metadata.channel_language)
    decision = detect_language_decision(metadata)
    if decision.final_language == "unknown":
        return None

    if decision.language_conflict:
        LOGGER.info(
            "recommended_video_language_ambiguous url=%s yt_lang=%s ch_lang=%s langdetect_lang=%s action=discard",
            cleaned_url,
            yt_lang or "none",
            ch_lang or "none",
            decision.langdetect_language or "none",
        )
        return None

    return decision.final_language


class AuthoritativeUrlSelector:
    """Selects and builds authoritative source URLs for publication."""

    @staticmethod
    def _extract_semantic_tokens(text: str) -> set[str]:
        tokens: set[str] = set()
        for token in SEMANTIC_TOKEN_PATTERN.findall(str(text or "").lower()):
            normalized_token: str = token.strip().lower()
            if not normalized_token or normalized_token in SEMANTIC_STOPWORDS:
                continue
            tokens.add(normalized_token)
        return tokens

    @staticmethod
    def _strip_urls_and_hashtags_for_context(text: str) -> str:
        cleaned_text: str = URL_PATTERN.sub(" ", str(text or ""))
        cleaned_text = re.sub(r"(?<!\w)#[^\s#]+", " ", cleaned_text, flags=re.UNICODE)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        return cleaned_text.strip(" ,;:-")

    @staticmethod
    def _extract_raw_description_urls(
        source_videos: Sequence[PlannedVideo],
    ) -> tuple[List[tuple[str, str, int, int]], List[tuple[str, str]], int]:
        all_occurrences: List[tuple[str, str, int, int]] = []
        non_youtube_occurrences: List[tuple[str, str]] = []
        raw_youtube_urls_found: int = 0
        for source_index, video in enumerate(source_videos):
            lines: List[str] = str(video.metadata.description or "").replace("\r\n", "\n").replace(
                "\r", "\n"
            ).split("\n")
            for line_index, raw_line in enumerate(lines):
                line: str = str(raw_line or "").strip()
                if not line:
                    continue
                urls: List[str] = [str(match.group(0) or "").strip() for match in URL_PATTERN.finditer(line)]
                if not urls:
                    continue
                fallback_context: str = (
                    AuthoritativeUrlSelector._strip_urls_and_hashtags_for_context(line)
                    or str(video.metadata.title or "").strip()
                )
                for raw_url in urls:
                    sanitized_url: Optional[str] = _sanitize_source_url(raw_url)
                    if sanitized_url is None:
                        continue
                    if _is_youtube_url(sanitized_url):
                        raw_youtube_urls_found += 1
                    all_occurrences.append((sanitized_url, fallback_context, source_index, line_index))
                    if not _is_youtube_url(sanitized_url):
                        non_youtube_occurrences.append((sanitized_url, line))
        return (all_occurrences, non_youtube_occurrences, raw_youtube_urls_found)

    @staticmethod
    def select_recommended_youtube(
        *,
        source_videos: Sequence[PlannedVideo],
        summary_text: str,
        source_count: int,
        target_language: str,
    ) -> tuple[List[str], int, int, int]:
        all_occurrences, _, raw_youtube_urls_found = AuthoritativeUrlSelector._extract_raw_description_urls(source_videos)
        summary_tokens: set[str] = AuthoritativeUrlSelector._extract_semantic_tokens(summary_text)
        candidate_contexts: dict[str, List[str]] = {}
        candidate_source_indices: dict[str, set[int]] = {}
        candidate_occurrence_counts: dict[str, int] = {}
        candidate_first_seen: dict[str, int] = {}

        for normalized_url, context_text, source_index, line_index in all_occurrences:
            if not _is_youtube_url(normalized_url):
                continue
            candidate_contexts.setdefault(normalized_url, []).append(context_text)
            candidate_source_indices.setdefault(normalized_url, set()).add(source_index)
            candidate_occurrence_counts[normalized_url] = (
                candidate_occurrence_counts.get(normalized_url, 0) + 1
            )
            candidate_first_seen.setdefault(normalized_url, len(candidate_first_seen))

        candidates: List[_RecommendedYouTubeCandidate] = []
        repeated_candidates: int = 0
        for url, contexts in candidate_contexts.items():
            source_hits: int = len(candidate_source_indices.get(url, set()))
            if source_hits >= 2:
                repeated_candidates += 1
            context_tokens: set[str] = set()
            for context_text in contexts:
                context_tokens.update(AuthoritativeUrlSelector._extract_semantic_tokens(context_text))
            semantic_overlap_count: int = len(summary_tokens & context_tokens)
            LOGGER.debug(
                "recommended_candidate_evaluated url=%s source_hits=%d semantic_overlap_count=%d total_occurrences=%d",
                url,
                source_hits,
                semantic_overlap_count,
                candidate_occurrence_counts.get(url, 0),
            )
            candidates.append(
                _RecommendedYouTubeCandidate(
                    url=url,
                    source_hits=source_hits,
                    total_occurrences=candidate_occurrence_counts.get(url, 0),
                    semantic_overlap_count=semantic_overlap_count,
                    first_seen_order=candidate_first_seen[url],
                )
            )

        candidates.sort(
            key=lambda candidate: (
                candidate.source_hits >= 2,
                candidate.source_hits,
                candidate.semantic_overlap_count,
                candidate.total_occurrences,
                -candidate.first_seen_order,
            ),
            reverse=True,
        )

        language_safe_candidates: List[_RecommendedYouTubeCandidate] = []
        for candidate in candidates:
            detected_language: Optional[str] = _determine_recommended_video_language(candidate.url)
            if detected_language is None:
                LOGGER.info(
                    "recommended_candidate_language_filtered url=%s target=%s detected=ambiguous action=discard",
                    candidate.url,
                    target_language,
                )
                continue
            if detected_language != target_language:
                LOGGER.info(
                    "recommended_candidate_language_filtered url=%s target=%s detected=%s action=discard",
                    candidate.url,
                    target_language,
                    detected_language,
                )
                continue
            language_safe_candidates.append(candidate)

        selected_urls: List[str] = []
        _low_source_mode: bool = source_count <= 2
        for candidate in language_safe_candidates:
            if _low_source_mode:
                passes_threshold: bool = (
                    candidate.source_hits >= 1 and candidate.semantic_overlap_count >= 1
                )
            else:
                passes_threshold = (
                    candidate.source_hits >= 2 or candidate.semantic_overlap_count >= 2
                )
            if not passes_threshold:
                continue
            selected_urls.append(candidate.url)
            if len(selected_urls) >= 2:
                break
        if not selected_urls:
            for candidate in language_safe_candidates:
                if candidate.source_hits >= 2:
                    selected_urls.append(candidate.url)
                    LOGGER.info(
                        "recommended_materials_fallback_applied url=%s source_hits=%d semantic_overlap=%d reason=no_candidates_passed_threshold",
                        candidate.url,
                        candidate.source_hits,
                        candidate.semantic_overlap_count,
                    )
                    break
        LOGGER.info(
            "recommended_materials_candidates_built summary_tokens=%d raw_youtube_urls_found=%d deduped_candidates=%d repeated_in_multiple_sources=%d selected=%d selection_mode=deterministic_source_hits_then_semantic_overlap",
            len(summary_tokens),
            raw_youtube_urls_found,
            len(candidates),
            repeated_candidates,
            len(selected_urls),
        )
        return (selected_urls, raw_youtube_urls_found, len(candidates), repeated_candidates)

    @staticmethod
    def select_authoritative_non_youtube(
        *,
        source_videos: Sequence[PlannedVideo],
        extracted_tail_urls: Sequence[str],
    ) -> tuple[List[str], int, int]:
        authoritative_urls: List[str] = []
        seen_canonical_urls: set[str] = set()
        duplicate_urls_removed: int = 0
        emitted_source_video_urls: int = 0
        _, non_youtube_occurrences, _ = AuthoritativeUrlSelector._extract_raw_description_urls(source_videos)
        domain_counts: dict[str, int] = {}
        raw_candidates: List[tuple[str, str]] = []

        for normalized_url, line_text in non_youtube_occurrences:
            raw_candidates.append((normalized_url, line_text))
            domain: str = urlsplit(normalized_url).netloc.lower().strip()
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        for video in source_videos:
            normalized_source_url: Optional[str] = _normalize_authoritative_video_url(video)
            if not normalized_source_url or _is_youtube_url(normalized_source_url):
                continue
            raw_candidates.append((normalized_source_url, str(video.metadata.title or "").strip()))

        scored_candidates: List[tuple[int, int, str]] = []
        for index, (url, line_text) in enumerate(raw_candidates):
            parts = urlsplit(url)
            domain = parts.netloc.lower().strip()
            score: int = 0
            if parts.scheme == "https":
                score += 20
            if is_official_links_heading(str(line_text or "")) or any(
                hint in str(line_text or "").lower()
                for hint in ("official", "resource", "resources", "details", "site", "website")
            ):
                score += 20
            score += min(20, domain_counts.get(domain, 1) * 5)
            if parts.query:
                score -= 5
            score += max(0, 15 - (len(url) // 15))
            scored_candidates.append((score, -index, url))

        scored_candidates.sort(reverse=True)
        for _, _, url in scored_candidates:
            normalized_official_url: str = normalize_official_link_display(url)
            canonical_key: str = _canonical_domain_key(normalized_official_url)
            if canonical_key in seen_canonical_urls:
                duplicate_urls_removed += 1
                continue
            seen_canonical_urls.add(canonical_key)
            authoritative_urls.append(normalized_official_url)
            emitted_source_video_urls += 1
            if len(authoritative_urls) >= 3:
                break

        preserved_non_youtube_tail_urls: List[str] = []
        for extracted_tail_url in extracted_tail_urls:
            cleaned_tail_url: str = str(extracted_tail_url or "").strip()
            if (
                not cleaned_tail_url
                or not _is_complete_source_url(cleaned_tail_url)
                or _is_youtube_url(cleaned_tail_url)
            ):
                continue
            normalized_tail_url: str = normalize_official_link_display(cleaned_tail_url)
            canonical_key = _canonical_domain_key(normalized_tail_url)
            if canonical_key in seen_canonical_urls:
                duplicate_urls_removed += 1
                continue
            seen_canonical_urls.add(canonical_key)
            authoritative_urls.append(normalized_tail_url)
            preserved_non_youtube_tail_urls.append(normalized_tail_url)

        return (authoritative_urls, emitted_source_video_urls, duplicate_urls_removed)

    @staticmethod
    def build_authoritative(
        *,
        language: str,
        source_videos: Sequence[PlannedVideo],
        extracted_tail_urls: Sequence[str],
        malformed_tail_urls_dropped: int,
        summary_text: str,
        cleanup_event_key: Optional[str] = None,
    ) -> AuthoritativeSourceUrlsResult:
        selected_youtube_urls, raw_youtube_urls_found, deduped_youtube_candidates, repeated_youtube_candidates = (
            AuthoritativeUrlSelector.select_recommended_youtube(
                source_videos=source_videos,
                summary_text=summary_text,
                source_count=len(source_videos),
                target_language=language,
            )
        )
        authoritative_urls, emitted_source_video_urls, duplicate_urls_removed = (
            AuthoritativeUrlSelector.select_authoritative_non_youtube(
                source_videos=source_videos,
                extracted_tail_urls=extracted_tail_urls,
            )
        )
        preserved_non_youtube_tail_urls: List[str] = [
            str(item or "").strip()
            for item in extracted_tail_urls
            if str(item or "").strip()
            and _is_complete_source_url(str(item or "").strip())
            and not _is_youtube_url(str(item or "").strip())
        ]
        ignored_llm_youtube_urls: int = sum(
            1 for item in extracted_tail_urls if _is_youtube_url(str(item or "").strip())
        )

        extracted_tail_dropped_count: int = malformed_tail_urls_dropped + sum(
            1 for item in extracted_tail_urls if not _is_complete_source_url(item)
        )
        LOGGER.info(
            "merged_source_urls_built lang=%s inspected=%d emitted=%d selected_youtube_urls=%d deduped=%d preserved_non_youtube_tail_urls=%d raw_youtube_urls_found=%d deduped_youtube_candidates=%d repeated_youtube_candidates=%d ignored_llm_youtube_urls=%d source_urls_mode=authoritative_non_youtube_from_inputs_plus_script_selected_recommended_materials",
            language,
            len(source_videos),
            len(authoritative_urls),
            len(selected_youtube_urls),
            duplicate_urls_removed,
            len(preserved_non_youtube_tail_urls),
            raw_youtube_urls_found,
            deduped_youtube_candidates,
            repeated_youtube_candidates,
            ignored_llm_youtube_urls,
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
            raw_youtube_urls_found=raw_youtube_urls_found,
            deduped_youtube_candidates=deduped_youtube_candidates,
            repeated_youtube_candidates=repeated_youtube_candidates,
            ignored_llm_youtube_urls=ignored_llm_youtube_urls,
        )


# Backward-compat module-level wrappers

def build_authoritative_merged_source_urls(
    *,
    language: str,
    source_videos: Sequence[PlannedVideo],
    extracted_tail_urls: Sequence[str],
    malformed_tail_urls_dropped: int,
    summary_text: str,
    cleanup_event_key: Optional[str] = None,
) -> AuthoritativeSourceUrlsResult:
    return AuthoritativeUrlSelector.build_authoritative(
        language=language,
        source_videos=source_videos,
        extracted_tail_urls=extracted_tail_urls,
        malformed_tail_urls_dropped=malformed_tail_urls_dropped,
        summary_text=summary_text,
        cleanup_event_key=cleanup_event_key,
    )


def _select_authoritative_non_youtube_urls(
    *,
    source_videos: Sequence[PlannedVideo],
    extracted_tail_urls: Sequence[str],
) -> tuple[List[str], int, int]:
    return AuthoritativeUrlSelector.select_authoritative_non_youtube(
        source_videos=source_videos,
        extracted_tail_urls=extracted_tail_urls,
    )
