from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import List, Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.cta_detection import looks_like_cta_line as _cta_gate_check
from app.core.models import (
    BLOCK_GENERATION_MODE_REAL_MERGE,
    LanguageMergeAttempt,
    MergedLanguageContent,
    MergedPublicationPayload as _CoreMergedPublicationPayload,
    PlannedVideo,
)
from app.core.text_utils import normalize_multiline_text
from app.ingest.youtube_metadata import YtDlpYouTubeMetadataFetcher
from app.llm.merges.merge_quality import normalize_merge_description

# Backward compat re-exports from submodules
from app.publish.sanitizers.tail_parser import (
    TailParser,
    TailParts as _TailParts,
    EmbeddedTailParts as _EmbeddedTailParts,
    TailParts,
    EmbeddedTailParts,
)
from app.publish.sanitizers.url_selector import (
    AuthoritativeSourceUrlsResult,
    _select_authoritative_non_youtube_urls,
    build_authoritative_merged_source_urls,
    _sanitize_urls_in_text,
    _sanitize_url,
    _sanitize_source_url,
    _is_youtube_url,
    _is_complete_source_url,
    _dedupe_nonempty,
)
from app.publish.sanitizers.description_composer import (
    DescriptionComposer,
    _OfficialLinksExtractionResult,
)
from app.publish.sanitizers.quality_gate import PublishQualityGate


LOGGER = _get_logger_impl(__name__)

# Keep old function names accessible for tests
_clean_double_bullet_markers = TailParser.clean_double_bullet_markers


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
class MergedPublicationPayload(_CoreMergedPublicationPayload):
    has_publish_stage_duplicate: bool = field(default=False)
    has_publish_stage_opener_cta: bool = field(default=False)


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


def resolve_block_generation_mode(
    *,
    merge_attempt: Optional[LanguageMergeAttempt],
    merged_content: Optional[MergedLanguageContent] = None,
) -> str:
    merge_attempt_mode: str = str(
        getattr(merge_attempt, "block_generation_mode", "") if merge_attempt is not None else ""
    ).strip()
    if merge_attempt_mode:
        return merge_attempt_mode
    merged_content_mode: str = str(
        getattr(merged_content, "block_generation_mode", "")
        if merged_content is not None
        else ""
    ).strip()
    if merged_content_mode:
        return merged_content_mode
    return BLOCK_GENERATION_MODE_REAL_MERGE


def should_suppress_raw_merge_attempt_publish(
    *,
    merge_attempt: Optional[LanguageMergeAttempt],
    merged_content_available: bool,
) -> bool:
    return merge_attempt is not None and not merged_content_available


def log_safe_merge_attempt_fallback(
    *,
    target: str,
    merge_attempt: LanguageMergeAttempt,
    fallback_label: str,
) -> None:
    source_label: str = resolve_post_llm_source_label(
        merge_attempt,
        default_label="merge_attempt_raw",
    )
    raw_response_text: str = str(merge_attempt.raw_response_text or "").strip()
    LOGGER.info(
        "merge_publish_fallback target=%s language=%s block_generation_mode=%s merge_source=%s merge_failed=yes raw_output_suppressed=%s raw_chars=%d fallback=%s",
        target,
        merge_attempt.language,
        resolve_block_generation_mode(merge_attempt=merge_attempt),
        source_label,
        "yes" if bool(raw_response_text) else "no",
        len(raw_response_text),
        fallback_label,
    )


def _final_description_has_duplicate_paragraphs(description_text: str) -> bool:
    return PublishQualityGate.has_duplicate_paragraphs(description_text)


def _final_description_has_opener_cta(description_text: str) -> bool:
    return PublishQualityGate.has_opener_cta(description_text)


def _fetch_recommended_youtube_title(url: str) -> Optional[str]:
    recommended_url: str = str(url or "").strip()
    if not recommended_url:
        return None
    LOGGER.info("recommended_title_fetch_started url=%s", recommended_url)
    try:
        metadata = YtDlpYouTubeMetadataFetcher().fetch(recommended_url)
    except Exception as error:
        LOGGER.info(
            "recommended_title_fetch_failed url=%s reason=%s",
            recommended_url,
            error,
        )
        return None
    title_text: str = str(metadata.title or "")
    if not title_text:
        LOGGER.info("recommended_title_fetch_failed url=%s reason=empty_title", recommended_url)
        return None
    LOGGER.info("recommended_title_fetch_success url=%s", recommended_url)
    return title_text


def _extract_official_links_blocks(text: str) -> _OfficialLinksExtractionResult:
    composer: DescriptionComposer = DescriptionComposer()
    return composer.extract_official_links(text)


def _resolve_tail_layout(
    *,
    body_text: str,
    cta_text: str,
    hashtags_line: str,
    recommended_youtube_urls: Sequence[str],
    source_urls: Sequence[str],
) -> str:
    return DescriptionComposer.resolve_layout(
        body_text=body_text,
        cta_text=cta_text,
        hashtags_line=hashtags_line,
        recommended_youtube_urls=recommended_youtube_urls,
        source_urls=source_urls,
    )


def _compose_full_text(
    *,
    language: str,
    body_text: str,
    cta_text: str,
    hashtags_line: str,
    recommended_youtube_urls: Sequence[str],
    source_urls: Sequence[str],
) -> str:
    """Module-level wrapper; passes module-level _fetch_recommended_youtube_title so tests can patch it."""
    composer: DescriptionComposer = DescriptionComposer()
    return composer.compose(
        language=language,
        body_text=body_text,
        cta_text=cta_text,
        hashtags_line=hashtags_line,
        recommended_youtube_urls=recommended_youtube_urls,
        source_urls=source_urls,
        fetch_title_fn=_fetch_recommended_youtube_title,
    )


def _apply_publish_cta_gate(
    *,
    language: str,
    source_label: str,
    cta_text: str,
) -> str:
    sanitized_cta_text: str = str(cta_text or "").strip()
    if sanitized_cta_text and _cta_gate_check(sanitized_cta_text):
        LOGGER.info(
            "merged_publish_cta_gate_dropped lang=%s source=%s cta_chars=%d",
            language,
            source_label,
            len(sanitized_cta_text),
        )
        return ""
    return sanitized_cta_text


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
    block_generation_mode: str = resolve_block_generation_mode(
        merge_attempt=merge_attempt,
        merged_content=merged_content,
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
    authoritative_source_urls_result: AuthoritativeSourceUrlsResult = (
        build_authoritative_merged_source_urls(
            language=language,
            source_videos=source_videos_sequence,
            extracted_tail_urls=extracted_text_source_urls,
            malformed_tail_urls_dropped=sanitization_result.malformed_source_urls_dropped,
            summary_text=f"{raw_title}\n{sanitization_result.body_text}".strip(),
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
    sanitized_cta_text: str = _apply_publish_cta_gate(
        language=language,
        source_label=source_label,
        cta_text=sanitization_result.cta_text,
    )
    final_description: str = _compose_full_text(
        language=language,
        body_text=normalized_body_text,
        cta_text=sanitized_cta_text,
        hashtags_line=sanitization_result.hashtags_line,
        recommended_youtube_urls=final_selected_youtube_urls,
        source_urls=final_official_links_urls,
    )
    has_publish_stage_duplicate: bool = _final_description_has_duplicate_paragraphs(
        final_description
    )
    has_publish_stage_opener_cta: bool = _final_description_has_opener_cta(
        final_description
    )
    if has_publish_stage_duplicate:
        LOGGER.error(
            "merged_publish_duplicate_paragraph_detected lang=%s source=%s description_chars=%d",
            language,
            source_label,
            len(final_description),
        )
    if has_publish_stage_opener_cta:
        LOGGER.error(
            "merged_publish_opener_cta_detected lang=%s source=%s description_chars=%d",
            language,
            source_label,
            len(final_description),
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
        cta_text=sanitized_cta_text,
        hashtags_line=sanitization_result.hashtags_line,
        recommended_youtube_urls=final_selected_youtube_urls,
        source_urls=final_official_links_urls,
    )
    recommended_block_status: str = "emitted" if final_selected_youtube_urls else "skipped"
    LOGGER.info(
        "merged_publish_sanitation_applied=yes lang=%s source=%s cta_found=%s hashtags_found=%s hashtags_split_from_cta=%s tail_layout=%s recommended_materials_text_candidates_ignored=%d raw_youtube_urls_found=%d deduped_youtube_candidates=%d repeated_youtube_candidates=%d recommended_materials_final_count=%d recommended_materials_block=%s official_links_heading_found=%s official_links_text_links=%d official_links_source_links=%d official_links_final_count=%d official_links_block=%s official_links_dedup_applied=%s official_links_non_youtube_only=yes empty_official_links_suppressed=%d ignored_llm_youtube_urls=%d",
        language,
        source_label,
        "yes" if sanitization_result.cta_found else "no",
        "yes" if sanitization_result.hashtags_found else "no",
        "yes" if sanitization_result.hashtags_split_from_cta else "no",
        final_layout,
        len([url for url in extracted_text_source_urls if _is_youtube_url(url)]),
        authoritative_source_urls_result.raw_youtube_urls_found,
        authoritative_source_urls_result.deduped_youtube_candidates,
        authoritative_source_urls_result.repeated_youtube_candidates,
        len(final_selected_youtube_urls),
        recommended_block_status,
        "yes" if official_links_extraction.heading_found else "no",
        len([url for url in extracted_text_source_urls if not _is_youtube_url(url)]),
        authoritative_source_urls_result.emitted_source_video_urls,
        official_links_count,
        official_links_block_status,
        "yes" if authoritative_source_urls_result.duplicate_urls_removed > 0 else "no",
        official_links_extraction.empty_blocks_suppressed,
        authoritative_source_urls_result.ignored_llm_youtube_urls,
    )
    return MergedPublicationPayload(
        title_text=sanitize_post_llm_title(raw_title),
        description_text=final_description,
        block_generation_mode=block_generation_mode,
        has_publish_stage_duplicate=has_publish_stage_duplicate,
        has_publish_stage_opener_cta=has_publish_stage_opener_cta,
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
            summary_text=sanitization_result.body_text,
            cleanup_event_key=cleanup_event_key,
        )
    )
    normalized_body_text: str = normalize_merge_description(
        description=sanitization_result.body_text,
        language=language,
        source_texts=(),
    ).description_text
    normalized_body_text = _clean_double_bullet_markers(normalized_body_text)
    sanitized_cta_text: str = _apply_publish_cta_gate(
        language=language,
        source_label=source_label,
        cta_text=sanitization_result.cta_text,
    )
    return _compose_full_text(
        language=language,
        body_text=normalized_body_text,
        cta_text=sanitized_cta_text,
        hashtags_line=sanitization_result.hashtags_line,
        recommended_youtube_urls=authoritative_source_urls.selected_youtube_urls,
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

    parser: TailParser = TailParser()
    lines: List[str] = normalized_text.split("\n")
    extracted_tail: _TailParts = parser.split_tail(lines)
    initial_body_text: str = _join_lines(lines[: extracted_tail.body_end_index])
    embedded_tail: _EmbeddedTailParts = parser.extract_embedded(initial_body_text)
    body_text, body_url_changes = _sanitize_urls_in_text(embedded_tail.body_text)
    body_text = parser.clean_double_bullet_markers(body_text)
    cta_lines: List[str] = parser.dedupe_cta_lines(
        embedded_tail.cta_lines + extracted_tail.cta_lines
    )
    cta_text, cta_url_changes = _sanitize_urls_in_text("\n".join(cta_lines).strip())
    hashtags_line: str = parser.merge_hashtag_lines(
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
        recommended_youtube_urls=youtube_urls,
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
    sanitized_cta_text: str = _apply_publish_cta_gate(
        language=language,
        source_label=source_label,
        cta_text=cta_text,
    )
    full_text: str = _compose_full_text(
        language=language,
        body_text=body_text,
        cta_text=sanitized_cta_text,
        hashtags_line=hashtags_line,
        recommended_youtube_urls=youtube_urls,
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


def _normalize_text(text: str) -> str:
    return normalize_multiline_text(text)


def _join_lines(lines: Sequence[str]) -> str:
    return "\n".join(lines).strip()


def _normalize_authoritative_video_url(video: PlannedVideo) -> Optional[str]:
    from app.publish.sanitizers.url_selector import _normalize_authoritative_video_url as _impl
    return _impl(video)


def _extract_semantic_tokens(text: str) -> set[str]:
    return AuthoritativeUrlSelector._extract_semantic_tokens(text)


def _strip_urls_and_hashtags_for_context(text: str) -> str:
    return AuthoritativeUrlSelector._strip_urls_and_hashtags_for_context(text)


def _extract_raw_description_urls(
    source_videos: Sequence[PlannedVideo],
) -> tuple[List[tuple[str, str, int, int]], List[tuple[str, str]], int]:
    return AuthoritativeUrlSelector._extract_raw_description_urls(source_videos)


# Import AuthoritativeUrlSelector for the _extract_semantic_tokens / _strip_urls_and_hashtags wrappers
from app.publish.sanitizers.url_selector import AuthoritativeUrlSelector


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
