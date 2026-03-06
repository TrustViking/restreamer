from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, List, Optional, Sequence

from app.core.models import LanguageMergeAttempt, MergedLanguageContent, PlannedVideo
from app.publish.post_llm_sanitation import (
    PostLlmSanitizationResult,
    resolve_post_llm_source_label,
    sanitize_post_llm_text,
)


@dataclass(frozen=True)
class ContentContractSnapshot:
    mode: str
    title_mode: str
    title_present: bool
    title_chars: int
    paragraph_count_min: int
    paragraph_count_max: int
    paragraph_count_actual: int
    paragraph_count_valid: bool
    cta_present: bool
    cta_chars: int
    hashtags_present: bool
    hashtags_count: int
    links_allowed_max: int
    links_actual: int
    links_valid: bool
    tail_source: str
    contract_status: str
    contract_reason_codes: List[str]


@dataclass(frozen=True)
class ContentContractTransition:
    pre_snapshot: ContentContractSnapshot
    post_snapshot: ContentContractSnapshot
    sanitization_result: PostLlmSanitizationResult
    transition_status: str


def build_pre_sanitation_text(
    *,
    merged_content: Optional[MergedLanguageContent],
    source_videos: Sequence[PlannedVideo],
    use_audit_text: bool,
) -> str:
    del source_videos
    if merged_content is None:
        return ""
    return (
        str(merged_content.description_audit or merged_content.description or "").strip()
        if use_audit_text
        else str(merged_content.description_selected or merged_content.description or "").strip()
    )


def infer_title_mode(
    *,
    source_videos: Sequence[PlannedVideo],
    merged_content: Optional[MergedLanguageContent],
    processing_mode: str,
) -> str:
    if processing_mode == "nomerge":
        return "enumerated" if len(source_videos) > 1 else "single_source"
    if merged_content is not None:
        return "merged"
    if len(source_videos) == 1:
        return "single_source"
    return "fallback_titles"


def infer_tail_source(
    *,
    merge_attempt: Optional[LanguageMergeAttempt],
    processing_mode: str,
) -> str:
    if processing_mode == "nomerge":
        return "fallback_from_sources"
    source_label: str = resolve_post_llm_source_label(
        merge_attempt,
        default_label="none",
    )
    return source_label or "none"


def analyze_content_contract(
    *,
    title_text: str,
    description_text: str,
    source_videos: Sequence[PlannedVideo],
    mode: str,
    title_mode: str,
    tail_source: str,
) -> tuple[ContentContractSnapshot, PostLlmSanitizationResult]:
    del source_videos
    sanitized_result: PostLlmSanitizationResult = sanitize_post_llm_text(
        description_text,
        language="contract_check",
        source_label=f"contract:{tail_source}",
        log_summary=False,
    )
    paragraph_count_actual: int = len(_split_paragraphs(sanitized_result.body_text))
    paragraph_count_min: int = 2 if mode == "merge" else 0
    paragraph_count_max: int = 4 if mode == "merge" else 0
    paragraph_count_valid: bool = (
        paragraph_count_min <= paragraph_count_actual <= paragraph_count_max
        if mode == "merge"
        else paragraph_count_actual >= 0
    )
    hashtags_count: int = len([token for token in sanitized_result.hashtags_line.split() if token.strip()])
    links_allowed_max: int = 0
    links_actual: int = len(sanitized_result.source_urls)
    links_valid: bool = links_actual == 0

    reason_codes: List[str] = []
    if not str(title_text or "").strip():
        reason_codes.append("title_missing")
    if mode == "merge" and not paragraph_count_valid:
        reason_codes.append("paragraph_count_invalid")
    if mode == "merge" and not links_valid:
        reason_codes.append("links_limit_exceeded")

    contract_status: str = _resolve_contract_status(
        reason_codes=reason_codes,
        title_present=bool(str(title_text or "").strip()),
        paragraph_count_actual=paragraph_count_actual,
    )
    snapshot: ContentContractSnapshot = ContentContractSnapshot(
        mode=mode,
        title_mode=title_mode,
        title_present=bool(str(title_text or "").strip()),
        title_chars=len(str(title_text or "").strip()),
        paragraph_count_min=paragraph_count_min,
        paragraph_count_max=paragraph_count_max,
        paragraph_count_actual=paragraph_count_actual,
        paragraph_count_valid=paragraph_count_valid,
        cta_present=sanitized_result.cta_found,
        cta_chars=len(sanitized_result.cta_text),
        hashtags_present=sanitized_result.hashtags_found,
        hashtags_count=hashtags_count,
        links_allowed_max=links_allowed_max,
        links_actual=links_actual,
        links_valid=links_valid,
        tail_source=tail_source,
        contract_status=contract_status,
        contract_reason_codes=reason_codes,
    )
    return (snapshot, sanitized_result)


def build_contract_transition(
    *,
    pre_snapshot: ContentContractSnapshot,
    post_snapshot: ContentContractSnapshot,
    sanitization_result: PostLlmSanitizationResult,
) -> ContentContractTransition:
    transition_status: str = post_snapshot.contract_status
    if pre_snapshot.contract_reason_codes and not post_snapshot.contract_reason_codes:
        transition_status = "degraded_recovered"
    return ContentContractTransition(
        pre_snapshot=pre_snapshot,
        post_snapshot=post_snapshot,
        sanitization_result=sanitization_result,
        transition_status=transition_status,
    )


def log_content_contract(
    *,
    logger: logging.Logger,
    event_name: str,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    snapshot: ContentContractSnapshot,
) -> None:
    logger.info(
        "%s branch=%s date_key=%s slot_key=%s lang=%s mode=%s title_mode=%s title_present=%s title_chars=%d paragraph_count_min=%d paragraph_count_max=%d paragraph_count_actual=%d paragraph_count_valid=%s cta_present=%s cta_chars=%d hashtags_present=%s hashtags_count=%d links_allowed_max=%d links_actual=%d links_valid=%s tail_source=%s contract_status=%s contract_reason_codes=%s",
        event_name,
        branch_label,
        date_key,
        slot_key,
        language,
        snapshot.mode,
        snapshot.title_mode,
        _yes_no(snapshot.title_present),
        snapshot.title_chars,
        snapshot.paragraph_count_min,
        snapshot.paragraph_count_max,
        snapshot.paragraph_count_actual,
        _yes_no(snapshot.paragraph_count_valid),
        _yes_no(snapshot.cta_present),
        snapshot.cta_chars,
        _yes_no(snapshot.hashtags_present),
        snapshot.hashtags_count,
        snapshot.links_allowed_max,
        snapshot.links_actual,
        _yes_no(snapshot.links_valid),
        snapshot.tail_source or "none",
        snapshot.contract_status,
        _join_codes(snapshot.contract_reason_codes),
    )


def log_tail_analysis(
    *,
    logger: logging.Logger,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    tail_source: str,
    sanitization_result: PostLlmSanitizationResult,
) -> None:
    cta_origin: str = _tail_origin_from_source(tail_source, found=sanitization_result.cta_found)
    hashtags_origin: str = _tail_origin_from_source(
        tail_source,
        found=sanitization_result.hashtags_found,
    )
    links_origin: str = "llm_links" if sanitization_result.source_urls_found > 0 else "none"
    reason_codes: List[str] = []
    if not sanitization_result.cta_found:
        reason_codes.append("cta_missing")
    if not sanitization_result.hashtags_found:
        reason_codes.append("hashtags_missing")
    if sanitization_result.malformed_source_urls_dropped > 0:
        reason_codes.append("malformed_links_dropped")
    logger.info(
        "tail_analysis branch=%s date_key=%s slot_key=%s lang=%s tail_source=%s cta_found=%s cta_origin=%s hashtags_found=%s hashtags_origin=%s hashtags_count=%d links_found=%d links_origin=%s urls_normalized_count=%d malformed_links_dropped=%d tail_separated=%s tail_reason_codes=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        tail_source or "none",
        _yes_no(sanitization_result.cta_found),
        cta_origin,
        _yes_no(sanitization_result.hashtags_found),
        hashtags_origin,
        len([token for token in sanitization_result.hashtags_line.split() if token.strip()]),
        sanitization_result.source_urls_found,
        links_origin,
        sanitization_result.normalized_url_count,
        sanitization_result.malformed_source_urls_dropped,
        _yes_no(sanitization_result.tail_was_separated),
        _join_codes(reason_codes),
    )


def log_publish_block_summary(
    *,
    logger: logging.Logger,
    event_name: str,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    title_text: str,
    description_text: str,
    snapshot: ContentContractSnapshot,
    source_video_count: int,
    preview_present: str,
) -> None:
    logger.info(
        "%s branch=%s date_key=%s slot_key=%s lang=%s title_mode=%s title_chars=%d description_chars=%d paragraph_count_actual=%d cta_present=%s hashtags_present=%s links_actual=%d preview_present=%s source_video_count=%d",
        event_name,
        branch_label,
        date_key,
        slot_key,
        language,
        snapshot.title_mode,
        len(str(title_text or "").strip()),
        len(str(description_text or "").strip()),
        snapshot.paragraph_count_actual,
        _yes_no(snapshot.cta_present),
        _yes_no(snapshot.hashtags_present),
        snapshot.links_actual,
        preview_present,
        source_video_count,
    )


def _resolve_contract_status(
    *,
    reason_codes: Sequence[str],
    title_present: bool,
    paragraph_count_actual: int,
) -> str:
    if not reason_codes:
        return "ok"
    if not title_present or paragraph_count_actual == 0:
        return "fail"
    return "degraded_unrecovered"


def _split_paragraphs(body_text: str) -> List[str]:
    return [part.strip() for part in str(body_text or "").split("\n\n") if part.strip()]


def _tail_origin_from_source(tail_source: str, *, found: bool) -> str:
    if not found:
        return "none"
    normalized_source: str = str(tail_source or "").strip().lower()
    if "structured" in normalized_source:
        return "structured"
    if "repair" in normalized_source:
        return "repair"
    if "fallback" in normalized_source or normalized_source == "none":
        return "fallback_source"
    return "unknown"


def _join_codes(reason_codes: Iterable[str]) -> str:
    cleaned_codes: List[str] = [str(code or "").strip() for code in reason_codes if str(code or "").strip()]
    return ",".join(cleaned_codes) if cleaned_codes else "none"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
