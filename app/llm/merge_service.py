from __future__ import annotations

import dataclasses
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.constants import URL_PATTERN
from app.core.models import LanguageMergeAttempt, MergedLanguageContent, PlannedVideo
from app.llm.merge_parser import (
    build_plain_merged_content_or_raise,
    clean_and_validate_llm_description,
    normalize_filtered_links,
    normalize_hashtags_line,
    normalize_single_line_text,
    parse_merge_response_or_raise,
)
from app.llm.merge_run_summary import MergeRunSummary
from app.llm.openai_client import LlmTraceContext, OpenAITransportResult, openai_request_merge

LOGGER = _get_logger_impl(__name__)
PRIMARY_ATTEMPTS: int = 2


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
    cta_text: Optional[str]
    hashtags_line: Optional[str]
    links: Tuple[str, ...]
    mutated: bool
    recovery_applied: bool
    note: str
    body_paragraphs_before: int
    body_paragraphs_after: int
    cta_present_before: bool
    cta_present_after: bool
    links_count_after: int
    hashtags_count_after: int


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
    if "title must be a non-empty string" in error_text or "title is invalid" in error_text:
        return "invalid_title"
    if "description must be a non-empty string" in error_text or "description paragraph count" in error_text or "description validation failed" in error_text:
        return "invalid_description"
    if "cta must be a non-empty string" in error_text:
        return "invalid_cta"
    if "hashtags must be a list of strings" in error_text:
        return "invalid_hashtags"
    if "links must be a list of strings" in error_text:
        return "invalid_links"
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


def _strip_urls(text: str) -> str:
    without_urls: str = URL_PATTERN.sub("", str(text or ""))
    normalized: str = without_urls.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _truncate_text(text: str, *, limit: int) -> str:
    normalized: str = normalize_single_line_text(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


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
    for index, video in enumerate(videos, start=1):
        source_blocks.append(
            "\n".join(
                [
                    f"SOURCE {index}",
                    f"TITLE: {video.metadata.title.strip()}",
                    (
                        "DESCRIPTION: "
                        f"{_truncate_text(_strip_urls(video.metadata.description.strip() or no_description_text), limit=config.llm_source_desc_max_chars)}"
                    ),
                    f"URL: {video.normalized_link.strip()}",
                ]
            )
        )
    template_prompt: str = str(config.templates.llm_merge_title_description_prompt or "").strip()
    if template_prompt:
        return template_prompt.format(
            language_name=language_name,
            sources_block="\n\n".join(source_blocks),
        ).strip()
    return (
        "You are a careful editorial writer.\n"
        f"Write output only in {language_name}.\n"
        "Use only facts explicitly present in the sources.\n"
        "Produce one final stream summary, not a per-source enumeration.\n"
        "Description must be a cohesive summary in 2 to 4 paragraphs.\n"
        "CTA must be moderate and concise.\n"
        "links must contain only valid external URLs worth showing to users; use [] if none.\n"
        "hashtags must be an array of hashtag strings.\n"
        'Output only one strict JSON object with exactly these keys: title, description, cta, hashtags, links.\n\n'
        f"{'\n\n'.join(source_blocks)}"
    ).strip()


def _structured_merge_schema() -> dict[str, object]:
    return {
        "name": "restreamer_merge_summary_v2",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "description", "cta", "hashtags", "links"],
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 98},
                "description": {"type": "string", "minLength": 1},
                "cta": {"type": "string", "minLength": 1},
                "hashtags": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "links": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
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
        "No headings, JSON, markdown, CTA, links, or hashtags.\n\n"
        f"{source_description.strip()}"
    ).strip()


def _request_plain_text(
    *,
    prompt_text: str,
    config: AppConfig,
    model_name: str,
    attempt_label: str,
    trace_context: LlmTraceContext,
) -> str:
    if config.openai_pre_delay_sec > 0:
        time.sleep(max(0.0, float(config.openai_pre_delay_sec)))
    response: OpenAITransportResult = openai_request_merge(
        prompt_text=prompt_text,
        model_name=model_name,
        timeout_sec=config.openai_timeout_sec,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=None,
        temperature=0.0,
        trace_context=trace_context,
    )
    return response.raw_text


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
    del merge_run_summary
    if len(videos) != 1:
        raise RuntimeError("single-source translate expects exactly one video")
    source_video: PlannedVideo = videos[0]
    model_name: str = str(config.openai_model_primary or "").strip() or "gpt-5.1"
    try:
        raw_text: str = _request_plain_text(
            prompt_text=_single_source_translate_prompt(
                source_language=source_video.language,
                target_language=language,
                source_description=source_video.metadata.description.strip() or no_description_text,
            ),
            config=config,
            model_name=model_name,
            attempt_label=attempt_label,
            trace_context=LlmTraceContext(
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                provider="openai",
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
            merged=merged_content,
            error_summary=None,
            salvaged_title=merged_content.title,
            publish_source_label="single_source_plain_ok",
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
        )


def _log_merge_attempt_start(
    *,
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
            "merge_llm_fallback_start branch=%s date_key=%s slot_key=%s language=%s stage=%s attempt=%d model=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            stage_name,
            attempt_index,
            model_name,
        )
        return
    LOGGER.info(
        "merge_llm_primary_attempt branch=%s date_key=%s slot_key=%s language=%s stage=%s attempt=%d model=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        stage_name,
        attempt_index,
        model_name,
    )


def _log_merge_attempt_invalid(
    *,
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
        "merge_llm_response_invalid branch=%s date_key=%s slot_key=%s language=%s stage=%s model=%s attempt=%d code=%s raw_response_received=%s reason=%s raw_chars=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        stage_name,
        model_name,
        attempt_index,
        reason_code,
        "yes" if bool(str(raw_response_text or "").strip()) else "no",
        reason,
        len(str(raw_response_text or "")),
    )


def _normalize_merge_links(
    *,
    merged_content: MergedLanguageContent,
    normalize_youtube_url: Callable[[str], str],
) -> MergedLanguageContent:
    link_stats = normalize_filtered_links(
        links=tuple(merged_content.links),
        normalize_link=normalize_youtube_url,
    )
    LOGGER.info(
        "merge_links_normalized links_count=%d duplicates_dropped=%d invalid_dropped=%d",
        len(link_stats.accepted_links),
        link_stats.duplicates_dropped,
        link_stats.invalid_dropped,
    )
    return dataclasses.replace(merged_content, links=link_stats.accepted_links)


def _split_description_paragraphs(text: str) -> List[str]:
    normalized_text: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []
    return [
        _normalize_paragraph_text(paragraph_text)
        for paragraph_text in re.split(r"\n\s*\n", normalized_text)
        if _normalize_paragraph_text(paragraph_text)
    ]


def _normalize_paragraph_text(text: str) -> str:
    stripped_lines: List[str] = [line.strip() for line in str(text or "").split("\n") if line.strip()]
    return re.sub(r"\s+", " ", " ".join(stripped_lines)).strip()


def _count_hashtags(line: Optional[str]) -> int:
    return len([token for token in str(line or "").split() if token.strip()])


def _split_single_paragraph_safely(paragraph_text: str) -> Optional[List[str]]:
    sentence_parts: List[str] = [
        part.strip()
        for part in re.split(r"(?<=[.!?…])\s+", str(paragraph_text or "").strip())
        if part.strip()
    ]
    if len(sentence_parts) < 4:
        return None
    split_index: int = max(2, len(sentence_parts) // 2)
    left_part: str = " ".join(sentence_parts[:split_index]).strip()
    right_part: str = " ".join(sentence_parts[split_index:]).strip()
    if not left_part or not right_part:
        return None
    return [left_part, right_part]


def _collapse_paragraphs_to_limit(
    paragraphs: Sequence[str],
    *,
    max_paragraphs: int,
) -> Optional[List[str]]:
    cleaned_paragraphs: List[str] = [str(paragraph or "").strip() for paragraph in paragraphs if str(paragraph or "").strip()]
    if not cleaned_paragraphs:
        return None
    if len(cleaned_paragraphs) <= max_paragraphs:
        return cleaned_paragraphs
    total_items: int = len(cleaned_paragraphs)
    base_group_size: int = total_items // max_paragraphs
    remainder: int = total_items % max_paragraphs
    collapsed: List[str] = []
    start_index: int = 0
    for group_index in range(max_paragraphs):
        group_size: int = base_group_size + (1 if group_index < remainder else 0)
        next_index: int = start_index + max(1, group_size)
        group_items: List[str] = cleaned_paragraphs[start_index:next_index]
        if not group_items:
            break
        collapsed.append("\n".join(group_items).strip())
        start_index = next_index
    return collapsed if len(collapsed) <= max_paragraphs else None


def _normalize_tail_fields(
    merged_content: MergedLanguageContent,
) -> tuple[Optional[str], Optional[str], Tuple[str, ...]]:
    normalized_cta_text: Optional[str] = normalize_single_line_text(
        str(merged_content.cta_text or "")
    ) or None
    normalized_hashtags_line: Optional[str] = normalize_hashtags_line(
        str(merged_content.hashtags_line or "")
    ) or None
    normalized_links: Tuple[str, ...] = tuple(
        str(link or "").strip() for link in merged_content.links if str(link or "").strip()
    )
    return (normalized_cta_text, normalized_hashtags_line, normalized_links)


def _enforce_merged_description_structure(
    *,
    merged_content: MergedLanguageContent,
) -> ParagraphEnforcementResult:
    original_description: str = str(merged_content.description or "").strip()
    normalized_cta_text, normalized_hashtags_line, normalized_links = _normalize_tail_fields(
        merged_content
    )
    paragraphs_before: List[str] = _split_description_paragraphs(original_description)
    body_paragraphs_before: int = len(paragraphs_before)
    if not paragraphs_before:
        return ParagraphEnforcementResult(
            description_text=original_description,
            cta_text=normalized_cta_text,
            hashtags_line=normalized_hashtags_line,
            links=normalized_links,
            mutated=False,
            recovery_applied=False,
            note="empty_description",
            body_paragraphs_before=0,
            body_paragraphs_after=0,
            cta_present_before=bool(str(merged_content.cta_text or "").strip()),
            cta_present_after=bool(normalized_cta_text),
            links_count_after=len(normalized_links),
            hashtags_count_after=_count_hashtags(normalized_hashtags_line),
        )

    enforced_paragraphs: List[str] = list(paragraphs_before)
    note: str = "already_structurally_valid"
    recovery_applied: bool = False

    if len(paragraphs_before) == 1:
        safely_split: Optional[List[str]] = _split_single_paragraph_safely(paragraphs_before[0])
        if safely_split is not None:
            enforced_paragraphs = safely_split
            note = "split_single_paragraph_into_two"
            recovery_applied = True
        else:
            note = "single_paragraph_not_safely_split"
    elif len(paragraphs_before) > 4:
        collapsed_paragraphs: Optional[List[str]] = _collapse_paragraphs_to_limit(
            paragraphs_before,
            max_paragraphs=4,
        )
        if collapsed_paragraphs is not None and len(collapsed_paragraphs) <= 4:
            enforced_paragraphs = collapsed_paragraphs
            note = "collapsed_excess_paragraphs_to_limit"
            recovery_applied = True
        else:
            note = "excess_paragraphs_not_safely_collapsed"

    normalized_description: str = "\n\n".join(paragraph for paragraph in enforced_paragraphs if paragraph).strip()
    mutated: bool = (
        normalized_description != original_description
        or normalized_cta_text != (str(merged_content.cta_text or "").strip() or None)
        or normalized_hashtags_line != (str(merged_content.hashtags_line or "").strip() or None)
        or normalized_links != tuple(merged_content.links)
    )
    return ParagraphEnforcementResult(
        description_text=normalized_description,
        cta_text=normalized_cta_text,
        hashtags_line=normalized_hashtags_line,
        links=normalized_links,
        mutated=mutated,
        recovery_applied=recovery_applied and mutated,
        note=note,
        body_paragraphs_before=body_paragraphs_before,
        body_paragraphs_after=len(_split_description_paragraphs(normalized_description)),
        cta_present_before=bool(str(merged_content.cta_text or "").strip()),
        cta_present_after=bool(normalized_cta_text),
        links_count_after=len(normalized_links),
        hashtags_count_after=_count_hashtags(normalized_hashtags_line),
    )


def _attempt_merge_once(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    model_name: str,
    attempt_index: int,
    attempt_label: str,
    branch_label: str,
    date_key: str,
    slot_key: str,
    no_description_text: str,
    normalize_youtube_url: Callable[[str], str],
) -> tuple[MergedLanguageContent, str]:
    prompt_text: str = build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
        no_description_text=no_description_text,
    )
    if config.openai_pre_delay_sec > 0:
        time.sleep(max(0.0, float(config.openai_pre_delay_sec)))
    response: OpenAITransportResult = openai_request_merge(
        prompt_text=prompt_text,
        model_name=model_name,
        timeout_sec=config.openai_timeout_sec,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=_structured_merge_schema(),
        temperature=0.0,
        trace_context=LlmTraceContext(
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
            provider="openai",
            model_name=model_name,
            attempt_index=attempt_index,
            request_kind="structured",
            source_count=len(videos),
        ),
    )
    raw_response_text: str = response.raw_text
    try:
        merged_content, _, paragraph_count = parse_merge_response_or_raise(
            provider_name="openai",
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
    normalized_content: MergedLanguageContent = _normalize_merge_links(
        merged_content=merged_content,
        normalize_youtube_url=normalize_youtube_url,
    )
    LOGGER.info(
        "merge_llm_response_valid model=%s attempt=%d title_length=%d description_length=%d paragraph_count=%d links_count=%d hashtags_count=%d",
        model_name,
        attempt_index,
        len(normalized_content.title),
        len(normalized_content.description),
        paragraph_count,
        len(normalized_content.links),
        len([token for token in str(normalized_content.hashtags_line or "").split() if token.strip()]),
    )
    return (normalized_content, raw_response_text)


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
    primary_model: str = str(config.openai_model_primary or "").strip() or "gpt-5.1"
    fallback_model: str = str(config.openai_model_fallback or "").strip() or "gpt-5-mini"
    last_raw_response: str = ""
    last_error_summary: str = "unknown error"

    for attempt_index in range(1, PRIMARY_ATTEMPTS + 1):
        _log_merge_attempt_start(
            model_name=primary_model,
            attempt_index=attempt_index,
            is_fallback=False,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
        )
        if attempt_index > 1:
            LOGGER.info("merge_llm_retry attempt=%d model=%s", attempt_index, primary_model)
            if merge_run_summary is not None:
                merge_run_summary.record_primary_retry_used()
        try:
            merged_content, raw_response_text = _attempt_merge_once(
                language=language,
                videos=videos,
                config=config,
                model_name=primary_model,
                attempt_index=attempt_index,
                attempt_label=f"{attempt_label}_PRIMARY_{attempt_index}",
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                no_description_text=no_description_text,
                normalize_youtube_url=normalize_youtube_url,
            )
            last_raw_response = raw_response_text
            if merge_run_summary is not None:
                merge_run_summary.record_primary_success()
            return LanguageMergeAttempt(
                language=language,
                model_name=primary_model,
                raw_response_text=raw_response_text,
                merged=merged_content,
                error_summary=None,
                salvaged_title=merged_content.title,
                publish_source_label="primary_success",
            )
        except MergeAttemptFailure as error:
            last_raw_response = error.raw_response_text
            last_error_summary = error.reason
            _log_merge_attempt_invalid(
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

    _log_merge_attempt_start(
        model_name=fallback_model,
        attempt_index=1,
        is_fallback=True,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
        language=language,
    )
    try:
        merged_content, raw_response_text = _attempt_merge_once(
            language=language,
            videos=videos,
            config=config,
            model_name=fallback_model,
            attempt_index=1,
            attempt_label=f"{attempt_label}_FALLBACK_1",
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            no_description_text=no_description_text,
            normalize_youtube_url=normalize_youtube_url,
        )
        last_raw_response = raw_response_text
        LOGGER.info("merge_llm_fallback_valid model=%s", fallback_model)
        if merge_run_summary is not None:
            merge_run_summary.record_fallback_success()
        return LanguageMergeAttempt(
            language=language,
            model_name=fallback_model,
            raw_response_text=raw_response_text,
            merged=merged_content,
            error_summary=None,
            salvaged_title=merged_content.title,
            publish_source_label="fallback_success",
        )
    except MergeAttemptFailure as error:
        last_raw_response = error.raw_response_text
        last_error_summary = error.reason
        _log_merge_attempt_invalid(
            model_name=fallback_model,
            attempt_index=1,
            is_fallback=True,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
            reason_code=error.reason_code,
            reason=error.reason,
            raw_response_text=error.raw_response_text,
        )
    except Exception as error:
        last_error_summary = summarize_error(error)
        _log_merge_attempt_invalid(
            model_name=fallback_model,
            attempt_index=1,
            is_fallback=True,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
            reason_code="unexpected_error",
            reason=last_error_summary,
            raw_response_text="",
        )

    LOGGER.error(
        "merge_llm_final_failure branch=%s date_key=%s slot_key=%s language=%s stage=fallback code=%s fallback_used=yes raw_response_received=%s reason=%s raw_chars=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        _reason_code_from_error(RuntimeError(last_error_summary)),
        "yes" if bool(last_raw_response.strip()) else "no",
        last_error_summary,
        len(last_raw_response),
    )
    if merge_run_summary is not None:
        merge_run_summary.record_final_failure()
    return LanguageMergeAttempt(
        language=language,
        model_name=fallback_model,
        raw_response_text=last_raw_response,
        merged=None,
        error_summary=last_error_summary,
        salvaged_title=None,
        publish_source_label="merge_failed",
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
        "merge_post_enforcement branch=%s date_key=%s slot_key=%s language=%s body_paragraphs_before=%d body_paragraphs_after=%d cta_present_before=%s cta_present_after=%s links_count=%d hashtags_count=%d mutated=%s recovery_applied=%s reason=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        enforcement_result.body_paragraphs_before,
        enforcement_result.body_paragraphs_after,
        "yes" if enforcement_result.cta_present_before else "no",
        "yes" if enforcement_result.cta_present_after else "no",
        enforcement_result.links_count_after,
        enforcement_result.hashtags_count_after,
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
        cta_text=enforcement_result.cta_text,
        hashtags_line=enforcement_result.hashtags_line,
        links=enforcement_result.links,
        description_selected=enforcement_result.description_text,
        description_audit=enforcement_result.description_text,
    )
    return updated_content
