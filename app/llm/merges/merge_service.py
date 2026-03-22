from __future__ import annotations

import dataclasses
import json
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE
from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    BLOCK_GENERATION_MODE_REAL_MERGE,
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
    RejectedMergeAttempt,
)
from app.llm.merges.merge_constants import (
    ALLOWED_BULLET_MARKERS,
    CTA_FIRST_PARAGRAPH_PREFIXES,
    PRIMARY_ATTEMPTS,
    PRIMARY_ATTEMPTS_EXTENDED,
    RECOVERABLE_REJECT_CODES,
    SEMANTIC_TOKEN_PATTERN,
    STYLE_CONTRACT_VERSION,
)
from app.llm.merges.merge_links import (
    OfficialLinksFillResult,
    OfficialLinksSelection,
    _count_output_official_links,
    _extract_official_links_from_sources,
)
from app.llm.merges.merge_prompt import (
    MergeContractMode,
    _select_merge_contract_mode,
    _single_source_translate_prompt,
    _source_texts_for_merge_quality,
    build_llm_merge_prompt_text,
)
from app.llm.merges.merge_quality import (
    MergeQualityDiagnostics,
    MergeQualityNormalizationResult,
    count_overloaded_bullets,
    normalize_merge_description,
)
from app.llm.merges.merge_retry import (
    ExpandedRetryProfile,
    _standard_expanded_retry_profile,
    _targeted_bullet_coverage_retry_profile,
    _targeted_compact_bullet_overflow_retry_profile,
    _targeted_cta_opener_retry_profile,
    _targeted_duplicate_retry_profile,
    _targeted_hook_echo_retry_profile,
    _targeted_overloaded_bullet_retry_profile,
    _targeted_paragraph_overflow_retry_profile,
    _targeted_paragraph_underflow_retry_profile,
)
from app.llm.merges.merge_text_utils import (
    _bullet_marker_for_line,
    _contains_agenda_heading,
    _extract_description_paragraphs_raw,
    _extract_named_entities,
    _looks_like_per_source_dump,
    _looks_like_service_tail_paragraph,
)
from app.llm.merges.merge_youtube import _count_youtube_urls_in_text
from app.llm.merges.merge_parser import (
    MergeTailSeparationResult,
    build_plain_merged_content_or_raise,
    clean_and_validate_llm_description,
    parse_json_tolerant,
    parse_merge_response_or_raise,
    separate_merge_body_and_tail,
)
from app.llm.models.model_compatibility import (
    LlmModelConfigurationError,
    classify_openai_request_error,
)
from app.llm.models.model_identity import resolve_effective_llm_model
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.llm_client import LlmTraceContext, OpenAITransportResult, openai_request_merge
from app.llm.llm_factory import get_llm_provider
from app.llm.providers.provider_base import LlmProvider
from app.observability.runtime_analytics import log_warning_operational
from app.llm.merges.merge_validation import (
    MergeAttemptFailure,
    DescriptionValidationFailure,
    MergeSemanticDiagnostics,
    _reason_code_from_error,
    _reason_codes_from_error,
    _normalize_description_validation_reason_code,
    _normalize_description_validation_reason_codes,
    _build_description_validation_failure,
    _extract_description_validation_reason_codes,
    _is_softened_distinctive_source_coverage_eligible,
    _best_effort_rejected_payload_fields,
    _build_rejected_merge_attempt,
    _looks_like_bad_hook_paragraph,
    _has_duplicate_paragraphs,
    _has_hook_echo_in_body,
    _attempt_hook_echo_repair,
    _starts_with_cta_prefix,
    _has_cta_in_opening_lines_before_hook_or_bullet,
    _has_adjacent_duplicate_lines,
    _validate_coverage_preserving_merge_or_raise,
    _count_emoji,
    _build_merge_semantic_diagnostics,
    _log_merge_style_diagnostics,
)
from app.llm.merges.merge_formatting import (
    FormattingNormalizationResult,
    ParagraphEnforcementResult,
    MergeValidationRecoveryAttempt,
    _strip_non_structural_emoji_from_line,
    _strip_non_structural_emoji_from_description,
    _normalize_formatting_only_description,
    _attempt_expanded_formatting_recovery,
)

LOGGER = _get_logger_impl(__name__)
_BULLET_PLAIN_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
)


def _fallback_title_source_label() -> str:
    return "fallback_titles"


def _fallback_hook_source_label() -> str:
    return "fallback_none"


def _fallback_body_source_label() -> str:
    return "fallback_source_descriptions"


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
    model_name: str = resolve_effective_llm_model(config)
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
                block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
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
            block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
        )
    except LlmModelConfigurationError:
        raise
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
            title_source=_fallback_title_source_label(),
            hook_source=_fallback_hook_source_label(),
            hashtags_source="fallback_none",
            body_source=_fallback_body_source_label(),
            block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
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


def _enforce_expanded_single_step_paragraph_overflow_policy(
    *,
    description_text: str,
    max_body_paragraphs: int,
) -> None:
    if max_body_paragraphs < 7:
        return
    preflight_tail_separation: MergeTailSeparationResult = separate_merge_body_and_tail(
        text=description_text,
        max_body_paragraphs=max_body_paragraphs,
    )
    preflight_body_paragraphs: int = preflight_tail_separation.body_paragraph_count
    if preflight_body_paragraphs == max_body_paragraphs + 1:
        raise RuntimeError(
            f"description body paragraph count must be between 2 and {max_body_paragraphs}, got {preflight_body_paragraphs}"
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
    expanded_retry_profile: Optional[ExpandedRetryProfile] = None,
    max_body_paragraphs: int = 4,
) -> tuple[MergedLanguageContent, str]:
    prompt_text: str = build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
        no_description_text=no_description_text,
        expanded_retry_profile=expanded_retry_profile,
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
        preflight_payload: Optional[dict[str, object]] = response.structured_payload
        if preflight_payload is None:
            preflight_payload, preflight_parse_mode = parse_json_tolerant(raw_response_text)
            del preflight_parse_mode
        if preflight_payload is not None:
            preflight_description_value: object = preflight_payload.get("description")
            if isinstance(preflight_description_value, str):
                _enforce_expanded_single_step_paragraph_overflow_policy(
                    description_text=preflight_description_value,
                    max_body_paragraphs=max_body_paragraphs,
                )
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name=provider.name,
            model_name=model_name,
            raw_text=raw_response_text,
            structured_payload=response.structured_payload,
            max_body_paragraphs=max_body_paragraphs,
        )
    except Exception as error:
        raise MergeAttemptFailure(
            reason_code=_reason_code_from_error(error),
            reason=str(error),
            model_name=model_name,
            attempt_stage="validation",
            raw_response_text=raw_response_text,
            reason_codes=_reason_codes_from_error(error),
            rejected_attempt=_build_rejected_merge_attempt(
                attempt_index=attempt_index,
                model_name=model_name,
                reason_codes=_reason_codes_from_error(error),
                fallback_reason_code=_reason_code_from_error(error),
                raw_response_text=raw_response_text,
                structured_payload=response.structured_payload,
            ),
        ) from error
    official_links_selection: OfficialLinksSelection = _extract_official_links_from_sources(
        videos
    )
    official_links_fill: OfficialLinksFillResult = OfficialLinksFillResult(
        description=merged_content.description,
        links_in_output=_count_output_official_links(merged_content.description),
        fill_applied=False,
    )
    source_texts: tuple[str, ...] = _source_texts_for_merge_quality(videos)
    quality_result: MergeQualityNormalizationResult = normalize_merge_description(
        description=merged_content.description,
        language=language,
        source_texts=source_texts,
    )
    if quality_result.description_text != merged_content.description:
        merged_content = dataclasses.replace(
            merged_content,
            description=quality_result.description_text,
            description_selected=quality_result.description_text,
            description_audit=quality_result.description_text,
        )
    diagnostics: MergeSemanticDiagnostics = _build_merge_semantic_diagnostics(
        merged_content=merged_content,
        videos=videos,
        official_links_selection=official_links_selection,
        official_links_fill=official_links_fill,
        merge_quality=quality_result.diagnostics,
    )
    raw_bullet_count: int = sum(
        1 for line in merged_content.description.split("\n")
        if _bullet_marker_for_line(line)
    )
    if raw_bullet_count != diagnostics.bullet_points_count:
        LOGGER.info(
            "merge_bullet_count_mismatch model=%s attempt=%d raw_bullets=%d normalized_bullets=%d",
            model_name,
            attempt_index,
            raw_bullet_count,
            diagnostics.bullet_points_count,
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
    try:
        diagnostics, merged_content = _validate_coverage_preserving_merge_or_raise(
            merged_content=merged_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=quality_result.diagnostics,
            precomputed_diagnostics=diagnostics,
        )
    except Exception as error:
        recovery_attempt: MergeValidationRecoveryAttempt = _attempt_expanded_formatting_recovery(
            validation_error=error,
            merged_content=merged_content,
            diagnostics=diagnostics,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            source_texts=source_texts,
            language=language,
            model_name=model_name,
            attempt_index=attempt_index,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
        )
        if (
            recovery_attempt.merged_content is not None
            and recovery_attempt.diagnostics is not None
        ):
            merged_content = recovery_attempt.merged_content
            diagnostics = recovery_attempt.diagnostics
        else:
            final_error: Exception = recovery_attempt.validation_error or error
            raise MergeAttemptFailure(
                reason_code=_reason_code_from_error(final_error),
                reason=str(final_error),
                model_name=model_name,
                attempt_stage="validation",
                raw_response_text=raw_response_text,
                reason_codes=_reason_codes_from_error(final_error),
                rejected_attempt=_build_rejected_merge_attempt(
                    attempt_index=attempt_index,
                    model_name=model_name,
                    reason_codes=_reason_codes_from_error(final_error),
                    fallback_reason_code=_reason_code_from_error(final_error),
                    raw_response_text=raw_response_text,
                    title_text=merged_content.title,
                    description_text=merged_content.description,
                ),
            ) from final_error
    LOGGER.info(
        "merge_llm_response_valid model=%s attempt=%d title_length=%d description_length=%d paragraph_count=%d youtube_links_in_llm_output=%d",
        model_name,
        attempt_index,
        len(merged_content.title),
        len(merged_content.description),
        paragraph_count,
        _count_youtube_urls_in_text(merged_content.description),
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
    primary_model: str = resolve_effective_llm_model(config)
    primary_provider: LlmProvider = get_llm_provider(config=config)
    main_source_label: str = _main_stage_field_source(provider_name=primary_provider.name)
    _pre_loop_contract_mode: MergeContractMode = _select_merge_contract_mode(
        source_count=len(videos),
        source_texts=[str(video.metadata.description or "") for video in videos],
        templates=config.templates,
    )
    contract_max_body_paragraphs: int = _pre_loop_contract_mode.max_body_paragraphs
    diagnostics: Optional[MergeSemanticDiagnostics] = None
    last_raw_response: str = ""
    last_error_summary: str = "unknown error"
    last_reason_code: str = "unexpected_error"
    last_reason_codes: tuple[str, ...] = ()
    pending_retry_profile: ExpandedRetryProfile = _standard_expanded_retry_profile()
    rejected_attempts: list[RejectedMergeAttempt] = []
    effective_max_attempts: int = PRIMARY_ATTEMPTS

    for attempt_index in range(1, PRIMARY_ATTEMPTS_EXTENDED + 1):
        if attempt_index > effective_max_attempts:
            break
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
        retry_profile: Optional[ExpandedRetryProfile] = (
            pending_retry_profile if attempt_index > 1 else None
        )
        if attempt_index > 1:
            LOGGER.info(
                "merge_llm_retry branch=%s date_key=%s slot_key=%s language=%s provider=%s attempt=%d model=%s retry_mode=%s retry_reason_codes=%s retry_focus=%s retry_structure=%s retry_source_count=%d",
                branch_label,
                date_key,
                slot_key,
                language,
                primary_provider.name,
                attempt_index,
                primary_model,
                (
                    retry_profile.retry_mode
                    if retry_profile is not None
                    else "standard"
                ),
                (
                    retry_profile.reject_signal_label
                    if retry_profile is not None
                    else "none"
                ),
                retry_profile.focus_label if retry_profile is not None else "none",
                (
                    "four_plus_structured"
                    if retry_profile is not None
                    and retry_profile.enabled
                    and len(videos) >= 4
                    else "standard"
                ),
                len(videos),
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
                expanded_retry_profile=retry_profile,
                max_body_paragraphs=contract_max_body_paragraphs,
            )
            last_raw_response = raw_response_text
            if merge_run_summary is not None and merged_content.tail_recovery_applied:
                merge_run_summary.record_paragraph_recovery_used()
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
                    block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
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
                block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
            )
        except LlmModelConfigurationError as error:
            LOGGER.error(
                "merge_llm_fatal_model_error branch=%s date_key=%s slot_key=%s language=%s provider=%s model=%s code=%s status_code=%s api_error_code=%s api_error_param=%s reason=%s",
                branch_label,
                date_key,
                slot_key,
                language,
                primary_provider.name,
                primary_model,
                error.reason_code,
                str(error.status_code if error.status_code is not None else "unknown"),
                error.api_error_code or "none",
                error.api_error_param or "none",
                error.detail,
            )
            raise
        except MergeAttemptFailure as error:
            last_raw_response = error.raw_response_text
            last_error_summary = error.reason
            last_reason_code = error.reason_code
            last_reason_codes = _reason_codes_from_error(error) or (error.reason_code,)
            if error.rejected_attempt is not None:
                rejected_attempts.append(error.rejected_attempt)
            if "overloaded_bullet" in (error.reason_codes or ()):
                _overloaded_description: str = (
                    error.rejected_attempt.description
                    if error.rejected_attempt is not None
                    else ""
                )
                _overloaded_count: int = (
                    count_overloaded_bullets(_overloaded_description)
                    if _overloaded_description
                    else 1
                )
                pending_retry_profile = _targeted_overloaded_bullet_retry_profile(
                    overloaded_count=_overloaded_count,
                    templates=config.templates,
                )
            elif "insufficient_bullet_coverage" in (error.reason_codes or ()):
                _actual_bullets: int = diagnostics.bullet_points_count if diagnostics is not None else 0
                _required_bullets: int = max(len(videos) + 1, 5)
                pending_retry_profile = _targeted_bullet_coverage_retry_profile(
                    actual_bullets=_actual_bullets,
                    required_bullets=_required_bullets,
                    source_count=len(videos),
                    templates=config.templates,
                )
            elif "cta_as_first_paragraph" in (error.reason_codes or (error.reason_code,)):
                pending_retry_profile = _targeted_cta_opener_retry_profile(
                    templates=config.templates,
                )
            elif "hook_echo_in_body" in (error.reason_codes or ()):
                pending_retry_profile = _targeted_hook_echo_retry_profile(
                    templates=config.templates,
                )
            elif any(
                code in (error.reason_codes or ())
                for code in ("duplicate_paragraph", "cta_in_hook")
            ):
                pending_retry_profile = _targeted_duplicate_retry_profile(
                    templates=config.templates,
                )
            elif "paragraph_underflow" in (error.reason_codes or (error.reason_code,)):
                pending_retry_profile = _targeted_paragraph_underflow_retry_profile(
                    templates=config.templates,
                )
            elif "paragraph_overflow" in (error.reason_codes or (error.reason_code,)):
                _para_match: Optional[re.Match[str]] = re.search(r"got (\d+)", str(error))
                _actual_paras: int = int(_para_match.group(1)) if _para_match else 8
                pending_retry_profile = _targeted_paragraph_overflow_retry_profile(
                    actual_paragraphs=_actual_paras,
                    max_paragraphs=contract_max_body_paragraphs,
                    templates=config.templates,
                )
            elif "compact_bullet_overflow" in (error.reason_codes or (error.reason_code,)):
                _actual_bullets_overflow: int = (
                    diagnostics.bullet_points_count if diagnostics is not None else 0
                )
                _contract_mode: MergeContractMode = _select_merge_contract_mode(
                    source_count=len(videos),
                    source_texts=[str(video.metadata.description or "") for video in videos],
                    templates=config.templates,
                )
                pending_retry_profile = _targeted_compact_bullet_overflow_retry_profile(
                    actual_bullets=_actual_bullets_overflow,
                    min_bullets=_contract_mode.bullet_range_min,
                    max_bullets=_contract_mode.bullet_range_max,
                    templates=config.templates,
                )
            else:
                pending_retry_profile = _standard_expanded_retry_profile(
                    reject_signals=error.reason_codes or (error.reason_code,)
                )
            if any(
                code in RECOVERABLE_REJECT_CODES
                for code in (error.reason_codes or (error.reason_code,))
            ):
                effective_max_attempts = PRIMARY_ATTEMPTS_EXTENDED
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
            error_classification = classify_openai_request_error(error)
            if error_classification.reason_code == "openai_quota_exhausted":
                last_error_summary = summarize_error(error)
                last_reason_code = "openai_quota_exhausted"
                last_reason_codes = ("openai_quota_exhausted",)
                _log_merge_attempt_invalid(
                    provider_name=primary_provider.name,
                    model_name=primary_model,
                    attempt_index=attempt_index,
                    is_fallback=False,
                    branch_label=branch_label,
                    date_key=date_key,
                    slot_key=slot_key,
                    language=language,
                    reason_code="openai_quota_exhausted",
                    reason=last_error_summary,
                    raw_response_text="",
                )
                if merge_run_summary is not None:
                    merge_run_summary.record_quota_exhausted()
                break
            last_error_summary = summarize_error(error)
            last_reason_code = "unexpected_error"
            last_reason_codes = ("unexpected_error",)
            pending_retry_profile = _standard_expanded_retry_profile(
                reject_signals=("unexpected_error",)
            )
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

    final_reason_code: str = last_reason_code
    log_warning_operational(
        LOGGER,
        "merge_llm_final_failure branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=primary code=%s fallback_used=no raw_response_received=%s reason=%s raw_chars=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        primary_provider.name,
        final_reason_code,
        "yes" if bool(last_raw_response.strip()) else "no",
        last_error_summary,
        len(last_raw_response),
        reason_code=final_reason_code,
    )
    LOGGER.info(
        "merge_provider_summary provider=%s model=%s structured_ok=no fallback_used=no parse_repair_used=no final_status=failed",
        primary_provider.name,
        primary_model,
    )
    if merge_run_summary is not None:
        merge_run_summary.record_final_failure()
    final_publish_source_label: str = "merge_failed"
    if final_reason_code == "openai_quota_exhausted":
        final_publish_source_label = "merge_failed_openai_quota_exhausted"
    return LanguageMergeAttempt(
        language=language,
        model_name=primary_model,
        raw_response_text=last_raw_response,
        merged=None,
        error_summary=last_error_summary,
        salvaged_title=None,
        publish_source_label=final_publish_source_label,
        validation_reasons=list(last_reason_codes),
        generator_model_name=primary_model,
        used_model_names=(primary_model,),
        branch_type=BRANCH_MERGE,
        title_source=_fallback_title_source_label(),
        hook_source=_fallback_hook_source_label(),
        hashtags_source="fallback_none",
        body_source=_fallback_body_source_label(),
        rejected_attempts=tuple(rejected_attempts),
        block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
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
