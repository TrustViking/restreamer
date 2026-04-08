from __future__ import annotations

import dataclasses
import re
import time
from typing import List, Optional

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.models import MergedLanguageContent, PlannedVideo
from app.llm.llm_client import LlmTraceContext, OpenAITransportResult
from app.llm.merges.merge_formatting import (
    FormattingNormalizationResult,
    MergeValidationRecoveryAttempt,
    ParagraphEnforcementResult,
    _attempt_expanded_formatting_recovery,
)
from app.llm.merges.merge_links import (
    OfficialLinksFillResult,
    OfficialLinksSelection,
    _count_output_official_links,
    _extract_official_links_from_sources,
)
from app.llm.merges.merge_parser import (
    MergeTailSeparationResult,
    parse_json_tolerant,
    parse_merge_response_or_raise,
    separate_merge_body_and_tail,
)
from app.llm.merges.merge_prompt import _source_texts_for_merge_quality, build_llm_merge_prompt_text
from app.llm.merges.merge_quality import MergeQualityNormalizationResult, normalize_merge_description
from app.llm.merges.merge_retry import ExpandedRetryProfile
from app.llm.merges.merge_text_utils import _bullet_marker_for_line
from app.llm.merges.merge_validation import (
    MergeAttemptFailure,
    MergeSemanticDiagnostics,
    _build_merge_semantic_diagnostics,
    _build_rejected_merge_attempt,
    _log_merge_style_diagnostics,
    _reason_code_from_error,
    _reason_codes_from_error,
    _validate_coverage_preserving_merge_or_raise,
)
from app.llm.merges.merge_youtube import _count_youtube_urls_in_text
from app.llm.providers.provider_base import LlmProvider

LOGGER = _get_logger_impl(__name__)

_BULLET_PLAIN_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
)


class MergeExecutor:
    """Executes a single merge attempt: LLM call -> parse -> validate -> normalize -> recover."""

    _STRUCTURED_SCHEMA: dict[str, object] = {
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
    _REQUEST_KIND: str = "structured"
    _ATTEMPT_STAGE_VALIDATION: str = "validation"

    def __init__(
        self,
        *,
        config: AppConfig,
        provider: LlmProvider,
        model_name: str,
        branch_label: str,
        date_key: str,
        slot_key: str,
    ) -> None:
        self._config: AppConfig = config
        self._provider: LlmProvider = provider
        self._model_name: str = model_name
        self._branch_label: str = branch_label
        self._date_key: str = date_key
        self._slot_key: str = slot_key

    def execute(
        self,
        *,
        language: str,
        videos: List[PlannedVideo],
        attempt_index: int,
        attempt_label: str,
        no_description_text: str,
        expanded_retry_profile: Optional[ExpandedRetryProfile] = None,
        max_body_paragraphs: int = 4,
    ) -> tuple[MergedLanguageContent, str]:
        prompt_text: str = build_llm_merge_prompt_text(
            language=language,
            videos=videos,
            config=self._config,
            no_description_text=no_description_text,
            expanded_retry_profile=expanded_retry_profile,
        )
        if self._provider.pre_delay_sec(config=self._config) > 0:
            time.sleep(max(0.0, float(self._provider.pre_delay_sec(config=self._config))))
        response: OpenAITransportResult = self._provider.request_merge(
            prompt_text=prompt_text,
            config=self._config,
            model_name=self._model_name,
            attempt_label=attempt_label,
            max_output_tokens=self._config.llm.max_output_tokens,
            structured_schema=self._STRUCTURED_SCHEMA,
            temperature=0.0,
            trace_context=LlmTraceContext(
                branch_label=self._branch_label,
                date_key=self._date_key,
                slot_key=self._slot_key,
                language=language,
                provider=self._provider.name,
                model_name=self._model_name,
                attempt_index=attempt_index,
                request_kind=self._REQUEST_KIND,
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
                    self._enforce_single_step_overflow_policy(
                        description_text=preflight_description_value,
                        max_body_paragraphs=max_body_paragraphs,
                    )
            merged_content, paragraph_count = parse_merge_response_or_raise(
                provider_name=self._provider.name,
                model_name=self._model_name,
                raw_text=raw_response_text,
                structured_payload=response.structured_payload,
                max_body_paragraphs=max_body_paragraphs,
            )
        except Exception as error:
            raise MergeAttemptFailure(
                reason_code=_reason_code_from_error(error),
                reason=str(error),
                model_name=self._model_name,
                attempt_stage=self._ATTEMPT_STAGE_VALIDATION,
                raw_response_text=raw_response_text,
                reason_codes=_reason_codes_from_error(error),
                rejected_attempt=_build_rejected_merge_attempt(
                    attempt_index=attempt_index,
                    model_name=self._model_name,
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
            title=merged_content.title,
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
                self._model_name,
                attempt_index,
                raw_bullet_count,
                diagnostics.bullet_points_count,
            )
        _log_merge_style_diagnostics(
            diagnostics=diagnostics,
            branch_label=self._branch_label,
            date_key=self._date_key,
            slot_key=self._slot_key,
            language=language,
            model_name=self._model_name,
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
                model_name=self._model_name,
                attempt_index=attempt_index,
                branch_label=self._branch_label,
                date_key=self._date_key,
                slot_key=self._slot_key,
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
                    model_name=self._model_name,
                    attempt_stage=self._ATTEMPT_STAGE_VALIDATION,
                    raw_response_text=raw_response_text,
                    reason_codes=_reason_codes_from_error(final_error),
                    rejected_attempt=_build_rejected_merge_attempt(
                        attempt_index=attempt_index,
                        model_name=self._model_name,
                        reason_codes=_reason_codes_from_error(final_error),
                        fallback_reason_code=_reason_code_from_error(final_error),
                        raw_response_text=raw_response_text,
                        title_text=merged_content.title,
                        description_text=merged_content.description,
                    ),
                ) from final_error
        LOGGER.info(
            "merge_llm_response_valid model=%s attempt=%d title_length=%d description_length=%d paragraph_count=%d youtube_links_in_llm_output=%d",
            self._model_name,
            attempt_index,
            len(merged_content.title),
            len(merged_content.description),
            paragraph_count,
            _count_youtube_urls_in_text(merged_content.description),
        )
        return (merged_content, raw_response_text)

    @staticmethod
    def _enforce_description_structure(
        *,
        merged_content: MergedLanguageContent,
    ) -> ParagraphEnforcementResult:
        original_description: str = str(merged_content.description or "").strip()
        tail_separation: MergeTailSeparationResult = separate_merge_body_and_tail(
            text=original_description
        )
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

    @staticmethod
    def _enforce_single_step_overflow_policy(
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
    executor: MergeExecutor = MergeExecutor(
        config=config,
        provider=provider,
        model_name=model_name,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )
    return executor.execute(
        language=language,
        videos=videos,
        attempt_index=attempt_index,
        attempt_label=attempt_label,
        no_description_text=no_description_text,
        expanded_retry_profile=expanded_retry_profile,
        max_body_paragraphs=max_body_paragraphs,
    )
