from __future__ import annotations

import dataclasses
import re
from typing import Callable, List, Optional

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
from app.llm.llm_factory import get_llm_provider
from app.llm.merges.merge_constants import (
    PRIMARY_ATTEMPTS,
    PRIMARY_ATTEMPTS_EXTENDED,
    RECOVERABLE_REJECT_CODES,
)
from app.llm.merges.merge_executor import MergeExecutor, _attempt_merge_once
from app.llm.merges.merge_prompt import MergeContractMode, _select_merge_contract_mode
from app.llm.merges.merge_quality import count_overloaded_bullets
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
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.merges.merge_single_source import _main_stage_field_source
from app.llm.merges.merge_validation import (
    MergeAttemptFailure,
    MergeSemanticDiagnostics,
    _reason_codes_from_error,
)
from app.llm.models.model_compatibility import LlmModelConfigurationError, classify_openai_request_error
from app.llm.models.model_identity import resolve_effective_llm_model
from app.llm.providers.provider_base import LlmProvider
from app.observability.runtime_analytics import log_warning_operational

LOGGER = _get_logger_impl(__name__)


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


class MergeOrchestrator:
    """Orchestrates multi-attempt merge with retry logic and reason-code routing."""

    _FALLBACK_TITLE_SOURCE: str = "fallback_titles"
    _FALLBACK_HOOK_SOURCE: str = "fallback_none"
    _FALLBACK_HASHTAGS_SOURCE: str = "fallback_none"
    _FALLBACK_BODY_SOURCE: str = "fallback_source_descriptions"
    _PRIMARY_BODY_SOURCE: str = "main_merge"
    _PRIMARY_PUBLISH_SOURCE: str = "primary_success"
    _FINAL_FAILED_PUBLISH_SOURCE: str = "merge_failed"
    _FINAL_FAILED_QUOTA_PUBLISH_SOURCE: str = "merge_failed_openai_quota_exhausted"
    _OPENAI_QUOTA_REASON_CODE: str = "openai_quota_exhausted"
    _UNEXPECTED_ERROR_REASON_CODE: str = "unexpected_error"

    def __init__(
        self,
        *,
        config: AppConfig,
        videos: List[PlannedVideo],
        language: str,
        attempt_label: str,
        summarize_error: Callable[[Exception], str],
        no_description_text: str,
        merge_run_summary: Optional[MergeRunSummary] = None,
        branch_label: str = "unknown",
        date_key: str = "unknown",
        slot_key: str = "unknown",
    ) -> None:
        self._config: AppConfig = config
        self._videos: List[PlannedVideo] = videos
        self._language: str = language
        self._attempt_label: str = attempt_label
        self._summarize_error: Callable[[Exception], str] = summarize_error
        self._no_description_text: str = no_description_text
        self._merge_run_summary: Optional[MergeRunSummary] = merge_run_summary
        self._branch_label: str = branch_label
        self._date_key: str = date_key
        self._slot_key: str = slot_key
        self._model: str = resolve_effective_llm_model(config)
        self._provider: LlmProvider = get_llm_provider(config=config)
        self._main_source_label: str = _main_stage_field_source(
            provider_name=self._provider.name
        )
        self._contract_mode: MergeContractMode = _select_merge_contract_mode(
            source_count=len(videos),
            source_texts=[str(video.metadata.description or "") for video in videos],
            templates=config.templates,
        )
        self._max_body_paragraphs: int = self._contract_mode.max_body_paragraphs

    def _select_retry_profile(
        self,
        error: MergeAttemptFailure,
        diagnostics: Optional[MergeSemanticDiagnostics] = None,
    ) -> ExpandedRetryProfile:
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
            return _targeted_overloaded_bullet_retry_profile(
                overloaded_count=_overloaded_count,
                templates=self._config.templates,
            )
        if "insufficient_bullet_coverage" in (error.reason_codes or ()):
            _actual_bullets: int = diagnostics.bullet_points_count if diagnostics is not None else 0
            _required_bullets: int = max(len(self._videos) + 1, 5)
            return _targeted_bullet_coverage_retry_profile(
                actual_bullets=_actual_bullets,
                required_bullets=_required_bullets,
                source_count=len(self._videos),
                templates=self._config.templates,
            )
        if "cta_as_first_paragraph" in (error.reason_codes or (error.reason_code,)):
            return _targeted_cta_opener_retry_profile(
                templates=self._config.templates,
            )
        if "hook_echo_in_body" in (error.reason_codes or ()):
            return _targeted_hook_echo_retry_profile(
                templates=self._config.templates,
            )
        if any(
            code in (error.reason_codes or ())
            for code in ("duplicate_paragraph", "cta_in_hook")
        ):
            return _targeted_duplicate_retry_profile(
                templates=self._config.templates,
            )
        if "paragraph_underflow" in (error.reason_codes or (error.reason_code,)):
            return _targeted_paragraph_underflow_retry_profile(
                templates=self._config.templates,
            )
        if "paragraph_overflow" in (error.reason_codes or (error.reason_code,)):
            _para_match: Optional[re.Match[str]] = re.search(r"got (\d+)", str(error))
            _actual_paras: int = int(_para_match.group(1)) if _para_match else 8
            return _targeted_paragraph_overflow_retry_profile(
                actual_paragraphs=_actual_paras,
                max_paragraphs=self._max_body_paragraphs,
                templates=self._config.templates,
            )
        if "compact_bullet_overflow" in (error.reason_codes or (error.reason_code,)):
            _actual_bullets_overflow: int = (
                diagnostics.bullet_points_count if diagnostics is not None else 0
            )
            _contract_mode: MergeContractMode = _select_merge_contract_mode(
                source_count=len(self._videos),
                source_texts=[str(video.metadata.description or "") for video in self._videos],
                templates=self._config.templates,
            )
            return _targeted_compact_bullet_overflow_retry_profile(
                actual_bullets=_actual_bullets_overflow,
                min_bullets=_contract_mode.bullet_range_min,
                max_bullets=_contract_mode.bullet_range_max,
                templates=self._config.templates,
            )
        return _standard_expanded_retry_profile(
            reject_signals=error.reason_codes or (error.reason_code,)
        )

    def _build_success_result(
        self,
        merged_content: MergedLanguageContent,
        raw_response_text: str,
    ) -> LanguageMergeAttempt:
        return LanguageMergeAttempt(
            language=self._language,
            model_name=self._model,
            raw_response_text=raw_response_text,
            merged=dataclasses.replace(
                merged_content,
                branch_type=BRANCH_MERGE,
                title_source=self._main_source_label,
                hook_source=self._main_source_label,
                hashtags_source=self._main_source_label,
                body_source=self._PRIMARY_BODY_SOURCE,
                block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
            ),
            error_summary=None,
            salvaged_title=merged_content.title,
            publish_source_label=self._PRIMARY_PUBLISH_SOURCE,
            generator_model_name=self._model,
            used_model_names=(self._model,),
            branch_type=BRANCH_MERGE,
            title_source=self._main_source_label,
            hook_source=self._main_source_label,
            hashtags_source=self._main_source_label,
            body_source=self._PRIMARY_BODY_SOURCE,
            block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
        )

    def _build_failure_result(
        self,
        *,
        last_error_summary: str,
        last_reason_code: str,
        last_reason_codes: tuple[str, ...],
        last_raw_response: str,
        rejected_attempts: list[RejectedMergeAttempt],
    ) -> LanguageMergeAttempt:
        final_publish_source_label: str = self._FINAL_FAILED_PUBLISH_SOURCE
        if last_reason_code == self._OPENAI_QUOTA_REASON_CODE:
            final_publish_source_label = self._FINAL_FAILED_QUOTA_PUBLISH_SOURCE
        return LanguageMergeAttempt(
            language=self._language,
            model_name=self._model,
            raw_response_text=last_raw_response,
            merged=None,
            error_summary=last_error_summary,
            salvaged_title=None,
            publish_source_label=final_publish_source_label,
            validation_reasons=list(last_reason_codes),
            generator_model_name=self._model,
            used_model_names=(self._model,),
            branch_type=BRANCH_MERGE,
            title_source=self._FALLBACK_TITLE_SOURCE,
            hook_source=self._FALLBACK_HOOK_SOURCE,
            hashtags_source=self._FALLBACK_HASHTAGS_SOURCE,
            body_source=self._FALLBACK_BODY_SOURCE,
            rejected_attempts=tuple(rejected_attempts),
            block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
        )

    def run(self) -> LanguageMergeAttempt:
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
                provider_name=self._provider.name,
                model_name=self._model,
                attempt_index=attempt_index,
                is_fallback=False,
                branch_label=self._branch_label,
                date_key=self._date_key,
                slot_key=self._slot_key,
                language=self._language,
            )
            retry_profile: Optional[ExpandedRetryProfile] = (
                pending_retry_profile if attempt_index > 1 else None
            )
            if attempt_index > 1:
                LOGGER.info(
                    "merge_llm_retry branch=%s date_key=%s slot_key=%s language=%s provider=%s attempt=%d model=%s retry_mode=%s retry_reason_codes=%s retry_focus=%s retry_structure=%s retry_source_count=%d",
                    self._branch_label,
                    self._date_key,
                    self._slot_key,
                    self._language,
                    self._provider.name,
                    attempt_index,
                    self._model,
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
                        and len(self._videos) >= 4
                        else "standard"
                    ),
                    len(self._videos),
                )
                if self._merge_run_summary is not None:
                    self._merge_run_summary.record_retry_used()
            try:
                merged_content, raw_response_text = _attempt_merge_once(
                    language=self._language,
                    videos=self._videos,
                    config=self._config,
                    provider=self._provider,
                    model_name=self._model,
                    attempt_index=attempt_index,
                    attempt_label=f"{self._attempt_label}_PRIMARY_{attempt_index}",
                    branch_label=self._branch_label,
                    date_key=self._date_key,
                    slot_key=self._slot_key,
                    no_description_text=self._no_description_text,
                    expanded_retry_profile=retry_profile,
                    max_body_paragraphs=self._max_body_paragraphs,
                )
                last_raw_response = raw_response_text
                if self._merge_run_summary is not None and merged_content.tail_recovery_applied:
                    self._merge_run_summary.record_paragraph_recovery_used()
                if self._merge_run_summary is not None:
                    self._merge_run_summary.record_merge_success()
                LOGGER.info(
                    "merge_branch_ready branch=%s date_key=%s slot_key=%s language=%s generator_model=%s",
                    self._branch_label,
                    self._date_key,
                    self._slot_key,
                    self._language,
                    self._model,
                )
                LOGGER.info(
                    "merge_provider_summary provider=%s model=%s structured_ok=yes fallback_used=no parse_repair_used=no final_status=success",
                    self._provider.name,
                    self._model,
                )
                return self._build_success_result(
                    merged_content=merged_content,
                    raw_response_text=raw_response_text,
                )
            except LlmModelConfigurationError as error:
                LOGGER.error(
                    "merge_llm_fatal_model_error branch=%s date_key=%s slot_key=%s language=%s provider=%s model=%s code=%s status_code=%s api_error_code=%s api_error_param=%s reason=%s",
                    self._branch_label,
                    self._date_key,
                    self._slot_key,
                    self._language,
                    self._provider.name,
                    self._model,
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
                pending_retry_profile = self._select_retry_profile(
                    error=error,
                    diagnostics=diagnostics,
                )
                if any(
                    code in RECOVERABLE_REJECT_CODES
                    for code in (error.reason_codes or (error.reason_code,))
                ):
                    effective_max_attempts = PRIMARY_ATTEMPTS_EXTENDED
                _log_merge_attempt_invalid(
                    provider_name=self._provider.name,
                    model_name=self._model,
                    attempt_index=attempt_index,
                    is_fallback=False,
                    branch_label=self._branch_label,
                    date_key=self._date_key,
                    slot_key=self._slot_key,
                    language=self._language,
                    reason_code=error.reason_code,
                    reason=error.reason,
                    raw_response_text=error.raw_response_text,
                )
                if self._merge_run_summary is not None:
                    self._merge_run_summary.record_validation_rejected()
            except Exception as error:
                error_classification = classify_openai_request_error(error)
                if error_classification.reason_code == self._OPENAI_QUOTA_REASON_CODE:
                    last_error_summary = self._summarize_error(error)
                    last_reason_code = self._OPENAI_QUOTA_REASON_CODE
                    last_reason_codes = (self._OPENAI_QUOTA_REASON_CODE,)
                    _log_merge_attempt_invalid(
                        provider_name=self._provider.name,
                        model_name=self._model,
                        attempt_index=attempt_index,
                        is_fallback=False,
                        branch_label=self._branch_label,
                        date_key=self._date_key,
                        slot_key=self._slot_key,
                        language=self._language,
                        reason_code=self._OPENAI_QUOTA_REASON_CODE,
                        reason=last_error_summary,
                        raw_response_text="",
                    )
                    if self._merge_run_summary is not None:
                        self._merge_run_summary.record_quota_exhausted()
                    break
                last_error_summary = self._summarize_error(error)
                last_reason_code = self._UNEXPECTED_ERROR_REASON_CODE
                last_reason_codes = (self._UNEXPECTED_ERROR_REASON_CODE,)
                pending_retry_profile = _standard_expanded_retry_profile(
                    reject_signals=(self._UNEXPECTED_ERROR_REASON_CODE,)
                )
                _log_merge_attempt_invalid(
                    provider_name=self._provider.name,
                    model_name=self._model,
                    attempt_index=attempt_index,
                    is_fallback=False,
                    branch_label=self._branch_label,
                    date_key=self._date_key,
                    slot_key=self._slot_key,
                    language=self._language,
                    reason_code=self._UNEXPECTED_ERROR_REASON_CODE,
                    reason=last_error_summary,
                    raw_response_text="",
                )

        final_reason_code: str = last_reason_code
        log_warning_operational(
            LOGGER,
            "merge_llm_final_failure branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=primary code=%s fallback_used=no raw_response_received=%s reason=%s raw_chars=%d",
            self._branch_label,
            self._date_key,
            self._slot_key,
            self._language,
            self._provider.name,
            final_reason_code,
            "yes" if bool(last_raw_response.strip()) else "no",
            last_error_summary,
            len(last_raw_response),
            reason_code=final_reason_code,
        )
        LOGGER.info(
            "merge_provider_summary provider=%s model=%s structured_ok=no fallback_used=no parse_repair_used=no final_status=failed",
            self._provider.name,
            self._model,
        )
        if self._merge_run_summary is not None:
            self._merge_run_summary.record_final_failure()
        return self._build_failure_result(
            last_error_summary=last_error_summary,
            last_reason_code=last_reason_code,
            last_reason_codes=last_reason_codes,
            last_raw_response=last_raw_response,
            rejected_attempts=rejected_attempts,
        )
