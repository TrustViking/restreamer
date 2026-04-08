from __future__ import annotations

import dataclasses
from typing import Callable, List, Optional

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
)
from app.llm.llm_client import openai_request_merge
from app.llm.merges.merge_executor import MergeExecutor
from app.llm.merges.merge_formatting import ParagraphEnforcementResult
from app.llm.merges.merge_orchestrator import MergeOrchestrator
from app.llm.merges.merge_run_summary import MergeRunSummary

LOGGER = _get_logger_impl(__name__)


def _fallback_title_source_label() -> str:
    return "fallback_titles"


def _fallback_hook_source_label() -> str:
    return "fallback_none"


def _fallback_body_source_label() -> str:
    return "fallback_source_descriptions"


def _enforce_merged_description_structure(
    *,
    merged_content: MergedLanguageContent,
) -> ParagraphEnforcementResult:
    return MergeExecutor._enforce_description_structure(
        merged_content=merged_content
    )


def _enforce_expanded_single_step_paragraph_overflow_policy(
    *,
    description_text: str,
    max_body_paragraphs: int,
) -> None:
    return MergeExecutor._enforce_single_step_overflow_policy(
        description_text=description_text,
        max_body_paragraphs=max_body_paragraphs,
    )


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
    orchestrator: MergeOrchestrator = MergeOrchestrator(
        config=config,
        videos=videos,
        language=language,
        attempt_label=attempt_label,
        summarize_error=summarize_error,
        no_description_text=no_description_text,
        merge_run_summary=merge_run_summary,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )
    return orchestrator.run()


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
    enforcement_result: ParagraphEnforcementResult = MergeExecutor._enforce_description_structure(
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
