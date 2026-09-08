from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.core.error_summary import summarize_error
from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
    SanitizedPublishBlock,
)
from app.core.text_utils import normalize_multiline_text
from app.ingest.youtube_metadata import normalize_youtube_video_url
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.merges.merge_service import (
    attempt_llm_merge_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.llm.models.model_identity import resolve_effective_llm_model
from app.planning import language_sort_key, planned_video_block_language
from app.publish.post_llm_sanitation import (
    MergedPublicationPayload,
    build_sanitized_merged_publication_payload,
)
from app.publish.shared_helpers import no_description_text as _publish_no_description_text
from app.publish.header_context import build_header_context
from app.observability.runtime_analytics import (
    log_stage_timing,
    record_branch_model_used,
    record_slot_total_ms,
)


MergeArtifactStatus = Literal["none", "full", "partial", "fallback_only"]


@dataclass(frozen=True)
class SlotProcessResult:
    slot_key: str
    slot_time_key: str
    header_context: Dict[str, str]
    day_videos: List[PlannedVideo]
    language_groups: Dict[str, List[PlannedVideo]]
    merged_content_by_language: Dict[str, MergedLanguageContent]
    merge_audit_by_language: Dict[str, LanguageMergeAttempt]
    real_merge_blocks: int
    merge_candidate_blocks: int = 0
    fallback_merge_blocks: int = 0
    merge_artifact_status: MergeArtifactStatus = "none"
    fallback_merge_targets: Tuple[str, ...] = ()
    merge_skipped_languages: Tuple[str, ...] = ()
    sanitized_blocks: Dict[str, SanitizedPublishBlock] = field(default_factory=dict)
    language: str = "unknown"


def resolve_merge_artifact_status(
    *,
    merge_candidate_blocks: int,
    real_merge_blocks: int,
    fallback_merge_blocks: int,
) -> MergeArtifactStatus:
    if merge_candidate_blocks <= 0:
        return "none"
    if real_merge_blocks > 0:
        if fallback_merge_blocks > 0:
            return "partial"
        return "full"
    if fallback_merge_blocks > 0:
        return "fallback_only"
    return "none"


def _merge_summary_snapshot(
    merge_run_summary: MergeRunSummary,
) -> Tuple[int, int, int, int]:
    return (
        merge_run_summary.merge_success,
        merge_run_summary.validation_rejected,
        merge_run_summary.retry_used,
        merge_run_summary.final_failure,
    )


def _log_merge_attempt_outcome(
    *,
    logger: logging.Logger,
    slot_key: str,
    language: str,
    source_count: int,
    merge_attempt: LanguageMergeAttempt,
    before_snapshot: Tuple[int, int, int, int],
    after_snapshot: Tuple[int, int, int, int],
) -> None:
    deltas: Tuple[int, int, int, int, int, int] = tuple(
        after_value - before_value
        for before_value, after_value in zip(before_snapshot, after_snapshot)
    )
    logger.info(
        "merge_attempt_outcome slot_key=%s language=%s source_count=%d success=%s merge_success=%d validation_rejected=%d retry_used=%d final_failure=%d",
        slot_key,
        language,
        source_count,
        "yes" if merge_attempt.merged is not None else "no",
        deltas[0],
        deltas[1],
        deltas[2],
        deltas[3],
    )


def _strip_urls_for_summary(text: str) -> str:
    normalized: str = normalize_multiline_text(text)
    return normalized


def _log_merge_input_summary(
    *,
    logger: logging.Logger,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    language_items_for_merge: List[PlannedVideo],
    desc_max_chars_limit: int,
    merge_expected: bool,
    merge_skip_reason: str,
) -> None:
    descriptions: List[str] = [
        str(item.metadata.description or "").strip() for item in language_items_for_merge
    ]
    titles_non_empty: int = sum(
        1 for item in language_items_for_merge if str(item.metadata.title or "").strip()
    )
    source_desc_chars_total: int = sum(len(description) for description in descriptions)
    stripped_descriptions: List[str] = [_strip_urls_for_summary(text) for text in descriptions]
    source_desc_chars_after_trim: int = sum(len(description) for description in stripped_descriptions)
    non_empty_descriptions: int = sum(1 for description in descriptions if description)
    logger.info(
        "merge_input_summary branch=%s date_key=%s slot_key=%s lang=%s source_count=%d non_empty_descriptions=%d titles_non_empty=%d source_desc_chars_total=%d source_desc_chars_passed_to_llm=%d hard_truncation=%s configured_desc_max_chars_limit=%d merge_expected=%s merge_skip_reason=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        len(language_items_for_merge),
        non_empty_descriptions,
        titles_non_empty,
        source_desc_chars_total,
        source_desc_chars_after_trim,
        "disabled",
        desc_max_chars_limit,
        "yes" if merge_expected else "no",
        merge_skip_reason or "not_applicable",
    )


def process_slot(
    *,
    logger: logging.Logger,
    config: AppConfig,
    videos: List[PlannedVideo],
    date_key: str,
    slot_time_key: str,
    llm_merge_enabled: bool,
    cet_tz: ZoneInfo,
    merge_run_summary: MergeRunSummary,
    branch_label: str,
) -> SlotProcessResult:
    slot_started_at: float = time.perf_counter()
    language_groups: Dict[str, List[PlannedVideo]] = {}
    raw_video: PlannedVideo
    for raw_video in videos:
        block_language: str = planned_video_block_language(raw_video)
        language_groups.setdefault(block_language, []).append(raw_video)
    ordered_slot_languages: List[str] = sorted(
        language_groups.keys(),
        key=language_sort_key,
    )
    slot_language: str = ordered_slot_languages[0] if ordered_slot_languages else "unknown"
    slot_key: str = f"{date_key}_{slot_time_key}_{slot_language}"
    logger.info(
        "[%s] slot_process_start date_key=%s slot_key=%s video_count=%d",
        branch_label,
        date_key,
        slot_key,
        len(videos),
    )
    day_videos: List[PlannedVideo] = sorted(
        videos,
        key=lambda item: (
            item.scheduled_at_kiev.time(),
            item.row_number,
        ),
    )
    slot_lang_counts: Dict[str, int] = {
        language: len(language_groups.get(language, ()))
        for language in ordered_slot_languages
    }
    counts_payload: str = ", ".join(
        f"{language}:{slot_lang_counts[language]}" for language in ordered_slot_languages
    )
    if len(ordered_slot_languages) > 1:
        logger.warning(
            "[%s] slot_language_mismatch slot_key=%s languages=%s",
            branch_label,
            slot_key,
            ordered_slot_languages,
        )
    logger.info(
        "[%s] Processing slot=%s date=%s time=%s counts_by_language=%s",
        branch_label,
        slot_key,
        date_key,
        slot_time_key,
        counts_payload or "none",
    )

    merged_content_by_language: Dict[str, MergedLanguageContent] = {}
    merge_audit_by_language: Dict[str, LanguageMergeAttempt] = {}
    real_merge_blocks: int = 0
    merge_candidate_blocks: int = 0
    fallback_merge_blocks: int = 0
    fallback_merge_targets: List[str] = []
    if llm_merge_enabled:
        merge_skipped_languages_list: List[str] = []
        for language in ordered_slot_languages:
            language_items_for_merge: List[PlannedVideo] = sorted(
                language_groups.get(language, []),
                key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
            )
            non_empty_descriptions_count: int = sum(
                1 for item in language_items_for_merge if item.metadata.description.strip()
            )
            run_classic_merge: bool = non_empty_descriptions_count >= 2
            merge_expected: bool = run_classic_merge
            merge_skip_reason: str = "not_applicable"
            if not merge_expected:
                merge_skip_reason = (
                    "insufficient_descriptions"
                    if llm_merge_enabled
                    else "llm_disabled_for_branch"
                )
            _log_merge_input_summary(
                logger=logger,
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                language_items_for_merge=language_items_for_merge,
                desc_max_chars_limit=config.llm.source_desc_max_chars,
                merge_expected=merge_expected,
                merge_skip_reason=merge_skip_reason,
            )
            if not run_classic_merge:
                logger.info(
                    "[%s] LLM merge skipped for language=%s: non-empty descriptions=%d reason=%s.",
                    branch_label,
                    language,
                    non_empty_descriptions_count,
                    merge_skip_reason,
                )
                if (
                    language_items_for_merge
                    and merge_skip_reason == "insufficient_descriptions"
                ):
                    merge_skipped_languages_list.append(language)
                continue
            merge_before_snapshot: Tuple[int, int, int, int, int, int] = _merge_summary_snapshot(
                merge_run_summary
            )
            merge_attempt: LanguageMergeAttempt
            is_multi_source_merge_candidate: bool = len(language_items_for_merge) > 1
            quota_stop_preexisting: bool = merge_run_summary.provider_quota_exhausted
            if quota_stop_preexisting:
                model_name: str = resolve_effective_llm_model(config)
                logger.error(
                    "merge_branch_aborted_quota_exhausted branch=%s date_key=%s slot_key=%s language=%s",
                    branch_label,
                    date_key,
                    slot_key,
                    language,
                )
                merge_attempt = LanguageMergeAttempt(
                    language=language,
                    model_name=model_name,
                    raw_response_text="",
                    merged=None,
                    error_summary="openai_quota_exhausted",
                    salvaged_title=None,
                    publish_source_label="merge_failed_openai_quota_exhausted",
                    validation_reasons=["openai_quota_exhausted"],
                    generator_model_name=model_name,
                    used_model_names=(model_name,) if model_name else (),
                    branch_type=BRANCH_MERGE,
                    title_source="fallback_titles",
                    hook_source="fallback_none",
                    hashtags_source="fallback_none",
                    body_source="fallback_source_descriptions",
                    block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
                )
            else:
                merge_attempt = attempt_llm_merge_with_audit(
                    language=language,
                    videos=language_items_for_merge,
                    config=config,
                    attempt_label=f"OPENAI_MERGE_{language.upper()}",
                    summarize_error=summarize_error,
                    normalize_youtube_url=normalize_youtube_video_url,
                    no_description_text=_publish_no_description_text(config.templates),
                    merge_run_summary=merge_run_summary,
                    branch_label=branch_label,
                    date_key=date_key,
                    slot_key=slot_key,
                )
            if (not quota_stop_preexisting) and merge_run_summary.provider_quota_exhausted:
                logger.error(
                    "merge_branch_aborted_quota_exhausted branch=%s date_key=%s slot_key=%s language=%s",
                    branch_label,
                    date_key,
                    slot_key,
                    language,
                )
            logger.info(
                "[%s] LLM merge result language=%s provider=%s model=%s success=%s",
                branch_label,
                language,
                config.llm.provider,
                merge_attempt.model_name,
                "yes" if merge_attempt.merged is not None else "no",
            )
            logger.info(
                "branch_field_sources branch_type=%s date_key=%s slot_key=%s language=%s block_generation_mode=%s title_source=%s hook_source=%s hashtags_source=%s body_source=%s",
                branch_label,
                date_key,
                slot_key,
                language,
                getattr(merge_attempt, "block_generation_mode", "unknown"),
                merge_attempt.title_source or "unknown",
                merge_attempt.hook_source or "unknown",
                merge_attempt.hashtags_source or "unknown",
                merge_attempt.body_source or "unknown",
            )
            merge_audit_by_language[language] = merge_attempt
            for used_model_name in merge_attempt.used_model_names or (merge_attempt.model_name,):
                record_branch_model_used(
                    date_key=date_key,
                    branch_label=branch_label,
                    model_name=used_model_name,
                )
            if is_multi_source_merge_candidate:
                merge_candidate_blocks += 1
                merge_run_summary.record_merge_candidate_block()
            if merge_attempt.merged is not None:
                merged_content_value: MergedLanguageContent = merge_attempt.merged
                if is_multi_source_merge_candidate:
                    real_merge_blocks += 1
                    merge_run_summary.record_real_merge_block()
                if is_multi_source_merge_candidate:
                    merged_content_value = enforce_openai_merged_paragraphs(
                        language=language,
                        merged_content=merged_content_value,
                        videos=language_items_for_merge,
                        config=config,
                        no_description_text=_publish_no_description_text(
                            config.templates
                        ),
                        merge_run_summary=merge_run_summary,
                        branch_label=branch_label,
                        date_key=date_key,
                        slot_key=slot_key,
                    )
                merged_content_by_language[language] = merged_content_value
            else:
                if is_multi_source_merge_candidate:
                    fallback_merge_blocks += 1
                    fallback_merge_targets.append(f"{slot_key}:{language}")
                    merge_run_summary.record_fallback_merge_block()
                logger.info(
                    "[%s] LLM merge failed for language=%s provider=%s reason=%s",
                    branch_label,
                    language,
                    config.llm.provider,
                    merge_attempt.error_summary or "unknown",
                )
            merge_after_snapshot: Tuple[int, int, int, int, int, int] = _merge_summary_snapshot(
                merge_run_summary
            )
            _log_merge_attempt_outcome(
                logger=logger,
                slot_key=slot_key,
                language=language,
                source_count=len(language_items_for_merge),
                merge_attempt=merge_attempt,
                before_snapshot=merge_before_snapshot,
                after_snapshot=merge_after_snapshot,
            )
    else:
        for language in ordered_slot_languages:
            language_items_for_merge = sorted(
                language_groups.get(language, []),
                key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
            )
            _log_merge_input_summary(
                logger=logger,
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                language_items_for_merge=language_items_for_merge,
                desc_max_chars_limit=config.llm.source_desc_max_chars,
                merge_expected=False,
                merge_skip_reason=(
                    "processing_mode_nomerge"
                    if branch_label.endswith(BRANCH_NOMERGE) or branch_label == BRANCH_NOMERGE
                    else "llm_unavailable"
                ),
            )
        if branch_label.endswith(BRANCH_NOMERGE) or branch_label == BRANCH_NOMERGE:
            logger.info(
                "[%s] slot=%s merge skipped by processing_mode=nomerge; append mode active.",
                branch_label,
                slot_key,
            )
            for language in ordered_slot_languages:
                if language_groups.get(language):
                    logger.info(
                        "branch_field_sources branch_type=%s date_key=%s slot_key=%s language=%s title_source=%s hook_source=%s hashtags_source=%s body_source=%s",
                        branch_label,
                        date_key,
                        slot_key,
                        language,
                        "nomerge",
                        "nomerge",
                        "fallback_none",
                        "nomerge",
                    )
        else:
            logger.info(
                "[%s] slot=%s merge skipped (LLM unavailable). Using append mode.",
                branch_label,
                slot_key,
            )

    header_context: Dict[str, str] = build_header_context(
        videos=day_videos,
        form_url=config.google.form_url,
        contacts=config.google.contacts,
        cet_tz=cet_tz,
    )
    sanitized_blocks: Dict[str, SanitizedPublishBlock] = {}
    for lang, merged_content_value in merged_content_by_language.items():
        lang_videos: List[PlannedVideo] = sorted(
            language_groups.get(lang, []),
            key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
        )
        merge_attempt_for_lang: Optional[LanguageMergeAttempt] = merge_audit_by_language.get(lang)
        try:
            payload: MergedPublicationPayload = build_sanitized_merged_publication_payload(
                language=lang,
                merged_content=merged_content_value,
                merge_attempt=merge_attempt_for_lang,
                use_audit_text=True,
                source_videos=lang_videos,
            )
            is_blocked: bool = bool(
                getattr(payload, "has_publish_stage_duplicate", False)
                or getattr(payload, "has_publish_stage_opener_cta", False)
            )
            sanitized_blocks[lang] = SanitizedPublishBlock(
                language=lang,
                title_text=payload.title_text,
                description_text=payload.description_text,
                block_generation_mode=payload.block_generation_mode,
                is_blocked=is_blocked,
            )
            logger.info(
                "sanitized_block_cached language=%s is_blocked=%s title_chars=%d description_chars=%d",
                lang,
                "yes" if is_blocked else "no",
                len(payload.title_text),
                len(payload.description_text),
            )
        except Exception as sanitation_error:
            logger.warning(
                "sanitized_block_failed language=%s error=%s",
                lang,
                sanitation_error,
            )
    logger.info(
        "[%s] slot_process_finish date_key=%s slot_key=%s video_count=%d merged_languages=%s",
        branch_label,
        date_key,
        slot_key,
        len(day_videos),
        sorted(merged_content_by_language.keys()),
    )
    merge_artifact_status: MergeArtifactStatus = resolve_merge_artifact_status(
        merge_candidate_blocks=merge_candidate_blocks,
        real_merge_blocks=real_merge_blocks,
        fallback_merge_blocks=fallback_merge_blocks,
    )
    logger.info(
        "[%s] slot_merge_decision slot_key=%s merge_candidate_blocks=%d real_merge_blocks=%d fallback_merge_blocks=%d merge_artifact_status=%s fallback_targets=%s merge_doc_eligible=%s",
        branch_label,
        slot_key,
        merge_candidate_blocks,
        real_merge_blocks,
        fallback_merge_blocks,
        merge_artifact_status,
        ",".join(fallback_merge_targets) or "none",
        "yes" if merge_artifact_status != "none" else "no",
    )
    slot_total_ms: int = int(round((time.perf_counter() - slot_started_at) * 1000.0))
    record_slot_total_ms(slot_key=slot_key, elapsed_ms=slot_total_ms)
    log_stage_timing(
        logger=logger,
        stage_name="slot_total",
        elapsed_ms=slot_total_ms,
        scope="slot",
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )
    return SlotProcessResult(
        slot_key=slot_key,
        slot_time_key=slot_time_key,
        header_context=header_context,
        day_videos=day_videos,
        language_groups=language_groups,
        merged_content_by_language=merged_content_by_language,
        merge_audit_by_language=merge_audit_by_language,
        real_merge_blocks=real_merge_blocks,
        merge_candidate_blocks=merge_candidate_blocks,
        fallback_merge_blocks=fallback_merge_blocks,
        merge_artifact_status=merge_artifact_status,
        fallback_merge_targets=tuple(fallback_merge_targets),
        merge_skipped_languages=tuple(merge_skipped_languages_list) if llm_merge_enabled else (),
        sanitized_blocks=sanitized_blocks,
        language=slot_language,
    )

