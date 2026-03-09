from __future__ import annotations

import dataclasses
import logging
from typing import Dict

from app.config.settings import AppConfig
from app.core.models import LanguageMergeAttempt, MergedLanguageContent
from app.llm.merge_packaging import build_packaging_overlay_attempt
from app.observability.runtime_analytics import record_branch_model_used

from .slot_processing import SlotProcessResult


def build_packaging_slot_result(
    *,
    logger: logging.Logger,
    config: AppConfig,
    source_slot_result: SlotProcessResult,
    branch_label: str,
    date_key: str,
) -> SlotProcessResult:
    merged_content_by_language: Dict[str, MergedLanguageContent] = {}
    merge_audit_by_language: Dict[str, LanguageMergeAttempt] = {}
    for language in ("uk", "en", "ru", "other"):
        source_merge_attempt: LanguageMergeAttempt | None = (
            source_slot_result.merge_audit_by_language.get(language)
        )
        source_merged_content: MergedLanguageContent | None = (
            source_slot_result.merged_content_by_language.get(language)
        )
        source_videos = source_slot_result.language_groups.get(language, [])
        if source_merge_attempt is None or source_merged_content is None or not source_videos:
            continue
        packaged_content, packaged_attempt = build_packaging_overlay_attempt(
            logger=logger,
            language=language,
            videos=source_videos,
            merge_attempt=source_merge_attempt,
            merged_content=source_merged_content,
            config=config,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=source_slot_result.slot_key,
        )
        merged_content_by_language[language] = packaged_content
        merge_audit_by_language[language] = packaged_attempt
        packaging_model_name: str = ""
        if packaged_attempt.packaging_audit is not None:
            packaging_model_name = str(packaged_attempt.packaging_audit.packaging_model or "").strip()
        if packaging_model_name:
            record_branch_model_used(
                date_key=date_key,
                branch_label=branch_label,
                model_name=packaging_model_name,
            )
        logger.info(
            "branch_field_sources branch_type=%s date_key=%s slot_key=%s language=%s title_source=%s hook_source=%s hashtags_source=%s body_source=%s",
            branch_label,
            date_key,
            source_slot_result.slot_key,
            language,
            packaged_content.title_source or "unknown",
            packaged_content.hook_source or "unknown",
            packaged_content.hashtags_source or "unknown",
            packaged_content.body_source or "unknown",
        )
    return dataclasses.replace(
        source_slot_result,
        merged_content_by_language=merged_content_by_language,
        merge_audit_by_language=merge_audit_by_language,
    )
