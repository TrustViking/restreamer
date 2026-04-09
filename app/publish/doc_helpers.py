from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppTemplates
from app.core.language_display import language_display_name
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    MergedPublicationPayload,
    PlannedVideo,
    SanitizedPublishBlock,
)
from app.planning import planned_video_block_language
from app.publish.post_llm_sanitation import (
    build_sanitized_merged_publication_payload,
    log_safe_merge_attempt_fallback,
    should_suppress_raw_merge_attempt_publish,
)
from app.publish.shared_helpers import (
    fallback_source_description_text as _fallback_source_description_text,
    is_merge_payload_blocked as _is_merge_payload_blocked,
    no_description_text as _no_description_text,
    normalize_urls_in_text as _normalize_urls_in_text,
    numbered_lines as _numbered_lines,
    numbered_original_titles as _numbered_original_titles,
)


LOGGER = _get_logger_impl(__name__)

def _language_heading(language: str, templates: Optional[AppTemplates]) -> str:
    return language_display_name(language)


def _build_guarded_merged_payload(
    *,
    target: str,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    merge_attempt: Optional[LanguageMergeAttempt],
    use_audit_text: bool,
) -> Optional[MergedPublicationPayload]:
    language: str = planned_video_block_language(videos[0]) if videos else "unknown"
    payload: MergedPublicationPayload = build_sanitized_merged_publication_payload(
        language=language,
        merged_content=merged_content,
        merge_attempt=merge_attempt,
        use_audit_text=use_audit_text,
        source_videos=videos,
    )
    if _is_merge_payload_blocked(payload):
        LOGGER.warning(
            "merge_publish_gate_blocked target=%s language=%s has_publish_stage_duplicate=%s has_publish_stage_opener_cta=%s fallback=source_descriptions",
            target,
            language,
            "yes" if bool(getattr(payload, "has_publish_stage_duplicate", False)) else "no",
            "yes" if bool(getattr(payload, "has_publish_stage_opener_cta", False)) else "no",
        )
        return None
    return payload


def _build_titles_summary(
    videos: List[PlannedVideo],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    use_audit_text: bool = True,
) -> str:
    if merged_content:
        payload: Optional[MergedPublicationPayload] = _build_guarded_merged_payload(
            target="doc",
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=use_audit_text,
        )
        if payload is not None:
            return payload.title_text
        return _numbered_original_titles(videos) if videos else "1) ..."
    if merge_attempt is not None:
        salvaged_title: str = str(merge_attempt.salvaged_title or "").strip()
        if salvaged_title:
            return salvaged_title
        return _numbered_original_titles(videos) if videos else "1) ..."
    if not videos:
        return "1) ..."
    return _numbered_original_titles(videos)


def _build_rejected_attempts_block(
    rejected_attempts: tuple["RejectedMergeAttempt", ...],
) -> str:
    if not rejected_attempts:
        return ""
    lines: List[str] = ["MODEL OUTPUTS REJECTED BY VALIDATION."]
    for attempt in rejected_attempts:
        reject_codes_label: str = ",".join(attempt.reject_reasons) if attempt.reject_reasons else "unknown"
        title_header: str = (
            f"REJECTED TITLE | ATTEMPT {attempt.attempt_index}"
            f" | MODEL: {attempt.model_name}"
            f" | REJECT: {reject_codes_label}"
        )
        description_header: str = (
            f"REJECTED DESCRIPTION | ATTEMPT {attempt.attempt_index}"
            f" | MODEL: {attempt.model_name}"
            f" | REJECT: {reject_codes_label}"
        )
        lines.append(title_header)
        if attempt.title:
            lines.append(attempt.title)
        lines.append(description_header)
        if attempt.description:
            lines.append(attempt.description)
    return "\n".join(lines)


def _build_descriptions_summary(
    videos: List[PlannedVideo],
    templates: Optional[AppTemplates],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    use_audit_text: bool = True,
) -> str:
    source_descriptions: List[str] = []
    for video in videos:
        source_descriptions.append(_fallback_source_description_text(video, templates))
    source_lines: str = _numbered_lines(source_descriptions)
    if merged_content:
        payload: Optional[MergedPublicationPayload] = _build_guarded_merged_payload(
            target="doc",
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=use_audit_text,
        )
        if payload is not None:
            return payload.description_text
    if merge_attempt is not None:
        if should_suppress_raw_merge_attempt_publish(
            merge_attempt=merge_attempt,
            merged_content_available=False,
        ):
            log_safe_merge_attempt_fallback(
                target="doc",
                merge_attempt=merge_attempt,
                fallback_label="source_descriptions",
            )
        rejected_block: str = _build_rejected_attempts_block(merge_attempt.rejected_attempts)
        if rejected_block:
            return f"{source_lines}\n\n{rejected_block}"
        return source_lines
    if not videos:
        return "1) ..."
    if len(source_descriptions) == 1:
        return source_descriptions[0]
    lines: List[str] = []
    for index, item in enumerate(source_descriptions, start=1):
        lines.append(f"{index}) {item}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_preview_placeholder_rows(
    videos: List[PlannedVideo],
) -> List[Tuple[str, bool]]:
    if not videos:
        return [(" ", False)]
    return [(" ", False) for _ in videos]


def _artifact_heading_for_block(
    *,
    base_heading: str,
    merged_payload: Optional[MergedPublicationPayload],
    merge_attempt: Optional[LanguageMergeAttempt],
    merged_content: Optional[MergedLanguageContent],
    artifact_status: str = "none",
) -> str:
    if merged_content is not None:
        return base_heading
    if merge_attempt is None:
        return base_heading
    if artifact_status in ("fallback_only", "partial"):
        return f"{base_heading} ⚠ [merge failed — source list]"
    return base_heading


def _extract_youtube_video_id(raw_value: str) -> Optional[str]:
    text: str = str(raw_value or "").strip()
    if not text:
        return None
    context_patterns: Tuple[Tuple[str, str], ...] = (
        (
            "context",
            r"(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/watch\?[^#\s]*?(?:[?&]v=|&v=)([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
    )
    first_context_match: Optional[Tuple[int, str, str]] = None
    for method_name, pattern in context_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate: str = str(match.group(1) or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                continue
            candidate_pos: int = match.start(1)
            if first_context_match is None or candidate_pos < first_context_match[0]:
                first_context_match = (candidate_pos, method_name, candidate)
    if first_context_match is not None:
        _, method_name, candidate = first_context_match
        LOGGER.debug("youtube_id_extracted method=%s id=%s", method_name, candidate)
        return candidate

    for match in re.finditer(r"([A-Za-z0-9_-]{11})", text):
        candidate = str(match.group(1) or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            continue
        start_index: int = match.start(1)
        end_index: int = match.end(1)
        char_before: str = text[start_index - 1] if start_index > 0 else ""
        char_after: str = text[end_index] if end_index < len(text) else ""
        if char_before and re.fullmatch(r"[A-Za-z0-9_-]", char_before):
            continue
        if char_after and re.fullmatch(r"[A-Za-z0-9_-]", char_after):
            continue
        LOGGER.debug("youtube_id_extracted method=raw id=%s", candidate)
        return candidate
    return None


def _youtube_video_id_from_url(video_url: str) -> Optional[str]:
    return _extract_youtube_video_id(video_url)


def _thumbnail_candidates(video: PlannedVideo) -> List[str]:
    candidates: List[str] = []
    video_id: Optional[str] = _youtube_video_id_from_url(video.normalized_link)
    if video_id:
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg")
    candidates.append(video.metadata.thumbnail_url)
    deduped: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        key: str = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _build_language_table_rows(
    language: str,
    videos: List[PlannedVideo],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    time_display: Optional[str] = None,
    templates: Optional[AppTemplates] = None,
    artifact_status: str = "none",
    sanitized_block: Optional[SanitizedPublishBlock] = None,
) -> List[Tuple[str, bool]]:
    if templates is None:
        raise RuntimeError("Templates are required for language table labels.")
    labels_payload: object = getattr(templates, "google_doc_table_labels", {})
    if not isinstance(labels_payload, dict):
        raise RuntimeError("Invalid template google_doc.table_labels: expected mapping")
    labels: Dict[str, Tuple[str, str, str]] = {}
    for key, value in labels_payload.items():
        if isinstance(value, list) and len(value) == 3:
            labels[str(key)] = (str(value[0]), str(value[1]), str(value[2]))
    title_label, desc_label, preview_label = labels.get(
        language,
        ("TITLE", "DESCRIPTION", "PREVIEW"),
    )

    heading: str = _language_heading(language, templates)
    if time_display:
        heading = f"{heading} - {time_display}"
    merged_payload: Optional[MergedPublicationPayload] = None
    if sanitized_block is not None and not sanitized_block.is_blocked:
        # Use the cached sanitation result - no re-computation
        merged_payload = MergedPublicationPayload(
            title_text=sanitized_block.title_text,
            description_text=sanitized_block.description_text,
            block_generation_mode=sanitized_block.block_generation_mode,
        )
    elif merged_content is not None:
        merged_payload = _build_guarded_merged_payload(
            target="doc",
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=True,
        )
    if sanitized_block is not None and not sanitized_block.is_blocked:
        LOGGER.info(
            "publish_sanitation_cache_hit lang=%s target=doc",
            language,
        )
    elif merged_content is not None and merged_payload is not None:
        LOGGER.info(
            "publish_sanitation_cache_miss lang=%s target=doc reason=fallback_to_live_sanitation",
            language,
        )
    heading_merged_content: Optional[MergedLanguageContent] = (
        merged_content if merged_payload is not None else None
    )
    heading = _artifact_heading_for_block(
        base_heading=heading,
        merged_payload=merged_payload,
        merge_attempt=merge_attempt,
        merged_content=heading_merged_content,
        artifact_status=artifact_status,
    )
    description_text: str = (
        merged_payload.description_text
        if merged_payload is not None
        else _build_descriptions_summary(
            videos=videos,
            templates=templates,
            merged_content=None,
            merge_attempt=merge_attempt,
        )
    )

    rows: List[Tuple[str, bool]] = [
        (heading, True),
        (title_label, True),
        (
            (
                merged_payload.title_text
                if merged_payload is not None
                else _build_titles_summary(
                    videos=videos,
                    merged_content=None,
                    merge_attempt=merge_attempt,
                )
            ),
            True,
        ),
        (desc_label, True),
        (description_text, False),
        (preview_label, True),
    ]
    rows.extend(_build_preview_placeholder_rows(videos))
    return rows
