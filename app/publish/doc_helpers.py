from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppTemplates
from app.core.env_flags import (
    strip_chapter_timestamps,
    strip_chapter_timestamps_enabled_from_env,
)
from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    FALLBACK_ONLY_ARTIFACT_MARKER,
    PARTIAL_FALLBACK_ARTIFACT_MARKER,
    LanguageMergeAttempt,
    MergedLanguageContent,
    MergedPublicationPayload,
    PlannedVideo,
    RejectedMergeAttempt,
)
from app.planning import planned_video_block_language
from app.publish.post_llm_sanitation import (
    build_sanitized_merged_publication_payload,
    log_safe_merge_attempt_fallback,
    resolve_block_generation_mode,
    should_suppress_raw_merge_attempt_publish,
)


LOGGER = _get_logger_impl(__name__)

_REJECTED_ATTEMPTS_NOTICE: str = (
    "MODEL OUTPUTS REJECTED BY VALIDATION. The script received these merge attempts, "
    "but all were rejected by the validator. Saved below for analysis."
)


def _language_heading(language: str, templates: Optional[AppTemplates]) -> str:
    if templates is None:
        return "OTHER"
    try:
        payload: Any = json.loads(templates.google_doc_language_headings_json)
        if isinstance(payload, dict):
            return str(payload.get(language, payload.get("other", "OTHER")))
    except Exception:
        pass
    return "OTHER"


def _numbered_lines(values: Sequence[str]) -> str:
    cleaned_values: List[str] = [
        str(item or "").strip() for item in values if str(item or "").strip()
    ]
    if not cleaned_values:
        return "1) ..."
    if len(cleaned_values) == 1:
        return cleaned_values[0]
    return "\n".join(
        [f"{index}) {item}" for index, item in enumerate(cleaned_values, start=1)]
    )


def _numbered_original_titles(videos: List[PlannedVideo]) -> str:
    source_titles: List[str] = [video.metadata.title for video in videos]
    return _numbered_lines(source_titles)


def _build_titles_summary(
    videos: List[PlannedVideo],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    use_audit_text: bool = True,
    prefer_rejected_attempts: bool = False,
) -> str:
    if merged_content:
        language: str = planned_video_block_language(videos[0]) if videos else "unknown"
        payload: MergedPublicationPayload = build_sanitized_merged_publication_payload(
            language=language,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=use_audit_text,
            source_videos=videos,
        )
        return payload.title_text
    if merge_attempt is not None:
        if prefer_rejected_attempts:
            rejected_titles_summary: str = _build_rejected_attempt_titles_summary(
                merge_attempt=merge_attempt
            )
            if rejected_titles_summary:
                return rejected_titles_summary
        salvaged_title: str = str(merge_attempt.salvaged_title or "").strip()
        if salvaged_title:
            return salvaged_title
        return _numbered_original_titles(videos) if videos else "1) ..."
    if not videos:
        return "1) ..."
    return _numbered_original_titles(videos)


def _fallback_source_description_text(
    video: PlannedVideo,
    templates: Optional[AppTemplates],
) -> str:
    description_text: str = video.metadata.description.strip() or _no_description_text(
        templates
    )
    if strip_chapter_timestamps_enabled_from_env():
        description_text = strip_chapter_timestamps(description_text)
    return description_text


def _has_rejected_attempt_details(merge_attempt: Optional[LanguageMergeAttempt]) -> bool:
    return bool(
        merge_attempt is not None
        and str(getattr(merge_attempt, "publish_source_label", "") or "").strip()
        == "merge_failed"
        and getattr(merge_attempt, "rejected_attempts", ())
    )


def _reject_reasons_label(rejected_attempt: RejectedMergeAttempt) -> str:
    return ",".join(rejected_attempt.reject_reasons) or "unknown"


def _build_rejected_attempt_titles_summary(
    *,
    merge_attempt: LanguageMergeAttempt,
) -> str:
    if not _has_rejected_attempt_details(merge_attempt):
        return ""
    lines: List[str] = [_REJECTED_ATTEMPTS_NOTICE, ""]
    for rejected_attempt in merge_attempt.rejected_attempts:
        title_text: str = str(rejected_attempt.title or "").strip() or "(empty title)"
        lines.append(
            " | ".join(
                (
                    f"attempt={rejected_attempt.attempt_index}",
                    f"model={rejected_attempt.model_name or 'unknown'}",
                    f"reject={_reject_reasons_label(rejected_attempt)}",
                )
            )
        )
        lines.append(title_text)
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_rejected_attempt_descriptions_summary(
    *,
    merge_attempt: LanguageMergeAttempt,
) -> str:
    if not _has_rejected_attempt_details(merge_attempt):
        return ""
    lines: List[str] = [_REJECTED_ATTEMPTS_NOTICE, ""]
    for rejected_attempt in merge_attempt.rejected_attempts:
        description_text: str = (
            str(rejected_attempt.description or "").strip()
            or "(empty description)"
        )
        lines.append(
            " | ".join(
                (
                    f"attempt={rejected_attempt.attempt_index}",
                    f"model={rejected_attempt.model_name or 'unknown'}",
                    f"reject={_reject_reasons_label(rejected_attempt)}",
                )
            )
        )
        lines.append(description_text)
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_rejected_attempt_rows(
    *,
    merge_attempt: Optional[LanguageMergeAttempt],
) -> List[Tuple[str, bool]]:
    if not _has_rejected_attempt_details(merge_attempt):
        return []
    assert merge_attempt is not None
    rows: List[Tuple[str, bool]] = [(_REJECTED_ATTEMPTS_NOTICE, True)]
    for rejected_attempt in merge_attempt.rejected_attempts:
        context_label: str = (
            f"ATTEMPT {rejected_attempt.attempt_index} | "
            f"MODEL: {rejected_attempt.model_name or 'unknown'} | "
            f"REJECT: {_reject_reasons_label(rejected_attempt)}"
        )
        rows.append((f"REJECTED TITLE | {context_label}", True))
        rows.append((str(rejected_attempt.title or "").strip() or "(empty title)", False))
        rows.append((f"REJECTED DESCRIPTION | {context_label}", True))
        rows.append(
            (
                str(rejected_attempt.description or "").strip()
                or "(empty description)",
                False,
            )
        )
    return rows


def _build_descriptions_summary(
    videos: List[PlannedVideo],
    templates: Optional[AppTemplates],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    use_audit_text: bool = True,
    prefer_rejected_attempts: bool = False,
) -> str:
    source_descriptions: List[str] = []
    for video in videos:
        source_descriptions.append(_fallback_source_description_text(video, templates))
    source_lines: str = _numbered_lines(source_descriptions)
    if merged_content:
        language: str = planned_video_block_language(videos[0]) if videos else "unknown"
        payload: MergedPublicationPayload = build_sanitized_merged_publication_payload(
            language=language,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=use_audit_text,
            source_videos=videos,
        )
        return payload.description_text
    if merge_attempt is not None:
        if prefer_rejected_attempts:
            rejected_descriptions_summary: str = _build_rejected_attempt_descriptions_summary(
                merge_attempt=merge_attempt
            )
            if rejected_descriptions_summary:
                return rejected_descriptions_summary
        if should_suppress_raw_merge_attempt_publish(
            merge_attempt=merge_attempt,
            merged_content_available=False,
        ):
            log_safe_merge_attempt_fallback(
                target="doc",
                merge_attempt=merge_attempt,
                fallback_label="source_descriptions",
            )
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
    block_generation_mode: str = (
        merged_payload.block_generation_mode
        if merged_payload is not None
        else resolve_block_generation_mode(
            merge_attempt=merge_attempt,
            merged_content=merged_content,
        )
    )
    if block_generation_mode == BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE:
        marker_text: str = (
            FALLBACK_ONLY_ARTIFACT_MARKER
            if artifact_status == "fallback_only"
            else PARTIAL_FALLBACK_ARTIFACT_MARKER
        )
        return f"{base_heading} ({marker_text})"
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


def _no_description_text(templates: Optional[AppTemplates] = None) -> str:
    if templates is None:
        return "no description"
    value: str = str(templates.common_no_description_text or "").strip()
    return value or "no description"


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
) -> List[Tuple[str, bool]]:
    if templates is None:
        raise RuntimeError("Templates are required for language table labels.")
    try:
        labels_payload: Any = json.loads(templates.google_doc_table_labels_json)
    except Exception as error:
        raise RuntimeError(
            f"Invalid template google_doc.table_labels: {error}"
        ) from error
    labels: Dict[str, Tuple[str, str, str]] = {}
    if isinstance(labels_payload, dict):
        for key, value in labels_payload.items():
            if isinstance(value, list) and len(value) == 3:
                labels[str(key)] = (str(value[0]), str(value[1]), str(value[2]))
    title_label, desc_label, preview_label = labels.get(
        language,
        labels.get("other", ("TITLE", "DESCRIPTION", "PREVIEW")),
    )

    heading: str = _language_heading(language, templates)
    if time_display:
        heading = f"{heading} - {time_display}"
    merged_payload: Optional[MergedPublicationPayload] = None
    if merged_content is not None:
        merged_payload = build_sanitized_merged_publication_payload(
            language=language,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=True,
            source_videos=videos,
        )
    heading = _artifact_heading_for_block(
        base_heading=heading,
        merged_payload=merged_payload,
        merge_attempt=merge_attempt,
        merged_content=merged_content,
        artifact_status=artifact_status,
    )
    description_text: str = (
        merged_payload.description_text
        if merged_payload is not None
        else _build_descriptions_summary(
            videos=videos,
            templates=templates,
            merged_content=merged_content,
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
                    merged_content=merged_content,
                    merge_attempt=merge_attempt,
                )
            ),
            False,
        ),
        (desc_label, True),
        (description_text, False),
        (preview_label, True),
    ]
    rows.extend(_build_preview_placeholder_rows(videos))
    rows.extend(_build_rejected_attempt_rows(merge_attempt=merge_attempt))
    return rows
