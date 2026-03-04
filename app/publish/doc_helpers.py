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
    LanguageMergeAttempt,
    MergedLanguageContent,
    MergedPublicationPayload,
    PlannedVideo,
)
from app.planning import planned_video_block_language
from app.publish.post_llm_sanitation import (
    build_sanitized_merged_publication_payload,
    resolve_post_llm_source_label,
    sanitize_post_llm_text_for_merged_publish,
)


LOGGER = _get_logger_impl(__name__)


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
        raw_text: str = sanitize_post_llm_text_for_merged_publish(
            text=str(merge_attempt.raw_response_text or "").strip(),
            language=merge_attempt.language,
            source_label=resolve_post_llm_source_label(
                merge_attempt,
                default_label="merge_attempt_raw",
            ),
            source_videos=videos,
        )
        if raw_text:
            return f"{raw_text}\n\n{source_lines}".strip()
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
    return rows
