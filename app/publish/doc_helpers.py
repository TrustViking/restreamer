from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppTemplates
from app.core.cta_detection import starts_with_cta_prefix
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
from app.core.text_utils import split_paragraphs
from app.core.url_normalizer import normalize_display_url
from app.core.constants import URL_PATTERN
from app.llm.merges.merge_text_utils import _looks_like_service_tail_paragraph
from app.planning import planned_video_block_language
from app.publish.post_llm_sanitation import (
    build_sanitized_merged_publication_payload,
    log_safe_merge_attempt_fallback,
    should_suppress_raw_merge_attempt_publish,
)
from app.resources import promotional_opener_phrases


LOGGER = _get_logger_impl(__name__)

def _language_heading(language: str, templates: Optional[AppTemplates]) -> str:
    if templates is None:
        return "OTHER"
    payload: object = getattr(templates, "google_doc_language_headings", {})
    if isinstance(payload, dict):
        return str(payload.get(language, payload.get("other", "OTHER")))
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


def _is_merge_payload_blocked(payload: MergedPublicationPayload) -> bool:
    has_publish_stage_duplicate: bool = bool(
        getattr(payload, "has_publish_stage_duplicate", False)
    )
    has_publish_stage_opener_cta: bool = bool(
        getattr(payload, "has_publish_stage_opener_cta", False)
    )
    return has_publish_stage_duplicate or has_publish_stage_opener_cta


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


def _fallback_source_description_text(
    video: PlannedVideo,
    templates: Optional[AppTemplates],
) -> str:
    description_text: str = video.metadata.description.strip() or _no_description_text(
        templates
    )
    if strip_chapter_timestamps_enabled_from_env():
        description_text = strip_chapter_timestamps(description_text)
    description_text = _light_polish_single_source_description(description_text)
    return description_text


def _looks_like_promotional_opener(text: str) -> bool:
    normalized: str = re.sub(r"\s+", " ", str(text or "").strip()).casefold()
    if not normalized:
        return False
    return any(phrase in normalized for phrase in promotional_opener_phrases())


def _extract_description_paragraphs_raw(text: str) -> List[str]:
    paragraphs: List[str] = split_paragraphs(text)
    return paragraphs


def _normalize_urls_in_text(text: str) -> str:
    def _replace_url(match: re.Match[str]) -> str:
        matched_url: str = str(match.group(0) or "")
        return normalize_display_url(matched_url)

    normalized_text: str = URL_PATTERN.sub(_replace_url, text)
    return normalized_text


def _light_polish_single_source_description(text: str) -> str:
    original_text: str = str(text or "")
    paragraphs: List[str] = _extract_description_paragraphs_raw(original_text)
    if not paragraphs:
        return original_text

    cleaned_paragraphs: List[str] = list(paragraphs)
    if cleaned_paragraphs and (
        starts_with_cta_prefix(cleaned_paragraphs[0])
        or _looks_like_promotional_opener(cleaned_paragraphs[0])
    ):
        cleaned_paragraphs = cleaned_paragraphs[1:]
    while cleaned_paragraphs and _looks_like_service_tail_paragraph(cleaned_paragraphs[-1]):
        cleaned_paragraphs = cleaned_paragraphs[:-1]

    polished_text: str = "\n\n".join(cleaned_paragraphs).strip()
    if not polished_text:
        return original_text
    polished_text = _normalize_urls_in_text(polished_text)
    return polished_text


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
    labels_payload: object = getattr(templates, "google_doc_table_labels", {})
    if not isinstance(labels_payload, dict):
        raise RuntimeError("Invalid template google_doc.table_labels: expected mapping")
    labels: Dict[str, Tuple[str, str, str]] = {}
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
        merged_payload = _build_guarded_merged_payload(
            target="doc",
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=True,
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
            False,
        ),
        (desc_label, True),
        (description_text, False),
        (preview_label, True),
    ]
    rows.extend(_build_preview_placeholder_rows(videos))
    return rows
