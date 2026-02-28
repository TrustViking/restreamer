from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

from app.config.settings import AppConfig, AppTemplates
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    MergedPublicationPayload,
    PlannedVideo,
    VideoMetadata,
)
from app.planning import planned_video_block_language
from app.publish.doc_helpers import _no_description_text as _publish_no_description_text


def _render_template(template: str, values: Dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as error:
        raise RuntimeError(f"Template render failed, missing key: {error}") from error


def _telegram_safe_time(value: str) -> str:
    return value.replace(":", ":\u2060")


def _load_bool_env(name: str, default: bool) -> bool:
    raw_value: str = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default


def _strip_chapter_timestamps_enabled_from_env() -> bool:
    return _load_bool_env("STG_STRIP_CHAPTER_TIMESTAMPS", True)


def strip_chapter_timestamps(text: str) -> str:
    raw_text: str = str(text or "")
    if not raw_text:
        return raw_text
    chapter_pattern: re.Pattern[str] = re.compile(
        r"^\s*(?:\d{1,2}\s*:\s*)?\d{1,2}\s*:\s*\d{2}\s+\S.*$"
    )
    cleaned_lines: List[str] = []
    for line in raw_text.splitlines(keepends=True):
        stripped_line: str = line.strip()
        if stripped_line and chapter_pattern.match(stripped_line):
            continue
        cleaned_lines.append(line)
    return "".join(cleaned_lines)


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


def _fallback_source_description_text(
    video: PlannedVideo,
    templates: Optional[AppTemplates],
) -> str:
    description_text: str = video.metadata.description.strip() or _publish_no_description_text(
        templates
    )
    if _strip_chapter_timestamps_enabled_from_env():
        description_text = strip_chapter_timestamps(description_text)
    return description_text


def _merged_title_for_docs(merged_content: Optional[MergedLanguageContent]) -> str:
    if not merged_content:
        return ""
    return str(merged_content.title_audit or merged_content.title or "").strip()


def _merged_title_for_selected(merged_content: Optional[MergedLanguageContent]) -> str:
    if not merged_content:
        return ""
    return str(merged_content.title_selected or merged_content.title or "").strip()


def _merged_description_for_docs(
    merged_content: Optional[MergedLanguageContent],
) -> str:
    if not merged_content:
        return ""
    return str(
        merged_content.description_audit or merged_content.description or ""
    ).strip()


def _merged_description_for_selected(
    merged_content: Optional[MergedLanguageContent],
) -> str:
    if not merged_content:
        return ""
    return str(
        merged_content.description_selected or merged_content.description or ""
    ).strip()


def _numbered_original_titles(videos: List[PlannedVideo]) -> str:
    source_titles: List[str] = [video.metadata.title for video in videos]
    return _numbered_lines(source_titles)


def _build_merged_publication_payload(
    *,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    use_audit_text: bool,
) -> MergedPublicationPayload:
    merged_title_text: str = (
        _merged_title_for_docs(merged_content)
        if use_audit_text
        else _merged_title_for_selected(merged_content)
    )
    merged_description_text: str = (
        _merged_description_for_docs(merged_content)
        if use_audit_text
        else _merged_description_for_selected(merged_content)
    )
    return MergedPublicationPayload(
        title_text=merged_title_text.strip(),
        description_text=merged_description_text.strip(),
    )


def build_titles_summary(
    videos: List[PlannedVideo],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    use_audit_text: bool = True,
) -> str:
    if merged_content:
        payload: MergedPublicationPayload = _build_merged_publication_payload(
            videos=videos,
            merged_content=merged_content,
            use_audit_text=use_audit_text,
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


def build_descriptions_summary(
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
        payload: MergedPublicationPayload = _build_merged_publication_payload(
            videos=videos,
            merged_content=merged_content,
            use_audit_text=use_audit_text,
        )
        return payload.description_text
    if merge_attempt is not None:
        raw_text: str = str(merge_attempt.raw_response_text or "").strip()
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


def _telegram_language_flag(language: str, config: AppConfig) -> str:
    flag_by_language: Dict[str, str] = {
        "uk": config.telegram_flag_uk,
        "en": config.telegram_flag_en,
        "ru": config.telegram_flag_ru,
        "other": config.telegram_flag_other,
    }
    return flag_by_language.get(language, config.telegram_flag_other)


def _telegram_language_flags(language: str, config: AppConfig) -> str:
    return _telegram_language_flag(language, config) * max(
        1, int(config.telegram_flag_repeat_count)
    )


def _telegram_language_name(language: str, config: AppConfig) -> str:
    name_by_language: Dict[str, str] = {
        "uk": config.telegram_language_name_uk,
        "en": config.telegram_language_name_en,
        "ru": config.telegram_language_name_ru,
        "other": config.telegram_language_name_other,
    }
    return name_by_language.get(language, config.telegram_language_name_other)


def build_telegram_header_text(
    context: Dict[str, str],
    generated_doc_url: str,
    config: AppConfig,
) -> str:
    return _render_template(
        config.templates.telegram_header,
        {
            "time_cet": _telegram_safe_time(context["time_cet"]),
            "time_kiev": _telegram_safe_time(context["time_kiev"]),
            "time_gmt": _telegram_safe_time(context["time_gmt"]),
            "symbol_broadcast": config.telegram_symbol_broadcast,
            "symbol_alert": config.telegram_symbol_alert,
            "date": context["date"],
            "symbol_form": config.telegram_symbol_form,
            "form_url": context["form_url"],
            "contacts": context["contacts"],
            "symbol_description": config.telegram_symbol_description,
            "generated_doc_url": generated_doc_url,
        },
    )


def build_telegram_language_block(
    video: PlannedVideo,
    config: AppConfig,
    templates: Optional[AppTemplates],
) -> str:
    language: str = planned_video_block_language(video)
    description_text: str = _fallback_source_description_text(video, templates)
    return _render_template(
        config.templates.telegram_language_block,
        {
            "date_display": video.date_display,
            "time_kiev": _telegram_safe_time(
                video.scheduled_at_kiev.strftime("%H:%M")
            ),
            "symbol_pin": config.telegram_symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": video.metadata.title,
            "description": description_text,
        },
    )


def build_telegram_language_merged_block(
    language: str,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    merge_attempt: Optional[LanguageMergeAttempt],
    config: AppConfig,
    templates: Optional[AppTemplates],
) -> str:
    if not videos:
        raise ValueError("videos must not be empty for merged telegram block")
    times_text: str = ", ".join(
        sorted({video.scheduled_at_kiev.strftime("%H:%M") for video in videos})
    )
    merged_title_text: str = build_titles_summary(
        videos,
        merged_content,
        merge_attempt,
        use_audit_text=config.telegram_use_audit,
    )
    merged_description_text: str = build_descriptions_summary(
        videos=videos,
        templates=templates,
        merged_content=merged_content,
        merge_attempt=merge_attempt,
        use_audit_text=config.telegram_use_audit,
    )
    return _render_template(
        config.templates.telegram_language_merged_block,
        {
            "date_display": videos[0].date_display,
            "time_kiev": _telegram_safe_time(times_text),
            "symbol_pin": config.telegram_symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": merged_title_text.strip(),
            "description": (
                merged_description_text.strip()
                or _publish_no_description_text(templates)
            ),
        },
    )


def build_telegram_language_nomerge_block(
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    templates: Optional[AppTemplates],
) -> str:
    if not videos:
        raise ValueError("videos must not be empty for nomerge telegram block")
    times_text: str = ", ".join(
        sorted({video.scheduled_at_kiev.strftime("%H:%M") for video in videos})
    )
    titles_text: str = _numbered_original_titles(videos).strip()
    descriptions_text: str = build_descriptions_summary(
        videos=videos,
        templates=templates,
    ).strip()
    return _render_template(
        config.templates.telegram_language_merged_block,
        {
            "date_display": videos[0].date_display,
            "time_kiev": _telegram_safe_time(times_text),
            "symbol_pin": config.telegram_symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": titles_text or "1) ...",
            "description": descriptions_text or _publish_no_description_text(templates),
        },
    )


def build_telegram_language_digest_block(
    language: str,
    videos: List[PlannedVideo],
    context: Dict[str, str],
    config: AppConfig,
) -> str:
    times_text: str = ", ".join(
        sorted({video.scheduled_at_kiev.strftime("%H:%M") for video in videos})
    )
    digest_header: str = _render_template(
        config.templates.telegram_language_digest_header,
        {
            "language_flags": _telegram_language_flags(language, config),
            "language_name": _telegram_language_name(language, config),
            "date": context["date"],
            "time_kiev": _telegram_safe_time(times_text),
        },
    )
    lines: List[str] = [digest_header, ""]
    for video in videos:
        lines.append(f"{config.telegram_symbol_done} {video.metadata.title}")
        lines.append(video.normalized_link)
        lines.append("")
    return "\n".join(lines).rstrip()


def build_telegram_key_form_reminder(
    context: Dict[str, str],
    config: AppConfig,
) -> str:
    return _render_template(
        config.templates.telegram_key_form_reminder,
        {"form_url": context["form_url"]},
    )


def build_telegram_post_header_text(
    header_context: Dict[str, str],
    config: AppConfig,
) -> str:
    return _render_template(
        config.templates.telegram_post_header,
        {
            "symbol_broadcast": config.telegram_symbol_broadcast,
            "date": header_context["date"],
            "time_kiev": _telegram_safe_time(header_context["time_kiev"]),
        },
    )


def build_single_mode_message(
    metadata: VideoMetadata,
    language: str,
    config: AppConfig,
    templates: Optional[AppTemplates],
) -> str:
    return _render_template(
        config.templates.common_single_mode_message,
        {
            "title": metadata.title,
            "language": language,
            "description": (
                metadata.description or _publish_no_description_text(templates)
            ),
            "url": metadata.url,
        },
    )
