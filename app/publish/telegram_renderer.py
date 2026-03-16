from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig, AppTemplates
from app.core.env_flags import (
    strip_chapter_timestamps,
    strip_chapter_timestamps_enabled_from_env,
)
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    MergedPublicationPayload,
    PlannedVideo,
    VideoMetadata,
)
from app.planning import planned_video_block_language
from app.publish.doc_helpers import _no_description_text as _publish_no_description_text
from app.publish.post_llm_sanitation import (
    build_sanitized_merged_publication_payload,
    log_safe_merge_attempt_fallback,
    should_suppress_raw_merge_attempt_publish,
)

LOGGER = _get_logger_impl(__name__)


def _render_template(template: str, values: Dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as error:
        raise RuntimeError(f"Template render failed, missing key: {error}") from error


def _telegram_safe_time(value: str) -> str:
    return value.replace(":", ":\u2060")


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
    if strip_chapter_timestamps_enabled_from_env():
        description_text = strip_chapter_timestamps(description_text)
    return description_text


def _numbered_original_titles(videos: List[PlannedVideo]) -> str:
    source_titles: List[str] = [video.metadata.title for video in videos]
    return _numbered_lines(source_titles)


def _build_merged_publication_payload(
    *,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    merge_attempt: Optional[LanguageMergeAttempt],
    use_audit_text: bool,
) -> MergedPublicationPayload:
    language: str = planned_video_block_language(videos[0]) if videos else "unknown"
    sanitized_payload: MergedPublicationPayload = (
        build_sanitized_merged_publication_payload(
            language=language,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=use_audit_text,
            source_videos=videos,
        )
    )
    return MergedPublicationPayload(
        title_text=sanitized_payload.title_text.strip(),
        description_text=sanitized_payload.description_text.strip(),
        block_generation_mode=sanitized_payload.block_generation_mode,
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
            merge_attempt=merge_attempt,
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
            merge_attempt=merge_attempt,
            use_audit_text=use_audit_text,
        )
        return payload.description_text
    if merge_attempt is not None:
        if should_suppress_raw_merge_attempt_publish(
            merge_attempt=merge_attempt,
            merged_content_available=False,
        ):
            log_safe_merge_attempt_fallback(
                target="telegram",
                merge_attempt=merge_attempt,
                fallback_label="source_descriptions",
            )
            LOGGER.debug(
                "telegram_safe_fallback_selected language=%s fallback=source_descriptions",
                merge_attempt.language,
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
    merged_payload: MergedPublicationPayload = _build_merged_publication_payload(
        videos=videos,
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
            "title": merged_payload.title_text.strip(),
            "description": (
                merged_payload.description_text.strip()
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
