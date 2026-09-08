from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig, AppTemplates
from app.core.language_display import language_display_name, language_to_flag_emoji
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
    SanitizedPublishBlock,
    VideoMetadata,
)
from app.observability.runtime_analytics import record_publish_gate_blocked
from app.planning import planned_video_block_language
from app.publish.post_llm_sanitation import (
    MergedPublicationPayload,
    build_sanitized_merged_publication_payload,
    log_safe_merge_attempt_fallback,
    should_suppress_raw_merge_attempt_publish,
)
from app.publish.shared_helpers import (
    fallback_source_description_text as _fallback_source_description_text,
    is_merge_payload_blocked as _is_merge_payload_blocked,
    no_description_text as _publish_no_description_text,
    numbered_lines as _numbered_lines,
    numbered_original_titles as _numbered_original_titles,
)

LOGGER = _get_logger_impl(__name__)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram parse_mode=HTML."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_BOLD_MARKER_PATTERN: re.Pattern[str] = re.compile(r"\*([^*]+)\*")


def telegram_safe_html(text: str) -> str:
    """Escape HTML entities and convert *text* to <b>text</b> for Telegram HTML mode."""
    escaped: str = _escape_html(text)
    return _BOLD_MARKER_PATTERN.sub(r"<b>\1</b>", escaped)


def _render_template(template: str, values: Dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as error:
        raise RuntimeError(f"Template render failed, missing key: {error}") from error


def _telegram_safe_time(value: str) -> str:
    return value.replace(":", ":\u2060")


def _build_merged_publication_payload(
    *,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    merge_attempt: Optional[LanguageMergeAttempt],
    use_audit_text: bool,
) -> Optional[MergedPublicationPayload]:
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
    if _is_merge_payload_blocked(sanitized_payload):
        LOGGER.warning(
            "merge_publish_gate_blocked target=telegram language=%s has_publish_stage_duplicate=%s has_publish_stage_opener_cta=%s fallback=nomerge",
            language,
            "yes"
            if bool(getattr(sanitized_payload, "has_publish_stage_duplicate", False))
            else "no",
            "yes"
            if bool(getattr(sanitized_payload, "has_publish_stage_opener_cta", False))
            else "no",
        )
        record_publish_gate_blocked(language=language, target="telegram")
        return None
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
        payload: Optional[MergedPublicationPayload] = _build_merged_publication_payload(
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
        payload: Optional[MergedPublicationPayload] = _build_merged_publication_payload(
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=use_audit_text,
        )
        if payload is not None:
            return payload.description_text
        return source_lines
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


def _telegram_language_flag(language: str) -> str:
    return language_to_flag_emoji(language)


def _telegram_language_flags(language: str, config: AppConfig) -> str:
    return _telegram_language_flag(language) * max(
        1, int(config.telegram.flag_repeat_count)
    )


def _telegram_language_name(language: str) -> str:
    return language_display_name(language)


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
            "symbol_broadcast": config.telegram.symbol_broadcast,
            "symbol_alert": config.telegram.symbol_alert,
            "date": context["date"],
            "symbol_form": config.telegram.symbol_form,
            "form_url": _escape_html(context["form_url"]),
            "contacts": _escape_html(context["contacts"]),
            "symbol_description": config.telegram.symbol_description,
            "generated_doc_url": _escape_html(generated_doc_url),
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
            "symbol_pin": config.telegram.symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": telegram_safe_html(video.metadata.title),
            "description": telegram_safe_html(description_text),
        },
    )


def build_telegram_language_merged_block(
    language: str,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    merge_attempt: Optional[LanguageMergeAttempt],
    config: AppConfig,
    templates: Optional[AppTemplates],
    sanitized_block: Optional[SanitizedPublishBlock] = None,
) -> str:
    if not videos:
        raise ValueError("videos must not be empty for merged telegram block")
    times_text: str = ", ".join(
        sorted({video.scheduled_at_kiev.strftime("%H:%M") for video in videos})
    )
    merged_payload: Optional[MergedPublicationPayload] = None
    if sanitized_block is not None and not sanitized_block.is_blocked:
        merged_payload = MergedPublicationPayload(
            title_text=sanitized_block.title_text,
            description_text=sanitized_block.description_text,
            block_generation_mode=sanitized_block.block_generation_mode,
        )
    else:
        merged_payload = _build_merged_publication_payload(
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            use_audit_text=config.telegram.use_audit,
        )
    if sanitized_block is not None and not sanitized_block.is_blocked:
        LOGGER.info(
            "publish_sanitation_cache_hit lang=%s target=telegram",
            language,
        )
    else:
        LOGGER.info(
            "publish_sanitation_cache_miss lang=%s target=telegram reason=fallback_to_live_sanitation",
            language,
        )
    if merged_payload is None:
        return build_telegram_language_nomerge_block(
            language=language,
            videos=videos,
            config=config,
            templates=templates,
        )
    return _render_template(
        config.templates.telegram_language_merged_block,
        {
            "date_display": videos[0].date_display,
            "time_kiev": _telegram_safe_time(times_text),
            "symbol_pin": config.telegram.symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": telegram_safe_html(merged_payload.title_text.strip()),
            "description": telegram_safe_html(
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
            "symbol_pin": config.telegram.symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": telegram_safe_html(titles_text or "1) ..."),
            "description": telegram_safe_html(
                descriptions_text or _publish_no_description_text(templates)
            ),
        },
    )


def build_telegram_key_form_reminder(
    context: Dict[str, str],
    config: AppConfig,
) -> str:
    return _render_template(
        config.templates.telegram_key_form_reminder,
        {"form_url": _escape_html(context["form_url"])},
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
            "title": telegram_safe_html(metadata.title),
            "language": telegram_safe_html(language),
            "description": telegram_safe_html(
                metadata.description or _publish_no_description_text(templates)
            ),
            "url": _escape_html(metadata.url),
        },
    )
