from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.constants import URL_PATTERN
from app.core.models import LanguageMergeAttempt, MergedLanguageContent, PlannedVideo
from app.llm.merge_parser import (
    build_plain_merged_content_or_raise,
    clean_and_validate_llm_description,
    enforce_merged_paragraphs_for_group,
    extract_title_and_description_payload_or_none,
    extract_valid_title_from_raw_or_none,
    parse_llm_merge_raw_or_raise,
    sanitize_title,
)
from app.llm.merge_run_summary import MergeRunSummary
from app.llm.openai_client import OpenAITransportResult, openai_request_merge

LOGGER = _get_logger_impl(__name__)


@dataclass(frozen=True)
class _MergeCallResult:
    raw_text: str
    structured_attempted: bool
    structured_used: bool
    structured_failure_reason: Optional[str] = None


def _language_name_for_merge_prompt(language: str, llm_language_names_json: str) -> str:
    default_names: Dict[str, str] = {
        "uk": "Ukrainian",
        "en": "English",
        "ru": "Russian",
        "other": "the original language of sources",
    }
    try:
        import json

        payload: Any = json.loads(llm_language_names_json)
        if isinstance(payload, dict):
            return str(payload.get(language, payload.get("other", default_names["other"])))
    except Exception:
        pass
    return default_names.get(language, default_names["other"])


def _strip_urls(text: str) -> str:
    without_urls: str = URL_PATTERN.sub("", str(text or ""))
    normalized: str = without_urls.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"\n\s+", "\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _truncate_head_tail(text: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 2:
        return "…"
    body_limit: int = limit - 1
    head_len: int = max(1, int(body_limit * 0.7))
    tail_len: int = max(1, body_limit - head_len)
    head_part: str = text[:head_len].rstrip()
    tail_part: str = text[-tail_len:].lstrip()
    if not head_part or not tail_part:
        return text[: limit - 1].rstrip() + "…"
    return f"{head_part}…{tail_part}"


def build_llm_merge_prompt_text(*, language: str, videos: List[PlannedVideo], config: AppConfig, no_description_text: str) -> str:
    if len(videos) < 2:
        raise ValueError("Expected at least 2 videos for merged generation.")
    unique_dates: List[str] = sorted({video.date_display for video in videos if video.date_display})
    date_or_period: str = ", ".join(unique_dates) if unique_dates else "n/a"
    language_name: str = _language_name_for_merge_prompt(language, config.templates.llm_language_names_json)
    sources_blocks: List[str] = []
    for index, video in enumerate(videos, start=1):
        title_text: str = video.metadata.title.strip()
        source_description: str = video.metadata.description.strip() or no_description_text
        description_text_raw: str = _strip_urls(source_description)
        description_text: str = _truncate_head_tail(description_text_raw, limit=config.llm_source_desc_max_chars)
        source_url: str = video.normalized_link.strip()
        sources_blocks.append(f"VIDEO {index}:\nURL: {source_url}\nTITLE: {title_text}\nDESCRIPTION: {description_text}")

    base_prompt: str = config.templates.llm_merge_title_description_prompt.format(
        language_name=language_name,
        sources_block="\n\n".join(sources_blocks),
    )
    strict_prompt: str = (
        "You are a careful editor. You MUST use ONLY facts explicitly present in source content.\n"
        "Do NOT invent names, dates, places, numbers, quotes, or events.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "Line 1: TITLE: <title>\n"
        "Line 2+: DESCRIPTION: <description text; may contain newlines>\n"
        "No other headers. No JSON. No markdown. No code fences.\n\n"
        f"LANGUAGE:\nWrite in: {language_name}.\n\n"
        f"Date/period (optional): {date_or_period}\n"
    )
    return f"{base_prompt}\n\n{strict_prompt}".strip()


def _build_llm_merge_repair_prompt_text(*, language: str, videos: List[PlannedVideo], config: AppConfig, previous_output: str, parse_error: str, no_description_text: str) -> str:
    base_prompt: str = build_llm_merge_prompt_text(language=language, videos=videos, config=config, no_description_text=no_description_text)
    return (
        f"{base_prompt}\n\n"
        "Your previous output was invalid. Rewrite the full answer from scratch.\n"
        "Do not explain errors. Return only TITLE and DESCRIPTION blocks.\n"
        "Keep one DESCRIPTION paragraph per source, in source order.\n\n"
        f"Validation error: {parse_error}\n\n"
        "Previous invalid output:\n"
        f"{previous_output.strip()}"
    ).strip()


def _build_plain_description_repair_prompt_text(*, target_language: str, invalid_text: str) -> str:
    return (
        "Remove any labels/headings/meta lines. "
        f"Return ONLY plain paragraph text in {target_language}.\n"
        "No TITLE/DESCRIPTION/CTA/HASHTAGS/PREVIEW labels.\n"
        "No markdown. No JSON. No commentary.\n\n"
        "Text:\n"
        f"{str(invalid_text or '').strip()}"
    ).strip()


def _build_single_source_translate_prompt_text(*, source_language: str, target_language: str, source_description: str) -> str:
    return (
        "You are a precise editor and translator.\n"
        f"Source language: {source_language}.\n"
        f"Target language: {target_language}.\n"
        "Task: translate and lightly rewrite for readability while preserving facts.\n"
        "Return ONLY plain paragraph text in target language.\n"
        "No labels/headings/meta lines.\n"
        "No TITLE/DESCRIPTION/CTA/HASHTAGS/PREVIEW.\n"
        "No markdown. No JSON.\n\n"
        "Source description:\n"
        f"{str(source_description or '').strip()}"
    ).strip()


def _build_hashtags_repair_prompt_text(*, invalid_hashtags_line: str) -> str:
    return (
        "You are a strict hashtag formatter.\n"
        "Return exactly one line with only space-separated hashtags.\n"
        "Rules:\n"
        "1) Keep existing hashtag words if possible.\n"
        "2) Remove all non-hashtag tokens.\n"
        "3) Do not output labels like 'HASHTAGS:'.\n"
        "4) No extra commentary.\n\n"
        "Invalid hashtags line:\n"
        f"{invalid_hashtags_line.strip()}"
    ).strip()


def _openai_merge_call_raw(*, language: str, videos: List[PlannedVideo], config: AppConfig, attempt_label: str, model_name: str, prompt_text_override: Optional[str] = None, no_description_text: str = "no description", merge_run_summary: Optional[MergeRunSummary] = None) -> _MergeCallResult:
    prompt_text: str = str(prompt_text_override or "").strip() or build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
        no_description_text=no_description_text,
    )
    pre_delay_sec: float = max(0.0, float(config.openai_pre_delay_sec))
    if pre_delay_sec > 0.0:
        LOGGER.info("LLM merge: sleeping %.2fs before request ... provider=openai language=%s sources=%d", pre_delay_sec, language, len(videos))
        time.sleep(pre_delay_sec)

    structured_schema: Optional[Dict[str, Any]] = None
    if prompt_text_override is None:
        expected_source_count: int = len(videos)
        structured_schema = {
            "name": f"streamertg_merge_{language}_v1",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "paragraphs", "cta", "hashtags", "sources"],
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 98},
                    "paragraphs": {
                        "type": "array",
                        "minItems": expected_source_count,
                        "maxItems": expected_source_count,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "cta": {"type": "string", "minLength": 1},
                    "hashtags": {"type": "string", "minLength": 1},
                    "sources": {
                        "type": "array",
                        "minItems": expected_source_count,
                        "maxItems": expected_source_count,
                        "items": {"type": "string", "minLength": 1, "pattern": r"^https?://\\S+$"},
                    },
                },
            },
        }

    try:
        result: OpenAITransportResult = openai_request_merge(
            prompt_text=prompt_text,
            model_name=model_name,
            timeout_sec=config.openai_timeout_sec,
            attempt_label=attempt_label,
            max_output_tokens=config.openai_max_output_tokens,
            structured_schema=structured_schema,
            temperature=0.0,
        )
    except Exception:
        if structured_schema is not None:
            LOGGER.info(
                "merge_path structured_failed lang=%s source_count=%d reason=api_error",
                language,
                len(videos),
            )
            if merge_run_summary is not None:
                merge_run_summary.record_structured_failed()
        raise

    if structured_schema is not None and result.structured_payload is not None:
        payload: Dict[str, Any] = result.structured_payload
        title_raw: str = str(payload.get("title") or "").strip()
        title_value: str = sanitize_title(title_raw, min_chars=1, max_chars=98, allow_emoji=False)
        paragraphs: List[str] = [str(item or "").strip() for item in cast(List[Any], payload.get("paragraphs") or []) if str(item or "").strip()]
        cta_line: str = str(payload.get("cta") or "").strip()
        hashtags_line: str = str(payload.get("hashtags") or "").strip()
        source_urls: List[str] = [str(item or "").strip() for item in cast(List[Any], payload.get("sources") or []) if str(item or "").strip()]
        if title_value and paragraphs and cta_line and hashtags_line and source_urls:
            description_text: str = "\n\n".join(paragraphs) + "\n" + cta_line + "\n" + hashtags_line + "\n" + "\n".join(source_urls)
            return _MergeCallResult(
                raw_text=f"TITLE: {title_value}\nDESCRIPTION:\n{description_text}".strip(),
                structured_attempted=True,
                structured_used=True,
            )

    if structured_schema is not None:
        LOGGER.info(
            "merge_path structured_failed lang=%s source_count=%d reason=unusable_structured_output",
            language,
            len(videos),
        )
        if merge_run_summary is not None:
            merge_run_summary.record_structured_failed()
    return _MergeCallResult(
        raw_text=result.raw_text,
        structured_attempted=structured_schema is not None,
        structured_used=False,
        structured_failure_reason=(
            "unusable_structured_output" if structured_schema is not None else None
        ),
    )


def _attempt_openai_plain_description_repair_once(*, language: str, videos: List[PlannedVideo], config: AppConfig, attempt_label: str, model_name: str, invalid_text: str, no_description_text: str, merge_run_summary: Optional[MergeRunSummary] = None) -> str:
    prompt_text: str = _build_plain_description_repair_prompt_text(target_language=language, invalid_text=invalid_text)
    repaired_result: _MergeCallResult = _openai_merge_call_raw(
        language=language,
        videos=videos,
        config=config,
        attempt_label=attempt_label,
        model_name=model_name,
        prompt_text_override=prompt_text,
        no_description_text=no_description_text,
        merge_run_summary=merge_run_summary,
    )
    cleaned_text, is_valid, reasons = clean_and_validate_llm_description(
        text=repaired_result.raw_text
    )
    if not is_valid:
        raise RuntimeError("repair output validation failed: " + ("; ".join(reasons) or "unknown"))
    if not cleaned_text:
        raise RuntimeError("repair output is empty")
    return cleaned_text


def attempt_openai_single_source_translate_with_audit(*, language: str, videos: List[PlannedVideo], config: AppConfig, attempt_label: str, summarize_error: Callable[[Exception], str], no_description_text: str, merge_run_summary: Optional[MergeRunSummary] = None) -> LanguageMergeAttempt:
    if len(videos) != 1:
        raise RuntimeError("single-source translate expects exactly one video.")
    source_video: PlannedVideo = videos[0]
    source_language: str = source_video.language
    model_sequence: List[str] = [str(config.openai_model_primary or "").strip() or "gpt-5-nano", "gpt-5-mini"]
    if model_sequence[1] == model_sequence[0]:
        model_sequence = [model_sequence[0]]
    source_description: str = source_video.metadata.description.strip() or no_description_text
    base_title: str = source_video.metadata.title.strip() or "Untitled"
    last_raw_response: str = ""
    last_error_summary: Optional[str] = None
    plain_repair_used: bool = False
    for attempt_index, model_name in enumerate(model_sequence, start=1):
        call_label: str = f"{attempt_label}_TRY{attempt_index}"
        prompt_text: str = _build_single_source_translate_prompt_text(source_language=source_language, target_language=language, source_description=source_description)
        try:
            raw_result: _MergeCallResult = _openai_merge_call_raw(language=language, videos=videos, config=config, attempt_label=call_label, model_name=model_name, prompt_text_override=prompt_text, no_description_text=no_description_text, merge_run_summary=merge_run_summary)
            last_raw_response = raw_result.raw_text
            cleaned_description, is_valid, _ = clean_and_validate_llm_description(text=last_raw_response)
            if not is_valid:
                plain_repair_used = True
                cleaned_description = _attempt_openai_plain_description_repair_once(
                    language=language,
                    videos=videos,
                    config=config,
                    attempt_label=f"{call_label}_REPAIR",
                    model_name=model_name,
                    invalid_text=last_raw_response,
                    no_description_text=no_description_text,
                    merge_run_summary=merge_run_summary,
                )
                LOGGER.info(
                    "merge_path repair_used lang=%s repair=plain_description source_count=%d",
                    language,
                    len(videos),
                )
                if merge_run_summary is not None:
                    merge_run_summary.record_repair_used()
            merged_content: MergedLanguageContent = build_plain_merged_content_or_raise(model_name=model_name, title_text=base_title, description_text=cleaned_description)
            return LanguageMergeAttempt(language=language, model_name=model_name, raw_response_text=last_raw_response, merged=merged_content, error_summary=None, salvaged_title=merged_content.title, plain_repair_used=plain_repair_used)
        except Exception as error:
            last_error_summary = summarize_error(error)
            continue
    return LanguageMergeAttempt(language=language, model_name=model_sequence[-1], raw_response_text=last_raw_response, merged=None, error_summary=last_error_summary or "unknown error", salvaged_title=base_title, plain_repair_used=plain_repair_used or None)


def attempt_openai_merge_with_audit(*, language: str, videos: List[PlannedVideo], config: AppConfig, attempt_label: str, summarize_error: Callable[[Exception], str], normalize_youtube_url: Callable[[str], str], no_description_text: str, merge_run_summary: Optional[MergeRunSummary] = None) -> LanguageMergeAttempt:
    model_sequence: List[str] = [str(config.openai_model_primary or "").strip() or "gpt-5-nano", str(config.openai_model_fallback or "").strip() or "gpt-5-mini"]
    if model_sequence[1] == model_sequence[0]:
        model_sequence = [model_sequence[0]]
    last_raw_response: str = ""
    last_error_summary: Optional[str] = None
    source_urls: List[str] = [item.normalized_link for item in videos]
    source_titles: List[str] = [item.metadata.title for item in videos]
    best_salvaged_title: Optional[str] = None

    for attempt_index, model_name in enumerate(model_sequence, start=1):
        current_label: str = f"{attempt_label}_TRY{attempt_index}"
        try:
            raw_result: _MergeCallResult = _openai_merge_call_raw(language=language, videos=videos, config=config, attempt_label=current_label, model_name=model_name, no_description_text=no_description_text, merge_run_summary=merge_run_summary)
            last_raw_response = raw_result.raw_text
            maybe_title: Optional[str] = extract_valid_title_from_raw_or_none(last_raw_response)
            if maybe_title:
                best_salvaged_title = maybe_title
        except Exception as error:
            last_error_summary = summarize_error(error)
            LOGGER.warning("OpenAI merge call failed language=%s model=%s reason=%s", language, model_name, last_error_summary)
            continue
        try:
            merged_content: MergedLanguageContent = parse_llm_merge_raw_or_raise(
                provider_name="openai",
                model_name=model_name,
                language=language,
                raw_text=last_raw_response,
                source_urls=source_urls,
                source_titles=source_titles,
                normalize_url=normalize_youtube_url,
            )
            if raw_result.structured_used:
                LOGGER.info(
                    "merge_path structured_ok lang=%s source_count=%d",
                    language,
                    len(videos),
                )
                if merge_run_summary is not None:
                    merge_run_summary.record_structured_ok()
            elif raw_result.structured_attempted:
                LOGGER.info(
                    "merge_path plain_fallback_ok lang=%s source_count=%d",
                    language,
                    len(videos),
                )
                if merge_run_summary is not None:
                    merge_run_summary.record_plain_fallback_ok()
            return LanguageMergeAttempt(language=language, model_name=model_name, raw_response_text=last_raw_response, merged=merged_content, error_summary=None, salvaged_title=merged_content.title)
        except Exception as parse_error:
            parse_error_summary: str = summarize_error(parse_error)
            if raw_result.structured_used:
                LOGGER.info(
                    "merge_path structured_failed lang=%s source_count=%d reason=parse_validation_failed",
                    language,
                    len(videos),
                )
                if merge_run_summary is not None:
                    merge_run_summary.record_structured_failed()
            payload_pair: Optional[Tuple[str, str]] = extract_title_and_description_payload_or_none(last_raw_response)
            if payload_pair is not None:
                payload_title, payload_description = payload_pair
                try:
                    repaired_description: str = _attempt_openai_plain_description_repair_once(
                        language=language,
                        videos=videos,
                        config=config,
                        attempt_label=f"{current_label}_PLAIN_REPAIR",
                        model_name=model_name,
                        invalid_text=payload_description,
                        no_description_text=no_description_text,
                        merge_run_summary=merge_run_summary,
                    )
                    repaired_content: MergedLanguageContent = build_plain_merged_content_or_raise(
                        model_name=model_name,
                        title_text=payload_title,
                        description_text=repaired_description,
                    )
                    LOGGER.info(
                        "merge_path repair_used lang=%s repair=plain_description source_count=%d",
                        language,
                        len(videos),
                    )
                    if merge_run_summary is not None:
                        merge_run_summary.record_repair_used()
                    return LanguageMergeAttempt(language=language, model_name=model_name, raw_response_text=last_raw_response, merged=repaired_content, error_summary=None, salvaged_title=repaired_content.title, plain_repair_used=True)
                except Exception:
                    pass
            repair_prompt: str = _build_llm_merge_repair_prompt_text(language=language, videos=videos, config=config, previous_output=last_raw_response, parse_error=parse_error_summary, no_description_text=no_description_text)
            try:
                repaired_result: _MergeCallResult = _openai_merge_call_raw(language=language, videos=videos, config=config, attempt_label=f"{current_label}_REPAIR", model_name=model_name, prompt_text_override=repair_prompt, no_description_text=no_description_text, merge_run_summary=merge_run_summary)
                repaired_raw_text: str = repaired_result.raw_text
                repaired_merged: MergedLanguageContent = parse_llm_merge_raw_or_raise(
                    provider_name="openai",
                    model_name=model_name,
                    language=language,
                    raw_text=repaired_raw_text,
                    source_urls=source_urls,
                    source_titles=source_titles,
                    normalize_url=normalize_youtube_url,
                )
                LOGGER.info(
                    "merge_path repair_used lang=%s repair=full_retry source_count=%d",
                    language,
                    len(videos),
                )
                if merge_run_summary is not None:
                    merge_run_summary.record_repair_used()
                return LanguageMergeAttempt(language=language, model_name=model_name, raw_response_text=repaired_raw_text, merged=repaired_merged, error_summary=None, salvaged_title=repaired_merged.title)
            except Exception as repair_error:
                last_error_summary = f"parse_error={parse_error_summary}; repair_error={summarize_error(repair_error)}"
                LOGGER.warning("OpenAI merge attempt failed language=%s model=%s reason=%s", language, model_name, last_error_summary)
                continue

    return LanguageMergeAttempt(language=language, model_name=model_sequence[-1], raw_response_text=last_raw_response, merged=None, error_summary=last_error_summary or "unknown error", salvaged_title=best_salvaged_title)


def enforce_openai_merged_paragraphs(*, language: str, merged_content: MergedLanguageContent, videos: List[PlannedVideo], config: AppConfig, no_description_text: str, merge_run_summary: Optional[MergeRunSummary] = None) -> MergedLanguageContent:
    return enforce_merged_paragraphs_for_group(
        provider_name="openai",
        language=language,
        merged_content=merged_content,
        videos=videos,
        paragraph_limit=config.llm_source_desc_max_chars,
        no_description_text=no_description_text,
        merge_run_summary=merge_run_summary,
    )
