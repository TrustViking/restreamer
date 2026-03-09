from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.config.settings import AppConfig
from app.config.model_resolution import resolve_provider_for_model
from app.core.branching import BRANCH_MERGE_MAIN_FALLBACK_PACKAGING
from app.core.models import (
    LanguageMergeAttempt,
    MergedLanguageContent,
    PackagingAudit,
    PlannedVideo,
)
from app.llm.merge_quality import normalize_merge_description
from app.llm.openai_client import LlmTraceContext, OpenAITransportResult
from app.llm.provider_factory import get_llm_provider_for_model
from app.llm.providers.base import LlmProvider
from app.publish.post_llm_sanitation import sanitize_post_llm_title, sanitize_post_llm_text


_HASHTAG_TOKEN_RE: re.Pattern[str] = re.compile(r"^#[^\s#]+$")
_MEANINGFUL_TEXT_RE: re.Pattern[str] = re.compile(r"[0-9A-Za-zА-Яа-яЇїІіЄєҐґ]")


@dataclass(frozen=True)
class PackagingPayload:
    title_text: str
    hook_text: str
    hashtags_line: str
    emoji_plan: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackagingEmojiNormalizationResult:
    description_text: str
    normalized: bool
    replaced_markers: tuple[str, ...]
    downgraded_markers: tuple[str, ...]


@dataclass(frozen=True)
class PackagingOverlayGuardResult:
    merged_content: MergedLanguageContent
    title_fallback_used: bool
    hook_fallback_used: bool
    hashtags_normalized: bool
    links_restored: bool
    emoji_normalized: bool
    marker_changes: tuple[str, ...]
    accent_downgrades: tuple[str, ...]


def _packaging_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "hook": {"type": "string"},
            "hashtags": {
                "type": "array",
                "items": {"type": "string"},
            },
            "emoji_suggestions": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["title", "hook", "hashtags"],
    }


def _language_name(language: str) -> str:
    mapping: dict[str, str] = {
        "uk": "Ukrainian",
        "en": "English",
        "ru": "Russian",
        "other": "Original language",
    }
    return mapping.get(language, mapping["other"])


def _dedupe_hashtags(tokens: Sequence[str]) -> str:
    ordered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        cleaned_token: str = str(token or "").strip()
        if not cleaned_token:
            continue
        if not cleaned_token.startswith("#"):
            cleaned_token = f"#{cleaned_token.lstrip('#')}"
        if not _HASHTAG_TOKEN_RE.fullmatch(cleaned_token):
            continue
        normalized_key: str = cleaned_token.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        ordered.append(cleaned_token)
        if len(ordered) >= 5:
            break
    return " ".join(ordered)


def _normalize_hashtags_line(text: str) -> str:
    return _dedupe_hashtags(str(text or "").split())


def _is_meaningful_packaging_text(text: str) -> bool:
    normalized_text: str = sanitize_post_llm_title(str(text or "").strip())
    if len(normalized_text) < 3:
        return False
    return _MEANINGFUL_TEXT_RE.search(normalized_text) is not None


def _looks_valid_packaging_hook(text: str) -> bool:
    normalized_text: str = str(text or "").strip()
    if not _is_meaningful_packaging_text(normalized_text):
        return False
    if "http://" in normalized_text.lower() or "https://" in normalized_text.lower():
        return False
    hashtag_tokens: list[str] = [
        token for token in normalized_text.split() if token.strip().startswith("#")
    ]
    if hashtag_tokens:
        return False
    return True


def _extract_payload_from_response(response: OpenAITransportResult) -> PackagingPayload:
    structured_payload: Any = response.structured_payload
    payload: Dict[str, Any]
    if isinstance(structured_payload, dict):
        payload = structured_payload
    else:
        raw_text: str = str(response.raw_text or "").strip()
        if not raw_text:
            raise RuntimeError("empty_packaging_response")
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as error:
            raise RuntimeError("packaging_json_parse_failed") from error
        if not isinstance(payload, dict):
            raise RuntimeError("packaging_json_not_object")
    title_text: str = sanitize_post_llm_title(str(payload.get("title", "")).strip())
    hook_text: str = str(payload.get("hook", "")).strip()
    hashtags_value: Any = payload.get("hashtags", [])
    if not title_text and not hook_text and not hashtags_value:
        raise RuntimeError("packaging_fields_missing")
    hashtags_tokens: list[str]
    if isinstance(hashtags_value, list):
        hashtags_tokens = [str(item or "").strip() for item in hashtags_value]
    else:
        hashtags_tokens = str(hashtags_value or "").split()
    emoji_value: Any = payload.get("emoji_suggestions", [])
    emoji_plan: tuple[str, ...] = ()
    if isinstance(emoji_value, list):
        emoji_plan = tuple(str(item or "").strip() for item in emoji_value if str(item or "").strip())
    return PackagingPayload(
        title_text=title_text,
        hook_text=hook_text,
        hashtags_line=_dedupe_hashtags(hashtags_tokens),
        emoji_plan=emoji_plan,
    )


def _build_packaging_prompt(
    *,
    language: str,
    nomerge_title: str,
    nomerge_description: str,
    merge_title: str,
    merge_description: str,
) -> str:
    return (
        "You are preparing packaging only for a livestream digest branch.\n"
        "Do not create a new semantic merge. Do not rewrite the full description. "
        "Do not add new facts, new speakers, new links, or new sections.\n\n"
        f"Target language: {_language_name(language)}\n\n"
        "Return one strict JSON object with fields:\n"
        '- \"title\": short polished title in the same language\n'
        '- \"hook\": one short factual hook paragraph in the same language\n'
        '- \"hashtags\": 2 to 5 relevant hashtags\n'
        '- \"emoji_suggestions\": optional short list\n\n'
        "Rules:\n"
        "- packaging only, no new facts\n"
        "- keep language consistent\n"
        "- no markdown fences\n"
        "- no full description\n"
        "- no source links block\n"
        "- no semantic rewrite of body\n\n"
        f"NOMERGE TITLE:\n{nomerge_title}\n\n"
        f"NOMERGE DESCRIPTION:\n{nomerge_description}\n\n"
        f"MERGE_MAIN TITLE:\n{merge_title}\n\n"
        f"MERGE_MAIN DESCRIPTION:\n{merge_description}\n"
    )


def _split_paragraphs(text: str) -> List[str]:
    normalized_text: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized_text) if part.strip()]


def _compose_full_text(
    *,
    body_text: str,
    cta_text: str,
    hashtags_line: str,
    source_urls: Sequence[str],
) -> str:
    parts: List[str] = []
    if body_text:
        parts.append(body_text)
    tail_lines: List[str] = []
    if cta_text:
        tail_lines.append(cta_text)
    if hashtags_line:
        tail_lines.append(hashtags_line)
    if source_urls:
        tail_lines.extend(str(item or "").strip() for item in source_urls if str(item or "").strip())
    if tail_lines:
        parts.append("\n".join(tail_lines).strip())
    return "\n\n".join(part for part in parts if part).strip()


def _extract_bullet_markers(text: str) -> list[str]:
    markers: list[str] = []
    for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped_line: str = line.strip()
        if len(stripped_line) < 2:
            continue
        if stripped_line[1:2] != " ":
            continue
        first_token: str = stripped_line[:1]
        if first_token in {"🔹", "📌", "🎤", "🎥", "⚖", "🌐", "✅"}:
            markers.append(first_token)
    return markers


def _normalize_packaging_markers(
    *,
    description_text: str,
    language: str,
) -> PackagingEmojiNormalizationResult:
    before_markers: list[str] = _extract_bullet_markers(description_text)
    normalization_result = normalize_merge_description(
        description=description_text,
        language=language,
        source_texts=(),
    )
    after_markers: list[str] = _extract_bullet_markers(normalization_result.description_text)
    replaced_markers: list[str] = []
    downgraded_markers: list[str] = []
    for before_marker, after_marker in zip(before_markers, after_markers):
        if before_marker == after_marker:
            continue
        replaced_markers.append(f"{before_marker}->{after_marker}")
        if before_marker != "🔹" and after_marker == "🔹":
            downgraded_markers.append(f"{before_marker}->{after_marker}")
    return PackagingEmojiNormalizationResult(
        description_text=normalization_result.description_text,
        normalized=normalization_result.normalization_applied,
        replaced_markers=tuple(replaced_markers),
        downgraded_markers=tuple(downgraded_markers),
    )


def _apply_packaging_overlay(
    *,
    language: str,
    merged_content: MergedLanguageContent,
    payload: PackagingPayload,
    main_source_label: str,
    packaging_source_label: str,
) -> PackagingOverlayGuardResult:
    base_sanitization_result = sanitize_post_llm_text(
        str(merged_content.description or ""),
        language="unknown",
        source_label="merge_main_packaging_overlay",
        log_summary=False,
    )
    body_paragraphs: List[str] = _split_paragraphs(base_sanitization_result.body_text)
    base_hook_text: str = body_paragraphs[0] if body_paragraphs else ""

    title_candidate: str = sanitize_post_llm_title(str(payload.title_text or "").strip())
    use_packaging_title: bool = _is_meaningful_packaging_text(title_candidate)
    final_title: str = (
        title_candidate if use_packaging_title else str(merged_content.title or "").strip()
    )
    title_source: str = packaging_source_label if use_packaging_title else main_source_label

    hook_candidate: str = str(payload.hook_text or "").strip()
    use_packaging_hook: bool = _looks_valid_packaging_hook(hook_candidate)
    selected_hook_text: str = hook_candidate if use_packaging_hook else base_hook_text
    if selected_hook_text:
        if body_paragraphs:
            body_paragraphs[0] = selected_hook_text
        else:
            body_paragraphs = [selected_hook_text]
    overlaid_body_text: str = "\n\n".join(body_paragraphs).strip()

    payload_hashtags_line: str = _normalize_hashtags_line(str(payload.hashtags_line or "").strip())
    base_hashtags_line: str = _normalize_hashtags_line(
        str(base_sanitization_result.hashtags_line or "").strip()
    )
    selected_hashtags_line: str = payload_hashtags_line or base_hashtags_line
    provisional_description: str = _compose_full_text(
        body_text=overlaid_body_text,
        cta_text=base_sanitization_result.cta_text,
        hashtags_line=selected_hashtags_line,
        source_urls=base_sanitization_result.source_urls,
    )
    provisional_sanitization_result = sanitize_post_llm_text(
        provisional_description,
        language="unknown",
        source_label="merge_main_packaging_overlay_guard",
        log_summary=False,
    )
    final_source_urls: Sequence[str]
    links_restored: bool = (
        len(provisional_sanitization_result.source_urls)
        < len(base_sanitization_result.source_urls)
    )
    if links_restored:
        final_source_urls = base_sanitization_result.source_urls
    else:
        final_source_urls = provisional_sanitization_result.source_urls
    final_hashtags_line: str = _normalize_hashtags_line(
        provisional_sanitization_result.hashtags_line or selected_hashtags_line
    )
    hashtags_normalized: bool = final_hashtags_line != str(payload.hashtags_line or "").strip()
    final_description: str = _compose_full_text(
        body_text=provisional_sanitization_result.body_text or overlaid_body_text,
        cta_text=provisional_sanitization_result.cta_text or base_sanitization_result.cta_text,
        hashtags_line=final_hashtags_line,
        source_urls=final_source_urls,
    )
    emoji_normalization_result: PackagingEmojiNormalizationResult = _normalize_packaging_markers(
        description_text=final_description,
        language=language,
    )
    final_description = emoji_normalization_result.description_text
    hook_source: str = packaging_source_label if use_packaging_hook else main_source_label
    if payload_hashtags_line:
        hashtags_source: str = packaging_source_label
    elif base_hashtags_line:
        hashtags_source = main_source_label
    else:
        hashtags_source = "fallback_none"

    return PackagingOverlayGuardResult(
        merged_content=dataclasses.replace(
            merged_content,
            title=final_title,
            description=final_description,
            hook_text=(selected_hook_text or body_paragraphs[0] if body_paragraphs else ""),
            hashtags_line=final_hashtags_line,
            branch_type=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
            title_source=title_source,
            hook_source=hook_source,
            hashtags_source=hashtags_source,
            body_source="main_merge",
            packaging_model_name=merged_content.packaging_model_name or "",
            title_selected=final_title,
            description_selected=final_description,
            title_audit=final_title,
            description_audit=final_description,
        ),
        title_fallback_used=bool(title_candidate) and not use_packaging_title,
        hook_fallback_used=bool(hook_candidate) and not use_packaging_hook,
        hashtags_normalized=hashtags_normalized,
        links_restored=links_restored,
        emoji_normalized=emoji_normalization_result.normalized,
        marker_changes=emoji_normalization_result.replaced_markers,
        accent_downgrades=emoji_normalization_result.downgraded_markers,
    )


def build_packaging_overlay_attempt(
    *,
    logger: logging.Logger,
    language: str,
    videos: List[PlannedVideo],
    merge_attempt: LanguageMergeAttempt,
    merged_content: MergedLanguageContent,
    config: AppConfig,
    branch_label: str,
    date_key: str,
    slot_key: str,
) -> tuple[MergedLanguageContent, LanguageMergeAttempt]:
    packaging_model: str = str(config.llm_fallback_model or "").strip()
    main_model: str = str(merge_attempt.generator_model_name or merge_attempt.model_name or "").strip()
    main_source_label: str = resolve_provider_for_model(main_model) if main_model else "main_model"
    packaging_source_label: str = (
        resolve_provider_for_model(packaging_model) if packaging_model else "packaging_model"
    )
    if not packaging_model:
        packaging_audit = PackagingAudit(
            packaging_model="",
            raw_response_text="",
            title_text="",
            hook_text="",
            hashtags_line="",
            requested=False,
            received=False,
            inserted=False,
            fallback_used=True,
            fallback_reason="packaging_model_missing",
        )
        logger.info(
            "llm_packaging_fallback_used branch=%s date_key=%s slot_key=%s language=%s reason=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            packaging_audit.fallback_reason,
        )
        return (
            dataclasses.replace(
                merged_content,
                branch_type=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                title_source=main_source_label,
                hook_source=main_source_label,
                hashtags_source="fallback_none",
                body_source="main_merge",
            ),
            dataclasses.replace(
                merge_attempt,
                branch_type=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                title_source=main_source_label,
                hook_source=main_source_label,
                hashtags_source="fallback_none",
                body_source="main_merge",
                packaging_audit=packaging_audit,
                used_model_names=merge_attempt.used_model_names or (merge_attempt.model_name,),
            ),
        )
    packaging_provider: LlmProvider = get_llm_provider_for_model(model_name=packaging_model)
    nomerge_title: str = "\n".join(
        str(video.metadata.title or "").strip() for video in videos if str(video.metadata.title or "").strip()
    ).strip()
    nomerge_description: str = "\n\n".join(
        str(video.metadata.description or "").strip() for video in videos if str(video.metadata.description or "").strip()
    ).strip()
    logger.info(
        "llm_packaging_requested branch=%s date_key=%s slot_key=%s language=%s source_model=%s packaging_model=%s packaging_provider=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        merge_attempt.generator_model_name or merge_attempt.model_name,
        packaging_model,
        packaging_provider.name,
    )
    try:
        response: OpenAITransportResult = packaging_provider.request_merge(
            prompt_text=_build_packaging_prompt(
                language=language,
                nomerge_title=nomerge_title,
                nomerge_description=nomerge_description,
                merge_title=str(merged_content.title or ""),
                merge_description=str(merged_content.description or ""),
            ),
            model_name=packaging_model,
            config=config,
            attempt_label=f"PACKAGING_{language.upper()}",
            max_output_tokens=1200,
            structured_schema=_packaging_schema(),
            temperature=0.0,
            trace_context=LlmTraceContext(
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                provider=packaging_provider.name,
                model_name=packaging_model,
                attempt_index=1,
                request_kind="packaging",
                source_count=len(videos),
            ),
        )
        payload: PackagingPayload = _extract_payload_from_response(response)
        logger.info(
            "llm_packaging_received branch=%s date_key=%s slot_key=%s language=%s title_length=%d hook_length=%d hashtags_count=%d",
            branch_label,
            date_key,
            slot_key,
            language,
            len(payload.title_text),
            len(payload.hook_text),
            len([token for token in payload.hashtags_line.split() if token.strip()]),
        )
        logger.debug(
            "llm_packaging_payload branch=%s date_key=%s slot_key=%s language=%s raw=%r",
            branch_label,
            date_key,
            slot_key,
            language,
            response.raw_text,
        )
        guard_result: PackagingOverlayGuardResult = _apply_packaging_overlay(
            language=language,
            merged_content=dataclasses.replace(
                merged_content,
                packaging_model_name=packaging_model,
            ),
            payload=payload,
            main_source_label=main_source_label,
            packaging_source_label=packaging_source_label,
        )
        packaged_content: MergedLanguageContent = guard_result.merged_content
        packaging_audit = PackagingAudit(
            packaging_model=packaging_model,
            raw_response_text=str(response.raw_text or ""),
            title_text=payload.title_text,
            hook_text=payload.hook_text,
            hashtags_line=payload.hashtags_line,
            emoji_plan=payload.emoji_plan,
            requested=True,
            received=True,
            inserted=True,
            fallback_used=False,
            fallback_reason=None,
        )
        logger.info(
            "llm_packaging_quality_guard branch=%s date_key=%s slot_key=%s language=%s title_fallback=%s hook_fallback=%s hashtags_normalized=%s links_restored=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            "yes" if guard_result.title_fallback_used else "no",
            "yes" if guard_result.hook_fallback_used else "no",
            "yes" if guard_result.hashtags_normalized else "no",
            "yes" if guard_result.links_restored else "no",
        )
        logger.info(
            "llm_packaging_inserted branch=%s date_key=%s slot_key=%s language=%s title_source=%s hook_source=%s hashtags_source=%s body_source=%s emoji_normalized=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            packaged_content.title_source or "unknown",
            packaged_content.hook_source or "unknown",
            packaged_content.hashtags_source or "unknown",
            packaged_content.body_source or "unknown",
            "yes" if guard_result.emoji_normalized else "no",
        )
        logger.info(
            "llm_packaging_emoji_markers branch=%s date_key=%s slot_key=%s language=%s normalized=%s marker_changes=%s accent_downgrades=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            "yes" if guard_result.emoji_normalized else "no",
            ",".join(guard_result.marker_changes) or "none",
            ",".join(guard_result.accent_downgrades) or "none",
        )
        used_model_names: tuple[str, ...] = tuple(
            item
            for item in (
                *(merge_attempt.used_model_names or (merge_attempt.model_name,)),
                packaging_model,
            )
            if str(item or "").strip()
        )
        return (
            packaged_content,
            dataclasses.replace(
                merge_attempt,
                branch_type=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                title_source=packaged_content.title_source,
                hook_source=packaged_content.hook_source,
                hashtags_source=packaged_content.hashtags_source,
                body_source=packaged_content.body_source,
                packaging_audit=packaging_audit,
                used_model_names=used_model_names,
            ),
        )
    except Exception as error:
        fallback_reason: str = str(error or "").strip() or "packaging_request_failed"
        logger.info(
            "llm_packaging_fallback_used branch=%s date_key=%s slot_key=%s language=%s reason=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            fallback_reason,
        )
        packaging_audit = PackagingAudit(
            packaging_model=packaging_model,
            raw_response_text="",
            title_text="",
            hook_text="",
            hashtags_line="",
            requested=True,
            received=False,
            inserted=False,
            fallback_used=True,
            fallback_reason=fallback_reason,
        )
        fallback_content: MergedLanguageContent = dataclasses.replace(
            merged_content,
            branch_type=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
            title_source=main_source_label,
            hook_source=main_source_label,
            hashtags_source=(
                main_source_label
                if sanitize_post_llm_text(
                    str(merged_content.description or ""),
                    language="unknown",
                    source_label="merge_main_packaging_fallback",
                    log_summary=False,
                ).hashtags_found
                else "fallback_none"
            ),
            body_source="main_merge",
            packaging_model_name=packaging_model,
        )
        return (
            fallback_content,
            dataclasses.replace(
                merge_attempt,
                branch_type=BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
                title_source=fallback_content.title_source,
                hook_source=fallback_content.hook_source,
                hashtags_source=fallback_content.hashtags_source,
                body_source=fallback_content.body_source,
                packaging_audit=packaging_audit,
                used_model_names=merge_attempt.used_model_names or (merge_attempt.model_name,),
            ),
        )
