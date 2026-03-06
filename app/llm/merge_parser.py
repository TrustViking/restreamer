from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import MergedLanguageContent

LOGGER = _get_logger_impl(__name__)

_FORBIDDEN_VARIANT_KEYS: set[str] = {
    "variants",
    "options",
    "alternatives",
    "titles",
    "descriptions",
}
_EMOJI_PATTERN: re.Pattern[str] = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
    flags=re.UNICODE,
)


def normalize_single_line_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_json_code_fences(text: str) -> str:
    cleaned: str = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def extract_json_object_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    in_string: bool = False
    escaped: bool = False
    depth: int = 0
    start_index: int = -1
    for index, char in enumerate(str(text or "")):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index >= 0:
                candidate: str = str(text)[start_index : index + 1].strip()
                if candidate:
                    candidates.append(candidate)
                start_index = -1
    return candidates


def parse_json_tolerant(raw_text: str) -> tuple[dict[str, object] | None, str]:
    text: str = strip_json_code_fences(raw_text)
    if not text:
        return (None, "fail")
    try:
        parsed: Any = json.loads(text)
    except Exception:
        candidates: List[str] = extract_json_object_candidates(text)
        if len(candidates) != 1:
            return (None, "fail")
        try:
            parsed = json.loads(candidates[0])
        except Exception:
            return (None, "fail")
        if isinstance(parsed, dict):
            return (cast(dict[str, object], parsed), "candidate")
        return (None, "fail")
    if isinstance(parsed, dict):
        return (cast(dict[str, object], parsed), "direct")
    return (None, "fail")


def validate_payload_keys(payload: Dict[str, Any]) -> None:
    allowed_keys: set[str] = {"title", "description"}
    payload_keys: set[str] = set(payload.keys())
    missing_keys: List[str] = sorted(key for key in allowed_keys if key not in payload_keys)
    extra_keys: List[str] = sorted(key for key in payload_keys if key not in allowed_keys)
    if missing_keys:
        raise RuntimeError("missing_keys:" + ",".join(missing_keys))
    if extra_keys:
        raise RuntimeError("extra_keys:" + ",".join(extra_keys))


def sanitize_title(title: str, *, min_chars: int, max_chars: int) -> str:
    normalized_title: str = re.sub(r"\s+", " ", str(title or "").strip())
    if len(normalized_title) > max_chars:
        normalized_title = normalized_title[:max_chars].rstrip()
    if len(normalized_title) < min_chars:
        return ""
    return normalized_title


def contains_emoji(text: str) -> bool:
    return bool(_EMOJI_PATTERN.search(str(text or "")))


def strip_meta_lines(text: str) -> str:
    lines: List[str] = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: List[str] = []
    meta_pattern: re.Pattern[str] = re.compile(r"(?i)^\s*(?:title|description|sources?)\s*:")
    for line in lines:
        if meta_pattern.match(line.strip()):
            continue
        cleaned_lines.append(line.rstrip())
    return "\n".join(cleaned_lines).strip()


def validate_description_plain(text: str) -> Tuple[bool, List[str]]:
    normalized: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    reasons: List[str] = []
    if not normalized:
        reasons.append("empty")
        return (False, reasons)
    for line in [line.strip() for line in normalized.split("\n") if line.strip()][:6]:
        if re.match(r"(?i)^\s*(?:title|description|sources?)\s*:", line):
            reasons.append("meta_header")
            break
    return (len(reasons) == 0, reasons)


def clean_and_validate_llm_description(*, text: str) -> Tuple[str, bool, List[str]]:
    cleaned: str = strip_meta_lines(str(text or ""))
    ok, reasons = validate_description_plain(cleaned)
    return (cleaned, ok, reasons)


def build_plain_merged_content_or_raise(
    *,
    model_name: str,
    title_text: str,
    description_text: str,
) -> MergedLanguageContent:
    title: str = sanitize_title(str(title_text or "").strip(), min_chars=1, max_chars=99)
    description, ok, reasons = clean_and_validate_llm_description(text=description_text)
    if not title:
        raise RuntimeError("plain merge title is invalid")
    if not ok:
        raise RuntimeError("plain description validation failed: " + ("; ".join(reasons) or "unknown"))
    return MergedLanguageContent(
        title=title,
        description=description,
        title_selected=title,
        description_selected=description,
        title_audit=title,
        description_audit=description,
        llm_model=model_name,
    )


def _is_forbidden_multi_variant_payload(payload: Dict[str, Any]) -> Optional[str]:
    for key in payload.keys():
        normalized_key: str = str(key or "").strip().lower()
        if normalized_key in _FORBIDDEN_VARIANT_KEYS:
            return f"forbidden_key:{normalized_key}"
    for key in ("title", "description"):
        value: Any = payload.get(key)
        if isinstance(value, list) or isinstance(value, dict):
            return f"invalid_type:{key}"
    return None


def _validate_paragraph_count(description: str) -> int:
    paragraphs: List[str] = [
        part.strip()
        for part in re.split(r"\n\s*\n", description.replace("\r\n", "\n").replace("\r", "\n"))
        if part.strip()
    ]
    if not 2 <= len(paragraphs) <= 4:
        raise RuntimeError(f"description paragraph count must be between 2 and 4, got {len(paragraphs)}")
    return len(paragraphs)


def parse_merge_response_or_raise(
    *,
    provider_name: str,
    model_name: str,
    raw_text: str,
    structured_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[MergedLanguageContent, int]:
    payload: Optional[Dict[str, Any]] = structured_payload
    parse_mode: str = "structured_payload"
    if payload is None:
        payload, parse_mode = parse_json_tolerant(raw_text)
    if payload is None:
        raise RuntimeError(f"{provider_name} merge output is not a valid single JSON object")
    validate_payload_keys(payload)
    variant_reason: Optional[str] = _is_forbidden_multi_variant_payload(payload)
    if variant_reason is not None:
        raise RuntimeError(f"{provider_name} merge output rejected: {variant_reason}")

    title_value: Any = payload.get("title")
    description_value: Any = payload.get("description")
    if not isinstance(title_value, str) or not title_value.strip():
        raise RuntimeError("title must be a non-empty string")
    if not isinstance(description_value, str) or not description_value.strip():
        raise RuntimeError("description must be a non-empty string")

    paragraph_count: int = _validate_paragraph_count(description_value)
    cleaned_description, ok, reasons = clean_and_validate_llm_description(text=description_value)
    if not ok or not cleaned_description:
        raise RuntimeError("description validation failed: " + ("; ".join(reasons) or "unknown"))

    title: str = sanitize_title(title_value, min_chars=1, max_chars=99)
    if not title:
        raise RuntimeError("title is invalid after normalization")
    if contains_emoji(title):
        raise RuntimeError("title must not contain emoji")
    LOGGER.info(
        "merge_payload_parsed model=%s parse_mode=%s title_length=%d description_length=%d paragraph_count=%d",
        model_name,
        parse_mode,
        len(title),
        len(cleaned_description),
        paragraph_count,
    )
    return (
        MergedLanguageContent(
            title=title,
            description=cleaned_description,
            title_selected=title,
            description_selected=cleaned_description,
            title_audit=title,
            description_audit=cleaned_description,
            llm_model=model_name,
        ),
        paragraph_count,
    )
