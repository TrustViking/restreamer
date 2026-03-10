from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
_URL_LINE_RE: re.Pattern[str] = re.compile(r"^https?://\S+$", re.IGNORECASE)
_OFFICIAL_LINKS_HEADING_RE: re.Pattern[str] = re.compile(
    r"(?im)^\s*(?:🌐\s*)?(?:official links|офіційні ресурси|официальные ссылки)\s*:\s*$"
)
_HASHTAG_TOKEN_RE: re.Pattern[str] = re.compile(r"^#[^\s#]+$")


@dataclass(frozen=True)
class MergeTailSeparationResult:
    body_text: str
    full_text: str
    raw_paragraph_count: int
    body_paragraph_count: int
    body_paragraph_count_after_recovery: int
    tail_blocks: tuple[str, ...]
    tail_detected: bool
    recovery_applied: bool
    recovery_note: str


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


def _normalize_multiline_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_paragraphs(text: str) -> List[str]:
    normalized_text: str = _normalize_multiline_text(text)
    if not normalized_text:
        return []
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", normalized_text)
        if paragraph.strip()
    ]


def _looks_like_hashtags_paragraph(paragraph_text: str) -> bool:
    tokens: List[str] = [token.strip() for token in str(paragraph_text or "").split() if token.strip()]
    return bool(tokens) and all(_HASHTAG_TOKEN_RE.fullmatch(token) for token in tokens)


def _looks_like_cta_paragraph(paragraph_text: str) -> bool:
    lines: List[str] = _paragraph_lines(paragraph_text)
    if not lines:
        return False
    if len(lines) > 2:
        return False
    if any(_OFFICIAL_LINKS_HEADING_RE.match(line) for line in lines):
        return False
    if any(line.startswith(tuple(("🔹 ", "📌 ", "🎤 ", "🎥 ", "⚖ ", "🌐 ", "✅ "))) for line in lines):
        return False
    if any(_URL_LINE_RE.fullmatch(line) for line in lines):
        return False
    normalized: str = re.sub(r"\s+", " ", str(paragraph_text or "")).strip().lower()
    if not normalized:
        return False
    if "#" in normalized and len(normalized) <= 220:
        return True
    return any(
        hint in normalized
        for hint in (
            "watch",
            "join",
            "share",
            "follow",
            "subscribe",
            "learn more",
            "дивіться",
            "долуч",
            "підпис",
            "смотрите",
            "присоединяйтесь",
            "делитесь",
            "подпис",
        )
    )


def _paragraph_lines(paragraph_text: str) -> List[str]:
    return [line.strip() for line in str(paragraph_text or "").split("\n") if line.strip()]


def _looks_like_official_links_paragraph(paragraph_text: str) -> bool:
    lines: List[str] = _paragraph_lines(paragraph_text)
    if not lines or not _OFFICIAL_LINKS_HEADING_RE.match(lines[0]):
        return False
    if len(lines) == 1:
        return True
    return all(_URL_LINE_RE.fullmatch(line) for line in lines[1:])


def _looks_like_youtube_links_paragraph(paragraph_text: str) -> bool:
    lines: List[str] = _paragraph_lines(paragraph_text)
    if not lines:
        return False
    for line in lines:
        if not _URL_LINE_RE.fullmatch(line):
            return False
        if "youtu" not in line.lower():
            return False
    return True


def _split_body_and_allowed_tail(text: str) -> tuple[List[str], List[str], List[str]]:
    paragraphs: List[str] = _split_paragraphs(text)
    if not paragraphs:
        return ([], [], [])
    body_paragraphs: List[str] = list(paragraphs)
    tail_paragraphs_reversed: List[str] = []
    tail_blocks_reversed: List[str] = []
    while body_paragraphs:
        candidate: str = body_paragraphs[-1]
        if _looks_like_hashtags_paragraph(candidate):
            tail_paragraphs_reversed.append(body_paragraphs.pop())
            tail_blocks_reversed.append("hashtags")
            continue
        if _looks_like_cta_paragraph(candidate):
            tail_paragraphs_reversed.append(body_paragraphs.pop())
            tail_blocks_reversed.append("cta")
            continue
        if _looks_like_official_links_paragraph(candidate):
            tail_paragraphs_reversed.append(body_paragraphs.pop())
            tail_blocks_reversed.append("official_links")
            continue
        if _looks_like_youtube_links_paragraph(candidate):
            tail_paragraphs_reversed.append(body_paragraphs.pop())
            tail_blocks_reversed.append("youtube_links")
            continue
        break
    tail_paragraphs: List[str] = list(reversed(tail_paragraphs_reversed))
    tail_blocks: List[str] = list(reversed(tail_blocks_reversed))
    return (body_paragraphs, tail_paragraphs, tail_blocks)


def _split_single_paragraph_safely(paragraph_text: str) -> Optional[List[str]]:
    sentence_parts: List[str] = [
        part.strip()
        for part in re.split(r"(?<=[.!?…])\s+", str(paragraph_text or "").strip())
        if part.strip()
    ]
    if len(sentence_parts) < 4:
        return None
    split_index: int = max(2, len(sentence_parts) // 2)
    left_part: str = " ".join(sentence_parts[:split_index]).strip()
    right_part: str = " ".join(sentence_parts[split_index:]).strip()
    if not left_part or not right_part:
        return None
    return [left_part, right_part]


def _collapse_paragraphs_to_limit(
    paragraphs: List[str],
    *,
    max_paragraphs: int,
) -> Optional[List[str]]:
    cleaned_paragraphs: List[str] = [item.strip() for item in paragraphs if item.strip()]
    if not cleaned_paragraphs:
        return None
    if len(cleaned_paragraphs) <= max_paragraphs:
        return cleaned_paragraphs
    if len(cleaned_paragraphs) > max_paragraphs + 2:
        return None
    total_items: int = len(cleaned_paragraphs)
    base_group_size: int = total_items // max_paragraphs
    remainder: int = total_items % max_paragraphs
    collapsed: List[str] = []
    start_index: int = 0
    for group_index in range(max_paragraphs):
        group_size: int = base_group_size + (1 if group_index < remainder else 0)
        next_index: int = start_index + max(1, group_size)
        group_items: List[str] = cleaned_paragraphs[start_index:next_index]
        if not group_items:
            break
        collapsed.append("\n".join(group_items).strip())
        start_index = next_index
    return collapsed if len(collapsed) <= max_paragraphs else None


def separate_merge_body_and_tail(*, text: str) -> MergeTailSeparationResult:
    normalized_text: str = _normalize_multiline_text(text)
    raw_paragraphs: List[str] = _split_paragraphs(normalized_text)
    body_paragraphs, tail_paragraphs, tail_blocks = _split_body_and_allowed_tail(normalized_text)
    recovered_body_paragraphs: List[str] = list(body_paragraphs)
    recovery_applied: bool = False
    recovery_note: str = "not_needed"

    if len(recovered_body_paragraphs) == 1:
        safely_split: Optional[List[str]] = _split_single_paragraph_safely(recovered_body_paragraphs[0])
        if safely_split is not None:
            recovered_body_paragraphs = safely_split
            recovery_applied = True
            recovery_note = "split_single_body_paragraph"
    elif len(recovered_body_paragraphs) > 4:
        collapsed_paragraphs: Optional[List[str]] = _collapse_paragraphs_to_limit(
            recovered_body_paragraphs,
            max_paragraphs=4,
        )
        if collapsed_paragraphs is not None and len(collapsed_paragraphs) <= 4:
            recovered_body_paragraphs = collapsed_paragraphs
            recovery_applied = True
            recovery_note = "collapsed_excess_body_paragraphs"
        else:
            recovery_note = "body_recovery_not_possible"

    body_text: str = "\n\n".join(recovered_body_paragraphs).strip()
    full_text: str = "\n\n".join(
        paragraph
        for paragraph in [body_text, *tail_paragraphs]
        if paragraph.strip()
    ).strip()
    return MergeTailSeparationResult(
        body_text=body_text,
        full_text=full_text,
        raw_paragraph_count=len(raw_paragraphs),
        body_paragraph_count=len(body_paragraphs),
        body_paragraph_count_after_recovery=len(recovered_body_paragraphs),
        tail_blocks=tuple(tail_blocks),
        tail_detected=bool(tail_blocks),
        recovery_applied=recovery_applied,
        recovery_note=recovery_note,
    )


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


def _validate_body_paragraph_count(body_text: str) -> int:
    body_paragraphs: List[str] = _split_paragraphs(body_text)
    if not 2 <= len(body_paragraphs) <= 4:
        raise RuntimeError(
            f"description body paragraph count must be between 2 and 4, got {len(body_paragraphs)}"
        )
    return len(body_paragraphs)


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

    tail_separation: MergeTailSeparationResult = separate_merge_body_and_tail(
        text=description_value
    )
    cleaned_description, ok, reasons = clean_and_validate_llm_description(
        text=tail_separation.full_text
    )
    if not ok or not cleaned_description:
        raise RuntimeError("description validation failed: " + ("; ".join(reasons) or "unknown"))
    try:
        paragraph_count: int = _validate_body_paragraph_count(tail_separation.body_text)
    except Exception as error:
        LOGGER.info(
            "merge_description_tail_analysis model=%s raw_paragraph_count=%d tail_separated=%s tail_blocks=%s body_paragraph_count=%d recovery_applied=%s body_paragraph_count_after_recovery=%d final_status=rejected reject_reason=%s",
            model_name,
            tail_separation.raw_paragraph_count,
            "yes" if tail_separation.tail_detected else "no",
            ",".join(tail_separation.tail_blocks) or "none",
            tail_separation.body_paragraph_count,
            "yes" if tail_separation.recovery_applied else "no",
            tail_separation.body_paragraph_count_after_recovery,
            str(error),
        )
        raise

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
    LOGGER.info(
        "merge_description_tail_analysis model=%s raw_paragraph_count=%d tail_separated=%s tail_blocks=%s body_paragraph_count=%d recovery_applied=%s body_paragraph_count_after_recovery=%d final_status=accepted",
        model_name,
        tail_separation.raw_paragraph_count,
        "yes" if tail_separation.tail_detected else "no",
        ",".join(tail_separation.tail_blocks) or "none",
        tail_separation.body_paragraph_count,
        "yes" if tail_separation.recovery_applied else "no",
        tail_separation.body_paragraph_count_after_recovery,
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
