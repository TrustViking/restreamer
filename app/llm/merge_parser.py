from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.constants import MERGED_DESCRIPTION_HARD_CEILING, URL_PATTERN, _SOURCE_URL_LINE_RE
from app.core.models import MergedLanguageContent, PlannedVideo
from app.llm.merge_run_summary import MergeRunSummary

LOGGER = _get_logger_impl(__name__)


def normalize_single_line_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def strip_json_code_fences(text: str) -> str:
    cleaned: str = text.strip()
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
    for index, char in enumerate(text):
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
                fragment: str = text[start_index : index + 1].strip()
                if fragment:
                    candidates.append(fragment)
                start_index = -1
    return candidates


def parse_json_tolerant(raw_text: str) -> tuple[dict[str, object] | None, str]:
    def _load_object(candidate_text: str) -> dict[str, object] | None:
        candidate: str = str(candidate_text or "").strip()
        if not candidate:
            return None
        try:
            parsed: Any = json.loads(candidate)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return cast(dict[str, object], parsed)
        return None

    text: str = str(raw_text or "").strip()
    if not text:
        return (None, "fail")

    direct_obj: dict[str, object] | None = _load_object(text)
    if direct_obj is not None:
        return (direct_obj, "direct")

    fence_matches: List[Tuple[int, str]] = []
    for fence_match in re.finditer(r"```([^\n`]*)\s*\n?(.*?)```", text, re.DOTALL):
        info_string: str = str(fence_match.group(1) or "").strip().lower()
        block_text: str = str(fence_match.group(2) or "").strip()
        if not block_text:
            continue
        priority: int = 0 if "json" in info_string else 1
        fence_matches.append((priority, block_text))
    if fence_matches:
        fence_matches.sort(key=lambda item: item[0])
        fence_obj: dict[str, object] | None = _load_object(fence_matches[0][1])
        if fence_obj is not None:
            return (fence_obj, "code_fence")

    first_brace_index: int = text.find("{")
    last_brace_index: int = text.rfind("}")
    if first_brace_index >= 0 and last_brace_index > first_brace_index:
        first_last_obj: dict[str, object] | None = _load_object(
            text[first_brace_index : last_brace_index + 1]
        )
        if first_last_obj is not None:
            return (first_last_obj, "first_brace")

        depth: int = 0
        in_string: bool = False
        escaped: bool = False
        for index in range(first_brace_index, len(text)):
            char: str = text[index]
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
                depth += 1
                continue
            if char == "}":
                if depth > 0:
                    depth -= 1
                if depth == 0:
                    balanced_obj: dict[str, object] | None = _load_object(
                        text[first_brace_index : index + 1]
                    )
                    if balanced_obj is not None:
                        return (balanced_obj, "first_brace")
                    break
    return (None, "fail")


def parse_merge_payload_tolerant(raw_text: str) -> Optional[Dict[str, str]]:
    text: str = strip_json_code_fences(str(raw_text or "").strip())
    if not text:
        return None

    candidates: List[str] = [text]
    candidates.extend(extract_json_object_candidates(text))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed: Any = json.loads(candidate)
            if isinstance(parsed, dict):
                title_raw: str = str(parsed.get("merged_title") or "").strip()
                description_raw: str = str(parsed.get("merged_description") or "").strip()
                if title_raw and description_raw:
                    return {
                        "merged_title": title_raw,
                        "merged_description": description_raw,
                    }
        except Exception:
            continue
    return None


def parse_merge_payload_plaintext_tolerant(raw_text: str) -> Optional[Dict[str, str]]:
    text: str = strip_json_code_fences(str(raw_text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip()
    if not text:
        return None

    lines: List[str] = text.split("\n")
    title_index: Optional[int] = None
    title_value: str = ""
    for index, line in enumerate(lines):
        title_match: Optional[re.Match[str]] = re.match(r"(?i)^\s*title\s*:\s*(.*)$", line)
        if not title_match:
            continue
        title_index = index
        title_value = title_match.group(1).strip()
        if not title_value:
            for tail_line in lines[index + 1 :]:
                candidate: str = tail_line.strip()
                if candidate:
                    title_value = candidate
                    break
        break
    if title_index is None or not title_value:
        return None

    description_index: Optional[int] = None
    description_first_line: str = ""
    for index in range(title_index + 1, len(lines)):
        desc_match: Optional[re.Match[str]] = re.match(r"(?i)^\s*description\s*:\s*(.*)$", lines[index])
        if not desc_match:
            continue
        description_index = index
        description_first_line = desc_match.group(1).strip()
        break
    if description_index is None:
        return None

    description_tail: str = "\n".join(lines[description_index + 1 :]).strip()
    description_value: str = (
        f"{description_first_line}\n{description_tail}".strip()
        if description_first_line and description_tail
        else (description_first_line or description_tail)
    )
    if title_value and description_value:
        return {
            "merged_title": title_value,
            "merged_description": description_value,
        }
    return None


def strip_paragraph_labels(text: str) -> str:
    lines: List[str] = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: List[str] = []
    pattern: re.Pattern[str] = re.compile(r"(?i)^\s*(?:video|paragraph|описание|абзац|видео)\s*\d+\s*[:\-]\s*")
    for line in lines:
        cleaned_lines.append(pattern.sub("", line))
    return "\n".join(cleaned_lines).strip()


def strip_meta_lines(text: str) -> str:
    lines: List[str] = str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cleaned_lines: List[str] = []
    meta_pattern: re.Pattern[str] = re.compile(
        r"(?i)^\s*(?:cta(?:\s*line)?|hashtags(?:\s*line)?|preview|title|description|sources?|source|название|описание|прев[ью'’]+)\b"
    )
    max_nonempty_guard_lines: int = 8
    nonempty_seen: int = 0
    for line in lines:
        normalized_line: str = line.strip()
        if not normalized_line:
            cleaned_lines.append("")
            continue
        nonempty_seen += 1
        in_guard_zone: bool = nonempty_seen <= max_nonempty_guard_lines
        if in_guard_zone and meta_pattern.match(normalized_line):
            continue
        if in_guard_zone and re.match(r"(?i)^https?://(?:www\.)?(?:youtu\.be|youtube\.com)/\S*$", normalized_line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def validate_description_plain(text: str) -> Tuple[bool, List[str]]:
    normalized: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    reasons: List[str] = []
    paragraph_label_pattern: re.Pattern[str] = re.compile(r"(?i)^\s*(?:video|paragraph|описание|абзац|видео)\s*\d+\s*[:\-]")
    nonempty_lines: List[str] = [line.strip() for line in normalized.split("\n") if line.strip()]
    for line in nonempty_lines[:8]:
        if paragraph_label_pattern.match(line):
            reasons.append("paragraph_label")
            break
        if re.match(r"(?i)^\s*(?:title|description|cta|hashtags|sources?)\s*:", line):
            reasons.append("meta_header")
            break
    if re.search(r"https?://\S+", normalized):
        reasons.append("url_leak")
    if not normalized.strip():
        reasons.append("empty")
    return (len(reasons) == 0, reasons)


def clean_and_validate_llm_description(*, text: str) -> Tuple[str, bool, List[str]]:
    cleaned: str = strip_meta_lines(strip_paragraph_labels(text))
    ok, reasons = validate_description_plain(cleaned)
    return (cleaned, ok, reasons)


def is_valid_hashtags_line(line: str) -> bool:
    cleaned_line: str = str(line or "").strip()
    if not cleaned_line:
        return False
    tokens: List[str] = [token for token in re.split(r"\s+", cleaned_line) if token]
    if not tokens:
        return False
    return all(re.match(r"^#[^\s#]+$", token) is not None for token in tokens)


def normalize_hashtags_line(line: str) -> str:
    tokens: List[str] = [token for token in re.split(r"\s+", str(line or "").strip()) if token]
    return " ".join(tokens)


def sanitize_forbidden_section_labels(text: str) -> Tuple[str, bool]:
    raw_text: str = str(text or "")
    sanitized_text: str = re.sub(r"(?im)^\s*(cta|hashtags|sources)\s*:\s*", "", raw_text)
    return (sanitized_text, sanitized_text != raw_text)


def remove_urls_from_paragraph(text: str) -> str:
    without_urls: str = URL_PATTERN.sub("", str(text or ""))
    normalized: str = re.sub(r"\s+", " ", without_urls).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized.strip()


def remove_cta_style_sentences_from_paragraphs(paragraphs: List[str]) -> Tuple[List[str], bool]:
    cta_patterns: Tuple[str, ...] = ("watch", "learn more", "join", "subscribe", "links below", "подписывай", "підпис", "долуч")
    filtered_paragraphs: List[str] = []
    changed: bool = False
    for paragraph_text in paragraphs:
        normalized_paragraph: str = normalize_single_line_text(paragraph_text)
        sentence_parts: List[str] = re.split(r"(?<=[.!?…])\s+", normalized_paragraph)
        kept_sentences: List[str] = []
        removed_any_sentence: bool = False
        for sentence in sentence_parts:
            sentence_clean: str = sentence.strip()
            if not sentence_clean:
                continue
            sentence_lower: str = sentence_clean.lower()
            if any(pattern in sentence_lower for pattern in cta_patterns):
                removed_any_sentence = True
                continue
            kept_sentences.append(sentence_clean)
        if removed_any_sentence and kept_sentences:
            cleaned_paragraph: str = normalize_single_line_text(" ".join(kept_sentences))
            if cleaned_paragraph != normalized_paragraph:
                changed = True
            filtered_paragraphs.append(cleaned_paragraph)
            continue
        filtered_paragraphs.append(normalized_paragraph)
    return (filtered_paragraphs, changed)


def sanitize_hashtag_token(raw_token: str) -> str:
    token: str = str(raw_token or "").strip().lower()
    token = token.strip("_")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"[^a-zа-яёіїєґ0-9_]+", "", token, flags=re.IGNORECASE)
    return token


def generate_hashtags_line_deterministic(language: str, merged_title: str, merged_description_paragraphs: List[str], source_titles: Optional[List[str]] = None) -> str:
    text_parts: List[str] = [str(merged_title or "")]
    text_parts.extend(str(item or "") for item in (merged_description_paragraphs or []))
    text_parts.extend(str(item or "") for item in (source_titles or []))
    combined_text: str = "\n".join(text_parts)
    raw_tokens: List[str] = [
        token for token in re.split(r"[^A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_]+", combined_text) if token
    ]
    selected_tokens: List[str] = []
    seen_tokens: set[str] = set()
    for raw_token in raw_tokens:
        normalized_token: str = sanitize_hashtag_token(raw_token)
        if not normalized_token or normalized_token in seen_tokens or len(normalized_token) < 3:
            continue
        if normalized_token.startswith(("http", "www")):
            continue
        seen_tokens.add(normalized_token)
        selected_tokens.append(normalized_token)
    default_tokens_by_language: Dict[str, List[str]] = {
        "en": ["news", "live", "update"],
        "ru": ["новости", "стрим", "обзор"],
        "uk": ["новини", "стрім", "огляд"],
    }
    hashtags_tokens: List[str] = selected_tokens[:10]
    if len(hashtags_tokens) < 3:
        for default_token in default_tokens_by_language.get(language, default_tokens_by_language["en"]):
            normalized_default: str = sanitize_hashtag_token(default_token)
            if normalized_default and normalized_default not in seen_tokens:
                seen_tokens.add(normalized_default)
                hashtags_tokens.append(normalized_default)
            if len(hashtags_tokens) >= 3:
                break
    hashtags_line: str = normalize_hashtags_line(" ".join(f"#{token}" for token in hashtags_tokens[:10]))
    if not is_valid_hashtags_line(hashtags_line):
        return "#news #live #update"
    return hashtags_line


def normalize_source_url_for_compare(url: str, normalize_url: Callable[[str], str]) -> str:
    cleaned: str = str(url or "").strip()
    if not cleaned:
        return ""
    try:
        return normalize_url(cleaned)
    except Exception:
        return cleaned.rstrip("/")


def normalize_source_url_for_validation(url: str) -> str:
    original: str = str(url or "")
    cleaned: str = original.strip()
    if cleaned.startswith("<") and cleaned.endswith(">") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("(") and cleaned.endswith(")") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.rstrip(".,;)")
    if cleaned != original:
        LOGGER.debug("Normalized source URL: old=%r new=%r", original, cleaned)
    return cleaned


def safe_split_paragraph_region_to_expected(*, paragraph_region_text: str, expected_source_count: int, provider_name: str) -> Tuple[List[str], int, str]:
    normalized_region: str = str(paragraph_region_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_region:
        raise RuntimeError(f"{provider_name} merge description is empty.")
    blank_split_paragraphs: List[str] = [
        normalize_single_line_text(item)
        for item in re.split(r"\n\s*\n", normalized_region)
        if str(item).strip()
    ]
    paragraphs_detected: int = len(blank_split_paragraphs)
    if len(blank_split_paragraphs) == expected_source_count:
        return (blank_split_paragraphs, paragraphs_detected, "")
    if len(blank_split_paragraphs) > expected_source_count:
        merged: List[str] = [normalize_single_line_text(item) for item in blank_split_paragraphs if item.strip()]
        while len(merged) > expected_source_count:
            idx: int = len(merged) - 1
            merged[idx - 1] = normalize_single_line_text(f"{merged[idx - 1]} {merged[idx]}")
            merged.pop(idx)
        return (merged, paragraphs_detected, "merge_tail")
    single_newline_paragraphs: List[str] = [normalize_single_line_text(line) for line in normalized_region.split("\n") if line.strip()]
    if len(single_newline_paragraphs) >= expected_source_count:
        merged2: List[str] = [normalize_single_line_text(item) for item in single_newline_paragraphs if item.strip()]
        while len(merged2) > expected_source_count:
            idx = len(merged2) - 1
            merged2[idx - 1] = normalize_single_line_text(f"{merged2[idx - 1]} {merged2[idx]}")
            merged2.pop(idx)
        return (merged2, paragraphs_detected, "split_newline")
    raise RuntimeError(f"{provider_name} merge description has {paragraphs_detected} paragraphs; expected {expected_source_count}.")


def parse_structured_merged_description_or_raise(*, provider_name: str, description_text: str, expected_source_count: int, expected_source_urls: Optional[List[str]] = None, normalize_url: Optional[Callable[[str], str]] = None) -> Tuple[List[str], str, str, List[str]]:
    if expected_source_count <= 0:
        raise RuntimeError("expected_source_count must be >= 1 for merge parsing.")
    normalized: str = str(description_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise RuntimeError(f"{provider_name} merge description is empty.")
    lines: List[str] = normalized.split("\n")
    indexed_non_empty: List[Tuple[int, str]] = [(line_index, line.strip()) for line_index, line in enumerate(lines) if line.strip()]
    required_non_empty: int = expected_source_count + 2
    if len(indexed_non_empty) < required_non_empty:
        raise RuntimeError(f"{provider_name} merge output has too few non-empty lines: {len(indexed_non_empty)} < {required_non_empty}.")
    source_pairs: List[Tuple[int, str]] = indexed_non_empty[-expected_source_count:]
    hashtags_pair: Tuple[int, str] = indexed_non_empty[-(expected_source_count + 1)]
    cta_pair: Tuple[int, str] = indexed_non_empty[-(expected_source_count + 2)]

    cta_line: str = normalize_single_line_text(cta_pair[1])
    hashtags_line: str = str(hashtags_pair[1] or "").strip()
    paragraph_region_text: str = "\n".join(lines[: cta_pair[0]]).strip()
    paragraphs, _, _ = safe_split_paragraph_region_to_expected(
        paragraph_region_text=paragraph_region_text,
        expected_source_count=expected_source_count,
        provider_name=provider_name,
    )
    parsed_source_urls: List[str] = []
    for source_index, source_pair in enumerate(source_pairs, start=1):
        candidate_url: str = normalize_source_url_for_validation(source_pair[1])
        if not _SOURCE_URL_LINE_RE.match(candidate_url):
            raise RuntimeError(f"{provider_name} merge source URL {source_index} is invalid: {candidate_url!r}.")
        parsed_source_urls.append(candidate_url)

    canonical_source_urls: List[str] = [item.strip() for item in parsed_source_urls]
    expected_urls_clean: List[str] = [str(item).strip() for item in (expected_source_urls or []) if str(item).strip()]
    if expected_urls_clean:
        if len(expected_urls_clean) != expected_source_count:
            raise RuntimeError(f"{provider_name} expected source URLs count mismatch: {len(expected_urls_clean)} vs {expected_source_count}.")
        for source_index in range(expected_source_count):
            if normalize_url is not None:
                expected_url: str = normalize_source_url_for_compare(expected_urls_clean[source_index], normalize_url)
                actual_url: str = normalize_source_url_for_compare(parsed_source_urls[source_index], normalize_url)
            else:
                expected_url = expected_urls_clean[source_index].rstrip("/")
                actual_url = parsed_source_urls[source_index].rstrip("/")
            if expected_url != actual_url:
                raise RuntimeError(f"{provider_name} merge source URL order mismatch at position {source_index + 1}.")
        canonical_source_urls = expected_urls_clean

    return (paragraphs, cta_line, hashtags_line, canonical_source_urls)


def format_structured_merged_description(*, paragraphs: List[str], cta_line: str, hashtags_line: str, source_urls: List[str]) -> str:
    paragraphs_block: str = "\n\n".join(item.strip() for item in paragraphs if item.strip()).strip()
    sources_block: str = "\n".join(item.strip() for item in source_urls if item.strip())
    return f"{paragraphs_block}\n{cta_line.strip()}\n{hashtags_line.strip()}\n{sources_block}".strip()


def auto_trim_structured_description_to_ceiling(*, paragraphs: List[str], cta_line: str, hashtags_line: str, source_urls: List[str], ceiling: int) -> Tuple[List[str], str, str, List[str], bool]:
    trimmed_paragraphs: List[str] = [normalize_single_line_text(item) for item in paragraphs]
    trimmed_cta: str = normalize_single_line_text(cta_line)
    trimmed_hashtags: str = normalize_hashtags_line(hashtags_line)
    trimmed_sources: List[str] = [str(item).strip() for item in source_urls if str(item).strip()]

    def _render_length() -> int:
        return len(format_structured_merged_description(paragraphs=trimmed_paragraphs, cta_line=trimmed_cta, hashtags_line=trimmed_hashtags, source_urls=trimmed_sources))

    before_length: int = _render_length()
    if before_length <= ceiling:
        return (trimmed_paragraphs, trimmed_cta, trimmed_hashtags, trimmed_sources, False)
    while _render_length() > ceiling and len(trimmed_hashtags.split()) > 1:
        parts: List[str] = trimmed_hashtags.split()
        parts.pop()
        trimmed_hashtags = " ".join(parts).strip()
    while _render_length() > ceiling and len(trimmed_cta) > 1:
        trimmed_cta = trimmed_cta[:-1].rstrip()
    while _render_length() > ceiling and any(len(item) > 1 for item in trimmed_paragraphs):
        trimmed_paragraphs = [item[:-1].rstrip() if len(item) > 1 else item for item in trimmed_paragraphs]
    after_length: int = _render_length()
    return (trimmed_paragraphs, trimmed_cta, trimmed_hashtags, trimmed_sources, after_length < before_length)


def sanitize_title(title: str, *, min_chars: int, max_chars: int, allow_emoji: bool) -> str:
    normalized_title: str = re.sub(r"\s+", " ", str(title or "").strip())
    if len(normalized_title) > max_chars:
        normalized_title = normalized_title[: max_chars - 1].rstrip() + "…"
    if not normalized_title or len(normalized_title) < min_chars:
        return ""
    return normalized_title


def parse_llm_merge_raw_or_raise(*, provider_name: str, model_name: str, language: str, raw_text: str, source_urls: List[str], source_titles: Optional[List[str]] = None, normalize_url: Optional[Callable[[str], str]] = None) -> MergedLanguageContent:
    payload: Optional[Dict[str, str]] = parse_merge_payload_plaintext_tolerant(raw_text)
    if payload is None:
        raise RuntimeError(f"{provider_name} merge output is not parseable as TITLE/DESCRIPTION text.")

    merged_title: str = str(payload.get("merged_title") or "").strip()
    merged_description: str = str(payload.get("merged_description") or "").strip()
    merged_description, _ = sanitize_forbidden_section_labels(merged_description)
    merged_title = sanitize_title(merged_title, min_chars=1, max_chars=98, allow_emoji=False)
    merged_title = re.sub(r"[.!?…]{2,}$", "", merged_title).strip()
    merged_title = sanitize_title(merged_title, min_chars=1, max_chars=98, allow_emoji=False)
    if not merged_title.strip("\"'«»` ").strip():
        raise RuntimeError(f"{provider_name} merge title is invalid (quotes-only or empty).")

    source_urls_clean: List[str] = [str(item).strip() for item in (source_urls or []) if str(item).strip()]
    if not source_urls_clean:
        raise RuntimeError(f"{provider_name} merge source URLs are missing.")

    description_paragraphs, cta_line, hashtags_line, description_source_urls = parse_structured_merged_description_or_raise(
        provider_name=provider_name,
        description_text=merged_description,
        expected_source_count=len(source_urls_clean),
        expected_source_urls=source_urls_clean,
        normalize_url=normalize_url,
    )

    cleaned_paragraphs: List[str] = []
    for paragraph_text in description_paragraphs:
        paragraph_after: str = remove_urls_from_paragraph(normalize_single_line_text(paragraph_text)) or "n/a"
        cleaned_paragraphs.append(paragraph_after)
    description_paragraphs = cleaned_paragraphs
    description_paragraphs, _ = remove_cta_style_sentences_from_paragraphs(description_paragraphs)

    if not is_valid_hashtags_line(hashtags_line):
        hashtags_line = generate_hashtags_line_deterministic(language, merged_title, description_paragraphs, source_titles)
    hashtags_line = normalize_hashtags_line(hashtags_line)

    description_paragraphs, cta_line, hashtags_line, description_source_urls, _ = auto_trim_structured_description_to_ceiling(
        paragraphs=description_paragraphs,
        cta_line=cta_line,
        hashtags_line=hashtags_line,
        source_urls=description_source_urls,
        ceiling=MERGED_DESCRIPTION_HARD_CEILING,
    )
    plain_description_text: str = "\n\n".join(description_paragraphs).strip()
    cleaned_plain_description, plain_ok, plain_reasons = clean_and_validate_llm_description(text=plain_description_text)
    if not plain_ok:
        raise RuntimeError(f"{provider_name} merge plain description validation failed: {'; '.join(plain_reasons) or 'unknown validation failure'}")
    if not cleaned_plain_description:
        raise RuntimeError(f"{provider_name} merge description is empty.")
    if len(cleaned_plain_description) > MERGED_DESCRIPTION_HARD_CEILING:
        raise RuntimeError(f"{provider_name} merge description exceeds {MERGED_DESCRIPTION_HARD_CEILING} chars.")

    return MergedLanguageContent(
        title=merged_title,
        description=cleaned_plain_description,
        title_selected=merged_title,
        description_selected=cleaned_plain_description,
        title_audit=merged_title,
        description_audit=cleaned_plain_description,
        llm_model=model_name,
    )


def extract_hashtags_line_from_repair_output(raw_text: str) -> str:
    cleaned: str = strip_json_code_fences(str(raw_text or ""))
    for line in cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        candidate: str = line.strip()
        if not candidate:
            continue
        if re.match(r"(?i)^hashtags\s*:", candidate):
            candidate = candidate.split(":", 1)[1].strip()
        candidate = candidate.strip("\"'` ")
        return candidate
    return ""


def extract_valid_title_from_raw_or_none(raw_text: str) -> Optional[str]:
    text: str = strip_json_code_fences(str(raw_text or ""))
    if not text.strip():
        return None
    payload: Optional[Dict[str, str]] = parse_merge_payload_plaintext_tolerant(text)
    if payload is None:
        return None
    payload_title: str = str(payload.get("merged_title") or "").strip()
    if not payload_title:
        return None
    sanitized_title: str = sanitize_title(re.sub(r"[.!?…]{2,}$", "", payload_title).strip(), min_chars=1, max_chars=98, allow_emoji=False)
    return sanitized_title or None


def build_plain_merged_content_or_raise(*, model_name: str, title_text: str, description_text: str) -> MergedLanguageContent:
    merged_title: str = sanitize_title(str(title_text or "").strip(), min_chars=1, max_chars=98, allow_emoji=False)
    merged_title = re.sub(r"[.!?…]{2,}$", "", merged_title).strip()
    merged_title = sanitize_title(merged_title, min_chars=1, max_chars=98, allow_emoji=False)
    cleaned_description, is_valid, reasons = clean_and_validate_llm_description(text=description_text)
    if not is_valid:
        raise RuntimeError("plain description validation failed: " + ("; ".join(reasons) or "unknown"))
    if not cleaned_description:
        raise RuntimeError("plain description is empty")
    return MergedLanguageContent(
        title=merged_title,
        description=cleaned_description,
        title_selected=merged_title,
        description_selected=cleaned_description,
        title_audit=merged_title,
        description_audit=cleaned_description,
        llm_model=model_name,
    )


def recover_merge_description_paragraphs(*, provider_name: str, merged_text: str, expected_paragraphs: int) -> Tuple[List[str], str]:
    paragraphs, _, recovery_path = safe_split_paragraph_region_to_expected(paragraph_region_text=str(merged_text or ""), expected_source_count=expected_paragraphs, provider_name=provider_name)
    if len(paragraphs) != expected_paragraphs:
        raise RuntimeError(f"{provider_name} paragraph recovery failed: {len(paragraphs)} != {expected_paragraphs}")
    return ([normalize_single_line_text(item) for item in paragraphs], recovery_path)


def build_deterministic_group_paragraphs(*, videos: List[PlannedVideo], paragraph_limit: int, no_description_text: str) -> List[str]:
    paragraphs: List[str] = []
    for video in videos:
        source_text: str = video.metadata.description.strip() or no_description_text
        source_text = normalize_single_line_text(URL_PATTERN.sub("", source_text))
        if not source_text:
            source_text = "n/a"
        paragraphs.append(source_text[: max(200, paragraph_limit)])
    return paragraphs


def enforce_merged_paragraphs_for_group(*, provider_name: str, language: str, merged_content: MergedLanguageContent, videos: List[PlannedVideo], paragraph_limit: int, no_description_text: str, merge_run_summary: Optional[MergeRunSummary] = None) -> MergedLanguageContent:
    expected_paragraphs: int = len(videos)
    if expected_paragraphs <= 1:
        return merged_content
    merged_text: str = str(merged_content.description or "").strip()
    recovery_path_used: str = ""
    try:
        recovered_paragraphs, recovery_path_used = recover_merge_description_paragraphs(provider_name=provider_name, merged_text=merged_text, expected_paragraphs=expected_paragraphs)
    except Exception:
        recovered_paragraphs = build_deterministic_group_paragraphs(videos=videos, paragraph_limit=paragraph_limit, no_description_text=no_description_text)
    enforced_description: str = "\n\n".join(recovered_paragraphs).strip()
    cleaned_description, ok, _ = clean_and_validate_llm_description(text=enforced_description)
    if not ok or not cleaned_description:
        fallback_paragraphs: List[str] = build_deterministic_group_paragraphs(videos=videos, paragraph_limit=paragraph_limit, no_description_text=no_description_text)
        cleaned_description = "\n\n".join(fallback_paragraphs).strip()
    elif recovery_path_used:
        LOGGER.info(
            "merge_path paragraph_recovery_used lang=%s expected=%d final=%d path=%s",
            language,
            expected_paragraphs,
            len(recovered_paragraphs),
            recovery_path_used,
        )
        if merge_run_summary is not None:
            merge_run_summary.record_paragraph_recovery_used()
    return dataclasses.replace(
        merged_content,
        description=cleaned_description,
        description_selected=cleaned_description,
        description_audit=cleaned_description,
    )


def extract_title_and_description_payload_or_none(raw_text: str) -> Optional[Tuple[str, str]]:
    payload: Optional[Dict[str, str]] = parse_merge_payload_plaintext_tolerant(raw_text)
    if payload is None:
        payload = parse_merge_payload_tolerant(raw_text)
    if payload is None:
        return None
    title_value: str = str(payload.get("merged_title") or "").strip()
    description_value: str = str(payload.get("merged_description") or "").strip()
    if not title_value or not description_value:
        return None
    return (title_value, description_value)
