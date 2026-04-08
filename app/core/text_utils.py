from __future__ import annotations

import re
from typing import List, Sequence
from urllib.parse import urlsplit


_YOUTUBE_HOSTS: frozenset[str] = frozenset({
    "youtu.be", "www.youtu.be",
    "youtube.com", "www.youtube.com", "m.youtube.com",
})


def normalize_newlines(text: str) -> str:
    normalized_text: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    return normalized_text


def normalize_multiline_text(text: str) -> str:
    normalized_text: str = normalize_newlines(text).strip()
    return normalized_text


def split_paragraphs(text: str) -> List[str]:
    normalized_text: str = normalize_multiline_text(text)
    if not normalized_text:
        return []
    paragraphs: List[str] = [
        part.strip()
        for part in re.split(r"\n\s*\n", normalized_text)
        if part.strip()
    ]
    return paragraphs


def starts_with_any_prefix(
    text: str,
    prefixes: Sequence[str],
    *,
    use_casefold: bool = False,
    collapse_whitespace: bool = False,
) -> bool:
    normalized_text: str = str(text or "").strip()
    if collapse_whitespace:
        normalized_text = re.sub(r"\s+", " ", normalized_text)
    if not normalized_text:
        return False
    comparable_text: str = (
        normalized_text.casefold()
        if use_casefold
        else normalized_text.lower()
    )
    for raw_prefix in prefixes:
        normalized_prefix: str = str(raw_prefix or "").strip()
        if not normalized_prefix:
            continue
        comparable_prefix: str = (
            normalized_prefix.casefold()
            if use_casefold
            else normalized_prefix.lower()
        )
        if comparable_text.startswith(comparable_prefix):
            return True
    return False


def is_youtube_url(url: str) -> bool:
    cleaned: str = str(url or "").strip().strip("<>()[]{}").rstrip(".,;")
    if not cleaned:
        return False
    try:
        host: str = urlsplit(cleaned).netloc.strip().lower()
    except Exception:
        return False
    return host in _YOUTUBE_HOSTS


def is_youtube_host(host: str) -> bool:
    return str(host or "").strip().lower() in _YOUTUBE_HOSTS


def has_duplicate_paragraphs(
    text: str,
    *,
    jaccard_threshold: float = 0.72,
    min_tokens: int = 6,
) -> bool:
    paragraphs: List[str] = split_paragraphs(text)
    seen_normalized: set[str] = set()
    token_sets: List[set[str]] = []
    for paragraph in paragraphs:
        normalized: str = re.sub(r"\s+", " ", paragraph.strip().lower())
        token_list: List[str] = [token for token in normalized.split(" ") if token]
        if len(token_list) < min_tokens:
            continue
        if normalized in seen_normalized:
            return True
        seen_normalized.add(normalized)
        current_set: set[str] = set(token_list)
        for prev_set in token_sets:
            union_size: int = len(current_set | prev_set)
            if union_size == 0:
                continue
            intersection_size: int = len(current_set & prev_set)
            if intersection_size / union_size >= jaccard_threshold:
                return True
        token_sets.append(current_set)
    return False


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def utf16_len(text: str) -> int:
    """Return the length of *text* in UTF-16 code units.

    Google Docs API indices are counted in UTF-16 code units, not Python
    (Unicode code-point) units.  Characters outside the Basic Multilingual
    Plane (e.g. emoji, flag sequences) occupy two UTF-16 code units each
    (a surrogate pair) but count as one ``len()`` unit in Python.

    Use this function instead of ``len()`` whenever computing
    ``startIndex`` / ``endIndex`` values for Google Docs API requests.
    """
    return len(text.encode("utf-16-le")) // 2


def format_date_key_for_display(date_key: str) -> str:
    """Convert internal date_key like '270426' to display format '27.04.2026'.

    Returns date_key unchanged if parsing fails.
    """
    from datetime import datetime
    raw: str = str(date_key or "").strip()
    if not raw:
        return raw
    try:
        return datetime.strptime(raw, "%d%m%y").strftime("%d.%m.%Y")
    except ValueError:
        return raw
