from __future__ import annotations

import re

from app.core.official_links import is_official_links_heading
from app.core.text_utils import starts_with_any_prefix
from app.llm.merges.merge_constants import ALLOWED_BULLET_MARKERS, CTA_FIRST_PARAGRAPH_PREFIXES, URL_LINE_PATTERN
from app.resources.resource_loader import load_lines_resource

CTA_HINTS: tuple[str, ...] = load_lines_resource("lexicon_cta_hints.txt")
_PREFIX_HINTS: frozenset[str] = frozenset(
    {"долуч", "підпис", "подпис", "коментар", "комментар", "comment"}
)
_PLAIN_BULLET_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
)
_HASHTAG_TOKEN_RE: re.Pattern[str] = re.compile(r"(?<!\w)#[^\s#]+")
_CTA_HINT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(
        rf"(?<!\w){re.escape(hint)}"
        if hint in _PREFIX_HINTS
        else rf"(?<!\w){re.escape(hint)}(?!\w)"
    )
    for hint in CTA_HINTS
)
_BULLET_PREFIXES: tuple[str, ...] = tuple(f"{marker} " for marker in ALLOWED_BULLET_MARKERS)


def _contains_cta_hint(normalized_text: str) -> bool:
    return any(pattern.search(normalized_text) for pattern in _CTA_HINT_PATTERNS)


def starts_with_cta_prefix(text: str) -> bool:
    return starts_with_any_prefix(text, CTA_FIRST_PARAGRAPH_PREFIXES)


def looks_like_cta_line(text: str) -> bool:
    normalized_text: str = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized_text:
        return False
    return _contains_cta_hint(normalized_text)


def looks_like_cta_paragraph(text: str) -> bool:
    lines: list[str] = [line.strip() for line in str(text or "").split("\n") if line.strip()]
    if not lines or len(lines) > 2:
        return False
    if any(is_official_links_heading(line) for line in lines):
        return False
    if any(
        line.startswith(_BULLET_PREFIXES)
        or _PLAIN_BULLET_PATTERN.match(line)
        for line in lines
    ):
        return False
    if any(URL_LINE_PATTERN.fullmatch(line) for line in lines):
        return False
    normalized_text: str = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized_text:
        return False
    if _HASHTAG_TOKEN_RE.search(normalized_text) and len(normalized_text) <= 220:
        return True
    return _contains_cta_hint(normalized_text)
