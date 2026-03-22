"""Shared constants for the merge pipeline.

All bullet markers, CTA prefixes, URL patterns, stopwords, and threshold values
live here. Other modules import from this file - never define copies.
"""
from __future__ import annotations

import re

from app.resources.resource_loader import load_lines_resource

# ---------------------------------------------------------------------------
# Bullet markers
# ---------------------------------------------------------------------------
NEUTRAL_BULLET_MARKER: str = "🔹"
ACCENT_BULLET_MARKERS: tuple[str, ...] = ("📌", "🎤", "🎥", "⚖", "🌐", "✅")
ALLOWED_BULLET_MARKERS: tuple[str, ...] = (NEUTRAL_BULLET_MARKER, *ACCENT_BULLET_MARKERS)
ACCENT_MARKER_CAP: int = 3

# ---------------------------------------------------------------------------
# CTA / promotional prefixes
# ---------------------------------------------------------------------------
CTA_FIRST_PARAGRAPH_PREFIXES: tuple[str, ...] = load_lines_resource("lexicon_cta_prefixes.txt")

# ---------------------------------------------------------------------------
# URL patterns
# ---------------------------------------------------------------------------
URL_PATTERN: re.Pattern[str] = re.compile(r"https?://\S+", flags=re.IGNORECASE)
URL_LINE_PATTERN: re.Pattern[str] = re.compile(r"^https?://\S+$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Semantic token analysis
# ---------------------------------------------------------------------------
SEMANTIC_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]{3,}",
    flags=re.UNICODE,
)

SEMANTIC_STOPWORDS: frozenset[str] = frozenset(load_lines_resource("lexicon_semantic_stopwords.txt"))

# ---------------------------------------------------------------------------
# Bullet overload thresholds
# ---------------------------------------------------------------------------
BULLET_OVERLOAD_CHAR_LIMIT: int = 280
BULLET_OVERLOAD_NAME_LIMIT: int = 3
BULLET_ABSOLUTE_MAX_CHAR_LIMIT: int = 500

# ---------------------------------------------------------------------------
# Merge orchestration
# ---------------------------------------------------------------------------
PRIMARY_ATTEMPTS: int = 2
PRIMARY_ATTEMPTS_EXTENDED: int = 3
STYLE_CONTRACT_VERSION: str = "v4_merge_quality_hardening"

RECOVERABLE_REJECT_CODES: frozenset[str] = frozenset({
    "overloaded_bullet",
    "compact_bullet_overflow",
    "duplicate_paragraph",
    "hook_echo_in_body",
    "cta_as_first_paragraph",
    "cta_in_hook",
    "paragraph_underflow",
    "paragraph_overflow",
})
