"""Constants for merge pipeline internals.

Shared structural constants are re-exported from ``app.core.constants`` for
backward compatibility and to avoid circular imports from ``app.core``.
"""
from __future__ import annotations

from app.core.constants import (
    ACCENT_BULLET_MARKERS,
    ACCENT_MARKER_CAP,
    ALLOWED_BULLET_MARKERS,
    CTA_FIRST_PARAGRAPH_PREFIXES,
    NEUTRAL_BULLET_MARKER,
    SEMANTIC_STOPWORDS,
    SEMANTIC_TOKEN_PATTERN,
    URL_LINE_PATTERN,
    URL_PATTERN,
)

BULLET_OVERLOAD_CHAR_LIMIT: int = 280
BULLET_OVERLOAD_NAME_LIMIT: int = 3
BULLET_ABSOLUTE_MAX_CHAR_LIMIT: int = 500

# Compact-mode bullet bounds. Used by the merge prompt contract,
# the validator's compact_bullet_overflow reject, and the normalizer's
# deterministic trim repair. Keep these three usages in sync via this constant.
COMPACT_BULLET_MIN: int = 4
COMPACT_BULLET_MAX: int = 7

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
