from __future__ import annotations

import re

from app.resources.resource_loader import load_lines_resource

LOGGER_NAME_ENV_VAR: str = "LOGGER_NAME"
LOGGER_NAME_DEFAULT: str = "pipeline"
MERGED_DESCRIPTION_HARD_CEILING: int = 4500

# Bullet markers (structural, shared across core and llm)
NEUTRAL_BULLET_MARKER: str = "🔹"
ACCENT_BULLET_MARKERS: tuple[str, ...] = ("📌", "🎤", "🎥", "⚖", "🌐", "✅")
ALLOWED_BULLET_MARKERS: tuple[str, ...] = (NEUTRAL_BULLET_MARKER, *ACCENT_BULLET_MARKERS)
ACCENT_MARKER_CAP: int = 3

# CTA prefixes
CTA_FIRST_PARAGRAPH_PREFIXES: tuple[str, ...] = load_lines_resource("lexicon_cta_prefixes.txt")

# URL patterns
URL_PATTERN: re.Pattern[str] = re.compile(r"https?://\S+", flags=re.IGNORECASE)
URL_LINE_PATTERN: re.Pattern[str] = re.compile(r"^https?://\S+$", re.IGNORECASE)

# Semantic analysis
SEMANTIC_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]{3,}",
    flags=re.UNICODE,
)
SEMANTIC_STOPWORDS: frozenset[str] = frozenset(load_lines_resource("lexicon_semantic_stopwords.txt"))
