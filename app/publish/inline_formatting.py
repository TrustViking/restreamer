from __future__ import annotations

import re
from typing import List, Tuple

_BOLD_MARKER_PATTERN: re.Pattern[str] = re.compile(r"\*([^*]+)\*")


def parse_inline_bold(text: str) -> List[Tuple[str, bool]]:
    """Parse *bold* markers into segments.

    Returns list of (text, is_bold) tuples.
    If no markers found, returns single segment with is_bold=False.
    """
    segments: List[Tuple[str, bool]] = []
    last_end: int = 0
    for match in _BOLD_MARKER_PATTERN.finditer(text):
        if match.start() > last_end:
            segments.append((text[last_end:match.start()], False))
        segments.append((match.group(1), True))
        last_end = match.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))
    if not segments:
        segments.append((text, False))
    return segments


def strip_bold_markers(text: str) -> str:
    """Remove *bold* markers, returning clean text without asterisks."""
    return _BOLD_MARKER_PATTERN.sub(r"\1", text)


def has_bold_markers(text: str) -> bool:
    """Check if text contains *bold* markers."""
    return bool(_BOLD_MARKER_PATTERN.search(text))
