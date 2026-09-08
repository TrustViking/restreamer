from __future__ import annotations

import re

# Structural detector: a heading line consists of a mandatory 🌐 prefix,
# arbitrary non-empty text in any language or script, and a trailing colon.
# This allows LLM-generated headings in any language to be recognized
# without maintaining a per-language phrase list.
#
# The 🌐 prefix is mandatory because without it the pattern would falsely
# match any random "Something:" line in the description body.
_HEADING_LINE_RE: re.Pattern[str] = re.compile(
    r"^\s*🌐\s*[^\s:][^:\n]*:\s*$"
)


def is_official_links_heading(text: str) -> bool:
    normalized_text: str = str(text or "").strip()
    if not normalized_text:
        return False
    return bool(_HEADING_LINE_RE.fullmatch(normalized_text))
