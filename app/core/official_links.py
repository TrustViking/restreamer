from __future__ import annotations

import re

# Must match entries in app/resources/text/official_links_headings.json.
OFFICIAL_LINKS_HEADING_RE: re.Pattern[str] = re.compile(
    r"(?im)^\s*(?:🌐\s*)?(?:official links|офіційні ресурси|официальные ссылки)\s*:\s*$"
)


def is_official_links_heading(text: str) -> bool:
    normalized_text: str = str(text or "").strip()
    if not normalized_text:
        return False
    return bool(OFFICIAL_LINKS_HEADING_RE.fullmatch(normalized_text))
