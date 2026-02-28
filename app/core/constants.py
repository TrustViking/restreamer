from __future__ import annotations

import re

LOGGER_NAME_ENV_VAR: str = "RESTREAMER_LOGGER_NAME"
LOGGER_NAME_DEFAULT: str = "restreamer"
_SOURCE_URL_LINE_RE: re.Pattern[str] = re.compile(r"^https?://\\S+$", re.IGNORECASE)
URL_PATTERN: re.Pattern[str] = re.compile(r"https?://\\S+", re.IGNORECASE)
MERGED_DESCRIPTION_HARD_CEILING: int = 4500
