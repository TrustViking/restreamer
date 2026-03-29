from __future__ import annotations

import re
from typing import Final, List, Optional

from app.bootstrap.logging_config import get_logger


LOGGER = get_logger(__name__)

HASHTAG_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"#\S+")
NON_WHITESPACE_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"\S+")
HASHTAG_LETTER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[^\W\d_]")
HASHTAG_DIGITS_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d+")
TRAILING_SEPARATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?:\s*[|—–-]+\s*)+$")


def _find_trailing_hashtag_zone_start(title: str) -> Optional[int]:
    token_matches: List[re.Match[str]] = list(NON_WHITESPACE_TOKEN_PATTERN.finditer(title))
    if not token_matches:
        return None

    zone_start: Optional[int] = None
    for match in reversed(token_matches):
        token: str = match.group(0)
        if HASHTAG_TOKEN_PATTERN.fullmatch(token) is None:
            break
        zone_start = match.start()

    return zone_start


def sanitize_source_video_title(raw_title: str) -> str:
    stripped_title: str = raw_title.strip()
    if not stripped_title:
        return ""

    zone_start: Optional[int] = _find_trailing_hashtag_zone_start(stripped_title)
    if zone_start is None:
        return stripped_title

    prefix: str = stripped_title[:zone_start].rstrip()
    trailing_zone: str = stripped_title[zone_start:]
    zone_tokens: List[str] = [match.group(0) for match in NON_WHITESPACE_TOKEN_PATTERN.finditer(trailing_zone)]

    kept_tokens: List[str] = []
    removed_count: int = 0
    for token in zone_tokens:
        hashtag_body: str = token[1:]
        has_letter: bool = HASHTAG_LETTER_PATTERN.search(hashtag_body) is not None
        is_protected: bool = HASHTAG_DIGITS_PATTERN.fullmatch(hashtag_body) is not None

        if is_protected:
            kept_tokens.append(token)
            continue
        if has_letter:
            removed_count += 1
            continue

        kept_tokens.append(token)

    if kept_tokens:
        kept_suffix: str = " ".join(kept_tokens)
        result: str = f"{prefix} {kept_suffix}" if prefix else kept_suffix
    else:
        result = prefix

    result = TRAILING_SEPARATOR_PATTERN.sub("", result)
    result = result.strip()

    if removed_count > 0:
        LOGGER.debug(
            "title_hashtag_cleanup: removed %d hashtag(s) from title: %r -> %r",
            removed_count,
            raw_title,
            result,
        )

    return result
