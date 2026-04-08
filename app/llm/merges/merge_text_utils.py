from __future__ import annotations

import re
from typing import List

from app.core.description_cleaner import (
    _is_official_links_heading_line,
    _looks_like_service_tail_paragraph,
)
from app.core.text_utils import normalize_newlines, split_paragraphs
from app.llm.merges.merge_constants import ALLOWED_BULLET_MARKERS
from app.resources import merge_agenda_headings

_BULLET_PLAIN_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
)

def _extract_description_paragraphs_raw(text: str) -> List[str]:
    paragraphs: List[str] = split_paragraphs(text)
    return paragraphs

def _contains_agenda_heading(text: str) -> bool:
    agenda_headings: tuple[str, ...] = merge_agenda_headings()
    lines: List[str] = [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in normalize_newlines(text).split("\n")
        if line.strip()
    ]
    for line in lines:
        normalized_line: str = line.strip(" -–—:;.!?")
        for heading in agenda_headings:
            if (
                normalized_line == heading
                or normalized_line.startswith(f"{heading}:")
                or normalized_line.startswith(f"{heading} -")
                or normalized_line.startswith(f"{heading} –")
                or normalized_line.startswith(f"{heading} —")
            ):
                return True
    return False

def _looks_like_per_source_dump(text: str) -> bool:
    source_line_pattern: re.Pattern[str] = re.compile(
        r"^\s*(?:source|video)\s*\d+[:.)-]?",
        flags=re.IGNORECASE,
    )
    lines: List[str] = [
        line.strip()
        for line in normalize_newlines(text).split("\n")
        if line.strip()
    ]
    source_line_hits: int = sum(1 for line in lines if source_line_pattern.match(line))
    if source_line_hits >= 2:
        return True
    lowered_text: str = str(text or "").lower()
    return ("source 1" in lowered_text and "source 2" in lowered_text) or (
        "video 1" in lowered_text and "video 2" in lowered_text
    )

def _extract_named_entities(text: str) -> set[str]:
    entity_pattern: re.Pattern[str] = re.compile(
        r"\b(?:[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'-]{2,})(?:\s+[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'-]{2,})+\b",
        flags=re.UNICODE,
    )
    return {match.group(0).strip().lower() for match in entity_pattern.finditer(str(text or ""))}

def _bullet_marker_for_line(line: str) -> str:
    stripped: str = str(line or "").strip()
    if not stripped:
        return ""
    for marker in ALLOWED_BULLET_MARKERS:
        if stripped.startswith(f"{marker} "):
            return marker
    if _BULLET_PLAIN_PATTERN.match(stripped):
        first_token: str = stripped.split(maxsplit=1)[0]
        return first_token
    return ""
