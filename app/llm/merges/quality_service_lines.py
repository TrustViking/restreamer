from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from app.core.cta_detection import looks_like_cta_paragraph
from app.core.language import detect_language_from_text
from app.core.official_links import is_official_links_heading
from app.llm.merges.merge_constants import ALLOWED_BULLET_MARKERS, URL_LINE_PATTERN
from app.resources import canonical_service_lines

_PLAIN_BULLET_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
)
_HASHTAG_PATTERN: re.Pattern[str] = re.compile(r"(?:^|\s)(#[^\s#]+)")
_EMAIL_PATTERN: re.Pattern[str] = re.compile(r"\b\S+@\S+\.\S+\b", re.IGNORECASE)
_UKRAINIAN_HINTS: tuple[str, ...] = (
    "у цьому стрімі",
    "офіційні ресурси",
    "дивіться ефір",
    "діліться думками",
)
_RUSSIAN_HINTS: tuple[str, ...] = (
    "в этом стриме",
    "официальные ссылки",
    "смотрите эфир",
    "делитесь мнением",
)
_ENGLISH_HINTS: tuple[str, ...] = (
    "in this stream",
    "official links",
    "watch the stream",
    "share your thoughts",
)
_UKRAINIAN_WORD_HINTS: tuple[str, ...] = (
    " це ",
    " про ",
    " у ",
    " та ",
    " ефір",
    " стрімі",
)
_RUSSIAN_WORD_HINTS: tuple[str, ...] = (
    " это ",
    " про ",
    " эфир",
    " стриме",
    " этом ",
)

CANONICAL_SERVICE_LINES: dict[str, dict[str, str]] = canonical_service_lines()


@dataclass(frozen=True)
class MergeBlocks:
    hook: str
    lead_in: str
    theses_lines: List[str]
    links_heading: str
    links_urls: List[str]
    cta: str


def is_bullet_line(line: str) -> bool:
    stripped: str = str(line or "").strip()
    if not stripped:
        return False
    if looks_like_links_heading(stripped):
        return False
    return any(stripped.startswith(f"{marker} ") for marker in ALLOWED_BULLET_MARKERS) or bool(
        _PLAIN_BULLET_PATTERN.match(stripped)
    )


def looks_like_links_heading(line: str) -> bool:
    return is_official_links_heading(str(line or ""))


def canonical_service_line(language: str, key: str) -> str:
    return CANONICAL_SERVICE_LINES.get(language, CANONICAL_SERVICE_LINES["other"])[key]


def detect_service_language(text: str) -> str:
    normalized: str = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized:
        return "none"
    if any(hint in normalized for hint in _UKRAINIAN_HINTS):
        return "uk"
    if any(hint in normalized for hint in _RUSSIAN_HINTS):
        return "ru"
    if any(hint in normalized for hint in _ENGLISH_HINTS):
        return "en"
    return _detect_text_language(normalized)


def detect_paragraph_language(text: str) -> str:
    normalized: str = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) < 12:
        return "none"
    return _detect_text_language(normalized.lower())


def is_wrong_service_language(detected: str, expected: str) -> bool:
    if detected in {"none", "other"}:
        return False
    normalized_expected: str = expected if expected in {"uk", "en", "ru"} else "other"
    if normalized_expected == "other":
        return False
    return detected != normalized_expected


def is_short_service_line(text: str) -> bool:
    return len(re.sub(r"\s+", " ", str(text or "")).strip()) <= 140


def replace_cta_preserving_hashtags(text: str, language: str) -> str:
    hashtags: List[str] = [match.group(1) for match in _HASHTAG_PATTERN.finditer(str(text or ""))]
    base_text: str = canonical_service_line(language, "cta")
    if hashtags:
        return f"{base_text} {' '.join(hashtags)}".strip()
    return base_text


def extract_merge_blocks(text: str, *, language: str) -> MergeBlocks:
    paragraphs: List[str] = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]
    hook: str = ""
    lead_in: str = ""
    theses_lines: List[str] = []
    links_heading: str = ""
    links_urls: List[str] = []
    cta: str = ""

    for paragraph in paragraphs:
        lines: List[str] = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if not lines:
            continue
        if looks_like_links_heading(lines[0]):
            links_heading = lines[0]
            trailing_after_urls: List[str] = []
            for line in lines[1:]:
                if URL_LINE_PATTERN.match(line):
                    links_urls.append(line)
                else:
                    trailing_after_urls.append(line)
            if trailing_after_urls and not cta:
                cta = re.sub(r"\s+", " ", " ".join(trailing_after_urls)).strip()
            continue
        if any(is_bullet_line(line) for line in lines):
            first_bullet_index: int = next(
                index for index, line in enumerate(lines) if is_bullet_line(line)
            )
            bullet_end_index: int = first_bullet_index
            while bullet_end_index < len(lines) and is_bullet_line(lines[bullet_end_index]):
                bullet_end_index += 1
            if first_bullet_index > 0:
                if not theses_lines:
                    candidate_lead_in: str = lines[first_bullet_index - 1]
                    if is_bullet_line(candidate_lead_in):
                        lead_in = ""
                        theses_lines = list(lines[first_bullet_index:bullet_end_index])
                    else:
                        lead_in = candidate_lead_in
                        theses_lines = [lead_in, *lines[first_bullet_index:bullet_end_index]]
                else:
                    theses_lines.extend(lines[first_bullet_index:bullet_end_index])
                if not hook:
                    hook = " ".join(lines[: first_bullet_index - 1]).strip()
            else:
                if not theses_lines:
                    theses_lines = list(lines[first_bullet_index:bullet_end_index])
                else:
                    theses_lines.extend(lines[first_bullet_index:bullet_end_index])
                if not hook:
                    hook = ""
            trailing_lines: List[str] = lines[bullet_end_index:]
            if trailing_lines:
                if looks_like_links_heading(trailing_lines[0]):
                    links_heading = trailing_lines[0]
                    trailing_after_urls = []
                    for line in trailing_lines[1:]:
                        if URL_LINE_PATTERN.match(line):
                            links_urls.append(line)
                        else:
                            trailing_after_urls.append(line)
                    if trailing_after_urls and not cta:
                        cta = re.sub(r"\s+", " ", " ".join(trailing_after_urls)).strip()
                elif not cta:
                    cta = re.sub(r"\s+", " ", " ".join(trailing_lines)).strip()
            continue
        if looks_like_cta_paragraph(paragraph):
            cta = re.sub(r"\s+", " ", paragraph).strip()
            continue
        if not hook:
            hook = " ".join(lines).strip()
        elif not cta:
            cta = re.sub(r"\s+", " ", paragraph).strip()

    if not theses_lines:
        for paragraph in paragraphs[1:2]:
            lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
            if lines:
                theses_lines = lines
                lead_in = lines[0]
                break
    if not hook and paragraphs:
        hook = re.sub(r"\s+", " ", paragraphs[0]).strip()
    return MergeBlocks(
        hook=hook,
        lead_in=lead_in,
        theses_lines=theses_lines,
        links_heading=links_heading,
        links_urls=links_urls,
        cta=cta,
    )
def _detect_text_language(normalized_text: str) -> str:
    if re.search(r"[іїєґ]", normalized_text):
        return "uk"
    if re.search(r"[ыэъ]", normalized_text):
        return "ru"
    if any(hint in f" {normalized_text} " for hint in _UKRAINIAN_WORD_HINTS):
        return "uk"
    if any(hint in f" {normalized_text} " for hint in _RUSSIAN_WORD_HINTS):
        return "ru"
    return detect_language_from_text(normalized_text)
