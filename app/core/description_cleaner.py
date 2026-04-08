from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.core.constants import SEMANTIC_TOKEN_PATTERN, URL_PATTERN
from app.core.official_links import is_official_links_heading
from app.core.text_utils import normalize_multiline_text, split_paragraphs
from app.resources import merge_service_hints


@dataclass(frozen=True)
class _DescriptionCleanReport:
    text: str
    urls_removed: int
    hashtags_removed: int
    service_paragraphs_dropped: int


def _strip_source_urls_from_text(text: str) -> tuple[str, int]:
    removed_urls: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed_urls
        removed_urls += 1
        return ""

    cleaned_text: str = URL_PATTERN.sub(_replace, str(text or ""))
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return (cleaned_text.strip(" ,;:-"), removed_urls)


def _strip_source_hashtags_from_text(text: str) -> tuple[str, int]:
    hashtag_pattern: re.Pattern[str] = re.compile(r"(?<!\w)#[^\s#]+", flags=re.UNICODE)
    removed_hashtags: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed_hashtags
        removed_hashtags += 1
        return ""

    cleaned_text: str = hashtag_pattern.sub(_replace, str(text or ""))
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    return (cleaned_text.strip(" ,;:-"), removed_hashtags)


def _is_official_links_heading_line(text: str) -> bool:
    return is_official_links_heading(str(text or ""))


def _looks_like_service_tail_paragraph(text: str) -> bool:
    normalized_text: str = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not normalized_text:
        return True
    if _is_official_links_heading_line(normalized_text):
        return True
    service_hints: tuple[str, ...] = merge_service_hints()
    semantic_tokens: List[str] = SEMANTIC_TOKEN_PATTERN.findall(normalized_text)
    return (
        len(normalized_text) <= 220
        and len(semantic_tokens) <= 12
        and any(hint in normalized_text for hint in service_hints)
    )


def _clean_description_for_analysis_report(text: str) -> _DescriptionCleanReport:
    normalized_text: str = normalize_multiline_text(text)
    if not normalized_text:
        return _DescriptionCleanReport(
            text="",
            urls_removed=0,
            hashtags_removed=0,
            service_paragraphs_dropped=0,
        )

    cleaned_paragraphs: List[str] = []
    urls_removed: int = 0
    hashtags_removed: int = 0
    service_paragraphs_dropped: int = 0
    paragraph: str
    for paragraph in split_paragraphs(normalized_text):
        cleaned_lines: List[str] = []
        raw_line: str
        for raw_line in str(paragraph or "").split("\n"):
            line: str = str(raw_line or "").strip()
            if not line:
                continue
            cleaned_line, line_urls_removed = _strip_source_urls_from_text(line)
            cleaned_line, line_hashtags_removed = _strip_source_hashtags_from_text(
                cleaned_line
            )
            urls_removed += line_urls_removed
            hashtags_removed += line_hashtags_removed
            cleaned_line = re.sub(r"\s{2,}", " ", cleaned_line).strip(" ,;:-")
            if not cleaned_line or _is_official_links_heading_line(cleaned_line):
                continue
            cleaned_lines.append(cleaned_line)
        cleaned_paragraph: str = "\n".join(cleaned_lines).strip()
        if not cleaned_paragraph:
            service_paragraphs_dropped += 1
            continue
        cleaned_paragraphs.append(cleaned_paragraph)

    while cleaned_paragraphs and _looks_like_service_tail_paragraph(cleaned_paragraphs[-1]):
        cleaned_paragraphs.pop()
        service_paragraphs_dropped += 1

    cleaned_text: str = "\n\n".join(
        paragraph for paragraph in cleaned_paragraphs if paragraph.strip()
    ).strip()
    return _DescriptionCleanReport(
        text=cleaned_text,
        urls_removed=urls_removed,
        hashtags_removed=hashtags_removed,
        service_paragraphs_dropped=service_paragraphs_dropped,
    )


def clean_description_for_analysis(text: str) -> str:
    return _clean_description_for_analysis_report(text).text
