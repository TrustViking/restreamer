from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.official_links import is_official_links_heading
from app.core.text_utils import split_paragraphs
from app.core.url_utils import normalize_official_link_display
from app.resources import resolve_official_links_heading, resolve_recommended_materials_heading
from app.publish.sanitizers.url_selector import (
    _sanitize_source_url,
    _is_youtube_url,
    _dedupe_nonempty,
)
from app.publish.sanitizers.tail_parser import TailParser


LOGGER = _get_logger_impl(__name__)


@dataclass(frozen=True)
class _OfficialLinksExtractionResult:
    cleaned_text: str
    heading_found: bool
    source_urls: List[str]
    empty_blocks_suppressed: int


def _extract_official_links_url_lines(lines: Sequence[str]) -> List[str]:
    source_urls: List[str] = []
    for line in lines:
        normalized_line: str = str(line or "").strip()
        if not normalized_line:
            continue
        if not TailParser.is_source_url_line(normalized_line):
            return []
        sanitized_url: Optional[str] = _sanitize_source_url(normalized_line)
        if sanitized_url is None or _is_youtube_url(sanitized_url):
            continue
        source_urls.append(sanitized_url)
    return _dedupe_nonempty(source_urls)


def _extract_official_links_from_heading_paragraph(
    paragraph: str,
) -> tuple[bool, List[str]]:
    lines: List[str] = [str(line or "").strip() for line in str(paragraph or "").splitlines()]
    nonempty_lines: List[str] = [line for line in lines if line]
    if not nonempty_lines:
        return (False, [])
    if not is_official_links_heading(nonempty_lines[0]):
        return (False, [])
    return (True, _extract_official_links_url_lines(nonempty_lines[1:]))


class DescriptionComposer:
    """Composes final publication text from body, CTA, hashtags, URLs."""

    def extract_official_links(self, text: str) -> _OfficialLinksExtractionResult:
        """Extract and remove official links blocks from text."""
        paragraphs: List[str] = split_paragraphs(text)
        if not paragraphs:
            return _OfficialLinksExtractionResult(
                cleaned_text="",
                heading_found=False,
                source_urls=[],
                empty_blocks_suppressed=0,
            )
        kept_paragraphs: List[str] = []
        extracted_source_urls: List[str] = []
        suppressed_count: int = 0
        heading_found: bool = False
        paragraph_index: int = 0
        while paragraph_index < len(paragraphs):
            paragraph: str = paragraphs[paragraph_index]
            normalized_paragraph: str = str(paragraph or "").strip()
            paragraph_heading_found, paragraph_source_urls = _extract_official_links_from_heading_paragraph(
                normalized_paragraph
            )
            if not paragraph_heading_found:
                kept_paragraphs.append(normalized_paragraph)
                paragraph_index += 1
                continue
            heading_found = True
            if paragraph_source_urls:
                extracted_source_urls.extend(paragraph_source_urls)
                paragraph_index += 1
                continue
            next_paragraph: str = (
                str(paragraphs[paragraph_index + 1] or "").strip()
                if paragraph_index + 1 < len(paragraphs)
                else ""
            )
            next_paragraph_source_urls: List[str] = _extract_official_links_url_lines(
                next_paragraph.splitlines()
            )
            if next_paragraph_source_urls:
                extracted_source_urls.extend(next_paragraph_source_urls)
                paragraph_index += 2
                continue
            suppressed_count += 1
            paragraph_index += 1
        return _OfficialLinksExtractionResult(
            cleaned_text="\n\n".join(paragraph for paragraph in kept_paragraphs if paragraph).strip(),
            heading_found=heading_found,
            source_urls=_dedupe_nonempty(extracted_source_urls),
            empty_blocks_suppressed=suppressed_count,
        )

    @staticmethod
    def resolve_layout(
        *,
        body_text: str,
        cta_text: str,
        hashtags_line: str,
        recommended_youtube_urls: Sequence[str],
        source_urls: Sequence[str],
    ) -> str:
        """Was `_resolve_tail_layout`."""
        layout_parts: List[str] = []
        if body_text:
            layout_parts.append("body")
        if recommended_youtube_urls:
            layout_parts.extend(("blank", "recommended_materials"))
        if source_urls:
            layout_parts.extend(("blank", "official_links"))
        if cta_text:
            layout_parts.extend(("blank", "cta"))
        if hashtags_line:
            layout_parts.extend(("blank", "hashtags"))
        return "_".join(layout_parts) if layout_parts else "empty"

    def render_recommended_block(
        self,
        *,
        language: str,
        recommended_youtube_urls: Sequence[str],
        fetch_title_fn: Callable[[str], Optional[str]],
    ) -> str:
        """Render the Recommended Materials block."""
        if not recommended_youtube_urls:
            return ""
        rendered_entries: List[str] = []
        titles_rendered: int = 0
        for recommended_url in recommended_youtube_urls:
            cleaned_url: str = str(recommended_url or "").strip()
            if not cleaned_url:
                continue
            try:
                title_text: Optional[str] = fetch_title_fn(cleaned_url)
            except Exception as error:
                LOGGER.info(
                    "recommended_title_fetch_failed url=%s reason=%s",
                    cleaned_url,
                    error,
                )
                title_text = None
            if title_text:
                rendered_entries.append(f"✅ {title_text}\n👉 {cleaned_url}")
                titles_rendered += 1
                continue
            rendered_entries.append(f"👉 {cleaned_url}")
        if not rendered_entries:
            return ""
        block_text: str = "\n\n".join(
            [resolve_recommended_materials_heading(language), *rendered_entries]
        ).strip()
        LOGGER.info(
            "recommended_block_rendered_with_titles urls=%d titles_rendered=%d title_fetch_failures=%d",
            len(rendered_entries),
            titles_rendered,
            len(rendered_entries) - titles_rendered,
        )
        return block_text

    def compose(
        self,
        *,
        language: str,
        body_text: str,
        cta_text: str,
        hashtags_line: str,
        recommended_youtube_urls: Sequence[str],
        source_urls: Sequence[str],
        fetch_title_fn: Callable[[str], Optional[str]],
    ) -> str:
        """Compose final publication text."""
        parts: List[str] = []
        if body_text:
            parts.append(body_text.strip())
        if recommended_youtube_urls:
            recommended_block_text: str = self.render_recommended_block(
                language=language,
                recommended_youtube_urls=recommended_youtube_urls,
                fetch_title_fn=fetch_title_fn,
            )
            if recommended_block_text:
                parts.append(recommended_block_text)
        if source_urls:
            parts.append(
                "\n".join(
                    [resolve_official_links_heading(language)]
                    + [
                        normalize_official_link_display(str(source_url or "").strip())
                        for source_url in source_urls
                        if str(source_url or "").strip()
                    ]
                ).strip()
            )
        if cta_text:
            parts.append(cta_text.strip())
        if hashtags_line:
            parts.append(hashtags_line.strip())
        return "\n\n".join(part for part in parts if part).strip()
