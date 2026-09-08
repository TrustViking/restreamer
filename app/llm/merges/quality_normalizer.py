from __future__ import annotations

import logging
import re
from typing import List, Sequence

from app.core.text_utils import normalize_newlines
from app.llm.merges.merge_constants import (
    ACCENT_BULLET_MARKERS,
    ACCENT_MARKER_CAP,
    ALLOWED_BULLET_MARKERS,
    COMPACT_BULLET_MAX,
    NEUTRAL_BULLET_MARKER,
)
from app.llm.merges.quality_diagnostics import (
    MergeQualityNormalizationResult,
    build_diagnostics,
)
from app.llm.merges.quality_service_lines import (
    _PLAIN_BULLET_PATTERN,
    canonical_service_line,
    detect_service_language,
    extract_merge_blocks,
    is_bullet_line,
    is_short_service_line,
    is_wrong_service_language,
    looks_like_links_heading,
    replace_cta_preserving_hashtags,
)

LOGGER = logging.getLogger(__name__)


def normalize_merge_description(
    *,
    description: str,
    language: str,
    source_texts: Sequence[str],
    title: str = "",
    source_count: int = 0,
) -> MergeQualityNormalizationResult:
    # source_count gates compact bullet trimming; 0 means unknown / do not trim.
    normalized_input: str = _normalize_text(description)
    del source_texts
    if not normalized_input:
        diagnostics = build_diagnostics(
            description_text="",
            title=title,
            language=language,
            block_spacing_ok=True,
            accent_overflow=False,
            wrong_language_heading_detected=False,
            official_links_heading_mismatch=False,
        )
        return MergeQualityNormalizationResult(
            description_text="",
            diagnostics=diagnostics,
            normalization_applied=False,
        )

    blocks = extract_merge_blocks(normalized_input, language=language)
    normalized_hook: str = blocks.hook
    normalized_theses_lines: List[str] = list(blocks.theses_lines)
    normalized_links_heading: str = blocks.links_heading
    normalized_links_urls: List[str] = list(blocks.links_urls)
    normalized_cta: str = blocks.cta
    if (
        normalized_hook
        and normalized_cta
        and normalized_hook == normalized_cta
        and not normalized_theses_lines
        and not normalized_links_heading
    ):
        normalized_cta = ""
    if (
        normalized_hook
        and len(normalized_theses_lines) == 1
        and normalized_theses_lines[0] == normalized_hook
        and not normalized_links_heading
    ):
        normalized_hook = ""

    wrong_language_heading_detected: bool = False
    official_links_heading_mismatch: bool = False
    normalization_applied: bool = False

    lead_in_language_detected: str = detect_service_language(blocks.lead_in)
    if blocks.lead_in and is_wrong_service_language(lead_in_language_detected, language):
        normalized_theses_lines = list(normalized_theses_lines)
        normalized_theses_lines[0] = canonical_service_line(language, "lead_in")
        wrong_language_heading_detected = True
        normalization_applied = True

    links_heading_language_detected: str = detect_service_language(blocks.links_heading)
    if blocks.links_heading:
        expected_heading: str = canonical_service_line(language, "links_heading")
        if blocks.links_heading != expected_heading:
            official_links_heading_mismatch = True
        if is_wrong_service_language(links_heading_language_detected, language) or blocks.links_heading != expected_heading:
            normalized_links_heading = expected_heading
            wrong_language_heading_detected = True
            normalization_applied = True

    cta_language_detected: str = detect_service_language(blocks.cta)
    if blocks.cta and is_short_service_line(blocks.cta) and is_wrong_service_language(
        cta_language_detected,
        language,
    ):
        normalized_cta = replace_cta_preserving_hashtags(blocks.cta, language)
        normalization_applied = True

    (
        normalized_theses_lines,
        accent_overflow,
        accent_marker_types,
        neutral_bullets_count,
        accent_bullets_count,
        bullet_changed,
    ) = _normalize_bullets(normalized_theses_lines)
    if bullet_changed:
        normalization_applied = True
    trimmed_theses_lines, trim_applied, bullets_before_trim, bullets_after_trim = (
        _trim_compact_bullet_overflow(
            normalized_theses_lines,
            source_count=source_count,
        )
    )
    if trim_applied:
        normalized_theses_lines = trimmed_theses_lines
        normalization_applied = True
        LOGGER.info(
            "merge_compact_bullet_trimmed source_count=%d bullets_before=%d bullets_after=%d cap=%d",
            source_count,
            bullets_before_trim,
            bullets_after_trim,
            COMPACT_BULLET_MAX,
        )

    normalized_description: str = _render_blocks(
        hook=normalized_hook,
        theses_lines=normalized_theses_lines,
        links_heading=normalized_links_heading,
        links_urls=normalized_links_urls,
        cta=normalized_cta,
    )
    # NOTE: имя `block_spacing_ok` историческое и неточное.
    # Здесь мы фиксируем не «правильность отступов между блоками», а сам факт
    # того, что normalizer ВООБЩЕ изменил текст по сравнению с входом LLM
    # (роли, отступы, бракованные фрагменты — что угодно). Любая правка → False.
    # Reject-код `missing_block_spacing` в quality_diagnostics.py поднимается
    # именно отсюда. Переименование кода — отдельная задача и в этом коммите не делается.
    block_spacing_ok: bool = normalized_description == normalized_input
    if not block_spacing_ok:
        normalization_applied = True

    diagnostics = build_diagnostics(
        description_text=normalized_description,
        title=title,
        language=language,
        block_spacing_ok=block_spacing_ok,
        accent_overflow=accent_overflow,
        wrong_language_heading_detected=wrong_language_heading_detected,
        official_links_heading_mismatch=official_links_heading_mismatch,
        accent_marker_types=accent_marker_types,
        neutral_bullets_count=neutral_bullets_count,
        accent_bullets_count=accent_bullets_count,
    )
    return MergeQualityNormalizationResult(
        description_text=normalized_description,
        diagnostics=diagnostics,
        normalization_applied=normalization_applied,
    )


def _normalize_text(text: str) -> str:
    lines: List[str] = [line.rstrip() for line in normalize_newlines(text).split("\n")]
    return "\n".join(lines).strip()


def _normalize_bullets(
    theses_lines: Sequence[str],
) -> tuple[List[str], bool, tuple[str, ...], int, int, bool]:
    if not theses_lines:
        return ([], False, (), 0, 0, False)
    normalized_lines: List[str] = []
    accent_overflow: bool = False
    accent_count: int = 0
    neutral_count: int = 0
    accent_marker_types: List[str] = []
    changed: bool = False

    for index, line in enumerate(theses_lines):
        if index == 0 and not is_bullet_line(line):
            normalized_lines.append(re.sub(r"\s+", " ", str(line or "")).strip())
            continue
        marker, content = _split_bullet_marker(line)
        next_marker: str = marker
        if marker not in ALLOWED_BULLET_MARKERS:
            next_marker = NEUTRAL_BULLET_MARKER
        if next_marker in ACCENT_BULLET_MARKERS:
            if accent_count >= ACCENT_MARKER_CAP:
                next_marker = NEUTRAL_BULLET_MARKER
                accent_overflow = True
            else:
                accent_count += 1
                if next_marker not in accent_marker_types:
                    accent_marker_types.append(next_marker)
        if next_marker == NEUTRAL_BULLET_MARKER:
            neutral_count += 1
        normalized_line: str = f"{next_marker} {content}".strip()
        if normalized_line != str(line or "").strip():
            changed = True
        normalized_lines.append(normalized_line)
    return (
        normalized_lines,
        accent_overflow,
        tuple(accent_marker_types),
        neutral_count,
        accent_count,
        changed,
    )


def _split_bullet_marker(line: str) -> tuple[str, str]:
    stripped: str = str(line or "").strip()
    for marker in ALLOWED_BULLET_MARKERS:
        if stripped.startswith(f"{marker} "):
            return (marker, stripped[len(marker) :].strip())
    content: str = _PLAIN_BULLET_PATTERN.sub("", stripped, count=1).strip()
    if content:
        return ("plain", content)
    return (NEUTRAL_BULLET_MARKER, stripped)


def _trim_compact_bullet_overflow(
    theses_lines: Sequence[str],
    *,
    source_count: int,
) -> tuple[List[str], bool, int, int]:
    """Trim trailing bullets when compact contract is exceeded.

    Returns:
        (trimmed_lines, trim_applied, bullets_before, bullets_after)

    Trim is applied only when:
      - source_count > 0 and source_count <= 2 (compact mode), AND
      - the count of bullet lines (per is_bullet_line) > COMPACT_BULLET_MAX.

    Trim policy:
      - Walk lines left to right; keep all non-bullet lines as-is and in place.
      - Keep the first COMPACT_BULLET_MAX bullets in their original order.
      - Drop every bullet after that, but keep any subsequent non-bullet
        lines that were already there (defensive - there should not be any
        in a well-formed thesis block, but do not silently delete them).
      - Do not modify bullet text.

    If no trim is needed, returns (list(theses_lines), False,
    bullet_count, bullet_count).
    """
    lines: List[str] = list(theses_lines)
    bullet_count: int = sum(1 for line in lines if is_bullet_line(line))
    if source_count <= 0 or source_count > 2 or bullet_count <= COMPACT_BULLET_MAX:
        return (lines, False, bullet_count, bullet_count)

    trimmed_lines: List[str] = []
    kept_bullets: int = 0
    for line in lines:
        if not is_bullet_line(line):
            trimmed_lines.append(line)
            continue
        if kept_bullets >= COMPACT_BULLET_MAX:
            continue
        trimmed_lines.append(line)
        kept_bullets += 1
    return (trimmed_lines, True, bullet_count, kept_bullets)


def _render_blocks(
    *,
    hook: str,
    theses_lines: Sequence[str],
    links_heading: str,
    links_urls: Sequence[str],
    cta: str,
) -> str:
    paragraphs: List[str] = []
    if hook:
        paragraphs.append(re.sub(r"\s+", " ", hook).strip())
    if theses_lines:
        paragraphs.append("\n".join(line.strip() for line in theses_lines if line.strip()).strip())
    if links_heading:
        links_lines: List[str] = [links_heading.strip()]
        for item in links_urls:
            url: str = item.strip()
            if url:
                if url.endswith("/") and url.count("/") == 3:
                    url = url.rstrip("/")
                links_lines.append(url)
        paragraphs.append("\n".join(links_lines).strip())
    if cta:
        paragraphs.append(re.sub(r"\s+", " ", cta).strip())
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip()
