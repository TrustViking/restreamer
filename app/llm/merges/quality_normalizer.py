from __future__ import annotations

import re
from typing import List, Optional, Sequence

from app.core.text_utils import normalize_newlines
from app.llm.merges.merge_constants import (
    ACCENT_BULLET_MARKERS,
    ACCENT_MARKER_CAP,
    ALLOWED_BULLET_MARKERS,
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

_ROLE_LABELS: tuple[str, ...] = (
    "pastor",
    "bishop",
    "human rights defender",
    "international expert",
    "official representative",
    "founder",
    "co-founder",
    "chair",
    "chairman",
    "chairwoman",
    "spiritual leader",
    "activist",
    "advocate",
    "пастор",
    "єпископ",
    "епископ",
    "правозахисник",
    "правозащитник",
    "міжнародний експерт",
    "международный эксперт",
    "офіційний представник",
    "официальный представитель",
    "засновник",
    "основатель",
    "співзасновник",
    "сооснователь",
    "голова",
    "председатель",
    "духовний лідер",
    "духовный лидер",
    "активіст",
    "активист",
    "адвокат",
)
_ROLE_PATTERN: re.Pattern[str] = re.compile(
    r"^(?P<prefix>(?:🔹|📌|🎤|🎥|⚖|🌐|✅)\s+)?"
    r"(?P<label>"
    + "|".join(re.escape(label) for label in sorted(_ROLE_LABELS, key=len, reverse=True))
    + r")\s+"
    r"(?P<name>[A-ZА-ЯЁІЇЄҐ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ'`-]+(?:\s+[A-ZА-ЯЁІЇЄҐ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ'`-]+){0,2})"
    r"(?P<rest>\b.*)$",
    flags=re.IGNORECASE | re.UNICODE,
)


def normalize_merge_description(
    *,
    description: str,
    language: str,
    source_texts: Sequence[str],
    title: str = "",
) -> MergeQualityNormalizationResult:
    normalized_input: str = _normalize_text(description)
    if not normalized_input:
        diagnostics = build_diagnostics(
            description_text="",
            title=title,
            language=language,
            block_spacing_ok=True,
            accent_overflow=False,
            wrong_language_heading_detected=False,
            official_links_heading_mismatch=False,
            person_role_claims_detected=0,
            suspicious_role_labels_detected=(),
            role_softening_applied=False,
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

    (
        normalized_hook,
        normalized_theses_lines,
        normalized_cta,
        person_role_claims_detected,
        suspicious_role_labels_detected,
        role_softening_applied,
    ) = _soften_person_roles(
        hook=normalized_hook,
        theses_lines=normalized_theses_lines,
        cta=normalized_cta,
        source_texts=source_texts,
    )
    if role_softening_applied:
        normalization_applied = True

    normalized_description: str = _render_blocks(
        hook=normalized_hook,
        theses_lines=normalized_theses_lines,
        links_heading=normalized_links_heading,
        links_urls=normalized_links_urls,
        cta=normalized_cta,
    )
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
        person_role_claims_detected=person_role_claims_detected,
        suspicious_role_labels_detected=suspicious_role_labels_detected,
        role_softening_applied=role_softening_applied,
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


def _soften_person_roles(
    *,
    hook: str,
    theses_lines: Sequence[str],
    cta: str,
    source_texts: Sequence[str],
) -> tuple[str, List[str], str, int, tuple[str, ...], bool]:
    detected_count: int = 0
    labels_detected: List[str] = []
    changed: bool = False

    new_hook, hook_count, hook_labels, hook_changed = _soften_roles_in_text(hook, source_texts)
    detected_count += hook_count
    labels_detected.extend(hook_labels)
    changed = changed or hook_changed

    new_theses_lines: List[str] = []
    for line in theses_lines:
        new_line, line_count, line_labels, line_changed = _soften_roles_in_text(
            line,
            source_texts,
        )
        new_theses_lines.append(new_line)
        detected_count += line_count
        labels_detected.extend(line_labels)
        changed = changed or line_changed

    new_cta, cta_count, cta_labels, cta_changed = _soften_roles_in_text(cta, source_texts)
    detected_count += cta_count
    labels_detected.extend(cta_labels)
    changed = changed or cta_changed

    unique_labels: List[str] = []
    for label in labels_detected:
        if label not in unique_labels:
            unique_labels.append(label)
    return (
        new_hook,
        new_theses_lines,
        new_cta,
        detected_count,
        tuple(unique_labels),
        changed,
    )


def _soften_roles_in_text(
    text: str,
    source_texts: Sequence[str],
) -> tuple[str, int, List[str], bool]:
    if not text:
        return ("", 0, [], False)
    match: Optional[re.Match[str]] = _ROLE_PATTERN.match(text.strip())
    if match is None:
        return (text, 0, [], False)
    label: str = str(match.group("label") or "").strip().lower()
    name: str = str(match.group("name") or "").strip()
    if _role_label_is_supported(label=label, name=name, source_texts=source_texts):
        return (text, 1, [label], False)
    prefix: str = str(match.group("prefix") or "")
    rest: str = str(match.group("rest") or "").lstrip(" ,:-")
    softened: str = f"{prefix}{name}"
    if rest:
        softened = f"{softened} {rest}"
    return (softened.strip(), 1, [label], softened.strip() != text.strip())


def _role_label_is_supported(
    *,
    label: str,
    name: str,
    source_texts: Sequence[str],
) -> bool:
    if not source_texts:
        return False
    support_hits: int = 0
    exact_pattern: re.Pattern[str] = re.compile(
        rf"\b{re.escape(label)}\s+{re.escape(name.lower())}\b",
        flags=re.IGNORECASE | re.UNICODE,
    )
    label_pattern: re.Pattern[str] = re.compile(
        rf"\b{re.escape(label)}\b",
        flags=re.IGNORECASE | re.UNICODE,
    )
    for source_text in source_texts:
        normalized: str = str(source_text or "").lower()
        if exact_pattern.search(normalized) or label_pattern.search(normalized):
            support_hits += 1
        if support_hits >= 2:
            return True
    return False


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
