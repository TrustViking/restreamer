from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Sequence

from app.core.language import detect_language_from_text
from app.llm.merges.merge_constants import (
    ACCENT_BULLET_MARKERS,
    ACCENT_MARKER_CAP,
    ALLOWED_BULLET_MARKERS,
    BULLET_ABSOLUTE_MAX_CHAR_LIMIT,
    BULLET_OVERLOAD_CHAR_LIMIT,
    BULLET_OVERLOAD_NAME_LIMIT,
    NEUTRAL_BULLET_MARKER,
    URL_LINE_PATTERN,
    URL_PATTERN,
)
_PLAIN_BULLET_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
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
_HASHTAG_PATTERN: re.Pattern[str] = re.compile(r"(?:^|\s)(#[^\s#]+)")
_EMAIL_PATTERN: re.Pattern[str] = re.compile(r"\b\S+@\S+\.\S+\b", re.IGNORECASE)
_SCRIPT_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[A-Za-zА-Яа-яЁёІіЇїЄєҐґ][A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9'_-]*",
    flags=re.UNICODE,
)
_CYRILLIC_PATTERN: re.Pattern[str] = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]", re.UNICODE)
_LATIN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z]", re.UNICODE)
_ALLOWED_LATIN_SCRIPT_TOKENS: set[str] = {
    "ai",
    "api",
    "docs",
    "gpt",
    "google",
    "nasa",
    "openai",
    "telegram",
    "youtube",
}
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

CANONICAL_SERVICE_LINES: dict[str, dict[str, str]] = {
    "uk": {
        "lead_in": "У цьому стрімі ви побачите:",
        "links_heading": "🌐 Офіційні ресурси:",
        "cta": "Дивіться ефір і діліться думками.",
    },
    "en": {
        "lead_in": "In this stream you'll see:",
        "links_heading": "🌐 Official links:",
        "cta": "Watch the stream and share your thoughts.",
    },
    "ru": {
        "lead_in": "В этом стриме вы увидите:",
        "links_heading": "🌐 Официальные ссылки:",
        "cta": "Смотрите эфир и делитесь мнением.",
    },
    "other": {
        "lead_in": "In this stream you'll see:",
        "links_heading": "🌐 Official links:",
        "cta": "Watch the stream and share your thoughts.",
    },
}


@dataclass(frozen=True)
class MergeQualityDiagnostics:
    neutral_bullets_count: int
    accent_bullets_count: int
    accent_marker_types: tuple[str, ...]
    accent_overflow: bool
    block_spacing_ok: bool
    block_language_expected: str
    hook_language_detected: str
    lead_in_language_detected: str
    links_heading_language_detected: str
    cta_language_detected: str
    language_consistency_ok: bool
    wrong_language_heading_detected: bool
    official_links_heading_mismatch: bool
    person_role_claims_detected: int
    suspicious_role_labels_detected: tuple[str, ...]
    role_softening_applied: bool
    script_mix_detected: bool
    script_mix_suspects: tuple[str, ...]
    semantic_gate_status: str
    semantic_gate_reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class MergeQualityNormalizationResult:
    description_text: str
    diagnostics: MergeQualityDiagnostics
    normalization_applied: bool


def normalize_merge_description(
    *,
    description: str,
    language: str,
    source_texts: Sequence[str],
) -> MergeQualityNormalizationResult:
    normalized_input: str = _normalize_text(description)
    if not normalized_input:
        diagnostics = _build_diagnostics(
            description_text="",
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

    blocks: _MergeBlocks = _extract_merge_blocks(normalized_input, language=language)
    normalized_hook: str = blocks.hook
    normalized_theses_lines: List[str] = list(blocks.theses_lines)
    normalized_links_heading: str = blocks.links_heading
    normalized_links_urls: List[str] = list(blocks.links_urls)
    normalized_cta: str = blocks.cta

    wrong_language_heading_detected: bool = False
    official_links_heading_mismatch: bool = False
    normalization_applied: bool = False

    lead_in_language_detected: str = _detect_service_language(blocks.lead_in)
    if blocks.lead_in and _is_wrong_service_language(lead_in_language_detected, language):
        normalized_theses_lines = list(normalized_theses_lines)
        normalized_theses_lines[0] = _canonical_service_line(language, "lead_in")
        wrong_language_heading_detected = True
        normalization_applied = True

    links_heading_language_detected: str = _detect_service_language(blocks.links_heading)
    if blocks.links_heading:
        expected_heading: str = _canonical_service_line(language, "links_heading")
        if blocks.links_heading != expected_heading:
            official_links_heading_mismatch = True
        if _is_wrong_service_language(links_heading_language_detected, language) or blocks.links_heading != expected_heading:
            normalized_links_heading = expected_heading
            wrong_language_heading_detected = True
            normalization_applied = True

    cta_language_detected: str = _detect_service_language(blocks.cta)
    if blocks.cta and _is_short_service_line(blocks.cta) and _is_wrong_service_language(
        cta_language_detected,
        language,
    ):
        normalized_cta = _replace_cta_preserving_hashtags(blocks.cta, language)
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

    diagnostics: MergeQualityDiagnostics = _build_diagnostics(
        description_text=normalized_description,
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


def inspect_merge_description(
    *,
    description: str,
    language: str,
    source_texts: Sequence[str] = (),
) -> MergeQualityDiagnostics:
    return normalize_merge_description(
        description=description,
        language=language,
        source_texts=source_texts,
    ).diagnostics


@dataclass(frozen=True)
class _MergeBlocks:
    hook: str
    lead_in: str
    theses_lines: List[str]
    links_heading: str
    links_urls: List[str]
    cta: str


def _normalize_text(text: str) -> str:
    lines: List[str] = [line.rstrip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def _extract_merge_blocks(text: str, *, language: str) -> _MergeBlocks:
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
        if _looks_like_links_heading(lines[0]):
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
        if any(_is_bullet_line(line) for line in lines):
            first_bullet_index: int = next(
                index for index, line in enumerate(lines) if _is_bullet_line(line)
            )
            bullet_end_index: int = first_bullet_index
            while bullet_end_index < len(lines) and _is_bullet_line(lines[bullet_end_index]):
                bullet_end_index += 1
            if first_bullet_index > 0:
                if not theses_lines:
                    _candidate_lead_in: str = lines[first_bullet_index - 1]
                    if _is_bullet_line(_candidate_lead_in):
                        lead_in: str = ""
                        theses_lines = list(lines[first_bullet_index:bullet_end_index])
                    else:
                        lead_in = _candidate_lead_in
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
                if _looks_like_links_heading(trailing_lines[0]):
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
        if _looks_like_cta_paragraph(paragraph, language=language):
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
    return _MergeBlocks(
        hook=hook,
        lead_in=lead_in,
        theses_lines=theses_lines,
        links_heading=links_heading,
        links_urls=links_urls,
        cta=cta,
    )


def _is_bullet_line(line: str) -> bool:
    stripped: str = str(line or "").strip()
    if not stripped:
        return False
    if _looks_like_links_heading(stripped):
        return False
    return any(stripped.startswith(f"{marker} ") for marker in ALLOWED_BULLET_MARKERS) or bool(
        _PLAIN_BULLET_PATTERN.match(stripped)
    )


def _looks_like_links_heading(line: str) -> bool:
    normalized: str = str(line or "").strip().lower()
    normalized = normalized.lstrip("🌐").strip()
    return normalized in {
        "official links:",
        "офіційні ресурси:",
        "официальные ссылки:",
    }


def _looks_like_cta_paragraph(paragraph: str, *, language: str) -> bool:
    normalized: str = re.sub(r"\s+", " ", str(paragraph or "")).strip().lower()
    if not normalized:
        return False
    if any(hint in normalized for hint in _cta_hints(language)):
        return True
    return "#" in normalized and len(normalized) <= 180


def _cta_hints(language: str) -> tuple[str, ...]:
    if language == "uk":
        return ("дивіться", "долучайтеся", "діліться")
    if language == "ru":
        return ("смотрите", "присоединяйтесь", "делитесь")
    return ("watch", "join", "share")


def _canonical_service_line(language: str, key: str) -> str:
    return CANONICAL_SERVICE_LINES.get(language, CANONICAL_SERVICE_LINES["other"])[key]


def _detect_service_language(text: str) -> str:
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


def _detect_paragraph_language(text: str) -> str:
    normalized: str = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) < 12:
        return "none"
    return _detect_text_language(normalized.lower())


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


def _is_wrong_service_language(detected_language: str, expected_language: str) -> bool:
    if detected_language in {"none", "other"}:
        return False
    normalized_expected: str = expected_language if expected_language in {"uk", "en", "ru"} else "other"
    if normalized_expected == "other":
        return False
    return detected_language != normalized_expected


def _is_short_service_line(text: str) -> bool:
    return len(re.sub(r"\s+", " ", str(text or "")).strip()) <= 140


def _replace_cta_preserving_hashtags(text: str, language: str) -> str:
    hashtags: List[str] = [match.group(1) for match in _HASHTAG_PATTERN.finditer(str(text or ""))]
    base_text: str = _canonical_service_line(language, "cta")
    if hashtags:
        return f"{base_text} {' '.join(hashtags)}".strip()
    return base_text


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
        if index == 0 and not _is_bullet_line(line):
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


def _build_diagnostics(
    *,
    description_text: str,
    language: str,
    block_spacing_ok: bool,
    accent_overflow: bool,
    wrong_language_heading_detected: bool,
    official_links_heading_mismatch: bool,
    person_role_claims_detected: int,
    suspicious_role_labels_detected: tuple[str, ...],
    role_softening_applied: bool,
    accent_marker_types: tuple[str, ...] = (),
    neutral_bullets_count: int = 0,
    accent_bullets_count: int = 0,
) -> MergeQualityDiagnostics:
    blocks: _MergeBlocks = _extract_merge_blocks(description_text, language=language)
    hook_language_detected: str = _detect_paragraph_language(blocks.hook)
    lead_in_language_detected: str = _detect_service_language(blocks.lead_in)
    links_heading_language_detected: str = _detect_service_language(blocks.links_heading)
    cta_language_detected: str = _detect_service_language(blocks.cta)
    script_mix_suspects: tuple[str, ...] = _detect_script_mix_suspects(
        hook=blocks.hook,
        theses_lines=blocks.theses_lines,
        language=language,
    )

    reason_codes: List[str] = []
    if accent_overflow:
        reason_codes.append("accent_marker_overflow")
    if not block_spacing_ok:
        reason_codes.append("missing_block_spacing")
    if wrong_language_heading_detected:
        reason_codes.append("wrong_language_heading_detected")
    if official_links_heading_mismatch:
        reason_codes.append("official_links_heading_mismatch")
    if role_softening_applied:
        reason_codes.append("suspicious_role_softened")
    if script_mix_suspects:
        reason_codes.append("script_mix_contamination")
    if _core_language_mismatch(hook_language_detected, language):
        reason_codes.append("inconsistent_block_language")
    elif suspicious_role_labels_detected and not role_softening_applied:
        reason_codes.append("suspicious_person_role_labels_detected")

    if "inconsistent_block_language" in reason_codes or "script_mix_contamination" in reason_codes:
        semantic_gate_status = "hard_reject"
    elif any(
        code in reason_codes
        for code in (
            "accent_marker_overflow",
            "missing_block_spacing",
            "wrong_language_heading_detected",
            "official_links_heading_mismatch",
            "suspicious_role_softened",
        )
    ):
        semantic_gate_status = "needs_normalization"
    elif "suspicious_person_role_labels_detected" in reason_codes:
        semantic_gate_status = "warning"
    else:
        semantic_gate_status = "ok"

    language_consistency_ok: bool = "inconsistent_block_language" not in reason_codes
    return MergeQualityDiagnostics(
        neutral_bullets_count=neutral_bullets_count,
        accent_bullets_count=accent_bullets_count,
        accent_marker_types=accent_marker_types,
        accent_overflow=accent_overflow,
        block_spacing_ok=block_spacing_ok,
        block_language_expected=language,
        hook_language_detected=hook_language_detected,
        lead_in_language_detected=lead_in_language_detected,
        links_heading_language_detected=links_heading_language_detected,
        cta_language_detected=cta_language_detected,
        language_consistency_ok=language_consistency_ok,
        wrong_language_heading_detected=wrong_language_heading_detected,
        official_links_heading_mismatch=official_links_heading_mismatch,
        person_role_claims_detected=person_role_claims_detected,
        suspicious_role_labels_detected=suspicious_role_labels_detected,
        role_softening_applied=role_softening_applied,
        script_mix_detected=bool(script_mix_suspects),
        script_mix_suspects=script_mix_suspects,
        semantic_gate_status=semantic_gate_status,
        semantic_gate_reason_codes=tuple(reason_codes),
    )


def _core_language_mismatch(detected_language: str, expected_language: str) -> bool:
    if detected_language in {"none", "other"}:
        return False
    if expected_language not in {"uk", "en", "ru"}:
        return False
    return detected_language != expected_language


def _normalize_script_mix_probe_text(text: str) -> str:
    normalized_text: str = URL_PATTERN.sub(" ", str(text or ""))
    normalized_text = _EMAIL_PATTERN.sub(" ", normalized_text)
    normalized_text = _HASHTAG_PATTERN.sub(" ", normalized_text)
    return normalized_text


def _is_allowed_latin_token_in_cyrillic_text(token: str) -> bool:
    normalized_token: str = str(token or "").strip()
    if not normalized_token:
        return True
    lowercase_token: str = normalized_token.lower()
    if lowercase_token in _ALLOWED_LATIN_SCRIPT_TOKENS:
        return True
    if re.fullmatch(r"[A-Z0-9]{2,6}", normalized_token):
        return True
    if re.fullmatch(r"[A-Z][a-z]{1,14}", normalized_token):
        return True
    return False


def _is_suspicious_script_token(*, token: str, language: str) -> bool:
    normalized_token: str = str(token or "").strip()
    if not normalized_token:
        return False
    has_cyrillic: bool = bool(_CYRILLIC_PATTERN.search(normalized_token))
    has_latin: bool = bool(_LATIN_PATTERN.search(normalized_token))
    if not has_cyrillic and not has_latin:
        return False
    if has_cyrillic and has_latin:
        return True
    if language == "en":
        return has_cyrillic and len(_CYRILLIC_PATTERN.findall(normalized_token)) >= 2
    if language not in {"uk", "ru"}:
        return False
    if not has_latin or has_cyrillic:
        return False
    pure_latin_token: str = re.sub(r"[^A-Za-z]", "", normalized_token)
    if len(pure_latin_token) < 4:
        return False
    if _is_allowed_latin_token_in_cyrillic_text(normalized_token):
        return False
    return normalized_token == normalized_token.lower()


def _detect_script_mix_suspects(
    *,
    hook: str,
    theses_lines: Sequence[str],
    language: str,
) -> tuple[str, ...]:
    if language not in {"uk", "en", "ru"}:
        return ()
    suspect_tokens: List[str] = []
    for raw_text in (hook, *theses_lines):
        probe_text: str = _normalize_script_mix_probe_text(raw_text)
        for token in _SCRIPT_TOKEN_PATTERN.findall(probe_text):
            cleaned_token: str = str(token or "").strip()
            if not cleaned_token:
                continue
            if not _is_suspicious_script_token(token=cleaned_token, language=language):
                continue
            if cleaned_token not in suspect_tokens:
                suspect_tokens.append(cleaned_token)
            if len(suspect_tokens) >= 5:
                return tuple(suspect_tokens)
    return tuple(suspect_tokens)


_PROPER_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'`-]{1,25}"
    r"(?:\s+[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'`-]{1,25}){1,3}\b",
    re.UNICODE,
)


def count_overloaded_bullets(description: str) -> int:
    overloaded_count: int = 0
    for line in str(description or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        stripped: str = line.strip()
        if not stripped:
            continue
        is_bullet: bool = any(
            stripped.startswith(f"{marker} ")
            for marker in (NEUTRAL_BULLET_MARKER, *ACCENT_BULLET_MARKERS)
        )
        if not is_bullet:
            continue
        if len(stripped) > BULLET_ABSOLUTE_MAX_CHAR_LIMIT:
            overloaded_count += 1
            continue
        if len(stripped) <= BULLET_OVERLOAD_CHAR_LIMIT:
            continue
        names_found: int = len(_PROPER_NAME_PATTERN.findall(stripped))
        if names_found >= BULLET_OVERLOAD_NAME_LIMIT:
            overloaded_count += 1
    return overloaded_count
