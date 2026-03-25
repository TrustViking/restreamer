from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence

from app.llm.merges.merge_constants import (
    ACCENT_BULLET_MARKERS,
    BULLET_ABSOLUTE_MAX_CHAR_LIMIT,
    BULLET_OVERLOAD_CHAR_LIMIT,
    BULLET_OVERLOAD_NAME_LIMIT,
    NEUTRAL_BULLET_MARKER,
    URL_PATTERN,
)
from app.llm.merges.quality_service_lines import (
    MergeBlocks,
    detect_paragraph_language,
    detect_service_language,
    extract_merge_blocks,
)

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
_HASHTAG_PATTERN: re.Pattern[str] = re.compile(r"(?:^|\s)(#[^\s#]+)")
_EMAIL_PATTERN: re.Pattern[str] = re.compile(r"\b\S+@\S+\.\S+\b", re.IGNORECASE)
_PROPER_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'`-]{1,25}"
    r"(?:\s+[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'`-]{1,25}){1,3}\b",
    re.UNICODE,
)


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


def build_diagnostics(
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
    blocks: MergeBlocks = extract_merge_blocks(description_text, language=language)
    hook_language_detected: str = detect_paragraph_language(blocks.hook)
    lead_in_language_detected: str = detect_service_language(blocks.lead_in)
    links_heading_language_detected: str = detect_service_language(blocks.links_heading)
    cta_language_detected: str = detect_service_language(blocks.cta)
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


def _core_language_mismatch(detected: str, expected: str) -> bool:
    if detected in {"none", "other"}:
        return False
    if expected not in {"uk", "en", "ru"}:
        return False
    return detected != expected


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
