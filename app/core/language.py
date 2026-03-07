from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import VideoMetadata

LOGGER = _get_logger_impl(__name__)

_LANGUAGE_ALIASES: dict[str, str] = {
    "ua": "uk",
    "uk": "uk",
    "ukr": "uk",
    "en": "en",
    "eng": "en",
    "ru": "ru",
    "rus": "ru",
}
_LANGUAGE_DECISION_SOURCES: set[str] = {
    "sheet_override",
    "metadata_agreement",
    "text_probe_override",
    "text_probe_only",
    "metadata_only",
    "fallback_default",
}


@dataclass(frozen=True)
class LanguageDecision:
    youtube_language_raw: str
    channel_language_raw: str
    metadata_language_candidates: tuple[str, ...]
    text_probe_language: str
    text_probe_confidence: str
    final_language: str
    language_decision_source: str
    language_conflict: bool


def normalize_language(raw_language: Optional[str]) -> Optional[str]:
    normalized_input: str = str(raw_language or "").strip().lower()
    if not normalized_input:
        return None
    if normalized_input in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[normalized_input]
    for prefix, canonical in (("uk", "uk"), ("ua", "uk"), ("en", "en"), ("ru", "ru")):
        if normalized_input.startswith(prefix):
            return canonical
    return None


def _text_probe_language_with_confidence(text: str) -> tuple[str, str]:
    low: str = str(text or "").lower()
    if not low:
        return ("other", "none")

    ukrainian_specific_count: int = len(re.findall(r"[іїєґ]", low))
    russian_specific_count: int = len(re.findall(r"[ыэъ]", low))
    cyrillic_count: int = len(re.findall(r"[а-яё]", low))
    latin_count: int = len(re.findall(r"[a-z]", low))

    if ukrainian_specific_count >= 2:
        return ("uk", "high")
    if ukrainian_specific_count == 1 and cyrillic_count >= 6:
        return ("uk", "medium")
    if russian_specific_count >= 1 and cyrillic_count >= 6:
        return ("ru", "high")
    if latin_count >= 12 and latin_count >= (cyrillic_count * 2):
        return ("en", "high")
    if latin_count >= 6 and latin_count >= cyrillic_count:
        return ("en", "medium")
    if cyrillic_count >= 8 and cyrillic_count > latin_count:
        return ("ru", "medium")
    if latin_count > 0 and cyrillic_count == 0:
        return ("en", "low")
    if cyrillic_count > 0 and latin_count == 0:
        return ("ru", "low")
    return ("other", "low")


def detect_language_from_text(text: str) -> str:
    language, _ = _text_probe_language_with_confidence(text)
    return language


def _is_confident_probe(confidence: str) -> bool:
    return confidence in {"high", "medium"}


def _metadata_candidates(metadata: VideoMetadata) -> tuple[str, ...]:
    ordered: List[str] = []
    for raw_value in (metadata.youtube_language, metadata.channel_language):
        normalized: Optional[str] = normalize_language(raw_value)
        if normalized is None or normalized in ordered:
            continue
        ordered.append(normalized)
    return tuple(ordered)


def detect_language_decision(
    metadata: VideoMetadata,
    *,
    sheet_override_language: Optional[str] = None,
) -> LanguageDecision:
    override_normalized: Optional[str] = normalize_language(sheet_override_language)
    youtube_language_raw: str = str(metadata.youtube_language or "").strip()
    channel_language_raw: str = str(metadata.channel_language or "").strip()
    metadata_candidates: tuple[str, ...] = _metadata_candidates(metadata)
    metadata_language: Optional[str] = metadata_candidates[0] if metadata_candidates else None
    text_probe_language, text_probe_confidence = _text_probe_language_with_confidence(
        f"{metadata.title}\n{metadata.description}".strip()
    )

    if override_normalized is not None:
        final_language: str = override_normalized
        decision_source: str = "sheet_override"
    elif metadata_language is not None and metadata_language == text_probe_language:
        final_language = metadata_language
        decision_source = "metadata_agreement"
    elif (
        metadata_language is not None
        and text_probe_language in {"uk", "en", "ru"}
        and _is_confident_probe(text_probe_confidence)
    ):
        final_language = text_probe_language
        decision_source = "text_probe_override"
    elif metadata_language is None and text_probe_language in {"uk", "en", "ru"} and _is_confident_probe(text_probe_confidence):
        final_language = text_probe_language
        decision_source = "text_probe_only"
    elif metadata_language is not None:
        final_language = metadata_language
        decision_source = "metadata_only"
    else:
        final_language = "other"
        decision_source = "fallback_default"

    if decision_source not in _LANGUAGE_DECISION_SOURCES:
        final_language = "other"
        decision_source = "fallback_default"

    language_conflict: bool = (
        metadata_language is not None
        and text_probe_language in {"uk", "en", "ru"}
        and metadata_language != text_probe_language
    )
    return LanguageDecision(
        youtube_language_raw=youtube_language_raw,
        channel_language_raw=channel_language_raw,
        metadata_language_candidates=metadata_candidates,
        text_probe_language=text_probe_language,
        text_probe_confidence=text_probe_confidence,
        final_language=final_language,
        language_decision_source=decision_source,
        language_conflict=language_conflict,
    )


def log_language_decision(*, row_number: int, decision: LanguageDecision) -> None:
    metadata_candidates: str = ",".join(decision.metadata_language_candidates) or "none"
    LOGGER.info(
        "language_decision row=%d youtube_language_raw=%s channel_language_raw=%s metadata_language_candidates=%s text_probe_language=%s text_probe_confidence=%s final_language=%s language_decision_source=%s language_conflict=%s",
        row_number,
        decision.youtube_language_raw or "none",
        decision.channel_language_raw or "none",
        metadata_candidates,
        decision.text_probe_language,
        decision.text_probe_confidence,
        decision.final_language,
        decision.language_decision_source,
        "yes" if decision.language_conflict else "no",
    )


def detect_language(
    metadata: VideoMetadata,
    *,
    sheet_override_language: Optional[str] = None,
) -> str:
    return detect_language_decision(
        metadata,
        sheet_override_language=sheet_override_language,
    ).final_language
