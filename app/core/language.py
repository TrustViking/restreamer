from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.description_cleaner import clean_description_for_analysis
from app.core.models import VideoMetadata

LOGGER = _get_logger_impl(__name__)
DetectorFactory.seed = 0

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
    "langdetect",
    "langdetect_metadata_agreement",
    "metadata_fallback",
    "undetected",
    "consensus",
    "metadata_arbitration",
    "langdetect_arbitration",
    "metadata_arbitration_fallback",
}


@dataclass(frozen=True)
class LanguageDecision:
    youtube_language_raw: str
    channel_language_raw: str
    metadata_language_candidates: tuple[str, ...]
    langdetect_language: str | None
    final_language: str
    language_decision_source: str
    language_conflict: bool
    description_language: str | None = None
    audio_language: str | None = None
    auto_caption_language: str | None = None
    title_language: str | None = None


def normalize_language(raw_language: Optional[str]) -> Optional[str]:
    normalized_input: str = str(raw_language or "").strip().lower().replace("_", "-")
    if not normalized_input:
        return None
    if normalized_input in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[normalized_input]
    iso_match: re.Match[str] | None = re.fullmatch(
        r"([a-z]{2,3})(?:-[a-z]{2,3})?",
        normalized_input,
    )
    if iso_match:
        return iso_match.group(1)
    for prefix, canonical in (("uk", "uk"), ("ua", "uk"), ("en", "en"), ("ru", "ru")):
        if normalized_input.startswith(prefix):
            return canonical
    return None


def _langdetect_language(text: str) -> str | None:
    cleaned_text: str = clean_description_for_analysis(text)
    if len(cleaned_text) < 20:
        return None
    try:
        detected: str = str(detect(cleaned_text) or "").strip().lower()
    except LangDetectException:
        return None
    return normalize_language(detected)


def detect_language_from_text(text: str) -> str:
    detected: str | None = _langdetect_language(text)
    return detected or "unknown"


def _metadata_candidates(metadata: VideoMetadata) -> tuple[str, ...]:
    ordered: list[str] = []
    for raw_value in (metadata.youtube_language, metadata.channel_language):
        normalized: Optional[str] = normalize_language(raw_value)
        if normalized is None or normalized in ordered:
            continue
        ordered.append(normalized)
    return tuple(ordered)


def detect_language_decision(
    metadata: VideoMetadata,
) -> LanguageDecision:
    from app.core.language_profile import VideoLanguageProfile
    from app.core.language_resolver import resolve_language

    youtube_language_raw: str = str(metadata.youtube_language or "").strip()
    channel_language_raw: str = str(metadata.channel_language or "").strip()
    metadata_candidates: tuple[str, ...] = _metadata_candidates(metadata)
    profile: VideoLanguageProfile = VideoLanguageProfile.from_metadata(metadata)
    final_language, decision_source, language_conflict = resolve_language(profile)

    if decision_source not in _LANGUAGE_DECISION_SOURCES:
        LOGGER.warning(
            "language_decision_source %r not in allowed set, falling back to undetected",
            decision_source,
        )
        final_language = "unknown"
        decision_source = "undetected"

    return LanguageDecision(
        youtube_language_raw=youtube_language_raw,
        channel_language_raw=channel_language_raw,
        metadata_language_candidates=metadata_candidates,
        langdetect_language=profile.langdetect_text_language,
        final_language=final_language,
        language_decision_source=decision_source,
        language_conflict=language_conflict,
        description_language=profile.description_language,
        audio_language=(profile.audio_languages[0] if profile.audio_languages else None),
        auto_caption_language=profile.auto_caption_orig_language,
        title_language=profile.title_language,
    )


def log_language_decision(*, row_number: int, decision: LanguageDecision) -> None:
    metadata_candidates: str = ",".join(decision.metadata_language_candidates) or "none"
    LOGGER.info(
        "language_decision row=%d final_language=%s source=%s conflict=%s "
        "youtube_raw=%s channel_raw=%s metadata_candidates=%s "
        "langdetect=%s description_lang=%s title_lang=%s "
        "audio_lang=%s auto_caption_orig=%s",
        row_number,
        decision.final_language,
        decision.language_decision_source,
        "yes" if decision.language_conflict else "no",
        decision.youtube_language_raw or "none",
        decision.channel_language_raw or "none",
        metadata_candidates,
        decision.langdetect_language or "none",
        decision.description_language or "none",
        decision.title_language or "none",
        decision.audio_language or "none",
        decision.auto_caption_language or "none",
    )


def detect_language(
    metadata: VideoMetadata,
) -> str:
    return detect_language_decision(metadata).final_language
