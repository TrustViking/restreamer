from __future__ import annotations

from dataclasses import dataclass

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import VideoMetadata

LOGGER = _get_logger_impl(__name__)


@dataclass(frozen=True)
class VideoLanguageProfile:
    """All language signals from yt-dlp and langdetect for a single video.

    Every field is normalized to ISO 639-1 two-letter codes via normalize_language().
    Fields are None or empty tuples when the source did not provide data.
    """

    # --- Group 1: metadata-level (root info dict fields) ---
    video_language: str | None
    channel_language: str | None

    # --- Group 2: audio tracks (info["formats"]) ---
    audio_languages: tuple[str, ...] = ()

    # --- Group 3: subtitles ---
    subtitle_languages: tuple[str, ...] = ()
    auto_caption_languages: tuple[str, ...] = ()
    auto_caption_orig_language: str | None = None

    # --- Group 4: langdetect results (computed during profile construction) ---
    title_language: str | None = None
    description_language: str | None = None
    langdetect_text_language: str | None = None

    @classmethod
    def from_metadata(cls, metadata: VideoMetadata) -> VideoLanguageProfile:
        """Build a language profile from VideoMetadata.

        Calls langdetect for title, description, and combined text.
        Normalizes all language codes.
        """
        from app.core.language import _langdetect_language, normalize_language

        video_language: str | None = normalize_language(metadata.youtube_language)
        channel_language: str | None = normalize_language(metadata.channel_language)

        audio_languages: tuple[str, ...] = tuple(
            lang
            for raw in metadata.audio_languages
            if (lang := normalize_language(raw)) is not None
        )
        # Deduplicate while preserving order
        seen_audio: list[str] = []
        for lang in audio_languages:
            if lang not in seen_audio:
                seen_audio.append(lang)
        audio_languages = tuple(seen_audio)

        subtitle_languages: tuple[str, ...] = tuple(
            lang
            for raw in metadata.subtitle_languages
            if (lang := normalize_language(raw)) is not None
        )
        seen_sub: list[str] = []
        for lang in subtitle_languages:
            if lang not in seen_sub:
                seen_sub.append(lang)
        subtitle_languages = tuple(seen_sub)

        auto_caption_languages: tuple[str, ...] = tuple(
            lang
            for raw in metadata.auto_caption_languages
            if (lang := normalize_language(raw)) is not None
        )
        seen_ac: list[str] = []
        for lang in auto_caption_languages:
            if lang not in seen_ac:
                seen_ac.append(lang)
        auto_caption_languages = tuple(seen_ac)
        # Extract *-orig key from auto_caption_languages (raw, before normalization)
        # YouTube marks the original audio language as "{lang}-orig" in auto_captions
        auto_caption_orig_language: str | None = None
        for raw_key in metadata.auto_caption_languages:
            raw_key_stripped: str = str(raw_key or "").strip()
            if raw_key_stripped.endswith("-orig"):
                orig_candidate: str = raw_key_stripped[: -len("-orig")]
                auto_caption_orig_language = normalize_language(orig_candidate)
                break

        title_language: str | None = _langdetect_language(metadata.title)
        description_language: str | None = _langdetect_language(metadata.description)
        langdetect_text_language: str | None = _langdetect_language(
            f"{metadata.title}\n{metadata.description}".strip()
        )

        profile = cls(
            video_language=video_language,
            channel_language=channel_language,
            audio_languages=audio_languages,
            subtitle_languages=subtitle_languages,
            auto_caption_languages=auto_caption_languages,
            auto_caption_orig_language=auto_caption_orig_language,
            title_language=title_language,
            description_language=description_language,
            langdetect_text_language=langdetect_text_language,
        )

        LOGGER.debug(
            "language_profile built: video_language=%s channel_language=%s "
            "audio_languages=%s subtitle_languages=%s auto_caption_orig=%s "
            "title_language=%s description_language=%s langdetect_text_language=%s",
            profile.video_language or "none",
            profile.channel_language or "none",
            profile.audio_languages or "none",
            profile.subtitle_languages or "none",
            profile.auto_caption_orig_language or "none",
            profile.title_language or "none",
            profile.description_language or "none",
            profile.langdetect_text_language or "none",
        )

        return profile
