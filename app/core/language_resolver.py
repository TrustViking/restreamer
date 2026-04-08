from __future__ import annotations

from collections import Counter

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.language_profile import VideoLanguageProfile

LOGGER = _get_logger_impl(__name__)


def resolve_language(profile: VideoLanguageProfile) -> tuple[str, str, bool]:
    """Resolve a single language from the profile using voting.

    Returns:
        (final_language, decision_source, language_conflict)

    Voting signals (each is an independent vote when present):
        - video_language: YouTube's language field (from ASR)
        - audio_first: first audio track language from formats
        - auto_caption_orig: language extracted from the *-orig key in auto_captions
        - description_language: langdetect on description only
        - langdetect_text: langdetect on title+description combined

    NOT used for voting (diagnostic only):
        - auto_caption_languages[0]: always 'ab' (alphabetical catalog, useless)
        - subtitle_languages[0]: often 'en' on educational videos (translation target, not content language)
    """
    video_lang: str | None = profile.video_language
    langdetect_lang: str | None = profile.langdetect_text_language
    audio_first: str | None = profile.audio_languages[0] if profile.audio_languages else None
    auto_caption_orig: str | None = profile.auto_caption_orig_language
    desc_lang: str | None = profile.description_language

    # Collect all available votes (each field is an independent vote)
    votes: list[tuple[str, str]] = []  # (source_name, language)
    if video_lang is not None:
        votes.append(("video_language", video_lang))
    if audio_first is not None:
        votes.append(("audio_first", audio_first))
    if auto_caption_orig is not None:
        votes.append(("auto_caption_orig", auto_caption_orig))
    if desc_lang is not None:
        votes.append(("description_language", desc_lang))
    if langdetect_lang is not None:
        votes.append(("langdetect_text", langdetect_lang))

    LOGGER.debug(
        "language_resolver votes: %s",
        " | ".join(f"{name}={lang}" for name, lang in votes) or "none",
    )

    # --- Priority 1: Consensus (≥3 independent signals agree) ---
    if len(votes) >= 3:
        lang_counts: Counter[str] = Counter(lang for _, lang in votes)
        top_lang, top_count = lang_counts.most_common(1)[0]
        if top_count >= 3:
            LOGGER.debug(
                "language_resolver result: consensus lang=%s count=%d/%d",
                top_lang, top_count, len(votes),
            )
            return top_lang, "consensus", False

    # --- Priority 2: langdetect agrees with video_language ---
    if langdetect_lang is not None and video_lang is not None:
        if langdetect_lang == video_lang:
            LOGGER.debug(
                "language_resolver result: langdetect_metadata_agreement lang=%s",
                langdetect_lang,
            )
            return langdetect_lang, "langdetect_metadata_agreement", False

        # --- Priority 3: Conflict -> arbitration ---
        arbiter_votes: list[tuple[str, str]] = []
        if audio_first is not None:
            arbiter_votes.append(("audio_first", audio_first))
        if auto_caption_orig is not None:
            arbiter_votes.append(("auto_caption_orig", auto_caption_orig))
        if desc_lang is not None:
            arbiter_votes.append(("description_language", desc_lang))

        LOGGER.debug(
            "language_resolver conflict: video_language=%s langdetect_text=%s "
            "arbiter_votes: %s",
            video_lang,
            langdetect_lang,
            " | ".join(f"{name}={lang}" for name, lang in arbiter_votes) or "none",
        )

        if arbiter_votes:
            arbiter_counts: Counter[str] = Counter(lang for _, lang in arbiter_votes)
            votes_for_metadata: int = arbiter_counts.get(video_lang, 0)
            votes_for_langdetect: int = arbiter_counts.get(langdetect_lang, 0)

            LOGGER.debug(
                "language_resolver arbitration: votes_for_metadata(%s)=%d "
                "votes_for_langdetect(%s)=%d total_arbiter_votes=%d",
                video_lang, votes_for_metadata,
                langdetect_lang, votes_for_langdetect,
                len(arbiter_votes),
            )

            if votes_for_metadata > votes_for_langdetect:
                return video_lang, "metadata_arbitration", True
            if votes_for_langdetect > votes_for_metadata:
                return langdetect_lang, "langdetect_arbitration", True

        # Tie or no arbiter votes -> fallback to video_language (trust the author)
        LOGGER.debug(
            "language_resolver arbitration_fallback: defaulting to video_language=%s",
            video_lang,
        )
        return video_lang, "metadata_arbitration_fallback", True

    # --- Priority 4: Only langdetect available ---
    if langdetect_lang is not None:
        LOGGER.debug(
            "language_resolver result: langdetect only, lang=%s",
            langdetect_lang,
        )
        return langdetect_lang, "langdetect", False

    # --- Priority 5: Only video_language available ---
    if video_lang is not None:
        LOGGER.debug(
            "language_resolver result: metadata_fallback lang=%s",
            video_lang,
        )
        return video_lang, "metadata_fallback", False

    # --- Priority 6: Nothing ---
    LOGGER.debug("language_resolver result: undetected")
    return "unknown", "undetected", False
