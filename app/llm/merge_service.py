from __future__ import annotations

import dataclasses
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE
from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    BLOCK_GENERATION_MODE_REAL_MERGE,
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
    RejectedMergeAttempt,
    VideoMetadata,
)
from app.ingest.youtube_metadata import YtDlpYouTubeMetadataFetcher, normalize_youtube_link
from app.llm.merge_polish import (
    MergePolishResult,
    build_merge_polish_prompt,
    bullet_marker_count,
    infer_polish_reject_reason,
    is_too_aggressive_rewrite,
)
from app.llm.merge_quality import (
    MergeQualityDiagnostics,
    MergeQualityNormalizationResult,
    normalize_merge_description,
)
from app.llm.merge_parser import (
    MergeTailSeparationResult,
    build_plain_merged_content_or_raise,
    clean_and_validate_llm_description,
    parse_json_tolerant,
    parse_merge_response_or_raise,
    separate_merge_body_and_tail,
)
from app.llm.model_compatibility import LlmModelConfigurationError
from app.llm.model_identity import resolve_effective_llm_model
from app.llm.merge_run_summary import MergeRunSummary
from app.llm.openai_client import LlmTraceContext, OpenAITransportResult, openai_request_merge
from app.llm.provider_factory import get_llm_provider
from app.llm.providers.base import LlmProvider
from app.observability.runtime_analytics import log_warning_operational

LOGGER = _get_logger_impl(__name__)
PRIMARY_ATTEMPTS: int = 2
STYLE_CONTRACT_VERSION: str = "v4_merge_quality_hardening"
_SEMANTIC_BULLET_MARKERS: tuple[str, ...] = ("🔹", "📌", "🎤", "🎥", "⚖", "🌐", "✅")
_SEMANTIC_BULLET_MARKER_SET: set[str] = set(_SEMANTIC_BULLET_MARKERS)
_BULLET_PLAIN_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:[-*•▪◦‣–—]|(?:\d+[.)]))\s+\S+",
    flags=re.UNICODE,
)
_URL_PATTERN: re.Pattern[str] = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_TRACKING_QUERY_KEYS: tuple[str, ...] = (
    "si",
    "feature",
    "pp",
    "fbclid",
    "gclid",
    "igsh",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref_src",
    "ref_url",
    "spm",
)


def _fallback_title_source_label() -> str:
    return "fallback_titles"


def _fallback_hook_source_label() -> str:
    return "fallback_none"


def _fallback_body_source_label() -> str:
    return "fallback_source_descriptions"
_OFFICIAL_LINK_CONTEXT_HINTS: tuple[str, ...] = (
    "official",
    "website",
    "initiative",
    "resource",
    "resources",
    "conference",
    "more information",
    "details",
    "site",
    "official links",
    "офіцій",
    "ініціатив",
    "ресурс",
    "сайт",
    "официал",
)
_SEMANTIC_TOKEN_PATTERN: re.Pattern[str] = re.compile(
    r"[0-9A-Za-zА-Яа-яЁёІіЇїЄєҐґ]{3,}",
    flags=re.UNICODE,
)
_SEMANTIC_STOPWORDS: set[str] = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "among",
    "and",
    "around",
    "because",
    "before",
    "between",
    "brief",
    "call",
    "conversation",
    "cover",
    "discussion",
    "during",
    "each",
    "from",
    "into",
    "more",
    "most",
    "other",
    "over",
    "stream",
    "talk",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "today",
    "topic",
    "topics",
    "update",
    "updates",
    "with",
    "будет",
    "более",
    "важный",
    "вместе",
    "всем",
    "всех",
    "главном",
    "диалог",
    "для",
    "день",
    "его",
    "или",
    "как",
    "который",
    "людей",
    "материал",
    "между",
    "миру",
    "наша",
    "наши",
    "нем",
    "них",
    "новый",
    "новости",
    "обзор",
    "общем",
    "почему",
    "разговор",
    "сегодня",
    "событие",
    "среди",
    "тема",
    "темы",
    "эфир",
    "этот",
    "важлива",
    "всіх",
    "головне",
    "діалог",
    "людей",
    "матеріал",
    "наші",
    "новий",
    "новини",
    "огляд",
    "подія",
    "потік",
    "розмова",
    "сьогодні",
    "теми",
    "цей",
    "ефір",
}
_EXPANDED_RETRY_FOCUS_ORDER: tuple[str, ...] = (
    "body_depth",
    "bullet_sufficiency",
    "source_specificity",
    "hook_restraint",
    "source_spread",
)
_EXPANDED_RETRY_REASON_TO_FOCUS: dict[str, str] = {
    "insufficient_expanded_body": "body_depth",
    "too_few_expanded_bullets": "bullet_sufficiency",
    "overly_generic_body": "source_specificity",
    "hook_dominates_body": "hook_restraint",
    "weak_source_coverage": "source_spread",
    "insufficient_topic_spread": "source_spread",
}
_EXPANDED_RETRY_FOCUS_LINES: dict[str, str] = {
    "body_depth": "Make the post-hook body clearly denser: after the opening, carry most of the useful information in a fuller factual agenda instead of a thin bridge into bullets.",
    "bullet_sufficiency": "Make sure the agenda reaches enough distinct, meaningful bullets; each bullet should add a separate fact, actor, place, event, timeline, or operational consequence rather than rephrasing one idea.",
    "source_specificity": "Use source-grounded specifics instead of broad editorial wording: preserve concrete names, places, institutions, numbers, timings, or clearly observable actions whenever the sources provide them.",
    "hook_restraint": "Keep the hook brief and functional so the main body carries the value; do not spend two long scene-setting sentences if that causes the agenda to stay thin.",
    "source_spread": "Restore distinguishable spread across source lines or topic nodes so the body does not collapse into one blended lane; keep separate angles recognizably separate without turning the text into SOURCE 1 / SOURCE 2 enumeration.",
}
_THREE_SOURCE_EXPANDED_RETRY_COMBINED_LINES: tuple[tuple[frozenset[str], str], ...] = (
    (
        frozenset({"body_depth", "bullet_sufficiency"}),
        "After the hook, move quickly into 2 to 3 short agenda tracks with enough fact-bearing bullets to sustain them; do not spend the remaining space on a thin bridge or a generic wrap-up.",
    ),
    (
        frozenset({"source_specificity", "source_spread"}),
        "Make the contribution of different sources visibly distinguishable across multiple meaningful lines of discussion; cut generic filler bridges and keep concrete source facts attached to the right lane.",
    ),
)
_FOUR_PLUS_EXPANDED_RETRY_REASON_CODES: frozenset[str] = frozenset(
    {
        "too_few_expanded_bullets",
        "insufficient_expanded_body",
        "overly_generic_body",
        "insufficient_topic_spread",
        "weak_source_coverage",
    }
)
_FORMATTING_ONLY_SALVAGE_REASON_CODES: frozenset[str] = frozenset(
    {"excessive_emoji_usage"}
)
_EMOJI_PATTERN: re.Pattern[str] = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class MergeAttemptFailure(RuntimeError):
    reason_code: str
    reason: str
    model_name: str
    attempt_stage: str
    raw_response_text: str
    reason_codes: tuple[str, ...] = ()
    rejected_attempt: Optional[RejectedMergeAttempt] = None

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True)
class DescriptionValidationFailure(RuntimeError):
    reason_codes: tuple[str, ...]
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class ParagraphEnforcementResult:
    description_text: str
    mutated: bool
    recovery_applied: bool
    note: str
    body_paragraphs_before: int
    body_paragraphs_after: int


@dataclass(frozen=True)
class MergeSemanticDiagnostics:
    hook_present: bool
    agenda_block_present: bool
    bullet_points_count: int
    semantic_bullets_count: int
    bullets_with_emoji_count: int
    bullets_with_plain_marker_count: int
    bullet_marker_types: tuple[str, ...]
    named_entities_preserved: int
    source_named_entities_total: int
    emoji_count: int
    source_coverage_hits: int
    source_coverage_total: int
    source_coverage_by_item: tuple[bool, ...]
    official_links_found_in_sources: int
    official_links_kept: int
    official_links_in_output: int
    official_links_fill_applied: bool
    merge_quality: MergeQualityDiagnostics
    expanded_quality: "ExpandedMergeDiagnostics"


@dataclass(frozen=True)
class ExpandedMergeDiagnostics:
    enabled: bool
    body_paragraph_count: int
    body_char_count: int
    body_non_hook_char_count: int
    compact_body_relaxed: bool
    hook_char_count: int
    hook_share: float
    bullet_lines_with_anchor_count: int
    detailed_bullet_count: int
    thematic_spread_count: int
    body_anchor_count: int
    distinctive_coverage_hits: int
    distinctive_coverage_total: int
    distinctive_coverage_by_item: tuple[int, ...]
    distinctive_coverage_targets: tuple[int, ...]
    softened_distinctive_source_coverage_applied: bool
    quality_gate_status: str
    quality_gate_reason_codes: tuple[str, ...]
    quality_gate_reason_details: tuple[str, ...]


@dataclass(frozen=True)
class OfficialLinksSelection:
    found_in_sources: int
    kept_links: tuple[str, ...]


@dataclass(frozen=True)
class OfficialLinksFillResult:
    description: str
    links_in_output: int
    fill_applied: bool


@dataclass(frozen=True)
class MergeYouTubeCandidate:
    url: str
    title: str
    duration_seconds: Optional[int]
    duration_label: str
    metadata_resolved: bool


@dataclass(frozen=True)
class MergeYouTubeCandidatesResult:
    raw_youtube_urls_found: int
    invalid_youtube_urls_skipped: int
    deduped_candidates: tuple[MergeYouTubeCandidate, ...]
    metadata_resolved_count: int


@dataclass(frozen=True)
class PreparedMergeSourceDescription:
    text: str
    raw_chars: int
    cleaned_chars: int
    urls_removed: int
    hashtags_removed: int
    service_paragraphs_dropped: int


@dataclass(frozen=True)
class MergeContractMode:
    mode_label: str
    source_count: int
    bullet_range_label: str
    expanded_structure_enabled: bool
    contract_block: str


@dataclass(frozen=True)
class ExpandedRetryProfile:
    retry_mode: str
    reject_signals: tuple[str, ...]
    focus_tags: tuple[str, ...]
    reinforcement_lines: tuple[str, ...]

    @property
    def enabled(self) -> bool:
        return self.retry_mode == "targeted" and bool(self.reinforcement_lines)

    @property
    def focus_label(self) -> str:
        return ",".join(self.focus_tags) or "none"

    @property
    def reject_signal_label(self) -> str:
        return ",".join(self.reject_signals) or "none"


@dataclass(frozen=True)
class FormattingNormalizationResult:
    description_text: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class MergeValidationRecoveryAttempt:
    merged_content: Optional[MergedLanguageContent]
    diagnostics: Optional[MergeSemanticDiagnostics]
    validation_error: Optional[Exception]
    actions: tuple[str, ...]


def _reason_code_from_error(error: Exception) -> str:
    error_text: str = str(error or "")
    validation_reason_codes: tuple[str, ...] = _reason_codes_from_error(error)
    if validation_reason_codes:
        return validation_reason_codes[0]
    if "not a valid single JSON object" in error_text:
        return "not_json_object"
    if error_text.startswith("missing_keys:"):
        return "missing_keys"
    if error_text.startswith("extra_keys:"):
        return "extra_keys"
    if "forbidden_key:" in error_text:
        return "forbidden_variants"
    if (
        "title must be a non-empty string" in error_text
        or "title is invalid" in error_text
        or "title must not contain emoji" in error_text
    ):
        return "invalid_title"
    if (
        "description must be a non-empty string" in error_text
        or "description paragraph count" in error_text
        or "description body paragraph count" in error_text
        or "description validation failed" in error_text
    ):
        return "invalid_description"
    return "unexpected_error"


def _reason_codes_from_error(error: Exception) -> tuple[str, ...]:
    if isinstance(error, MergeAttemptFailure):
        if error.reason_codes:
            return error.reason_codes
        # Defensive fallback for legacy failures that still carry only human-readable text.
        return _extract_description_validation_reason_codes(str(error or ""))
    if isinstance(error, DescriptionValidationFailure):
        if error.reason_codes:
            return error.reason_codes
        # Defensive fallback for legacy failures that still carry only human-readable text.
        return _extract_description_validation_reason_codes(str(error or ""))
    # Defensive fallback for legacy string-only validation errors.
    return _extract_description_validation_reason_codes(str(error or ""))


def _normalize_description_validation_reason_code(reason_code: str) -> str:
    normalized_reason_code: str = str(reason_code or "").strip()
    if not normalized_reason_code:
        return ""
    if re.fullmatch(r"semantic_source_\d+_coverage_missing", normalized_reason_code):
        return "weak_source_coverage"
    reason_aliases: dict[str, str] = {
        "semantic_source_grounding_too_low": "weak_source_coverage",
        "semantic_too_generic": "overly_generic_body",
    }
    return reason_aliases.get(normalized_reason_code, normalized_reason_code)


def _normalize_description_validation_reason_codes(
    reason_codes: Sequence[str],
) -> tuple[str, ...]:
    normalized_reason_codes: list[str] = []
    for raw_reason_code in reason_codes:
        normalized_reason_code: str = _normalize_description_validation_reason_code(
            raw_reason_code
        )
        if normalized_reason_code and normalized_reason_code not in normalized_reason_codes:
            normalized_reason_codes.append(normalized_reason_code)
    return tuple(normalized_reason_codes)


def _build_description_validation_failure(
    *,
    reason_codes: Sequence[str],
    message: str,
) -> DescriptionValidationFailure:
    return DescriptionValidationFailure(
        reason_codes=_normalize_description_validation_reason_codes(reason_codes),
        message=message,
    )


def _extract_description_validation_reason_codes(error_text: str) -> tuple[str, ...]:
    marker: str = "description validation failed:"
    if marker not in error_text:
        return ()
    raw_reason_codes: str = error_text.split(marker, maxsplit=1)[1].strip()
    if not raw_reason_codes:
        return ()
    return _normalize_description_validation_reason_codes(
        raw_reason_codes.split(",")
    )

def _standard_expanded_retry_profile(
    *,
    reject_signals: Sequence[str] = (),
) -> ExpandedRetryProfile:
    normalized_reject_signals: tuple[str, ...] = tuple(
        signal for signal in dict.fromkeys(reject_signals) if str(signal or "").strip()
    )
    return ExpandedRetryProfile(
        retry_mode="standard",
        reject_signals=normalized_reject_signals,
        focus_tags=(),
        reinforcement_lines=(),
    )


def _build_expanded_retry_profile(
    *,
    source_count: int,
    reject_signals: Sequence[str],
) -> ExpandedRetryProfile:
    normalized_reject_signals: tuple[str, ...] = tuple(
        _normalize_description_validation_reason_code(signal)
        for signal in dict.fromkeys(reject_signals)
        if _normalize_description_validation_reason_code(signal)
    )
    if source_count < 3:
        return _standard_expanded_retry_profile(
            reject_signals=normalized_reject_signals
        )
    focus_tags: list[str] = []
    for focus_tag in _EXPANDED_RETRY_FOCUS_ORDER:
        if any(
            _EXPANDED_RETRY_REASON_TO_FOCUS.get(reason_code) == focus_tag
            for reason_code in normalized_reject_signals
        ):
            focus_tags.append(focus_tag)
    if not focus_tags:
        return _standard_expanded_retry_profile(
            reject_signals=normalized_reject_signals
        )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=normalized_reject_signals,
        focus_tags=tuple(focus_tags),
        reinforcement_lines=_build_expanded_retry_reinforcement_lines(
            source_count=source_count,
            focus_tags=tuple(focus_tags),
            reject_signals=normalized_reject_signals,
        ),
    )


def _build_retry_profile_from_reason_codes(
    *,
    source_count: int,
    reject_signals: Sequence[str],
    fallback_reason_code: str,
) -> ExpandedRetryProfile:
    normalized_reject_signals: tuple[str, ...] = (
        _normalize_description_validation_reason_codes(reject_signals)
    )
    if not normalized_reject_signals and str(fallback_reason_code or "").strip():
        normalized_reject_signals = (
            _normalize_description_validation_reason_code(fallback_reason_code),
        )
    return _build_expanded_retry_profile(
        source_count=source_count,
        reject_signals=normalized_reject_signals,
    )


def _expanded_retry_reinforcement_block(
    profile: Optional[ExpandedRetryProfile],
) -> str:
    if profile is None or not profile.enabled:
        return ""
    return (
        "EXPANDED RETRY FOCUS\n"
        "Keep the same expanded contract, but correct the weak points from the previous draft.\n"
        f"{'\n'.join(profile.reinforcement_lines)}"
    ).strip()


def _build_expanded_retry_reinforcement_lines(
    *,
    source_count: int,
    focus_tags: Sequence[str],
    reject_signals: Sequence[str],
) -> tuple[str, ...]:
    ordered_focus_tags: tuple[str, ...] = tuple(
        tag for tag in dict.fromkeys(focus_tags) if str(tag or "").strip()
    )
    ordered_reject_signals: tuple[str, ...] = tuple(
        signal for signal in dict.fromkeys(reject_signals) if str(signal or "").strip()
    )
    reinforcement_lines: list[str] = [
        _EXPANDED_RETRY_FOCUS_LINES[focus_tag]
        for focus_tag in ordered_focus_tags
        if focus_tag in _EXPANDED_RETRY_FOCUS_LINES
    ]
    if source_count < 3:
        return tuple(reinforcement_lines)
    focus_set: set[str] = set(ordered_focus_tags)
    for required_tags, combined_line in _THREE_SOURCE_EXPANDED_RETRY_COMBINED_LINES:
        if required_tags.issubset(focus_set):
            reinforcement_lines.append(combined_line)
    if source_count >= 4:
        reinforcement_lines.extend(
            _build_four_plus_expanded_retry_reinforcement_lines(
                reject_signals=ordered_reject_signals,
            )
        )
    return tuple(reinforcement_lines)


def _build_four_plus_expanded_retry_reinforcement_lines(
    *,
    reject_signals: Sequence[str],
) -> tuple[str, ...]:
    ordered_reject_signals: tuple[str, ...] = tuple(
        signal for signal in dict.fromkeys(reject_signals) if str(signal or "").strip()
    )
    reject_signal_set: set[str] = set(ordered_reject_signals)
    if not reject_signal_set.intersection(_FOUR_PLUS_EXPANDED_RETRY_REASON_CODES):
        return ()
    reinforcement_lines: list[str] = [
        "For 4 or more sources, do not collapse the post-hook body into one umbrella summary. Build 2 to 3 meaningful thematic micro-blocks after the hook and let each micro-block carry distinguishable source-specific contributions.",
    ]
    if {
        "too_few_expanded_bullets",
        "insufficient_expanded_body",
    }.intersection(reject_signal_set):
        reinforcement_lines.append(
            "Do not compress 4 or 5 sources into only 3 generic bullets. Restore lost source lines with enough fact-bearing bullets inside the micro-blocks so the body gains source-specific density, not just extra length."
        )
    if "overly_generic_body" in reject_signal_set:
        reinforcement_lines.append(
            "Increase source-specific density inside the micro-blocks: use concrete names, places, institutions, counts, timings, events, or operational consequences from the actual sources instead of broad framing language."
        )
    if {
        "weak_source_coverage",
        "insufficient_topic_spread",
    }.intersection(reject_signal_set):
        reinforcement_lines.append(
            "Make every source leave a recognizable trace in the body and keep independent topic nodes separate. Do not collapse several inputs into one generic moral, one universal frame, or one blended concluding block."
        )
    return tuple(reinforcement_lines)


def _source_texts_for_merge_quality(videos: Sequence[PlannedVideo]) -> tuple[str, ...]:
    return tuple(
        (
            f"{video.metadata.title.strip()}\n"
            f"{_clean_source_description_for_llm(video.metadata.description.strip()).text}"
        ).strip()
        for video in videos
    )


def _best_effort_rejected_payload_fields(
    *,
    raw_response_text: str,
    structured_payload: object,
) -> tuple[str, str]:
    payload: object = structured_payload
    if not isinstance(payload, dict):
        payload, _ = parse_json_tolerant(raw_response_text)
    if not isinstance(payload, dict):
        return ("", "")
    title_text: str = (
        str(payload.get("title") or "").strip()
        if isinstance(payload.get("title"), str)
        else ""
    )
    description_text: str = (
        str(payload.get("description") or "").strip()
        if isinstance(payload.get("description"), str)
        else ""
    )
    return (title_text, description_text)


def _build_rejected_merge_attempt(
    *,
    attempt_index: int,
    model_name: str,
    reason_codes: Sequence[str],
    fallback_reason_code: str = "",
    raw_response_text: str,
    title_text: str = "",
    description_text: str = "",
    structured_payload: object = None,
) -> Optional[RejectedMergeAttempt]:
    normalized_reason_codes: tuple[str, ...] = _normalize_description_validation_reason_codes(
        reason_codes
    )
    if not normalized_reason_codes and str(fallback_reason_code or "").strip():
        normalized_reason_codes = (str(fallback_reason_code).strip(),)
    best_effort_title: str = str(title_text or "").strip()
    best_effort_description: str = str(description_text or "").strip()
    if not best_effort_title and not best_effort_description:
        best_effort_title, best_effort_description = _best_effort_rejected_payload_fields(
            raw_response_text=raw_response_text,
            structured_payload=structured_payload,
        )
    if (
        not best_effort_title
        and not best_effort_description
        and not str(raw_response_text or "").strip()
    ):
        return None
    return RejectedMergeAttempt(
        attempt_index=attempt_index,
        model_name=str(model_name or "").strip(),
        reject_reasons=normalized_reason_codes,
        title=best_effort_title,
        description=best_effort_description,
        raw_response_text=str(raw_response_text or ""),
    )


def _is_formatting_only_validation_failure(reason_codes: Sequence[str]) -> bool:
    normalized_reason_codes: tuple[str, ...] = _normalize_description_validation_reason_codes(
        reason_codes
    )
    return bool(normalized_reason_codes) and all(
        reason_code in _FORMATTING_ONLY_SALVAGE_REASON_CODES
        for reason_code in normalized_reason_codes
    )


def _strip_non_structural_emoji_from_line(line: str) -> tuple[str, bool]:
    raw_line: str = str(line or "")
    indent_length: int = len(raw_line) - len(raw_line.lstrip())
    indent: str = raw_line[:indent_length]
    stripped_line: str = raw_line[indent_length:]
    protected_prefix: str = ""
    for marker in _SEMANTIC_BULLET_MARKERS:
        marker_with_space: str = f"{marker} "
        if stripped_line.startswith(marker_with_space):
            protected_prefix = f"{indent}{marker_with_space}"
            stripped_line = stripped_line[len(marker_with_space) :]
            break
    sanitized_tail: str = _EMOJI_PATTERN.sub("", stripped_line)
    sanitized_tail = re.sub(r"\s+([,.;:!?])", r"\1", sanitized_tail)
    sanitized_tail = re.sub(r"\s{2,}", " ", sanitized_tail).strip()
    sanitized_line: str = f"{protected_prefix}{sanitized_tail}".strip()
    return (sanitized_line, sanitized_line != raw_line.strip())


def _strip_non_structural_emoji_from_description(description_text: str) -> tuple[str, bool]:
    normalized_description: str = str(description_text or "").replace("\r\n", "\n").replace("\r", "\n")
    sanitized_lines: list[str] = []
    changed: bool = False
    for raw_line in normalized_description.split("\n"):
        if not raw_line.strip():
            sanitized_lines.append("")
            continue
        sanitized_line, line_changed = _strip_non_structural_emoji_from_line(raw_line)
        sanitized_lines.append(sanitized_line)
        changed = changed or line_changed
    sanitized_description: str = "\n".join(sanitized_lines).strip()
    return (sanitized_description, changed)


def _normalize_formatting_only_description(
    *,
    description_text: str,
    language: str,
    source_texts: Sequence[str],
    reason_codes: Sequence[str],
) -> FormattingNormalizationResult:
    normalized_reason_codes: tuple[str, ...] = _normalize_description_validation_reason_codes(
        reason_codes
    )
    normalized_description_text: str = str(description_text or "").strip()
    actions: list[str] = []
    if "excessive_emoji_usage" in normalized_reason_codes:
        stripped_description_text, emoji_changed = _strip_non_structural_emoji_from_description(
            normalized_description_text
        )
        if emoji_changed:
            normalized_description_text = stripped_description_text
            actions.append("reduced_non_structural_emoji")
    quality_result: MergeQualityNormalizationResult = normalize_merge_description(
        description=normalized_description_text,
        language=language,
        source_texts=source_texts,
    )
    if quality_result.description_text != normalized_description_text:
        normalized_description_text = quality_result.description_text
        actions.append("reapplied_merge_quality_normalization")
    return FormattingNormalizationResult(
        description_text=normalized_description_text,
        actions=tuple(actions),
    )


def _attempt_expanded_formatting_recovery(
    *,
    validation_error: Exception,
    merged_content: MergedLanguageContent,
    diagnostics: MergeSemanticDiagnostics,
    videos: Sequence[PlannedVideo],
    official_links_selection: OfficialLinksSelection,
    official_links_fill: OfficialLinksFillResult,
    source_texts: Sequence[str],
    language: str,
    model_name: str,
    attempt_index: int,
    branch_label: str,
    date_key: str,
    slot_key: str,
) -> MergeValidationRecoveryAttempt:
    reason_codes: tuple[str, ...] = _reason_codes_from_error(validation_error)
    if len(videos) < 3 or not _is_formatting_only_validation_failure(reason_codes):
        return MergeValidationRecoveryAttempt(
            merged_content=None,
            diagnostics=None,
            validation_error=validation_error,
            actions=(),
        )
    normalization_result: FormattingNormalizationResult = _normalize_formatting_only_description(
        description_text=merged_content.description,
        language=language,
        source_texts=source_texts,
        reason_codes=reason_codes,
    )
    if (
        normalization_result.description_text == merged_content.description
        or not normalization_result.actions
    ):
        return MergeValidationRecoveryAttempt(
            merged_content=None,
            diagnostics=None,
            validation_error=validation_error,
            actions=normalization_result.actions,
        )
    recovered_content: MergedLanguageContent = dataclasses.replace(
        merged_content,
        description=normalization_result.description_text,
        description_selected=normalization_result.description_text,
        description_audit=normalization_result.description_text,
    )
    recovery_quality_result: MergeQualityNormalizationResult = normalize_merge_description(
        description=recovered_content.description,
        language=language,
        source_texts=source_texts,
    )
    if recovery_quality_result.description_text != recovered_content.description:
        recovered_content = dataclasses.replace(
            recovered_content,
            description=recovery_quality_result.description_text,
            description_selected=recovery_quality_result.description_text,
            description_audit=recovery_quality_result.description_text,
        )
    recovered_diagnostics: MergeSemanticDiagnostics = _build_merge_semantic_diagnostics(
        merged_content=recovered_content,
        videos=videos,
        official_links_selection=official_links_selection,
        official_links_fill=official_links_fill,
        merge_quality=recovery_quality_result.diagnostics,
    )
    try:
        _validate_coverage_preserving_merge_or_raise(
            merged_content=recovered_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=recovery_quality_result.diagnostics,
            precomputed_diagnostics=recovered_diagnostics,
        )
    except Exception as recovered_error:
        LOGGER.info(
            "merge_llm_validation_salvage branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d outcome=revealed_non_formatting_issue reason_codes=%s replacement_reason_codes=%s actions=%s emoji_before=%d emoji_after=%d",
            branch_label,
            date_key,
            slot_key,
            language,
            model_name,
            attempt_index,
            ",".join(reason_codes) or "none",
            ",".join(_reason_codes_from_error(recovered_error)) or "none",
            ",".join(normalization_result.actions) or "none",
            diagnostics.emoji_count,
            recovered_diagnostics.emoji_count,
        )
        return MergeValidationRecoveryAttempt(
            merged_content=None,
            diagnostics=None,
            validation_error=recovered_error,
            actions=normalization_result.actions,
        )
    LOGGER.info(
        "merge_llm_validation_salvage branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d outcome=applied reason_codes=%s actions=%s emoji_before=%d emoji_after=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        model_name,
        attempt_index,
        ",".join(reason_codes) or "none",
        ",".join(normalization_result.actions) or "none",
        diagnostics.emoji_count,
        recovered_diagnostics.emoji_count,
    )
    return MergeValidationRecoveryAttempt(
        merged_content=recovered_content,
        diagnostics=recovered_diagnostics,
        validation_error=None,
        actions=normalization_result.actions,
    )


def _language_name_for_merge_prompt(language: str, llm_language_names_json: str) -> str:
    default_names: dict[str, str] = {
        "uk": "Ukrainian",
        "en": "English",
        "ru": "Russian",
        "other": "the original language of sources",
    }
    try:
        import json

        payload = json.loads(llm_language_names_json)
        if isinstance(payload, dict):
            return str(payload.get(language, payload.get("other", default_names["other"])))
    except Exception:
        pass
    return default_names.get(language, default_names["other"])


def _normalize_source_description_text(text: str) -> str:
    normalized: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def _strip_source_urls_from_text(text: str) -> tuple[str, int]:
    removed_urls: int = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal removed_urls
        removed_urls += 1
        return ""

    cleaned_text: str = _URL_PATTERN.sub(_replace, str(text or ""))
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
    return bool(
        re.fullmatch(
            r"(?im)(?:🌐\s*)?(?:official links|офіційні ресурси|официальные ссылки)\s*:",
            str(text or "").strip(),
        )
    )


def _looks_like_service_tail_paragraph(text: str) -> bool:
    normalized_text: str = re.sub(r"\s+", " ", str(text or "").strip()).lower()
    if not normalized_text:
        return True
    if _is_official_links_heading_line(normalized_text):
        return True
    service_hints: tuple[str, ...] = (
        "watch",
        "join",
        "share",
        "follow",
        "subscribe",
        "learn more",
        "links below",
        "details below",
        "диві",
        "долуч",
        "підпис",
        "смотрите",
        "подпис",
        "подробности",
    )
    semantic_tokens: List[str] = _SEMANTIC_TOKEN_PATTERN.findall(normalized_text)
    return (
        len(normalized_text) <= 220
        and len(semantic_tokens) <= 12
        and any(hint in normalized_text for hint in service_hints)
    )


def _clean_source_description_for_llm(text: str) -> PreparedMergeSourceDescription:
    normalized_text: str = _normalize_source_description_text(text)
    raw_chars: int = len(normalized_text)
    if not normalized_text:
        return PreparedMergeSourceDescription(
            text="",
            raw_chars=0,
            cleaned_chars=0,
            urls_removed=0,
            hashtags_removed=0,
            service_paragraphs_dropped=0,
        )

    cleaned_paragraphs: List[str] = []
    urls_removed: int = 0
    hashtags_removed: int = 0
    service_paragraphs_dropped: int = 0

    for paragraph in _extract_description_paragraphs_raw(normalized_text):
        cleaned_lines: List[str] = []
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
    return PreparedMergeSourceDescription(
        text=cleaned_text,
        raw_chars=raw_chars,
        cleaned_chars=len(cleaned_text),
        urls_removed=urls_removed,
        hashtags_removed=hashtags_removed,
        service_paragraphs_dropped=service_paragraphs_dropped,
    )


def _extract_description_paragraphs_raw(text: str) -> List[str]:
    normalized_text: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized_text) if part.strip()]


def _count_emoji(text: str) -> int:
    emoji_pattern: re.Pattern[str] = re.compile(
        r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
        flags=re.UNICODE,
    )
    return len(emoji_pattern.findall(str(text or "")))


def _bullet_marker_for_line(line: str) -> str:
    stripped: str = str(line or "").strip()
    if not stripped:
        return ""
    for marker in _SEMANTIC_BULLET_MARKERS:
        if stripped.startswith(f"{marker} "):
            return marker
    if _BULLET_PLAIN_PATTERN.match(stripped):
        first_token: str = stripped.split(maxsplit=1)[0]
        return first_token
    return ""


def _is_youtube_host(host: str) -> bool:
    normalized_host: str = str(host or "").strip().lower()
    if not normalized_host:
        return False
    return (
        normalized_host.endswith("youtube.com")
        or normalized_host.endswith("youtu.be")
        or normalized_host.endswith("youtube-nocookie.com")
    )


def _normalize_link_candidate(url: str) -> Optional[str]:
    raw_url: str = str(url or "").strip().strip("<>()[]{}").rstrip(".,;")
    if not raw_url:
        return None
    parts = urlsplit(raw_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    filtered_query_items: List[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key: str = key.lower().strip()
        if normalized_key.startswith("utm_") or normalized_key in _TRACKING_QUERY_KEYS:
            continue
        filtered_query_items.append((key, value))
    sanitized_query: str = urlencode(filtered_query_items, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, sanitized_query, ""))


def _format_duration_label(duration_seconds: Optional[int]) -> str:
    if duration_seconds is None or duration_seconds <= 0:
        return "unknown"
    total_seconds: int = int(duration_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def _fetch_merge_youtube_candidate_metadata(video_url: str) -> VideoMetadata:
    fetcher = YtDlpYouTubeMetadataFetcher()
    return fetcher.fetch(video_url)


def _extract_merge_youtube_candidates(
    videos: Sequence[PlannedVideo],
) -> MergeYouTubeCandidatesResult:
    raw_youtube_urls_found: int = 0
    invalid_youtube_urls_skipped: int = 0
    candidate_urls_by_key: dict[str, str] = {}

    for video in videos:
        description_text: str = str(video.metadata.description or "")
        for match in _URL_PATTERN.finditer(description_text):
            raw_url: str = str(match.group(0) or "").strip()
            if not raw_url:
                continue
            try:
                netloc: str = urlsplit(raw_url).netloc
            except Exception:
                netloc = ""
            if not _is_youtube_host(netloc):
                continue
            raw_youtube_urls_found += 1
            normalized_url: Optional[str] = normalize_youtube_link(raw_url)
            if normalized_url is None:
                invalid_youtube_urls_skipped += 1
                continue
            candidate_urls_by_key.setdefault(normalized_url, normalized_url)

    candidates: List[MergeYouTubeCandidate] = []
    metadata_resolved_count: int = 0
    for normalized_url in candidate_urls_by_key.values():
        candidate_title: str = ""
        candidate_duration_seconds: Optional[int] = None
        candidate_url: str = normalized_url
        metadata_resolved: bool = False
        try:
            metadata: VideoMetadata = _fetch_merge_youtube_candidate_metadata(normalized_url)
            candidate_title = str(metadata.title or "").strip()
            candidate_duration_seconds = metadata.duration_seconds
            candidate_url = (
                normalize_youtube_link(
                    str(metadata.canonical_url or metadata.url or normalized_url).strip()
                )
                or normalized_url
            )
            metadata_resolved = True
            metadata_resolved_count += 1
        except Exception as error:
            LOGGER.info(
                "merge_youtube_candidate_metadata_failed url=%s reason=%s",
                normalized_url,
                error,
            )
        candidates.append(
            MergeYouTubeCandidate(
                url=candidate_url,
                title=candidate_title or "Unknown YouTube stream",
                duration_seconds=candidate_duration_seconds,
                duration_label=_format_duration_label(candidate_duration_seconds),
                metadata_resolved=metadata_resolved,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.duration_seconds is not None,
            candidate.duration_seconds or -1,
            candidate.title.lower(),
            candidate.url,
        ),
        reverse=True,
    )
    return MergeYouTubeCandidatesResult(
        raw_youtube_urls_found=raw_youtube_urls_found,
        invalid_youtube_urls_skipped=invalid_youtube_urls_skipped,
        deduped_candidates=tuple(candidates),
        metadata_resolved_count=metadata_resolved_count,
    )


def _build_youtube_candidates_block(
    candidates_result: MergeYouTubeCandidatesResult,
) -> str:
    if not candidates_result.deduped_candidates:
        return (
            "YOUTUBE CANDIDATES (optional)\n"
            "No valid YouTube candidates were found in merged source descriptions."
        )
    block_lines: List[str] = [
        "YOUTUBE CANDIDATES (optional)",
        "You may include 0, 1, or 2 YouTube URLs from this list only.",
        "Choosing none is valid.",
        "Pick only the most relevant main streams for the final summary.",
        "Prefer longer broadcasts. Do not pick short promo, teaser, clip, or secondary videos.",
        "If you include selected YouTube URLs, place each selected URL on its own line near the end of the description before any official links block or CTA.",
    ]
    for index, candidate in enumerate(candidates_result.deduped_candidates, start=1):
        block_lines.extend(
            (
                f"CANDIDATE {index}",
                f"TITLE: {candidate.title}",
                f"DURATION_SECONDS: {candidate.duration_seconds if candidate.duration_seconds is not None else 'unknown'}",
                f"DURATION_TEXT: {candidate.duration_label}",
                f"URL: {candidate.url}",
            )
        )
    return "\n".join(block_lines).strip()


def _count_youtube_urls_in_text(text: str) -> int:
    seen_urls: set[str] = set()
    for match in _URL_PATTERN.finditer(str(text or "")):
        raw_url: str = str(match.group(0) or "").strip()
        if not raw_url:
            continue
        try:
            netloc: str = urlsplit(raw_url).netloc
        except Exception:
            netloc = ""
        if not _is_youtube_host(netloc):
            continue
        normalized_url: Optional[str] = normalize_youtube_link(raw_url)
        if normalized_url is not None:
            seen_urls.add(normalized_url)
    return len(seen_urls)


def _official_links_heading(language: str) -> str:
    if language == "uk":
        return "🌐 Офіційні ресурси:"
    if language == "ru":
        return "🌐 Официальные ссылки:"
    return "🌐 Official links:"


def _canonical_link_key(url: str) -> str:
    parts = urlsplit(url)
    host: str = parts.netloc.lower().strip()
    path: str = (parts.path or "/").rstrip("/")
    return f"{host}{path}"


def _line_has_official_context(line_text: str) -> bool:
    normalized_line: str = str(line_text or "").strip().lower()
    if not normalized_line:
        return False
    return any(hint in normalized_line for hint in _OFFICIAL_LINK_CONTEXT_HINTS)


def _extract_official_links_from_sources(videos: Sequence[PlannedVideo]) -> OfficialLinksSelection:
    raw_candidates: List[tuple[str, str]] = []
    domain_counts: dict[str, int] = {}
    for video in videos:
        for line in str(video.metadata.description or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            urls: List[str] = [match.group(0) for match in _URL_PATTERN.finditer(line)]
            if not urls:
                continue
            for url in urls:
                normalized_url: Optional[str] = _normalize_link_candidate(url)
                if normalized_url is None:
                    continue
                parts = urlsplit(normalized_url)
                if _is_youtube_host(parts.netloc):
                    continue
                raw_candidates.append((normalized_url, line))
                domain: str = parts.netloc.lower().strip()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if not raw_candidates:
        return OfficialLinksSelection(found_in_sources=0, kept_links=())

    scored_candidates: List[tuple[int, int, str]] = []
    for index, (url, line_text) in enumerate(raw_candidates):
        parts = urlsplit(url)
        domain: str = parts.netloc.lower().strip()
        score: int = 0
        if parts.scheme == "https":
            score += 20
        if _line_has_official_context(line_text):
            score += 20
        score += min(20, domain_counts.get(domain, 1) * 5)
        if parts.query:
            score -= 5
        score += max(0, 15 - (len(url) // 15))
        scored_candidates.append((score, -index, url))

    scored_candidates.sort(reverse=True)
    selected_links: List[str] = []
    seen_keys: set[str] = set()
    for _, _, url in scored_candidates:
        canonical_key: str = _canonical_link_key(url)
        if canonical_key in seen_keys:
            continue
        seen_keys.add(canonical_key)
        selected_links.append(url)
        if len(selected_links) >= 3:
            break
    return OfficialLinksSelection(
        found_in_sources=len(raw_candidates),
        kept_links=tuple(selected_links),
    )


def _description_has_official_links_block(description: str) -> bool:
    normalized: str = str(description or "")
    if not normalized:
        return False
    heading_present: bool = bool(
        re.search(
            r"(?im)^\s*(?:🌐\s*)?(?:official links|офіційні ресурси|официальные ссылки)\s*:\s*$",
            normalized,
        )
    )
    if not heading_present:
        return False
    return any(
        not _is_youtube_host(urlsplit(match.group(0)).netloc)
        for match in _URL_PATTERN.finditer(normalized)
    )


def _count_output_official_links(description: str) -> int:
    seen_keys: set[str] = set()
    for match in _URL_PATTERN.finditer(str(description or "")):
        normalized_url: Optional[str] = _normalize_link_candidate(match.group(0))
        if normalized_url is None:
            continue
        if _is_youtube_host(urlsplit(normalized_url).netloc):
            continue
        seen_keys.add(_canonical_link_key(normalized_url))
    return len(seen_keys)


def _looks_like_close_paragraph(text: str) -> bool:
    normalized: str = str(text or "").strip().lower()
    if not normalized:
        return False
    if "#" in normalized:
        return True
    return any(hint in normalized for hint in ("subscribe", "join", "watch", "follow", "диві", "долуч", "смотрите", "подпис"))


def _inject_official_links_block_if_missing(
    *,
    description: str,
    language: str,
    official_links: Sequence[str],
) -> OfficialLinksFillResult:
    if not official_links:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    if _description_has_official_links_block(description):
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    paragraphs: List[str] = _extract_description_paragraphs_raw(description)
    if not paragraphs:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=0,
            fill_applied=False,
        )
    links_block: str = "\n".join([_official_links_heading(language), *official_links]).strip()
    updated_paragraphs: List[str] = list(paragraphs)
    if len(updated_paragraphs) <= 3:
        if _looks_like_close_paragraph(updated_paragraphs[-1]):
            updated_paragraphs.insert(-1, links_block)
        else:
            updated_paragraphs.append(links_block)
    elif _looks_like_close_paragraph(updated_paragraphs[-1]):
        updated_paragraphs[-1] = f"{links_block}\n{updated_paragraphs[-1].strip()}".strip()
    else:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    updated_description: str = "\n\n".join(
        paragraph for paragraph in updated_paragraphs if paragraph.strip()
    ).strip()
    return OfficialLinksFillResult(
        description=updated_description,
        links_in_output=_count_output_official_links(updated_description),
        fill_applied=True,
    )


def _contains_agenda_heading(text: str) -> bool:
    agenda_headings: tuple[str, ...] = (
        "что в этом стриме",
        "в этом выпуске",
        "о чем поговорим",
        "що в цьому стрімі",
        "про що поговоримо",
        "what's in this stream",
        "what’s in this stream",
        "in this stream",
    )
    lines: List[str] = [
        re.sub(r"\s+", " ", line.strip().lower())
        for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    for line in lines:
        normalized_line: str = line.strip(" -–—:;.!?")
        for heading in agenda_headings:
            if (
                normalized_line == heading
                or normalized_line.startswith(f"{heading}:")
                or normalized_line.startswith(f"{heading} -")
                or normalized_line.startswith(f"{heading} –")
                or normalized_line.startswith(f"{heading} —")
            ):
                return True
    return False


def _looks_like_per_source_dump(text: str) -> bool:
    source_line_pattern: re.Pattern[str] = re.compile(
        r"^\s*(?:source|video)\s*\d+[:.)-]?",
        flags=re.IGNORECASE,
    )
    lines: List[str] = [line.strip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    source_line_hits: int = sum(1 for line in lines if source_line_pattern.match(line))
    if source_line_hits >= 2:
        return True
    lowered_text: str = str(text or "").lower()
    return ("source 1" in lowered_text and "source 2" in lowered_text) or (
        "video 1" in lowered_text and "video 2" in lowered_text
    )


def _extract_named_entities(text: str) -> set[str]:
    entity_pattern: re.Pattern[str] = re.compile(
        r"\b(?:[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'-]{2,})(?:\s+[A-ZА-ЯЁІЇЄҐ][a-zа-яёіїєґ'-]{2,})+\b",
        flags=re.UNICODE,
    )
    return {match.group(0).strip().lower() for match in entity_pattern.finditer(str(text or ""))}


def _source_text_for_semantic_analysis(video: PlannedVideo) -> str:
    cleaned_description: str = _clean_source_description_for_llm(
        video.metadata.description.strip()
    ).text
    description_text: str = cleaned_description or video.metadata.description.strip()
    return f"{video.metadata.title.strip()}\n{description_text}".strip()


def _build_source_coverage_flags(
    *,
    merged_anchors: set[str],
    source_anchors: Sequence[set[str]],
) -> tuple[bool, ...]:
    coverage_flags: List[bool] = []
    for source_index, anchors in enumerate(source_anchors, start=1):
        if len(anchors) < 3:
            coverage_flags.append(True)
            continue
        others: set[str] = set()
        for other_index, other_anchors in enumerate(source_anchors, start=1):
            if other_index == source_index:
                continue
            others.update(other_anchors)
        source_specific: set[str] = anchors - others
        probe: set[str] = source_specific if len(source_specific) >= 3 else anchors
        coverage_flags.append(bool(merged_anchors & probe))
    return tuple(coverage_flags)


def _build_distinctive_source_coverage_metrics(
    *,
    merged_anchors: set[str],
    source_anchors: Sequence[set[str]],
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    coverage_hits_by_item: List[int] = []
    coverage_targets_by_item: List[int] = []
    covered_items: int = 0
    covered_items_total: int = 0
    for source_index, anchors in enumerate(source_anchors, start=1):
        others: set[str] = set()
        for other_index, other_anchors in enumerate(source_anchors, start=1):
            if other_index == source_index:
                continue
            others.update(other_anchors)
        distinctive_anchors: set[str] = anchors - others
        probe_anchors: set[str] = distinctive_anchors if len(distinctive_anchors) >= 3 else set()
        coverage_hit_count: int = len(merged_anchors & probe_anchors)
        if len(probe_anchors) >= 6:
            coverage_target: int = 2
        elif len(probe_anchors) >= 3:
            coverage_target = 1
        else:
            coverage_target = 0
        coverage_hits_by_item.append(coverage_hit_count)
        coverage_targets_by_item.append(coverage_target)
        if coverage_target <= 0:
            continue
        covered_items_total += 1
        if coverage_hit_count >= coverage_target:
            covered_items += 1
    return (
        tuple(coverage_hits_by_item),
        tuple(coverage_targets_by_item),
        covered_items,
        covered_items_total,
    )


def _normalized_char_count(text: str) -> int:
    return len(re.sub(r"\s+", " ", str(text or "").strip()))


def _is_detailed_bullet_line(*, line_text: str, line_anchors: set[str]) -> bool:
    if len(line_anchors) >= 2:
        return True
    if any(character.isdigit() for character in line_text):
        return True
    return bool(_extract_named_entities(line_text))


def _expanded_quality_has_full_source_coverage(
    *,
    source_coverage_by_item: tuple[bool, ...],
    distinctive_coverage_by_item: tuple[int, ...],
    distinctive_coverage_targets: tuple[int, ...],
) -> bool:
    if any(not covered for covered in source_coverage_by_item):
        return False
    for hit_count, target_count in zip(
        distinctive_coverage_by_item,
        distinctive_coverage_targets,
    ):
        if target_count > 0 and hit_count < target_count:
            return False
    return True


def _expanded_compact_body_is_content_rich(
    *,
    body_non_hook_char_count: int,
    bullet_lines_with_anchor_count: int,
    detailed_bullet_count: int,
    thematic_spread_count: int,
    body_anchor_count: int,
    source_coverage_by_item: tuple[bool, ...],
    distinctive_coverage_by_item: tuple[int, ...],
    distinctive_coverage_targets: tuple[int, ...],
    min_detailed_bullets: int,
    min_topic_spread: int,
    min_body_anchor_count: int,
) -> bool:
    if body_non_hook_char_count < 320:
        return False
    if bullet_lines_with_anchor_count < min_detailed_bullets:
        return False
    if detailed_bullet_count < min_detailed_bullets:
        return False
    if thematic_spread_count < min_topic_spread:
        return False
    if body_anchor_count < min_body_anchor_count:
        return False
    return _expanded_quality_has_full_source_coverage(
        source_coverage_by_item=source_coverage_by_item,
        distinctive_coverage_by_item=distinctive_coverage_by_item,
        distinctive_coverage_targets=distinctive_coverage_targets,
    )


def _expanded_quality_reason_detail(reason_code: str) -> str:
    reason_details: dict[str, str] = {
        "too_few_expanded_bullets": "Expanded merge did not produce enough substantive agenda bullets.",
        "insufficient_expanded_body": "Expanded merge body is too thin or too structurally compact without enough dense content after the hook.",
        "weak_source_coverage": "Expanded merge does not preserve distinguishable source-specific coverage.",
        "overly_generic_body": "Expanded merge body stays too generic for a multi-source summary.",
        "insufficient_topic_spread": "Expanded merge agenda does not spread across enough distinct topic nodes.",
        "hook_dominates_body": "Expanded merge hook takes too much of the useful summary volume.",
    }
    return reason_details.get(reason_code, reason_code.replace("_", " "))


def _is_softened_distinctive_source_coverage_eligible(
    *,
    source_count: int,
    source_coverage_by_item: tuple[bool, ...],
    distinctive_coverage_hits: int,
    distinctive_coverage_total: int,
    reason_codes: Sequence[str],
) -> bool:
    normalized_reason_codes: tuple[str, ...] = tuple(
        code for code in dict.fromkeys(reason_codes) if str(code or "").strip()
    )
    return (
        source_count == 3
        and distinctive_coverage_hits == 2
        and distinctive_coverage_total == 3
        and all(source_coverage_by_item)
        and normalized_reason_codes == ("weak_source_coverage",)
    )


def _inactive_expanded_merge_diagnostics() -> ExpandedMergeDiagnostics:
    return ExpandedMergeDiagnostics(
        enabled=False,
        body_paragraph_count=0,
        body_char_count=0,
        body_non_hook_char_count=0,
        compact_body_relaxed=False,
        hook_char_count=0,
        hook_share=0.0,
        bullet_lines_with_anchor_count=0,
        detailed_bullet_count=0,
        thematic_spread_count=0,
        body_anchor_count=0,
        distinctive_coverage_hits=0,
        distinctive_coverage_total=0,
        distinctive_coverage_by_item=(),
        distinctive_coverage_targets=(),
        softened_distinctive_source_coverage_applied=False,
        quality_gate_status="not_applicable",
        quality_gate_reason_codes=(),
        quality_gate_reason_details=(),
    )


def _build_expanded_merge_diagnostics(
    *,
    description_text: str,
    source_count: int,
    merged_anchors: set[str],
    source_anchors: Sequence[set[str]],
    bullet_points_count: int,
    source_coverage_by_item: tuple[bool, ...],
) -> ExpandedMergeDiagnostics:
    if source_count < 3:
        return _inactive_expanded_merge_diagnostics()

    tail_separation: MergeTailSeparationResult = separate_merge_body_and_tail(
        text=description_text
    )
    body_paragraphs: List[str] = _extract_description_paragraphs_raw(
        tail_separation.body_text
    )
    hook_paragraph: str = body_paragraphs[0] if body_paragraphs else ""
    non_hook_body_text: str = "\n\n".join(body_paragraphs[1:]).strip()
    bullet_lines: List[str] = []
    for paragraph in body_paragraphs[1:] if len(body_paragraphs) > 1 else body_paragraphs:
        for raw_line in paragraph.split("\n"):
            stripped_line: str = str(raw_line or "").strip()
            if _bullet_marker_for_line(stripped_line):
                bullet_lines.append(stripped_line)

    bullet_lines_with_anchor_count: int = 0
    detailed_bullet_count: int = 0
    thematic_spread_count: int = 0
    body_anchor_count: int = len(
        _extract_semantic_anchors(non_hook_body_text or tail_separation.body_text)
    )
    seen_bullet_anchors: set[str] = set()
    for bullet_line in bullet_lines:
        line_anchors: set[str] = _extract_semantic_anchors(bullet_line)
        if line_anchors:
            bullet_lines_with_anchor_count += 1
        if not _is_detailed_bullet_line(line_text=bullet_line, line_anchors=line_anchors):
            continue
        detailed_bullet_count += 1
        if line_anchors - seen_bullet_anchors:
            thematic_spread_count += 1
        seen_bullet_anchors.update(line_anchors)

    (
        distinctive_coverage_by_item,
        distinctive_coverage_targets,
        distinctive_coverage_hits,
        distinctive_coverage_total,
    ) = _build_distinctive_source_coverage_metrics(
        merged_anchors=merged_anchors,
        source_anchors=source_anchors,
    )

    min_bullet_points: int = 5 if source_count == 3 else 6
    min_detailed_bullets: int = max(4, min_bullet_points - 1)
    min_topic_spread: int = 3 if source_count == 3 else 4
    min_body_anchor_count: int = max(8, source_count * 3)
    has_full_source_coverage: bool = _expanded_quality_has_full_source_coverage(
        source_coverage_by_item=source_coverage_by_item,
        distinctive_coverage_by_item=distinctive_coverage_by_item,
        distinctive_coverage_targets=distinctive_coverage_targets,
    )
    compact_body_relaxed: bool = (
        tail_separation.body_paragraph_count_after_recovery < 3
        and _expanded_compact_body_is_content_rich(
            body_non_hook_char_count=_normalized_char_count(non_hook_body_text),
            bullet_lines_with_anchor_count=bullet_lines_with_anchor_count,
            detailed_bullet_count=detailed_bullet_count,
            thematic_spread_count=thematic_spread_count,
            body_anchor_count=body_anchor_count,
            source_coverage_by_item=source_coverage_by_item,
            distinctive_coverage_by_item=distinctive_coverage_by_item,
            distinctive_coverage_targets=distinctive_coverage_targets,
            min_detailed_bullets=min_detailed_bullets,
            min_topic_spread=min_topic_spread,
            min_body_anchor_count=min_body_anchor_count,
        )
    )

    reason_codes: List[str] = []
    if (
        bullet_points_count < min_bullet_points
        or detailed_bullet_count < min_detailed_bullets
    ):
        reason_codes.append("too_few_expanded_bullets")
    if (
        _normalized_char_count(non_hook_body_text) < 220
        or (
            tail_separation.body_paragraph_count_after_recovery < 3
            and not compact_body_relaxed
        )
    ):
        reason_codes.append("insufficient_expanded_body")
    if (
        body_anchor_count < min_body_anchor_count
        or bullet_lines_with_anchor_count < min_detailed_bullets
    ):
        reason_codes.append("overly_generic_body")
    if thematic_spread_count < min_topic_spread:
        reason_codes.append("insufficient_topic_spread")
    if not has_full_source_coverage:
        reason_codes.append("weak_source_coverage")

    hook_char_count: int = _normalized_char_count(hook_paragraph)
    body_char_count: int = _normalized_char_count(tail_separation.body_text)
    body_non_hook_char_count: int = _normalized_char_count(non_hook_body_text)
    hook_share: float = (
        (hook_char_count / body_char_count)
        if body_char_count > 0
        else 0.0
    )
    if (
        hook_char_count >= 140
        and hook_share > 0.38
        and body_non_hook_char_count < 320
    ):
        reason_codes.append("hook_dominates_body")

    softened_distinctive_source_coverage_applied: bool = (
        _is_softened_distinctive_source_coverage_eligible(
            source_count=source_count,
            source_coverage_by_item=source_coverage_by_item,
            distinctive_coverage_hits=distinctive_coverage_hits,
            distinctive_coverage_total=distinctive_coverage_total,
            reason_codes=reason_codes,
        )
    )
    if softened_distinctive_source_coverage_applied:
        reason_codes = [
            reason_code
            for reason_code in reason_codes
            if reason_code != "weak_source_coverage"
        ]

    unique_reason_codes: tuple[str, ...] = tuple(dict.fromkeys(reason_codes))
    return ExpandedMergeDiagnostics(
        enabled=True,
        body_paragraph_count=tail_separation.body_paragraph_count_after_recovery,
        body_char_count=body_char_count,
        body_non_hook_char_count=body_non_hook_char_count,
        compact_body_relaxed=compact_body_relaxed,
        hook_char_count=hook_char_count,
        hook_share=hook_share,
        bullet_lines_with_anchor_count=bullet_lines_with_anchor_count,
        detailed_bullet_count=detailed_bullet_count,
        thematic_spread_count=thematic_spread_count,
        body_anchor_count=body_anchor_count,
        distinctive_coverage_hits=distinctive_coverage_hits,
        distinctive_coverage_total=distinctive_coverage_total,
        distinctive_coverage_by_item=distinctive_coverage_by_item,
        distinctive_coverage_targets=distinctive_coverage_targets,
        softened_distinctive_source_coverage_applied=softened_distinctive_source_coverage_applied,
        quality_gate_status="hard_reject" if unique_reason_codes else "pass",
        quality_gate_reason_codes=unique_reason_codes,
        quality_gate_reason_details=tuple(
            _expanded_quality_reason_detail(reason_code)
            for reason_code in unique_reason_codes
        ),
    )


def _build_merge_semantic_diagnostics(
    *,
    merged_content: MergedLanguageContent,
    videos: Sequence[PlannedVideo],
    official_links_selection: OfficialLinksSelection,
    official_links_fill: OfficialLinksFillResult,
    merge_quality: MergeQualityDiagnostics,
) -> MergeSemanticDiagnostics:
    description_text: str = str(merged_content.description or "").strip()
    merged_text_for_anchors: str = f"{merged_content.title.strip()}\n{description_text}"
    merged_anchors: set[str] = _extract_semantic_anchors(merged_text_for_anchors)
    source_anchors: List[set[str]] = [
        _extract_semantic_anchors(_source_text_for_semantic_analysis(video))
        for video in videos
    ]
    source_coverage_by_item: tuple[bool, ...] = _build_source_coverage_flags(
        merged_anchors=merged_anchors,
        source_anchors=source_anchors,
    )
    source_entity_candidates: set[str] = set()
    for video in videos:
        source_entity_candidates.update(
            _extract_named_entities(
                f"{video.metadata.title.strip()}\n{video.metadata.description.strip()}"
            )
        )
    merged_entities: set[str] = _extract_named_entities(
        f"{merged_content.title.strip()}\n{description_text}"
    )
    named_entities_preserved: int = len(source_entity_candidates & merged_entities)
    marker_types: List[str] = []
    semantic_bullets_count: int = 0
    bullets_with_plain_marker_count: int = 0
    for line in description_text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        marker: str = _bullet_marker_for_line(line)
        if not marker:
            continue
        if marker in _SEMANTIC_BULLET_MARKER_SET:
            semantic_bullets_count += 1
        else:
            bullets_with_plain_marker_count += 1
        if marker not in marker_types:
            marker_types.append(marker)
    bullet_points_count: int = semantic_bullets_count + bullets_with_plain_marker_count
    agenda_heading_present: bool = _contains_agenda_heading(description_text)
    paragraphs: List[str] = _extract_description_paragraphs_raw(description_text)
    first_paragraph: str = paragraphs[0] if paragraphs else ""
    hook_present: bool = len(first_paragraph) >= 60 and (
        "!" in first_paragraph or "?" in first_paragraph or ":" in first_paragraph
    )
    expanded_quality: ExpandedMergeDiagnostics = _build_expanded_merge_diagnostics(
        description_text=description_text,
        source_count=len(videos),
        merged_anchors=merged_anchors,
        source_anchors=source_anchors,
        bullet_points_count=bullet_points_count,
        source_coverage_by_item=source_coverage_by_item,
    )
    return MergeSemanticDiagnostics(
        hook_present=hook_present,
        agenda_block_present=(bullet_points_count >= 3) or agenda_heading_present,
        bullet_points_count=bullet_points_count,
        semantic_bullets_count=semantic_bullets_count,
        bullets_with_emoji_count=semantic_bullets_count,
        bullets_with_plain_marker_count=bullets_with_plain_marker_count,
        bullet_marker_types=tuple(marker_types),
        named_entities_preserved=named_entities_preserved,
        source_named_entities_total=len(source_entity_candidates),
        emoji_count=_count_emoji(description_text),
        source_coverage_hits=sum(1 for value in source_coverage_by_item if value),
        source_coverage_total=len(source_coverage_by_item),
        source_coverage_by_item=source_coverage_by_item,
        official_links_found_in_sources=official_links_selection.found_in_sources,
        official_links_kept=len(official_links_selection.kept_links),
        official_links_in_output=official_links_fill.links_in_output,
        official_links_fill_applied=official_links_fill.fill_applied,
        merge_quality=merge_quality,
        expanded_quality=expanded_quality,
    )


def _log_merge_style_diagnostics(
    *,
    diagnostics: MergeSemanticDiagnostics,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    model_name: str,
    attempt_index: int,
) -> None:
    coverage_by_item_text: str = ",".join(
        f"{index}:{'yes' if covered else 'no'}"
        for index, covered in enumerate(diagnostics.source_coverage_by_item, start=1)
    )
    marker_types_text: str = ",".join(diagnostics.bullet_marker_types) or "none"
    LOGGER.info(
        "merge_style_coverage branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d style_contract_version=%s hook_present=%s agenda_block_present=%s bullet_points_count=%d semantic_bullets_count=%d bullets_with_emoji_count=%d bullets_with_plain_marker_count=%d bullet_marker_types=%s neutral_bullets_count=%d accent_bullets_count=%d accent_marker_types=%s accent_overflow=%s block_spacing_ok=%s named_entities_preserved=%d source_named_entities_total=%d named_entities_metric=%s emoji_count=%d source_coverage_total=%d/%d source_coverage_ok=%s source_coverage_by_item=%s official_links_found_in_sources=%d official_links_kept=%d official_links_in_output=%d official_links_fill_applied=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        model_name,
        attempt_index,
        STYLE_CONTRACT_VERSION,
        "yes" if diagnostics.hook_present else "no",
        "yes" if diagnostics.agenda_block_present else "no",
        diagnostics.bullet_points_count,
        diagnostics.semantic_bullets_count,
        diagnostics.bullets_with_emoji_count,
        diagnostics.bullets_with_plain_marker_count,
        marker_types_text,
        diagnostics.merge_quality.neutral_bullets_count,
        diagnostics.merge_quality.accent_bullets_count,
        ",".join(diagnostics.merge_quality.accent_marker_types) or "none",
        "yes" if diagnostics.merge_quality.accent_overflow else "no",
        "yes" if diagnostics.merge_quality.block_spacing_ok else "no",
        diagnostics.named_entities_preserved,
        diagnostics.source_named_entities_total,
        "informational",
        diagnostics.emoji_count,
        diagnostics.source_coverage_hits,
        diagnostics.source_coverage_total,
        (
            "yes"
            if diagnostics.source_coverage_hits == diagnostics.source_coverage_total
            else "no"
        ),
        coverage_by_item_text or "none",
        diagnostics.official_links_found_in_sources,
        diagnostics.official_links_kept,
        diagnostics.official_links_in_output,
        "yes" if diagnostics.official_links_fill_applied else "no",
    )
    LOGGER.info(
        "merge_semantic_gate branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d block_language_expected=%s hook_language_detected=%s lead_in_language_detected=%s links_heading_language_detected=%s cta_language_detected=%s language_consistency_ok=%s wrong_language_heading_detected=%s person_role_claims_detected=%d suspicious_role_labels_detected=%s role_softening_applied=%s script_mix_detected=%s script_mix_suspects=%s semantic_gate_status=%s semantic_gate_reason_codes=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        model_name,
        attempt_index,
        diagnostics.merge_quality.block_language_expected,
        diagnostics.merge_quality.hook_language_detected,
        diagnostics.merge_quality.lead_in_language_detected,
        diagnostics.merge_quality.links_heading_language_detected,
        diagnostics.merge_quality.cta_language_detected,
        "yes" if diagnostics.merge_quality.language_consistency_ok else "no",
        "yes" if diagnostics.merge_quality.wrong_language_heading_detected else "no",
        diagnostics.merge_quality.person_role_claims_detected,
        ",".join(diagnostics.merge_quality.suspicious_role_labels_detected) or "none",
        "yes" if diagnostics.merge_quality.role_softening_applied else "no",
        "yes" if diagnostics.merge_quality.script_mix_detected else "no",
        ",".join(diagnostics.merge_quality.script_mix_suspects) or "none",
        diagnostics.merge_quality.semantic_gate_status,
        ",".join(diagnostics.merge_quality.semantic_gate_reason_codes) or "none",
    )
    if diagnostics.expanded_quality.enabled:
        distinctive_coverage_text: str = ",".join(
            f"{index}:{hit_count}/{target_count}"
            for index, (hit_count, target_count) in enumerate(
                zip(
                    diagnostics.expanded_quality.distinctive_coverage_by_item,
                    diagnostics.expanded_quality.distinctive_coverage_targets,
                ),
                start=1,
            )
        ) or "none"
        LOGGER.info(
            "merge_expanded_quality_gate branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d source_count=%d body_paragraph_count=%d body_char_count=%d body_non_hook_char_count=%d compact_body_relaxed=%s hook_char_count=%d hook_share=%.2f bullet_lines_with_anchor_count=%d detailed_bullet_count=%d thematic_spread_count=%d body_anchor_count=%d distinctive_source_coverage=%d/%d distinctive_source_coverage_by_item=%s softened_distinctive_source_coverage_applied=%s quality_gate_status=%s quality_gate_reason_codes=%s quality_gate_reason_details=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            model_name,
            attempt_index,
            diagnostics.source_coverage_total,
            diagnostics.expanded_quality.body_paragraph_count,
            diagnostics.expanded_quality.body_char_count,
            diagnostics.expanded_quality.body_non_hook_char_count,
            "yes" if diagnostics.expanded_quality.compact_body_relaxed else "no",
            diagnostics.expanded_quality.hook_char_count,
            diagnostics.expanded_quality.hook_share,
            diagnostics.expanded_quality.bullet_lines_with_anchor_count,
            diagnostics.expanded_quality.detailed_bullet_count,
            diagnostics.expanded_quality.thematic_spread_count,
            diagnostics.expanded_quality.body_anchor_count,
            diagnostics.expanded_quality.distinctive_coverage_hits,
            diagnostics.expanded_quality.distinctive_coverage_total,
            distinctive_coverage_text,
            (
                "yes"
                if diagnostics.expanded_quality.softened_distinctive_source_coverage_applied
                else "no"
            ),
            diagnostics.expanded_quality.quality_gate_status,
            ",".join(diagnostics.expanded_quality.quality_gate_reason_codes) or "none",
            " | ".join(diagnostics.expanded_quality.quality_gate_reason_details) or "none",
        )


def _token_to_semantic_anchor(token: str) -> str:
    token_lower: str = str(token or "").lower().strip()
    if not token_lower:
        return ""
    if token_lower in _SEMANTIC_STOPWORDS:
        return ""
    has_digit: bool = any(char.isdigit() for char in token_lower)
    min_length: int = 2 if has_digit else 4
    if len(token_lower) < min_length:
        return ""
    if has_digit:
        return token_lower
    return token_lower[:6] if len(token_lower) > 6 else token_lower


def _extract_semantic_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    for token in _SEMANTIC_TOKEN_PATTERN.findall(str(text or "")):
        anchor: str = _token_to_semantic_anchor(token)
        if anchor:
            anchors.add(anchor)
    return anchors


def _validate_coverage_preserving_merge_or_raise(
    *,
    merged_content: MergedLanguageContent,
    videos: Sequence[PlannedVideo],
    official_links_selection: OfficialLinksSelection,
    official_links_fill: OfficialLinksFillResult,
    merge_quality: MergeQualityDiagnostics,
    precomputed_diagnostics: Optional[MergeSemanticDiagnostics] = None,
) -> MergeSemanticDiagnostics:
    diagnostics: MergeSemanticDiagnostics = (
        precomputed_diagnostics
        if precomputed_diagnostics is not None
        else _build_merge_semantic_diagnostics(
            merged_content=merged_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=merge_quality,
        )
    )
    merged_anchors: set[str] = _extract_semantic_anchors(
        f"{merged_content.title.strip()}\n{merged_content.description.strip()}"
    )
    if _looks_like_per_source_dump(merged_content.description):
        raise _build_description_validation_failure(
            reason_codes=("per_source_enumeration",),
            message="description validation failed: per_source_enumeration",
        )
    if len(merged_anchors) < 3:
        raise _build_description_validation_failure(
            reason_codes=("semantic_too_generic",),
            message="description validation failed: semantic_too_generic",
        )
    if diagnostics.emoji_count > 10:
        raise _build_description_validation_failure(
            reason_codes=("excessive_emoji_usage",),
            message="description validation failed: excessive_emoji_usage",
        )
    if merge_quality.semantic_gate_status == "hard_reject":
        reason_codes: str = ",".join(merge_quality.semantic_gate_reason_codes) or "semantic_gate"
        raise _build_description_validation_failure(
            reason_codes=merge_quality.semantic_gate_reason_codes or ("semantic_gate",),
            message=f"description validation failed: {reason_codes}",
        )
    if diagnostics.expanded_quality.quality_gate_status == "hard_reject":
        reason_codes = ",".join(diagnostics.expanded_quality.quality_gate_reason_codes)
        raise _build_description_validation_failure(
            reason_codes=diagnostics.expanded_quality.quality_gate_reason_codes,
            message=f"description validation failed: {reason_codes}",
        )

    source_anchors: List[set[str]] = [
        _extract_semantic_anchors(_source_text_for_semantic_analysis(video))
        for video in videos
    ]
    all_source_anchors: set[str] = set().union(*source_anchors) if source_anchors else set()
    if all_source_anchors:
        overlap_count: int = len(merged_anchors & all_source_anchors)
        source_anchor_count: int = len(all_source_anchors)
        if source_anchor_count <= 6:
            min_overlap = 1
        elif source_anchor_count <= 12:
            min_overlap = 2
        else:
            min_overlap = max(3, min(12, source_anchor_count // 6))
        if overlap_count < min_overlap:
            raise _build_description_validation_failure(
                reason_codes=("semantic_source_grounding_too_low",),
                message="description validation failed: semantic_source_grounding_too_low",
            )

    for source_index, covered in enumerate(diagnostics.source_coverage_by_item, start=1):
        if not covered:
            raise _build_description_validation_failure(
                reason_codes=("weak_source_coverage",),
                message=(
                    "description validation failed: "
                    f"semantic_source_{source_index}_coverage_missing"
                ),
            )
    return diagnostics


def build_llm_merge_prompt_text(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    no_description_text: str,
    expanded_retry_profile: Optional[ExpandedRetryProfile] = None,
) -> str:
    if len(videos) < 2:
        raise ValueError("Expected at least 2 videos for merged generation.")
    language_name: str = _language_name_for_merge_prompt(
        language,
        config.templates.llm_language_names_json,
    )
    source_blocks: List[str] = []
    raw_source_chars_total: int = 0
    cleaned_source_chars_total: int = 0
    for index, video in enumerate(videos, start=1):
        prepared_description: PreparedMergeSourceDescription = _clean_source_description_for_llm(
            video.metadata.description.strip() or no_description_text
        )
        description_for_prompt: str = prepared_description.text or no_description_text
        raw_source_chars_total += prepared_description.raw_chars
        cleaned_source_chars_total += len(description_for_prompt)
        LOGGER.info(
            "merge_source_text_prepared language=%s source_index=%d row=%s raw_chars=%d cleaned_chars=%d urls_removed=%d hashtags_removed=%d service_paragraphs_dropped=%d hard_truncation=disabled",
            language,
            index,
            getattr(video, "row_number", "unknown"),
            prepared_description.raw_chars,
            len(description_for_prompt),
            prepared_description.urls_removed,
            prepared_description.hashtags_removed,
            prepared_description.service_paragraphs_dropped,
        )
        source_blocks.append(
            "\n".join(
                [
                    f"SOURCE {index}",
                    f"TITLE: {video.metadata.title.strip()}",
                    f"DESCRIPTION: {description_for_prompt}",
                ]
            )
        )
    LOGGER.info(
        "merge_prompt_sources_ready language=%s source_count=%d raw_source_chars_total=%d cleaned_source_chars_total=%d hard_truncation=disabled",
        language,
        len(videos),
        raw_source_chars_total,
        cleaned_source_chars_total,
    )
    contract_mode: MergeContractMode = _select_merge_contract_mode(
        source_count=len(videos)
    )
    LOGGER.info(
        "merge_prompt_contract_selected language=%s source_count=%d contract_mode=%s expected_bullet_range=%s expanded_structure_enabled=%s",
        language,
        contract_mode.source_count,
        contract_mode.mode_label,
        contract_mode.bullet_range_label,
        "yes" if contract_mode.expanded_structure_enabled else "no",
    )
    contract_block: str = _merge_contract_block_with_retry(
        contract_mode=contract_mode,
        expanded_retry_profile=expanded_retry_profile,
    )
    link_policy_block: str = (
        "SYSTEM LINK POLICY\n"
        "Do not include any URLs in the output.\n"
        "Do not add a recommended materials block or an official links block.\n"
        "Link blocks will be assembled later by the system."
    )
    cross_domain_sentence_policy_block: str = (
        "CROSS-DOMAIN SENTENCE POLICY\n"
        "If the sources touch different semantic domains, do not compress them into one sentence.\n"
        "Especially do not merge medicine or biology or neurobiology, climate or weather or ecology, disasters or geophysics or natural hazards, psychology or thinking or behavior, and social or moral or civilizational conclusions into one sentence.\n"
        "These topics may stay in one final description, but present them as separate lines of discussion in separate sentences.\n"
        "Do not build one long cause-and-effect chain across all of those domains in a single sentence.\n"
        "Several calm sentences are better than one overloaded super-sentence."
    )
    template_prompt: str = str(config.templates.llm_merge_title_description_prompt or "").strip()
    if template_prompt:
        formatted_template: str = template_prompt.format(
            language_name=language_name,
            sources_block="\n\n".join(source_blocks),
            youtube_candidates_block="",
            merge_contract_block=contract_block,
        ).strip()
        if "{merge_contract_block}" not in template_prompt:
            formatted_template = (
                f"{formatted_template}\n\n{contract_block}"
            ).strip()
        return f"{formatted_template}\n\n{cross_domain_sentence_policy_block}\n\n{link_policy_block}".strip()
    return (
        "You are writing a YouTube stream title and description.\n"
        f"Write output only in {language_name}.\n"
        "Use only facts explicitly present in the source descriptions.\n"
        "Treat the sources as one complete stream, not as a list of separate videos.\n"
        "Generate a new final title, not a copy of any single source title.\n"
        "Do not use emoji in the title.\n"
        "Mentally extract key points from each source, preserve all non-trivial source-specific points,\n"
        "combine overlaps, compress repetition, and produce one coherent final description.\n"
        "Write a strong native YouTube title no longer than 99 characters.\n"
        f"{contract_block}\n"
        "The description must cover all source inputs that were merged.\n"
        "Do not drop a source-specific fact, event, or angle without clear overlap-based reason.\n"
        "Preserve important recognizable names from sources when relevant; never invent names.\n"
        "Avoid asserting strong person titles or role labels unless they are clearly necessary and well-supported by the sources.\n"
        f"{cross_domain_sentence_policy_block}\n"
        f"{link_policy_block}\n"
        "An optional one-line closing sentence should be a light practical CTA with 2 to 5 hashtags.\n"
        "Do not enumerate sources as 1) 2) 3).\n"
        "Do not write a dry digest, protocol, or generic CTA block.\n"
        "Do not output generic slogans, abstract editorial text, or propagandistic phrasing.\n"
        "Do not replace concrete facts with broad statements like 'an important conversation about everything'.\n"
        'Output only one strict JSON object with exactly these keys: title, description.\n\n'
        f"{'\n\n'.join(source_blocks)}"
    ).strip()


def _merge_contract_block_with_retry(
    *,
    contract_mode: MergeContractMode,
    expanded_retry_profile: Optional[ExpandedRetryProfile],
) -> str:
    retry_block: str = _expanded_retry_reinforcement_block(expanded_retry_profile)
    if not contract_mode.expanded_structure_enabled or not retry_block:
        return contract_mode.contract_block
    return f"{contract_mode.contract_block}\n{retry_block}".strip()


def _select_merge_contract_mode(*, source_count: int) -> MergeContractMode:
    if source_count >= 3:
        return MergeContractMode(
            mode_label="expanded",
            source_count=source_count,
            bullet_range_label="6-9",
            expanded_structure_enabled=True,
            contract_block=(
                "Use the expanded merge contract for 3 or more source items.\n"
                "Write one cohesive stream description in 3 to 4 compact paragraphs.\n"
                "Paragraph 1 (hook): write 1 to 2 sentences grounded in the main tension, risk, or key conflict.\n"
                "Keep the hook editorial and readable, but never clickbait.\n"
                "After the hook, use a more open agenda structure instead of one overloaded thesis block.\n"
                "Write 6 to 9 short bullet lines total.\n"
                "You may organize the bullets into 2 to 3 thematic micro-blocks when that improves clarity.\n"
                "Each bullet line must start with exactly one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.\n"
                "Most bullets should start with 🔹.\n"
                "Accent markers are rare and optional; use no more than 3 accent markers per description.\n"
                "Keep marker usage controlled and readable; do not use dash-only bullets as the sole style.\n"
                "Do not present the agenda as SOURCE 1 / SOURCE 2 / SOURCE 3.\n"
                "Keep agenda points specific and factual, not generic placeholders.\n"
                "Do not let the opening hook consume most of the useful summary space.\n"
                "A polished opening is never a substitute for a concrete multi-angle summary.\n"
                "Treat the post-hook body as the main payload and let it carry most of the concrete information.\n"
                "Make most bullets fact-bearing: anchor them with names, places, institutions, numbers, timings, events, or operational consequences whenever the sources provide them.\n"
                "Give the body at least two clearly substantive agenda lanes after the hook instead of one thin run of near-duplicate bullets.\n"
                "Across the agenda, preserve distinguishable source details such as names, places, numbers, events, or clearly separate thematic nodes whenever the sources provide them.\n"
                "Across 3 or more sources, spread the bullets across multiple source lines or topic nodes so the summary does not collapse into one generic lane.\n"
                "If the merged sources span different domains, separate them across different bullets or short thematic blocks instead of compressing them into one universal bullet.\n"
                "Do not combine science or medicine, climate or environment, disasters or catastrophic hazards, psychology or cognition or behavior, and broad social or moral conclusions into one bullet or one cause-and-effect chain unless the sources explicitly require that connection.\n"
                "Thematic grouping is encouraged when useful: research, hazards, human behavior, practical risk, public meaning, or response can be separated into different bullets or micro-blocks.\n"
                "The description must still cover all merged source items and preserve key concrete facts from each source."
            ),
        )
    return MergeContractMode(
        mode_label="compact",
        source_count=source_count,
        bullet_range_label="4-7",
        expanded_structure_enabled=False,
        contract_block=(
            "Use the compact merge contract for 1 to 2 source items.\n"
            "Write one cohesive stream description in 2 to 3 compact paragraphs.\n"
            "Paragraph 1 (hook): write 1 to 2 sentences grounded in the main tension, risk, or key conflict.\n"
            "Keep the hook editorial and readable, but never clickbait.\n"
            "Paragraph 2 (theses block): start with one short lead-in sentence like 'In this stream you'll see:' in the target language.\n"
            "Then write 4 to 7 short thesis bullet lines.\n"
            "Each bullet line must start with exactly one allowed marker: 🔹 📌 🎤 🎥 ⚖ 🌐 ✅.\n"
            "Most bullets should start with 🔹.\n"
            "Accent markers are rare and optional; use no more than 3 accent markers per theses block.\n"
            "Keep marker usage controlled and readable; do not use dash-only bullets as the sole style.\n"
            "Do not present the agenda as SOURCE 1 / SOURCE 2.\n"
            "Keep agenda points specific and factual, not generic placeholders.\n"
            "The description must still cover all merged source items and preserve key concrete facts from each source."
        ),
    )


def _structured_merge_schema() -> dict[str, object]:
    return {
        "name": "restreamer_merge_summary_v2",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "description"],
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 99},
                "description": {"type": "string", "minLength": 1},
            },
        },
    }


def _single_source_translate_prompt(
    *,
    source_language: str,
    target_language: str,
    source_description: str,
) -> str:
    return (
        "You are a precise editor and translator.\n"
        f"Source language: {source_language}.\n"
        f"Target language: {target_language}.\n"
        "Task: translate and lightly rewrite for readability while preserving facts.\n"
        "Return only plain paragraph text.\n"
        "No headings, JSON, markdown, CTA, hashtags, or links.\n\n"
        f"{source_description.strip()}"
    ).strip()


def _request_plain_text(
    *,
    prompt_text: str,
    config: AppConfig,
    provider: LlmProvider,
    model_name: str,
    attempt_label: str,
    trace_context: LlmTraceContext,
) -> str:
    if provider.pre_delay_sec(config=config) > 0:
        time.sleep(max(0.0, float(provider.pre_delay_sec(config=config))))
    response: OpenAITransportResult = provider.request_merge(
        prompt_text=prompt_text,
        config=config,
        model_name=model_name,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=None,
        temperature=0.0,
        trace_context=trace_context,
    )
    return response.raw_text


def _request_structured_merge_payload(
    *,
    prompt_text: str,
    config: AppConfig,
    provider: LlmProvider,
    model_name: str,
    attempt_label: str,
    trace_context: LlmTraceContext,
) -> OpenAITransportResult:
    if provider.pre_delay_sec(config=config) > 0:
        time.sleep(max(0.0, float(provider.pre_delay_sec(config=config))))
    return provider.request_merge(
        prompt_text=prompt_text,
        config=config,
        model_name=model_name,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=_structured_merge_schema(),
        temperature=0.0,
        trace_context=trace_context,
    )


def _apply_merge_polish_stage(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    source_model: str,
    polish_provider: LlmProvider,
    polish_model: str,
    merged_content: MergedLanguageContent,
    attempt_label: str,
    branch_label: str,
    date_key: str,
    slot_key: str,
    merge_run_summary: Optional[MergeRunSummary],
) -> tuple[MergedLanguageContent, MergePolishResult]:
    if merge_run_summary is not None:
        merge_run_summary.record_polish_requested()
    LOGGER.info(
        "merge_polish_requested branch=%s date_key=%s slot_key=%s language=%s source_model=%s polish_model=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        source_model,
        polish_model,
    )
    original_title: str = str(merged_content.title or "").strip()
    original_description: str = str(merged_content.description or "").strip()
    try:
        response: OpenAITransportResult = _request_structured_merge_payload(
            prompt_text=build_merge_polish_prompt(
                language=language,
                title_text=original_title,
                description_text=original_description,
            ),
            config=config,
            provider=polish_provider,
            model_name=polish_model,
            attempt_label=attempt_label,
            trace_context=LlmTraceContext(
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                provider=polish_provider.name,
                model_name=polish_model,
                attempt_index=1,
                request_kind="merge_polish",
                source_count=len(videos),
            ),
        )
        polished_content, _ = parse_merge_response_or_raise(
            provider_name=polish_provider.name,
            model_name=polish_model,
            raw_text=response.raw_text,
            structured_payload=response.structured_payload,
        )
        original_body = separate_merge_body_and_tail(text=original_description)
        polished_body = separate_merge_body_and_tail(text=polished_content.description)
        if original_body.body_paragraph_count_after_recovery != polished_body.body_paragraph_count_after_recovery:
            raise RuntimeError("structure_changed")
        if bullet_marker_count(original_description) != bullet_marker_count(polished_content.description):
            raise RuntimeError("structure_changed")
        if is_too_aggressive_rewrite(
            original_text=f"{original_title}\n{original_description}",
            polished_text=f"{polished_content.title}\n{polished_content.description}",
        ):
            raise RuntimeError("too_aggressive_rewrite")
        quality_result: MergeQualityNormalizationResult = normalize_merge_description(
            description=polished_content.description,
            language=language,
            source_texts=tuple(
                f"{video.metadata.title.strip()}\n{video.metadata.description.strip()}"
                for video in videos
            ),
        )
        if quality_result.description_text != polished_content.description:
            polished_content = dataclasses.replace(
                polished_content,
                description=quality_result.description_text,
                description_selected=quality_result.description_text,
                description_audit=quality_result.description_text,
            )
        official_links_selection: OfficialLinksSelection = _extract_official_links_from_sources(
            videos
        )
        official_links_fill: OfficialLinksFillResult = _inject_official_links_block_if_missing(
            description=polished_content.description,
            language=language,
            official_links=official_links_selection.kept_links,
        )
        if official_links_fill.fill_applied:
            polished_content = dataclasses.replace(
                polished_content,
                description=official_links_fill.description,
                description_selected=official_links_fill.description,
                description_audit=official_links_fill.description,
            )
        _validate_coverage_preserving_merge_or_raise(
            merged_content=polished_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=quality_result.diagnostics,
        )
        if merge_run_summary is not None:
            merge_run_summary.record_polish_accepted()
        LOGGER.info(
            "merge_polish_accepted branch=%s date_key=%s slot_key=%s language=%s source_model=%s polish_model=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            source_model,
            polish_model,
        )
        return (
            polished_content,
            MergePolishResult(
                source_model=source_model,
                polish_model=polish_model,
                original_text=f"{original_title}\n{original_description}",
                polished_text=f"{polished_content.title}\n{polished_content.description}",
                accepted=True,
                reject_reason=None,
                validation_passed=True,
            ),
        )
    except LlmModelConfigurationError:
        raise
    except Exception as error:
        reject_reason: str = (
            "too_aggressive_rewrite"
            if str(error) == "too_aggressive_rewrite"
            else infer_polish_reject_reason(str(error))
        )
        if merge_run_summary is not None:
            merge_run_summary.record_polish_rejected()
        LOGGER.warning(
            "merge_polish_discarded branch=%s date_key=%s slot_key=%s language=%s source_model=%s polish_model=%s reason=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            source_model,
            polish_model,
            reject_reason,
        )
        return (
            merged_content,
            MergePolishResult(
                source_model=source_model,
                polish_model=polish_model,
                original_text=f"{original_title}\n{original_description}",
                polished_text="",
                accepted=False,
                reject_reason=reject_reason,
                validation_passed=False,
            ),
        )


def attempt_openai_single_source_translate_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    return attempt_llm_single_source_translate_with_audit(
        language=language,
        videos=videos,
        config=config,
        attempt_label=attempt_label,
        summarize_error=summarize_error,
        no_description_text=no_description_text,
        merge_run_summary=merge_run_summary,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )


def attempt_llm_single_source_translate_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    del merge_run_summary
    if len(videos) != 1:
        raise RuntimeError("single-source translate expects exactly one video")
    source_video: PlannedVideo = videos[0]
    model_name: str = resolve_effective_llm_model(config)
    provider: LlmProvider = get_llm_provider(config=config)
    main_source_label: str = _main_stage_field_source(provider_name=provider.name)
    try:
        raw_text: str = _request_plain_text(
            prompt_text=_single_source_translate_prompt(
                source_language=source_video.language,
                target_language=language,
                source_description=source_video.metadata.description.strip() or no_description_text,
            ),
            config=config,
            provider=provider,
            model_name=model_name,
            attempt_label=attempt_label,
            trace_context=LlmTraceContext(
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                provider=provider.name,
                model_name=model_name,
                attempt_index=1,
                request_kind="single_source_plain",
                source_count=1,
            ),
        )
        cleaned_text, is_valid, reasons = clean_and_validate_llm_description(text=raw_text)
        if not is_valid:
            raise RuntimeError(
                "single-source plain validation failed: " + ("; ".join(reasons) or "unknown")
            )
        merged_content: MergedLanguageContent = build_plain_merged_content_or_raise(
            model_name=model_name,
            title_text=source_video.metadata.title.strip() or "Untitled",
            description_text=cleaned_text,
        )
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text=raw_text,
            merged=dataclasses.replace(
                merged_content,
                branch_type=BRANCH_MERGE,
                title_source=main_source_label,
                hook_source=main_source_label,
                hashtags_source=main_source_label,
                body_source="main_merge",
                block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
            ),
            error_summary=None,
            salvaged_title=merged_content.title,
            publish_source_label="single_source_plain_ok",
            generator_model_name=model_name,
            used_model_names=(model_name,),
            branch_type=BRANCH_MERGE,
            title_source=main_source_label,
            hook_source=main_source_label,
            hashtags_source=main_source_label,
            body_source="main_merge",
            block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
        )
    except LlmModelConfigurationError:
        raise
    except Exception as error:
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text="",
            merged=None,
            error_summary=summarize_error(error),
            salvaged_title=source_video.metadata.title.strip() or "Untitled",
            publish_source_label="single_source_plain_failed",
            generator_model_name=model_name,
            used_model_names=(model_name,),
            branch_type=BRANCH_MERGE,
            title_source=_fallback_title_source_label(),
            hook_source=_fallback_hook_source_label(),
            hashtags_source="fallback_none",
            body_source=_fallback_body_source_label(),
            block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
        )


def _log_merge_attempt_start(
    *,
    provider_name: str,
    model_name: str,
    attempt_index: int,
    is_fallback: bool,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
) -> None:
    stage_name: str = "fallback" if is_fallback else "primary"
    if is_fallback:
        LOGGER.info(
            "merge_llm_fallback_start branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=%s attempt=%d model=%s",
            branch_label,
            date_key,
            slot_key,
            language,
            provider_name,
            stage_name,
            attempt_index,
            model_name,
        )
        return
    LOGGER.info(
        "merge_llm_primary_attempt branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=%s attempt=%d model=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        provider_name,
        stage_name,
        attempt_index,
        model_name,
    )


def _log_merge_attempt_invalid(
    *,
    provider_name: str,
    model_name: str,
    attempt_index: int,
    is_fallback: bool,
    branch_label: str,
    date_key: str,
    slot_key: str,
    language: str,
    reason_code: str,
    reason: str,
    raw_response_text: str,
) -> None:
    stage_name: str = "fallback" if is_fallback else "primary"
    LOGGER.info(
        "merge_llm_response_invalid branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=%s model=%s attempt=%d code=%s raw_response_received=%s reason=%s raw_chars=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        provider_name,
        stage_name,
        model_name,
        attempt_index,
        reason_code,
        "yes" if bool(str(raw_response_text or "").strip()) else "no",
        reason,
        len(str(raw_response_text or "")),
    )


def _enforce_merged_description_structure(
    *,
    merged_content: MergedLanguageContent,
) -> ParagraphEnforcementResult:
    original_description: str = str(merged_content.description or "").strip()
    tail_separation = separate_merge_body_and_tail(text=original_description)
    body_paragraphs_before: int = tail_separation.body_paragraph_count
    body_paragraphs_after: int = tail_separation.body_paragraph_count_after_recovery
    if not tail_separation.body_text.strip():
        return ParagraphEnforcementResult(
            description_text=original_description,
            mutated=False,
            recovery_applied=False,
            note="empty_description",
            body_paragraphs_before=0,
            body_paragraphs_after=0,
        )
    normalized_description: str = tail_separation.full_text or original_description
    mutated: bool = normalized_description != original_description
    return ParagraphEnforcementResult(
        description_text=normalized_description,
        mutated=mutated,
        recovery_applied=tail_separation.recovery_applied and mutated,
        note=tail_separation.recovery_note,
        body_paragraphs_before=body_paragraphs_before,
        body_paragraphs_after=body_paragraphs_after,
    )


def _attempt_merge_once(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    provider: LlmProvider,
    model_name: str,
    attempt_index: int,
    attempt_label: str,
    branch_label: str,
    date_key: str,
    slot_key: str,
    no_description_text: str,
    expanded_retry_profile: Optional[ExpandedRetryProfile] = None,
) -> tuple[MergedLanguageContent, str]:
    prompt_text: str = build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
        no_description_text=no_description_text,
        expanded_retry_profile=expanded_retry_profile,
    )
    if provider.pre_delay_sec(config=config) > 0:
        time.sleep(max(0.0, float(provider.pre_delay_sec(config=config))))
    response: OpenAITransportResult = provider.request_merge(
        prompt_text=prompt_text,
        config=config,
        model_name=model_name,
        attempt_label=attempt_label,
        max_output_tokens=config.openai_max_output_tokens,
        structured_schema=_structured_merge_schema(),
        temperature=0.0,
        trace_context=LlmTraceContext(
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
            provider=provider.name,
            model_name=model_name,
            attempt_index=attempt_index,
            request_kind="structured",
            source_count=len(videos),
        ),
    )
    raw_response_text: str = response.raw_text
    try:
        merged_content, paragraph_count = parse_merge_response_or_raise(
            provider_name=provider.name,
            model_name=model_name,
            raw_text=raw_response_text,
            structured_payload=response.structured_payload,
        )
    except Exception as error:
        raise MergeAttemptFailure(
            reason_code=_reason_code_from_error(error),
            reason=str(error),
            model_name=model_name,
            attempt_stage="validation",
            raw_response_text=raw_response_text,
            reason_codes=_reason_codes_from_error(error),
            rejected_attempt=_build_rejected_merge_attempt(
                attempt_index=attempt_index,
                model_name=model_name,
                reason_codes=_reason_codes_from_error(error),
                fallback_reason_code=_reason_code_from_error(error),
                raw_response_text=raw_response_text,
                structured_payload=response.structured_payload,
            ),
        ) from error
    official_links_selection: OfficialLinksSelection = _extract_official_links_from_sources(
        videos
    )
    official_links_fill: OfficialLinksFillResult = OfficialLinksFillResult(
        description=merged_content.description,
        links_in_output=_count_output_official_links(merged_content.description),
        fill_applied=False,
    )
    source_texts: tuple[str, ...] = _source_texts_for_merge_quality(videos)
    quality_result: MergeQualityNormalizationResult = normalize_merge_description(
        description=merged_content.description,
        language=language,
        source_texts=source_texts,
    )
    if quality_result.description_text != merged_content.description:
        merged_content = dataclasses.replace(
            merged_content,
            description=quality_result.description_text,
            description_selected=quality_result.description_text,
            description_audit=quality_result.description_text,
        )
    diagnostics: MergeSemanticDiagnostics = _build_merge_semantic_diagnostics(
        merged_content=merged_content,
        videos=videos,
        official_links_selection=official_links_selection,
        official_links_fill=official_links_fill,
        merge_quality=quality_result.diagnostics,
    )
    _log_merge_style_diagnostics(
        diagnostics=diagnostics,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
        language=language,
        model_name=model_name,
        attempt_index=attempt_index,
    )
    try:
        _validate_coverage_preserving_merge_or_raise(
            merged_content=merged_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=quality_result.diagnostics,
            precomputed_diagnostics=diagnostics,
        )
    except Exception as error:
        recovery_attempt: MergeValidationRecoveryAttempt = _attempt_expanded_formatting_recovery(
            validation_error=error,
            merged_content=merged_content,
            diagnostics=diagnostics,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            source_texts=source_texts,
            language=language,
            model_name=model_name,
            attempt_index=attempt_index,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
        )
        if (
            recovery_attempt.merged_content is not None
            and recovery_attempt.diagnostics is not None
        ):
            merged_content = recovery_attempt.merged_content
            diagnostics = recovery_attempt.diagnostics
        else:
            final_error: Exception = recovery_attempt.validation_error or error
            raise MergeAttemptFailure(
                reason_code=_reason_code_from_error(final_error),
                reason=str(final_error),
                model_name=model_name,
                attempt_stage="validation",
                raw_response_text=raw_response_text,
                reason_codes=_reason_codes_from_error(final_error),
                rejected_attempt=_build_rejected_merge_attempt(
                    attempt_index=attempt_index,
                    model_name=model_name,
                    reason_codes=_reason_codes_from_error(final_error),
                    fallback_reason_code=_reason_code_from_error(final_error),
                    raw_response_text=raw_response_text,
                    title_text=merged_content.title,
                    description_text=merged_content.description,
                ),
            ) from final_error
    LOGGER.info(
        "merge_llm_response_valid model=%s attempt=%d title_length=%d description_length=%d paragraph_count=%d youtube_links_in_llm_output=%d",
        model_name,
        attempt_index,
        len(merged_content.title),
        len(merged_content.description),
        paragraph_count,
        _count_youtube_urls_in_text(merged_content.description),
    )
    return (merged_content, raw_response_text)


def attempt_openai_merge_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    normalize_youtube_url: Callable[[str], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    return attempt_llm_merge_with_audit(
        language=language,
        videos=videos,
        config=config,
        attempt_label=attempt_label,
        summarize_error=summarize_error,
        normalize_youtube_url=normalize_youtube_url,
        no_description_text=no_description_text,
        merge_run_summary=merge_run_summary,
        branch_label=branch_label,
        date_key=date_key,
        slot_key=slot_key,
    )


def attempt_llm_merge_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: AppConfig,
    attempt_label: str,
    summarize_error: Callable[[Exception], str],
    normalize_youtube_url: Callable[[str], str],
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> LanguageMergeAttempt:
    del normalize_youtube_url
    primary_model: str = resolve_effective_llm_model(config)
    primary_provider: LlmProvider = get_llm_provider(config=config)
    main_source_label: str = _main_stage_field_source(provider_name=primary_provider.name)
    last_raw_response: str = ""
    last_error_summary: str = "unknown error"
    last_reason_code: str = "unexpected_error"
    last_reason_codes: tuple[str, ...] = ()
    pending_retry_profile: ExpandedRetryProfile = _standard_expanded_retry_profile()
    rejected_attempts: list[RejectedMergeAttempt] = []

    for attempt_index in range(1, PRIMARY_ATTEMPTS + 1):
        _log_merge_attempt_start(
            provider_name=primary_provider.name,
            model_name=primary_model,
            attempt_index=attempt_index,
            is_fallback=False,
            branch_label=branch_label,
            date_key=date_key,
            slot_key=slot_key,
            language=language,
        )
        retry_profile: Optional[ExpandedRetryProfile] = (
            pending_retry_profile if attempt_index > 1 else None
        )
        if attempt_index > 1:
            LOGGER.info(
                "merge_llm_retry branch=%s date_key=%s slot_key=%s language=%s provider=%s attempt=%d model=%s retry_mode=%s retry_reason_codes=%s retry_focus=%s retry_structure=%s retry_source_count=%d",
                branch_label,
                date_key,
                slot_key,
                language,
                primary_provider.name,
                attempt_index,
                primary_model,
                (
                    retry_profile.retry_mode
                    if retry_profile is not None
                    else "standard"
                ),
                (
                    retry_profile.reject_signal_label
                    if retry_profile is not None
                    else "none"
                ),
                retry_profile.focus_label if retry_profile is not None else "none",
                (
                    "four_plus_structured"
                    if retry_profile is not None
                    and retry_profile.enabled
                    and len(videos) >= 4
                    else "standard"
                ),
                len(videos),
            )
            if merge_run_summary is not None:
                merge_run_summary.record_retry_used()
        try:
            merged_content, raw_response_text = _attempt_merge_once(
                language=language,
                videos=videos,
                config=config,
                provider=primary_provider,
                model_name=primary_model,
                attempt_index=attempt_index,
                attempt_label=f"{attempt_label}_PRIMARY_{attempt_index}",
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                no_description_text=no_description_text,
                expanded_retry_profile=retry_profile,
            )
            last_raw_response = raw_response_text
            if merge_run_summary is not None:
                merge_run_summary.record_merge_success()
            LOGGER.info(
                "merge_branch_ready branch=%s date_key=%s slot_key=%s language=%s generator_model=%s",
                branch_label,
                date_key,
                slot_key,
                language,
                primary_model,
            )
            LOGGER.info(
                "merge_provider_summary provider=%s model=%s structured_ok=yes fallback_used=no parse_repair_used=no final_status=success",
                primary_provider.name,
                primary_model,
            )
            return LanguageMergeAttempt(
                language=language,
                model_name=primary_model,
                raw_response_text=raw_response_text,
                merged=dataclasses.replace(
                    merged_content,
                    branch_type=BRANCH_MERGE,
                    title_source=main_source_label,
                    hook_source=main_source_label,
                    hashtags_source=main_source_label,
                    body_source="main_merge",
                    block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
                ),
                error_summary=None,
                salvaged_title=merged_content.title,
                publish_source_label="primary_success",
                generator_model_name=primary_model,
                used_model_names=(primary_model,),
                branch_type=BRANCH_MERGE,
                title_source=main_source_label,
                hook_source=main_source_label,
                hashtags_source=main_source_label,
                body_source="main_merge",
                block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
            )
        except LlmModelConfigurationError as error:
            LOGGER.error(
                "merge_llm_fatal_model_error branch=%s date_key=%s slot_key=%s language=%s provider=%s model=%s code=%s status_code=%s api_error_code=%s api_error_param=%s reason=%s",
                branch_label,
                date_key,
                slot_key,
                language,
                primary_provider.name,
                primary_model,
                error.reason_code,
                str(error.status_code if error.status_code is not None else "unknown"),
                error.api_error_code or "none",
                error.api_error_param or "none",
                error.detail,
            )
            raise
        except MergeAttemptFailure as error:
            last_raw_response = error.raw_response_text
            last_error_summary = error.reason
            last_reason_code = error.reason_code
            last_reason_codes = _reason_codes_from_error(error) or (error.reason_code,)
            if error.rejected_attempt is not None:
                rejected_attempts.append(error.rejected_attempt)
            pending_retry_profile = _build_retry_profile_from_reason_codes(
                source_count=len(videos),
                reject_signals=error.reason_codes,
                fallback_reason_code=error.reason_code,
            )
            _log_merge_attempt_invalid(
                provider_name=primary_provider.name,
                model_name=primary_model,
                attempt_index=attempt_index,
                is_fallback=False,
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                reason_code=error.reason_code,
                reason=error.reason,
                raw_response_text=error.raw_response_text,
            )
            if merge_run_summary is not None:
                merge_run_summary.record_validation_rejected()
        except Exception as error:
            last_error_summary = summarize_error(error)
            last_reason_code = "unexpected_error"
            last_reason_codes = ("unexpected_error",)
            pending_retry_profile = _standard_expanded_retry_profile(
                reject_signals=("unexpected_error",)
            )
            _log_merge_attempt_invalid(
                provider_name=primary_provider.name,
                model_name=primary_model,
                attempt_index=attempt_index,
                is_fallback=False,
                branch_label=branch_label,
                date_key=date_key,
                slot_key=slot_key,
                language=language,
                reason_code="unexpected_error",
                reason=last_error_summary,
                raw_response_text="",
            )

    final_reason_code: str = last_reason_code
    log_warning_operational(
        LOGGER,
        "merge_llm_final_failure branch=%s date_key=%s slot_key=%s language=%s provider=%s stage=primary code=%s fallback_used=no raw_response_received=%s reason=%s raw_chars=%d",
        branch_label,
        date_key,
        slot_key,
        language,
        primary_provider.name,
        final_reason_code,
        "yes" if bool(last_raw_response.strip()) else "no",
        last_error_summary,
        len(last_raw_response),
        reason_code=final_reason_code,
    )
    LOGGER.info(
        "merge_provider_summary provider=%s model=%s structured_ok=no fallback_used=no parse_repair_used=no final_status=failed",
        primary_provider.name,
        primary_model,
    )
    if merge_run_summary is not None:
        merge_run_summary.record_final_failure()
    return LanguageMergeAttempt(
        language=language,
        model_name=primary_model,
        raw_response_text=last_raw_response,
        merged=None,
        error_summary=last_error_summary,
        salvaged_title=None,
        publish_source_label="merge_failed",
        validation_reasons=list(last_reason_codes),
        generator_model_name=primary_model,
        used_model_names=(primary_model,),
        branch_type=BRANCH_MERGE,
        title_source=_fallback_title_source_label(),
        hook_source=_fallback_hook_source_label(),
        hashtags_source="fallback_none",
        body_source=_fallback_body_source_label(),
        rejected_attempts=tuple(rejected_attempts),
        block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    )


def enforce_openai_merged_paragraphs(
    *,
    language: str,
    merged_content: MergedLanguageContent,
    videos: List[PlannedVideo],
    config: AppConfig,
    no_description_text: str,
    merge_run_summary: Optional[MergeRunSummary] = None,
    branch_label: str = "unknown",
    date_key: str = "unknown",
    slot_key: str = "unknown",
) -> MergedLanguageContent:
    del videos, config, no_description_text
    enforcement_result: ParagraphEnforcementResult = _enforce_merged_description_structure(
        merged_content=merged_content
    )
    LOGGER.info(
        "merge_post_enforcement branch=%s date_key=%s slot_key=%s language=%s body_paragraphs_before=%d body_paragraphs_after=%d mutated=%s recovery_applied=%s reason=%s",
        branch_label,
        date_key,
        slot_key,
        language,
        enforcement_result.body_paragraphs_before,
        enforcement_result.body_paragraphs_after,
        "yes" if enforcement_result.mutated else "no",
        "yes" if enforcement_result.recovery_applied else "no",
        enforcement_result.note,
    )
    if not enforcement_result.mutated:
        return merged_content
    if enforcement_result.recovery_applied and merge_run_summary is not None:
        merge_run_summary.record_paragraph_recovery_used()
    updated_content: MergedLanguageContent = dataclasses.replace(
        merged_content,
        description=enforcement_result.description_text,
        description_selected=enforcement_result.description_text,
        description_audit=enforcement_result.description_text,
    )
    return updated_content
def _main_stage_field_source(*, provider_name: str) -> str:
    normalized_provider_name: str = str(provider_name or "").strip().lower()
    return normalized_provider_name or "main_model"
