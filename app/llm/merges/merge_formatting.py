from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.models import (
    MergedLanguageContent,
    PlannedVideo,
)
from app.llm.merges.merge_constants import ALLOWED_BULLET_MARKERS
from app.llm.merges.merge_links import (
    OfficialLinksFillResult,
    OfficialLinksSelection,
)
from app.llm.merges.merge_quality import (
    MergeQualityNormalizationResult,
    normalize_merge_description,
)
from app.llm.merges.merge_validation import (
    MergeSemanticDiagnostics,
    _build_merge_semantic_diagnostics,
    _normalize_description_validation_reason_codes,
    _reason_codes_from_error,
    _validate_coverage_preserving_merge_or_raise,
)

LOGGER = _get_logger_impl(__name__)

_EMOJI_PATTERN: re.Pattern[str] = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class FormattingNormalizationResult:
    description_text: str
    actions: tuple[str, ...]


@dataclass(frozen=True)
class ParagraphEnforcementResult:
    description_text: str
    mutated: bool
    recovery_applied: bool
    note: str
    body_paragraphs_before: int
    body_paragraphs_after: int


@dataclass(frozen=True)
class MergeValidationRecoveryAttempt:
    merged_content: Optional[MergedLanguageContent]
    diagnostics: Optional[MergeSemanticDiagnostics]
    validation_error: Optional[Exception]
    actions: tuple[str, ...]


def _strip_non_structural_emoji_from_line(line: str) -> tuple[str, bool]:
    raw_line: str = str(line or "")
    indent_length: int = len(raw_line) - len(raw_line.lstrip())
    indent: str = raw_line[:indent_length]
    stripped_line: str = raw_line[indent_length:]
    protected_prefix: str = ""
    for marker in ALLOWED_BULLET_MARKERS:
        marker_with_space: str = f"{marker} "
        if stripped_line.startswith(marker_with_space):
            protected_prefix = f"{indent}{marker_with_space}"
            stripped_line = stripped_line[len(marker_with_space):]
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
        title="",
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
    _FORMATTING_ONLY_REASON_CODES: frozenset[str] = frozenset({"excessive_emoji_usage"})
    is_formatting_only: bool = bool(set(reason_codes) & _FORMATTING_ONLY_REASON_CODES)
    if len(videos) < 3 or not is_formatting_only:
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
        title=recovered_content.title,
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
        revalidated_diagnostics: MergeSemanticDiagnostics
        revalidated_diagnostics, recovered_content = _validate_coverage_preserving_merge_or_raise(
            merged_content=recovered_content,
            videos=videos,
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=recovery_quality_result.diagnostics,
            precomputed_diagnostics=recovered_diagnostics,
        )
        recovered_diagnostics = revalidated_diagnostics
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
