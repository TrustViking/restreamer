from __future__ import annotations

import dataclasses
import re
from typing import List, Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.core.cta_detection import starts_with_cta_prefix
from app.core.text_utils import normalize_newlines
from app.core.models import (
    MergedLanguageContent,
    PlannedVideo,
    RejectedMergeAttempt,
)
from app.llm.merges.merge_constants import (
    ALLOWED_BULLET_MARKERS,
    RECOVERABLE_REJECT_CODES,
    SEMANTIC_TOKEN_PATTERN,
    STYLE_CONTRACT_VERSION,
)
from app.llm.merges.merge_links import (
    OfficialLinksFillResult,
    OfficialLinksSelection,
)
from app.llm.merges.merge_parser import parse_json_tolerant
from app.llm.merges.merge_quality import (
    MergeQualityDiagnostics,
    count_overloaded_bullets,
)
from app.llm.merges.merge_text_utils import (
    _bullet_marker_for_line,
    _contains_agenda_heading,
    _extract_description_paragraphs_raw,
    _extract_named_entities,
    _looks_like_per_source_dump,
    _looks_like_service_tail_paragraph,
)
from app.llm.merges.merge_validation_helpers import (
    attempt_hook_echo_repair as _shared_attempt_hook_echo_repair,
    has_duplicate_paragraphs as _shared_has_duplicate_paragraphs,
    has_hook_echo_in_body as _shared_has_hook_echo_in_body,
    looks_like_bad_hook_paragraph as _shared_looks_like_bad_hook_paragraph,
)

LOGGER = _get_logger_impl(__name__)

_NUMBERED_DUMP_PATTERN: re.Pattern[str] = re.compile(r"\b\d\)\s")


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class DescriptionValidationFailure(RuntimeError):
    reason_codes: tuple[str, ...]
    message: str

    def __str__(self) -> str:
        return self.message


@dataclasses.dataclass(frozen=True)
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
    official_links_found_in_sources: int
    official_links_kept: int
    official_links_in_output: int
    official_links_fill_applied: bool
    merge_quality: MergeQualityDiagnostics


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
    if "body paragraph count" in error_text:
        _paragraph_count_match: Optional[re.Match[str]] = re.search(r"got (\d+)", error_text)
        if _paragraph_count_match is not None and int(_paragraph_count_match.group(1)) < 2:
            return "paragraph_underflow"
        return "paragraph_overflow"
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


def _is_softened_distinctive_source_coverage_eligible(
    *,
    source_count: int,
    source_coverage_by_item: tuple[bool, ...],
    distinctive_coverage_hits: int,
    distinctive_coverage_total: int,
    reason_codes: tuple[str, ...],
) -> bool:
    if source_count <= 0:
        return False
    if source_count >= 4:
        return False
    if "weak_source_coverage" not in reason_codes:
        return False
    if distinctive_coverage_total <= 0:
        return False
    distinctive_coverage_ratio: float = distinctive_coverage_hits / distinctive_coverage_total
    if distinctive_coverage_ratio < 0.5:
        return False
    covered_sources_count: int = sum(1 for is_covered in source_coverage_by_item if is_covered)
    minimum_required_coverage_ratio: float = (source_count - 1) / source_count
    source_coverage_ratio: float = covered_sources_count / source_count
    if source_coverage_ratio < minimum_required_coverage_ratio:
        return False
    return True


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


def _looks_like_bad_hook_paragraph(text: str) -> bool:
    return _shared_looks_like_bad_hook_paragraph(text)


def _has_duplicate_paragraphs(description_text: str) -> bool:
    return _shared_has_duplicate_paragraphs(description_text)


def _has_hook_echo_in_body(description_text: str) -> bool:
    return _shared_has_hook_echo_in_body(description_text)


def _attempt_hook_echo_repair(description_text: str) -> Optional[str]:
    return _shared_attempt_hook_echo_repair(description_text)


def _has_cta_in_opening_lines_before_hook_or_bullet(
    description_paragraphs: Sequence[str],
) -> bool:
    opening_lines: list[str] = []
    for paragraph_text in list(description_paragraphs)[:2]:
        for raw_line_text in str(paragraph_text or "").split("\n"):
            normalized_line_text: str = str(raw_line_text or "").strip()
            if normalized_line_text:
                opening_lines.append(normalized_line_text)
    opening_line_window: list[str] = opening_lines[:3]
    if not opening_line_window:
        return False
    cta_line_indexes: list[int] = [
        line_index
        for line_index, line_text in enumerate(opening_line_window)
        if starts_with_cta_prefix(line_text) or _looks_like_bad_hook_paragraph(line_text)
    ]
    if not cta_line_indexes:
        return False
    first_hook_or_bullet_line_index: Optional[int] = None
    for line_index, line_text in enumerate(opening_line_window):
        if _bullet_marker_for_line(line_text):
            first_hook_or_bullet_line_index = line_index
            break
        if (
            starts_with_cta_prefix(line_text)
            or _looks_like_bad_hook_paragraph(line_text)
            or _looks_like_service_tail_paragraph(line_text)
        ):
            continue
        first_hook_or_bullet_line_index = line_index
        break
    first_cta_line_index: int = min(cta_line_indexes)
    if first_hook_or_bullet_line_index is None:
        return True
    return first_cta_line_index < first_hook_or_bullet_line_index


def _has_adjacent_duplicate_lines(description_text: str) -> bool:
    normalized_description_text: str = normalize_newlines(description_text)
    lines: list[str] = [str(line_text or "").strip() for line_text in normalized_description_text.split("\n")]
    for line_index in range(len(lines) - 1):
        current_line: str = lines[line_index]
        next_line: str = lines[line_index + 1]
        if not current_line or not next_line:
            continue
        current_line_length: int = len(current_line)
        next_line_length: int = len(next_line)
        shorter_length: int = min(current_line_length, next_line_length)
        if shorter_length < 60:
            continue
        common_prefix_length: int = 0
        for current_char, next_char in zip(current_line, next_line):
            if current_char != next_char:
                break
            common_prefix_length += 1
        common_prefix_ratio: float = common_prefix_length / shorter_length
        if common_prefix_ratio > 0.65:
            return True
        current_line_tokens: list[str] = [
            token.lower() for token in SEMANTIC_TOKEN_PATTERN.findall(current_line)
        ]
        next_line_tokens: list[str] = [
            token.lower() for token in SEMANTIC_TOKEN_PATTERN.findall(next_line)
        ]
        if len(current_line_tokens) < 8 or len(next_line_tokens) < 8:
            continue
        current_token_set: set[str] = set(current_line_tokens)
        next_token_set: set[str] = set(next_line_tokens)
        union_size: int = len(current_token_set | next_token_set)
        if union_size == 0:
            continue
        jaccard_similarity: float = len(current_token_set & next_token_set) / union_size
        if jaccard_similarity >= 0.75:
            return True
    return False


def _validate_coverage_preserving_merge_or_raise(
    *,
    merged_content: MergedLanguageContent,
    videos: Sequence[PlannedVideo],
    official_links_selection: OfficialLinksSelection,
    official_links_fill: OfficialLinksFillResult,
    merge_quality: MergeQualityDiagnostics,
    precomputed_diagnostics: Optional[MergeSemanticDiagnostics] = None,
) -> tuple[MergeSemanticDiagnostics, MergedLanguageContent]:
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
    if _looks_like_per_source_dump(merged_content.description):
        raise _build_description_validation_failure(
            reason_codes=("per_source_enumeration",),
            message="description validation failed: per_source_enumeration",
        )
    if _has_duplicate_paragraphs(merged_content.description):
        raise _build_description_validation_failure(
            reason_codes=("duplicate_paragraph",),
            message="description validation failed: duplicate_paragraph",
        )
    if _has_hook_echo_in_body(merged_content.description):
        _repaired_description: Optional[str] = _attempt_hook_echo_repair(merged_content.description)
        if _repaired_description is not None:
            merged_content = dataclasses.replace(
                merged_content,
                description=_repaired_description,
                description_selected=_repaired_description,
                description_audit=_repaired_description,
            )
            LOGGER.info("hook_echo_repair_applied=yes")
        else:
            raise _build_description_validation_failure(
                reason_codes=("hook_echo_in_body",),
                message="description validation failed: duplicate_paragraph (hook echo in body)",
            )
    if _has_adjacent_duplicate_lines(merged_content.description):
        raise _build_description_validation_failure(
            reason_codes=("duplicate_paragraph",),
            message="description validation failed: duplicate_paragraph",
        )
    description_paragraphs: List[str] = _extract_description_paragraphs_raw(merged_content.description)
    if _has_cta_in_opening_lines_before_hook_or_bullet(description_paragraphs):
        raise _build_description_validation_failure(
            reason_codes=("cta_as_first_paragraph",),
            message="description validation failed: cta_as_first_paragraph",
        )
    first_paragraph: str = description_paragraphs[0] if description_paragraphs else ""
    if first_paragraph and (
        _looks_like_service_tail_paragraph(first_paragraph)
        or _looks_like_bad_hook_paragraph(first_paragraph)
    ):
        raise _build_description_validation_failure(
            reason_codes=("cta_in_hook",),
            message="description validation failed: cta_in_hook",
        )
    source_count: int = len(videos)
    if source_count >= 2:
        min_required_bullets: int = max(source_count + 1, 5)
        if diagnostics.bullet_points_count < min_required_bullets:
            raise _build_description_validation_failure(
                reason_codes=("insufficient_bullet_coverage",),
                message="description validation failed: insufficient_bullet_coverage",
            )
    if source_count <= 2 and diagnostics.bullet_points_count > 7:
        raise _build_description_validation_failure(
            reason_codes=("compact_bullet_overflow",),
            message="description validation failed: compact_bullet_overflow",
        )
    if diagnostics.emoji_count > 10:
        raise _build_description_validation_failure(
            reason_codes=("excessive_emoji_usage",),
            message="description validation failed: excessive_emoji_usage",
        )
    overloaded_bullet_count: int = count_overloaded_bullets(merged_content.description)
    if overloaded_bullet_count >= 2 and diagnostics.bullet_points_count >= 4:
        raise _build_description_validation_failure(
            reason_codes=("overloaded_bullet",),
            message=f"description validation failed: overloaded_bullet count={overloaded_bullet_count}",
        )
    title: str = merged_content.title
    if _NUMBERED_DUMP_PATTERN.search(str(title or "")):
        raise _build_description_validation_failure(
            reason_codes=("numbered_title_dump",),
            message="description validation failed: numbered_title_dump",
        )
    if first_paragraph and starts_with_cta_prefix(first_paragraph):
        raise _build_description_validation_failure(
            reason_codes=("cta_as_first_paragraph",),
            message="description validation failed: cta_as_first_paragraph",
        )
    if len(description_paragraphs) >= 2:
        for paragraph_index_a in range(len(description_paragraphs)):
            for paragraph_index_b in range(paragraph_index_a + 1, len(description_paragraphs)):
                paragraph_a: str = description_paragraphs[paragraph_index_a]
                paragraph_b: str = description_paragraphs[paragraph_index_b]
                shorter_length: int = min(len(paragraph_a), len(paragraph_b))
                if shorter_length < 80:
                    continue
                common_prefix_length: int = 0
                for char_a, char_b in zip(paragraph_a, paragraph_b):
                    if char_a != char_b:
                        break
                    common_prefix_length += 1
                similarity_ratio: float = common_prefix_length / shorter_length
                if similarity_ratio > 0.7:
                    raise _build_description_validation_failure(
                        reason_codes=("duplicate_paragraph",),
                        message="description validation failed: duplicate_paragraph",
                    )
    if merge_quality.semantic_gate_status == "hard_reject":
        reason_codes: str = ",".join(merge_quality.semantic_gate_reason_codes) or "semantic_gate"
        raise _build_description_validation_failure(
            reason_codes=merge_quality.semantic_gate_reason_codes or ("semantic_gate",),
            message=f"description validation failed: {reason_codes}",
        )
    return (diagnostics, merged_content)


def _count_emoji(text: str) -> int:
    emoji_pattern: re.Pattern[str] = re.compile(
        r"[\U0001F300-\U0001FAFF\u2600-\u27BF]",
        flags=re.UNICODE,
    )
    raw_count: int = len(emoji_pattern.findall(str(text or "")))
    structural_count: int = sum(str(text or "").count(marker) for marker in ALLOWED_BULLET_MARKERS)
    return max(0, raw_count - structural_count)


def _build_merge_semantic_diagnostics(
    *,
    merged_content: MergedLanguageContent,
    videos: Sequence[PlannedVideo],
    official_links_selection: OfficialLinksSelection,
    official_links_fill: OfficialLinksFillResult,
    merge_quality: MergeQualityDiagnostics,
) -> MergeSemanticDiagnostics:
    description_text: str = str(merged_content.description or "").strip()
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
        if marker in ALLOWED_BULLET_MARKERS:
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
        official_links_found_in_sources=official_links_selection.found_in_sources,
        official_links_kept=len(official_links_selection.kept_links),
        official_links_in_output=official_links_fill.links_in_output,
        official_links_fill_applied=official_links_fill.fill_applied,
        merge_quality=merge_quality,
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
    marker_types_text: str = ",".join(diagnostics.bullet_marker_types) or "none"
    LOGGER.info(
        "merge_style_coverage branch=%s date_key=%s slot_key=%s language=%s model=%s attempt=%d style_contract_version=%s hook_present=%s agenda_block_present=%s bullet_points_count=%d semantic_bullets_count=%d bullets_with_emoji_count=%d bullets_with_plain_marker_count=%d bullet_marker_types=%s neutral_bullets_count=%d accent_bullets_count=%d accent_marker_types=%s accent_overflow=%s block_spacing_ok=%s named_entities_preserved=%d source_named_entities_total=%d named_entities_metric=%s emoji_count=%d official_links_found_in_sources=%d official_links_kept=%d official_links_in_output=%d official_links_fill_applied=%s",
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
