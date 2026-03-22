from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.llm.merges.merge_constants import BULLET_OVERLOAD_CHAR_LIMIT, BULLET_OVERLOAD_NAME_LIMIT

LOGGER = _get_logger_impl(__name__)

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

def _template_field_text(templates: Optional[object], field_name: str) -> str:
    if templates is None:
        return ""
    return str(getattr(templates, field_name, "") or "").strip()

def _template_json_object(templates: Optional[object], field_name: str) -> dict[str, object]:
    if templates is None:
        return {}
    direct_value: object = getattr(templates, field_name, None)
    if isinstance(direct_value, dict):
        return {str(key): value for key, value in direct_value.items()}
    if isinstance(direct_value, str):
        raw_text: str = str(direct_value or "").strip()
    else:
        raw_text = _template_field_text(templates, field_name)
    if not raw_text and not field_name.endswith("_json"):
        raw_text = _template_field_text(templates, f"{field_name}_json")
    if not raw_text:
        return {}
    try:
        parsed_payload: object = json.loads(raw_text)
    except Exception:
        LOGGER.warning(
            "template_json_parse_failed field=%s", field_name
        )
        return {}
    if not isinstance(parsed_payload, dict):
        LOGGER.warning(
            "template_json_invalid_type field=%s type=%s",
            field_name,
            type(parsed_payload).__name__,
        )
        return {}
    return {str(key): value for key, value in parsed_payload.items()}

def _format_template_placeholders(
    template_text: str,
    placeholder_values: dict[str, object],
) -> str:
    formatted_text: str = str(template_text or "")
    for placeholder_key, placeholder_value in placeholder_values.items():
        formatted_text = formatted_text.replace(
            "{" + placeholder_key + "}",
            str(placeholder_value),
        )
    return formatted_text.strip()

def _load_retry_reinforcement_lines(
    *,
    templates: Optional[object],
    signal: str,
    placeholder_values: dict[str, object],
    fallback_lines: tuple[str, ...],
) -> tuple[str, ...]:
    reinforcements_payload: dict[str, object] = _template_json_object(
        templates,
        "llm_merge_retry_reinforcements",
    )
    raw_template_lines: object = reinforcements_payload.get(signal)
    template_lines: list[str] = []
    if isinstance(raw_template_lines, list):
        for raw_template_line in raw_template_lines:
            normalized_template_line: str = str(raw_template_line or "").strip()
            if normalized_template_line:
                template_lines.append(normalized_template_line)
    elif isinstance(raw_template_lines, str):
        normalized_template_line = str(raw_template_lines or "").strip()
        if normalized_template_line:
            template_lines.append(normalized_template_line)
    selected_lines: tuple[str, ...] = (
        tuple(template_lines) if template_lines else fallback_lines
    )
    return tuple(
        _format_template_placeholders(
            template_line,
            placeholder_values,
        )
        for template_line in selected_lines
    )

def _targeted_bullet_coverage_retry_profile(
    *,
    actual_bullets: int,
    required_bullets: int,
    source_count: int,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        f"Previous attempt produced only {actual_bullets} bullet lines. Rewrite with at least {required_bullets} bullets.",
        f"Spread bullets across all {source_count} sources. Do not collapse multiple sources into one generic lane.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="insufficient_bullet_coverage",
        placeholder_values={
            "actual_bullets": actual_bullets,
            "required_bullets": required_bullets,
            "source_count": source_count,
        },
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("insufficient_bullet_coverage",),
        focus_tags=("bullet_coverage",),
        reinforcement_lines=reinforcement_lines,
    )

def _targeted_duplicate_retry_profile(
    *,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        "CRITICAL: Previous attempt had a structural error — the opening paragraph was repeated or a CTA appeared where the hook should be.",
        "RULE 1 — HOOK STRUCTURE: The very first paragraph must be the hook: a question, tension, or key thesis. It appears exactly ONCE.",
        "RULE 2 — NO DUPLICATION: Paragraph 2 and later must NOT restate, paraphrase, or echo any sentence from paragraph 1. If paragraph 1 ends with a question, paragraph 2 must answer it with new facts — never repeat the question.",
        "RULE 3 — NO CTA IN HOOK POSITION: Do NOT place 'Subscribe', 'Follow', 'Watch', 'Share', 'Подпишитесь', 'Поширюйте', 'Смотрите', or any call to action as the first paragraph. CTA belongs only at the very end, after all bullets.",
        "RULE 4 — PARAGRAPH 2 MUST BE BULLETS: Immediately after the hook, start the bullet block. The second paragraph must begin with a bullet marker (🔹, ⚖, 📌, etc.), not with another prose sentence.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="duplicate_paragraph",
        placeholder_values={},
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("duplicate_paragraph", "cta_as_first_paragraph", "cta_in_hook"),
        focus_tags=("no_hook_duplication", "hook_first"),
        reinforcement_lines=reinforcement_lines,
    )

def _targeted_hook_echo_retry_profile(
    *,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        "CRITICAL: Previous attempt repeated the hook thesis in the bullet block opening.",
        "RULE 5 — NO HOOK ECHO: The first line of the bullet block must NOT restate the hook. "
        "If the hook asks a question, the first bullet must answer with a new fact, not repeat the question.",
        "Paragraph 2 must start with a bullet marker and a NEW fact that was not in paragraph 1.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="hook_echo_in_body",
        placeholder_values={},
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("hook_echo_in_body",),
        focus_tags=("no_hook_echo",),
        reinforcement_lines=reinforcement_lines,
    )

def _targeted_overloaded_bullet_retry_profile(
    *,
    overloaded_count: int,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        f"Previous attempt had {overloaded_count} overloaded bullet(s) — single bullets listing {BULLET_OVERLOAD_NAME_LIMIT}+ multi-word names or spanning {BULLET_OVERLOAD_CHAR_LIMIT}+ characters.",
        f"RULE — SPLIT OVERLOADED BULLETS: If a bullet contains {BULLET_OVERLOAD_NAME_LIMIT} or more full names (first + last name) or is longer than {BULLET_OVERLOAD_CHAR_LIMIT} characters, split it into 2 separate bullets.",
        "Each bullet must carry ONE clear idea or ONE person's contribution — not a container for everything that didn't fit elsewhere.",
        "It is better to have 7 focused bullets than 5 bullets where one is a dump of everything.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="overloaded_bullet",
        placeholder_values={
            "overloaded_count": overloaded_count,
            "bullet_name_limit": BULLET_OVERLOAD_NAME_LIMIT,
            "bullet_char_limit": BULLET_OVERLOAD_CHAR_LIMIT,
        },
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("overloaded_bullet",),
        focus_tags=("split_overloaded_bullet",),
        reinforcement_lines=reinforcement_lines,
    )

def _targeted_paragraph_overflow_retry_profile(
    *,
    actual_paragraphs: int,
    max_paragraphs: int,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        f"Previous attempt produced {actual_paragraphs} body paragraphs but the maximum is {max_paragraphs}.",
        f"STRUCTURE RULE: The output must have at most {max_paragraphs} visual paragraphs (blocks separated by blank lines).",
        "Paragraph 1 = hook (1-2 sentences). Paragraph 2 = bullet block. All bullet lines within one thematic group must be separated by single newlines, NOT double newlines.",
        "If you use thematic micro-blocks, keep them inside the same paragraph — separate sub-groups with a single blank line only if absolutely needed, but the total paragraph count must stay within the limit.",
        "Do NOT put each bullet on its own paragraph. Group related bullets together.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="paragraph_overflow",
        placeholder_values={
            "actual_paragraphs": actual_paragraphs,
            "max_paragraphs": max_paragraphs,
        },
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("paragraph_overflow",),
        focus_tags=("paragraph_structure",),
        reinforcement_lines=reinforcement_lines,
    )

def _targeted_paragraph_underflow_retry_profile(
    *,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        "Previous attempt produced only 1 body paragraph. The minimum is 2.",
        "STRUCTURE RULE: The output must have at least 2 visual paragraphs (blocks separated by blank lines).",
        "Paragraph 1 = hook (1-2 editorial sentences setting the tension or key question).",
        "Paragraph 2 = bullet block starting with the allowed markers (🔹, 📌, 🎤, 🎥, ⚖, 🌐, ✅). Separate the hook from the bullets with one blank line.",
        "Do NOT merge the hook and bullets into a single block of text.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="paragraph_underflow",
        placeholder_values={},
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("paragraph_underflow",),
        focus_tags=("paragraph_structure",),
        reinforcement_lines=reinforcement_lines,
    )

def _targeted_compact_bullet_overflow_retry_profile(
    *,
    actual_bullets: int,
    min_bullets: int,
    max_bullets: int,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        f"Previous attempt had {actual_bullets} bullets but the compact contract allows maximum {max_bullets}.",
        f"RULE: For 2-source merge, the bullet block must have exactly {min_bullets} to {max_bullets} bullets.",
        "Merge related points into fewer, denser bullets. Each bullet = one distinct fact or angle. Do not pad.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="compact_bullet_overflow",
        placeholder_values={
            "actual_bullets": actual_bullets,
            "min_bullets": min_bullets,
            "max_bullets": max_bullets,
        },
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("compact_bullet_overflow",),
        focus_tags=("bullet_count",),
        reinforcement_lines=reinforcement_lines,
    )

def _targeted_cta_opener_retry_profile(
    *,
    templates: Optional[object] = None,
) -> ExpandedRetryProfile:
    fallback_lines: tuple[str, ...] = (
        "CRITICAL: Previous attempt placed a CTA (subscribe/follow/watch) as the first paragraph.",
        "RULE: The first paragraph must be the editorial hook — a question, tension, or key thesis.",
        "CTA belongs ONLY at the very end, after all bullets and before hashtags.",
        "Rewrite so paragraph 1 is the hook, and any CTA is the last line before hashtags.",
    )
    reinforcement_lines: tuple[str, ...] = _load_retry_reinforcement_lines(
        templates=templates,
        signal="cta_as_first_paragraph",
        placeholder_values={},
        fallback_lines=fallback_lines,
    )
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=("cta_as_first_paragraph",),
        focus_tags=("cta_position",),
        reinforcement_lines=reinforcement_lines,
    )

def _build_expanded_retry_profile(
    *,
    source_count: int,
    reject_signals: Sequence[str] = (),
) -> ExpandedRetryProfile:
    normalized_signals: tuple[str, ...] = tuple(
        cleaned_signal
        for cleaned_signal in (
            str(raw_signal or "").strip()
            for raw_signal in dict.fromkeys(reject_signals)
        )
        if cleaned_signal
    )
    primary_signal: str = normalized_signals[0] if normalized_signals else ""
    focus_tags: tuple[str, ...]
    base_lines: tuple[str, ...]

    if primary_signal == "insufficient_expanded_body":
        focus_tags = ("body_depth",)
        base_lines = (
            "The previous attempt's post-hook body was too thin. Make it clearly denser.",
            "Expand each bullet with a concrete fact, name, or outcome drawn from the sources.",
            "post-hook body clearly denser than the hook itself.",
        )
    elif primary_signal == "too_few_expanded_bullets":
        focus_tags = ("bullet_sufficiency",)
        base_lines = (
            "The previous attempt had too few bullets. Add enough distinct, meaningful bullets.",
            "Every source must leave at least one visible trace in the bullet block.",
            "enough distinct, meaningful bullets to cover all sources.",
        )
    elif primary_signal == "overly_generic_body":
        focus_tags = ("source_specificity",)
        base_lines = (
            "The previous body was too generic. Replace broad statements with source-grounded specifics.",
            "Each bullet must name a person, fact, event, or decision from the sources.",
            "source-grounded specifics instead of editorial summaries.",
        )
    elif primary_signal == "hook_dominates_body":
        focus_tags = ("hook_restraint",)
        base_lines = (
            "The hook was too long and crowded out the body. Keep the hook brief and functional.",
            "The hook must not exceed 3 sentences. Move facts into bullets, not into the hook.",
            "Keep the hook brief and functional — move specifics into bullets.",
        )
    elif primary_signal == "weak_source_coverage":
        focus_tags = ("source_spread",)
        base_lines = (
            "The previous attempt collapsed sources into a single undifferentiated lane.",
            "Restore distinguishable spread across source lines or topic nodes.",
            "Restore distinguishable spread across source lines or topic nodes.",
        )
    else:
        focus_tags = ()
        base_lines = (
            f"Previous attempt was rejected: {', '.join(normalized_signals) or 'unknown'}.",
            "Rewrite with concrete facts from each source. Avoid generic editorial phrasing.",
        )

    multi_source_lines: tuple[str, ...] = ()
    if source_count >= 3:
        multi_source_lines = (
            "Structure the body as 2 to 3 short agenda tracks — one per source or topic cluster.",
            "cut generic filler bridges between bullets.",
        )
    if source_count >= 4:
        multi_source_lines = multi_source_lines + (
            "do not collapse the post-hook body into one umbrella summary.",
            "Build 2 to 3 meaningful thematic micro-blocks after the hook.",
            "source-specific density, not just extra length.",
            "Make every source leave a recognizable trace in the body.",
        )

    reinforcement_lines: tuple[str, ...] = base_lines + multi_source_lines
    return ExpandedRetryProfile(
        retry_mode="targeted",
        reject_signals=normalized_signals,
        focus_tags=focus_tags,
        reinforcement_lines=reinforcement_lines,
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

def _merge_contract_block_with_retry(
    *,
    contract_mode: MergeContractMode,
    expanded_retry_profile: Optional[ExpandedRetryProfile],
    templates: Optional[object] = None,
) -> str:
    structural_rules_block: str = _template_field_text(
        templates,
        "llm_merge_structural_rules",
    )
    contract_block: str = contract_mode.contract_block.strip()
    if structural_rules_block:
        contract_block = f"{contract_block}\n\n{structural_rules_block.strip()}".strip()
    if expanded_retry_profile is not None and expanded_retry_profile.enabled:
        reinforcement_block: str = "\n".join(expanded_retry_profile.reinforcement_lines)
        return f"{contract_block}\n\nRETRY INSTRUCTION:\n{reinforcement_block}"
    return contract_block
