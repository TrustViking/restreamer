from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


@dataclass(frozen=True)
class MergePolishResult:
    source_model: str
    polish_model: str
    original_text: str
    polished_text: str
    accepted: bool
    reject_reason: str | None
    validation_passed: bool


def build_merge_polish_prompt(
    *,
    language: str,
    title_text: str,
    description_text: str,
) -> str:
    return (
        "You are a careful final-stage style editor for a merged YouTube stream summary.\n"
        f"Target language: {language}.\n"
        "Your task is to lightly polish phrasing and flow while preserving meaning exactly.\n"
        "Do not add facts, do not remove source-backed facts, do not change language, "
        "do not change the overall structure, and do not rewrite aggressively.\n"
        "Keep the same number of paragraphs and preserve agenda bullets if present.\n"
        "Return only one strict JSON object with exactly these keys: title, description.\n\n"
        f'TITLE: "{title_text.strip()}"\n\n'
        f"DESCRIPTION:\n{description_text.strip()}"
    ).strip()


def paragraph_count(text: str) -> int:
    normalized_text: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized_text:
        return 0
    return len([part for part in re.split(r"\n\s*\n", normalized_text) if part.strip()])


def bullet_marker_count(text: str) -> int:
    markers = ("🔹", "📌", "🎤", "🎥", "⚖", "🌐", "✅")
    return sum(1 for line in str(text or "").splitlines() if line.lstrip().startswith(markers))


def infer_polish_reject_reason(error_text: str) -> str:
    normalized_error: str = str(error_text or "").strip().lower()
    if "semantic_source_" in normalized_error or "semantic_source_grounding_too_low" in normalized_error:
        return "facts_lost"
    if "language" in normalized_error and "drift" in normalized_error:
        return "language_drift"
    if "format" in normalized_error or "paragraph count" in normalized_error:
        return "formatting_degraded"
    if "structure_changed" in normalized_error:
        return "structure_changed"
    return "validation_failed"


def is_too_aggressive_rewrite(*, original_text: str, polished_text: str) -> bool:
    original_normalized: str = str(original_text or "").strip()
    polished_normalized: str = str(polished_text or "").strip()
    if not original_normalized or not polished_normalized:
        return True
    similarity_ratio: float = SequenceMatcher(
        a=original_normalized,
        b=polished_normalized,
    ).ratio()
    length_delta: float = abs(len(original_normalized) - len(polished_normalized)) / max(
        1,
        len(original_normalized),
    )
    return similarity_ratio < 0.72 or length_delta > 0.35
