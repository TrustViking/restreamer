from __future__ import annotations

from typing import Sequence

from app.llm.merges.quality_diagnostics import (
    MergeQualityDiagnostics,
    MergeQualityNormalizationResult,
    count_overloaded_bullets,
)
from app.llm.merges.quality_normalizer import normalize_merge_description


def inspect_merge_description(
    *,
    description: str,
    language: str,
    source_texts: Sequence[str] = (),
) -> MergeQualityDiagnostics:
    return normalize_merge_description(
        description=description,
        language=language,
        source_texts=source_texts,
    ).diagnostics


__all__ = [
    "MergeQualityDiagnostics",
    "MergeQualityNormalizationResult",
    "normalize_merge_description",
    "inspect_merge_description",
    "count_overloaded_bullets",
]
