from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from app.pipeline.slot_processing import SlotProcessResult


@dataclass(frozen=True)
class UsedRuntimeModels:
    configured_primary_model: str
    configured_fallback_model: str
    used_generation_models: tuple[str, ...]
    used_polish_models: tuple[str, ...]
    used_packaging_models: tuple[str, ...]
    all_used_models: tuple[str, ...]


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for value in values:
        normalized_value: str = str(value or "").strip()
        if normalized_value and normalized_value not in ordered:
            ordered.append(normalized_value)
    return tuple(ordered)


def collect_used_runtime_models(
    *,
    slot_results: Sequence[SlotProcessResult],
    configured_primary_model: str,
    configured_fallback_model: str,
) -> UsedRuntimeModels:
    generation_models: list[str] = []
    polish_models: list[str] = []
    packaging_models: list[str] = []
    all_used_models: list[str] = []
    for slot in slot_results:
        for merge_attempt in slot.merge_audit_by_language.values():
            generator_model_name: str = (
                str(merge_attempt.generator_model_name or "").strip()
                or str(merge_attempt.model_name or "").strip()
            )
            if generator_model_name:
                generation_models.append(generator_model_name)
            if merge_attempt.polish_accepted and str(merge_attempt.polish_model_name or "").strip():
                polish_models.append(str(merge_attempt.polish_model_name or "").strip())
            for model_name in merge_attempt.used_model_names:
                all_used_models.append(str(model_name or "").strip())
            packaging_audit = merge_attempt.packaging_audit
            if packaging_audit is not None and str(packaging_audit.packaging_model or "").strip():
                packaging_models.append(str(packaging_audit.packaging_model or "").strip())
                all_used_models.append(str(packaging_audit.packaging_model or "").strip())
    ordered_all_used_models: tuple[str, ...] = _ordered_unique(all_used_models)
    return UsedRuntimeModels(
        configured_primary_model=str(configured_primary_model or "").strip(),
        configured_fallback_model=str(configured_fallback_model or "").strip(),
        used_generation_models=_ordered_unique(generation_models),
        used_polish_models=_ordered_unique(polish_models),
        used_packaging_models=_ordered_unique(packaging_models),
        all_used_models=ordered_all_used_models,
    )


def render_doc_title_models_segment(*, used_runtime_models: UsedRuntimeModels) -> str:
    if not used_runtime_models.all_used_models:
        return ""
    return "_[" + ",".join(used_runtime_models.all_used_models) + "]"
