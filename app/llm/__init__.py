"""LLM integration: merges, models, providers."""

from __future__ import annotations

from typing import Any

# Backward-compatible lazy re-exports.
# All production code already imports directly from submodules.
# These exist only for backward compatibility if any external consumer
# uses `from app.llm import merge_service` etc.

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "merge_service": ("app.llm.merges", "merge_service"),
    "MergeRunSummary": ("app.llm.merges.merge_run_summary", "MergeRunSummary"),
    "attempt_llm_merge_with_audit": ("app.llm.merges.merge_service", "attempt_llm_merge_with_audit"),
    "attempt_openai_merge_with_audit": ("app.llm.merges.merge_service", "attempt_openai_merge_with_audit"),
    "enforce_openai_merged_paragraphs": ("app.llm.merges.merge_service", "enforce_openai_merged_paragraphs"),
}

__all__: list[str] = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module 'app.llm' has no attribute {name!r}")
