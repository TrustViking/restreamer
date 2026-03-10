__all__ = [
    "MergeRunSummary",
    "attempt_llm_merge_with_audit",
    "attempt_llm_single_source_translate_with_audit",
    "attempt_openai_merge_with_audit",
    "attempt_openai_single_source_translate_with_audit",
    "enforce_openai_merged_paragraphs",
]


def __getattr__(name: str) -> object:
    if name == "MergeRunSummary":
        from .merge_run_summary import MergeRunSummary

        return MergeRunSummary
    if name in {
        "attempt_llm_merge_with_audit",
        "attempt_llm_single_source_translate_with_audit",
        "attempt_openai_merge_with_audit",
        "attempt_openai_single_source_translate_with_audit",
        "enforce_openai_merged_paragraphs",
    }:
        from . import merge_service

        return getattr(merge_service, name)
    raise AttributeError(name)
