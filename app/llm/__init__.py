"""LLM integration: merges, models, providers."""

# Backward-compatible re-exports from merges subpackage
from app.llm.merges import merge_service
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.merges.merge_service import (
    attempt_llm_merge_with_audit,
    attempt_llm_single_source_translate_with_audit,
    attempt_openai_merge_with_audit,
    attempt_openai_single_source_translate_with_audit,
    enforce_openai_merged_paragraphs,
)
