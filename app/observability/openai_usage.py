from __future__ import annotations

import logging
from typing import Set

from app.llm.llm_client import get_run_local_openai_usage
from app.llm.llm_usage_tracker import RunLocalOpenAIUsageState


def format_run_usage_totals(usage_state: RunLocalOpenAIUsageState) -> str:
    """One-line token/cost summary; cost is accumulated per response (served model + service tier)."""
    cost_text: str = (
        f"${usage_state.estimated_cost_usd:.4f}" if usage_state.cost_known else "unknown"
    )
    return (
        f"input_tokens={usage_state.input_tokens} cached_input_tokens={usage_state.cached_input_tokens} "
        f"cache_write_tokens={usage_state.cache_write_tokens} output_tokens={usage_state.output_tokens} "
        f"reasoning_tokens={usage_state.reasoning_tokens} total_tokens={usage_state.total_tokens} "
        f"estimated_cost_usd={cost_text}"
    )


def log_run_local_openai_usage(logger: logging.Logger, *, effective_model: str) -> None:
    usage_state = get_run_local_openai_usage()
    models_used: Set[str] = set(usage_state.models_used)
    model_names: str = ",".join(sorted(models_used)) if models_used else "none"
    served_names: str = ",".join(sorted(usage_state.served_models)) if usage_state.served_models else "none"
    parts: list[str] = [
        "OPENAI RUN USAGE",
        "scope=run_local",
        "source_of_truth_for_run=yes",
        f"effective_model={str(effective_model or '').strip() or 'unknown'}",
        f"requests_sent={usage_state.requests_sent}",
        f"repair_calls={usage_state.repair_calls}",
        f"structured_calls={usage_state.structured_calls}",
        f"fallback_calls={usage_state.fallback_calls}",
        f"heading_translation_calls={usage_state.heading_translation_calls}",
        f"models_used={model_names}",
        f"served_models={served_names}",
        f"service_tiers={','.join(sorted(usage_state.service_tiers)) or 'none'}",
        format_run_usage_totals(usage_state),
        f"tokens_known={'yes' if usage_state.tokens_known else 'no'}",
    ]
    logger.info("%s", " ".join(parts))
