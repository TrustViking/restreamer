from __future__ import annotations

import logging
from typing import Dict, Optional, Set

from app.llm.llm_client import get_run_local_openai_usage


def log_run_local_openai_usage(logger: logging.Logger, *, effective_model: str) -> None:
    usage_state = get_run_local_openai_usage()
    models_used: Set[str] = set(usage_state.models_used)
    model_names: str = ",".join(sorted(models_used)) if models_used else "none"
    token_fields: Dict[str, Optional[int]] = {
        "input_tokens": usage_state.input_tokens,
        "output_tokens": usage_state.output_tokens,
    }
    parts: list[str] = [
        "OPENAI RUN USAGE",
        "scope=run_local",
        "source_of_truth_for_run=yes",
        f"effective_model={str(effective_model or '').strip() or 'unknown'}",
        f"requests_sent={usage_state.requests_sent}",
        f"repair_calls={usage_state.repair_calls}",
        f"structured_calls={usage_state.structured_calls}",
        f"fallback_calls={usage_state.fallback_calls}",
        f"models_used={model_names}",
    ]
    tokens_known: bool = (
        token_fields["input_tokens"] is not None
        and token_fields["output_tokens"] is not None
    )
    if token_fields["input_tokens"] is not None:
        parts.append(f"input_tokens={token_fields['input_tokens']}")
    if token_fields["output_tokens"] is not None:
        parts.append(f"output_tokens={token_fields['output_tokens']}")
    parts.append(f"tokens_known={'yes' if tokens_known else 'no'}")
    logger.info("%s", " ".join(parts))
