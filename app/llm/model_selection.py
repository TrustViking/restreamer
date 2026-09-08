from __future__ import annotations

import dataclasses
import logging
import os

from app.config.settings import AppConfig
from app.core.env_flags import llm_allow_in_dry_run_from_env
from app.core.error_summary import summarize_error
from app.llm.llm_client import probe_openai_model_access
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.observability.runtime_analytics import log_warning_operational

# Only "this project cannot use this model" errors justify switching to the
# fallback model. Timeouts, rate limits and server errors say nothing about
# access, so the primary model is kept for those.
FALLBACK_REASON_CODES: frozenset[str] = frozenset(
    {"openai_model_not_found", "openai_model_access_denied"}
)
_PROBE_TIMEOUT_SEC: float = 30.0


def llm_merge_requested(*, audit_mode: str, dry_run: bool) -> bool:
    if audit_mode not in {"merge", "audit"}:
        return False
    if dry_run and not llm_allow_in_dry_run_from_env():
        return False
    return bool(os.getenv("GPT_API_KEY", "").strip())


def select_llm_model(*, config: AppConfig, logger: logging.Logger) -> AppConfig:
    """Return the config whose llm.model passed one real request.

    Order: primary (llm.model), then llm.fallback_model. The fallback is tried
    only when the primary fails with an access error; the last candidate raises.
    """
    primary: str = config.llm.model
    fallback: str = config.llm.fallback_model
    candidates: list[str] = [primary] if fallback in {"", primary} else [primary, fallback]
    for index, model_name in enumerate(candidates):
        next_model: str = candidates[index + 1] if index + 1 < len(candidates) else ""
        try:
            probe_openai_model_access(
                provider_name=config.llm.provider,
                model_name=model_name,
                timeout_sec=min(_PROBE_TIMEOUT_SEC, float(config.llm.timeout_sec)),
                reasoning_effort=config.llm.reasoning_effort,
            )
        except LlmModelConfigurationError as error:
            if next_model and error.reason_code in FALLBACK_REASON_CODES:
                log_warning_operational(
                    logger,
                    "llm_model_fallback from=%s to=%s reason_code=%s status_code=%s detail=%s",
                    model_name,
                    next_model,
                    error.reason_code,
                    str(error.status_code if error.status_code is not None else "unknown"),
                    error.detail,
                    reason_code="llm_model_fallback",
                )
                continue
            raise
        except Exception as error:
            logger.warning(
                "llm_model_probe_inconclusive model=%s error=%s decision=keep_model",
                model_name,
                summarize_error(error),
                extra={"warning_category": "informational"},
            )
        logger.info(
            "llm_model_selected model=%s primary=%s fallback=%s reasoning_effort=%s",
            model_name,
            primary,
            fallback or "none",
            config.llm.reasoning_effort,
        )
        if model_name == primary:
            return config
        return dataclasses.replace(config, llm=dataclasses.replace(config.llm, model=model_name))
    return config
