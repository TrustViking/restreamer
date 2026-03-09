from __future__ import annotations

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.llm_routing import (
    ResolvedLlmRouting,
    ResolvedLlmTarget,
    coerce_llm_routing_from_config,
    resolve_provider_for_model,
)
from app.config.settings import AppConfig
from app.llm.providers import DeepSeekProvider, LlmProvider, OpenAIProvider

LOGGER = _get_logger_impl(__name__)


def get_llm_provider_by_name(*, provider_name: str) -> LlmProvider:
    normalized_provider_name: str = str(provider_name or "").strip().lower() or "openai"
    if normalized_provider_name == "openai":
        return OpenAIProvider()
    if normalized_provider_name == "deepseek":
        return DeepSeekProvider()
    raise RuntimeError(f"Unsupported llm_provider={normalized_provider_name!r}.")


def get_llm_routing(*, config: object) -> ResolvedLlmRouting:
    return coerce_llm_routing_from_config(config)


def get_llm_provider(*, config: AppConfig) -> LlmProvider:
    routing: ResolvedLlmRouting = get_llm_routing(config=config)
    provider: LlmProvider = get_llm_provider_by_name(provider_name=routing.primary.provider)
    LOGGER.info(
        "llm_provider_selected provider=%s primary_model=%s fallback_model=%s",
        provider.name,
        routing.primary.model,
        routing.fallback.model,
    )
    return provider


def get_llm_provider_for_target(*, target: ResolvedLlmTarget) -> LlmProvider:
    provider: LlmProvider = get_llm_provider_by_name(provider_name=target.provider)
    LOGGER.info(
        "llm_provider_selected provider=%s primary_model=%s fallback_model=not_applicable selection_source=routing_target",
        provider.name,
        target.model,
    )
    return provider


def get_llm_provider_for_model(*, model_name: str) -> LlmProvider:
    return get_llm_provider_for_target(
        target=ResolvedLlmTarget(
            provider=resolve_provider_for_model(model_name),
            model=str(model_name or "").strip(),
        )
    )
