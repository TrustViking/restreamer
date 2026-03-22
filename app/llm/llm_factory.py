from __future__ import annotations

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.llm.models.model_identity import build_effective_llm_model_identity
from app.llm.providers import LlmProvider, OpenAIProvider

LOGGER = _get_logger_impl(__name__)


def get_llm_provider(*, config: AppConfig) -> LlmProvider:
    provider: LlmProvider = OpenAIProvider()
    model_identity = build_effective_llm_model_identity(config)
    LOGGER.info(
        "llm_provider_selected provider=%s effective_model=%s configured_model=%s provider_model=%s",
        provider.name,
        model_identity.effective_model or "unknown",
        model_identity.configured_model or "unknown",
        model_identity.provider_model or "unknown",
    )
    return provider
