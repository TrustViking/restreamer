from __future__ import annotations

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.llm.providers import LlmProvider, OpenAIProvider

LOGGER = _get_logger_impl(__name__)


def get_llm_provider(*, config: AppConfig) -> LlmProvider:
    provider: LlmProvider = OpenAIProvider()
    LOGGER.info(
        "llm_provider_selected provider=%s model=%s",
        provider.name,
        config.llm_model,
    )
    return provider
