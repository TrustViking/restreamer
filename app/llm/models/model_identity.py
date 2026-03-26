from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import AppConfig

DEFAULT_OPENAI_MODEL: str = "gpt-5.1"


@dataclass(frozen=True)
class EffectiveLlmModelIdentity:
    provider: str
    configured_model: str
    provider_model: str
    effective_model: str


def build_effective_llm_model_identity(
    config: AppConfig,
) -> EffectiveLlmModelIdentity:
    provider: str = str(config.llm.provider or "").strip() or "openai"
    configured_model: str = str(config.llm.model or "").strip()
    provider_model: str = configured_model
    effective_model: str = configured_model or provider_model
    if not effective_model and provider == "openai":
        effective_model = DEFAULT_OPENAI_MODEL
    return EffectiveLlmModelIdentity(
        provider=provider,
        configured_model=configured_model,
        provider_model=provider_model,
        effective_model=effective_model,
    )


def resolve_effective_llm_model(config: AppConfig) -> str:
    return build_effective_llm_model_identity(config).effective_model
