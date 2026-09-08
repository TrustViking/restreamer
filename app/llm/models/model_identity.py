from __future__ import annotations

from dataclasses import dataclass

from app.config.settings import AppConfig

DEFAULT_OPENAI_MODEL: str = "gpt-5.2"
DEFAULT_REASONING_EFFORT: str = "medium"
REASONING_EFFORT_VALUES: frozenset[str] = frozenset(
    {"none", "low", "medium", "high", "xhigh", "max"}
)
SERVICE_TIER_DEFAULT: str = "default"
SERVICE_TIER_FLEX: str = "flex"
DEFAULT_SERVICE_TIER: str = SERVICE_TIER_DEFAULT
# OpenAI `service_tier`: flex = -50% price, slower, may answer 429 resource_unavailable;
# priority/fast = faster, +100% price ("priority" was renamed "fast" in July 2026, both accepted).
SERVICE_TIER_VALUES: frozenset[str] = frozenset({"auto", "default", "flex", "priority", "fast"})
# Waits before each flex retry on 429 resource_unavailable; after the last one the
# request is repeated on the default tier.
FLEX_RETRY_DELAYS_SEC: tuple[float, ...] = (20.0, 40.0, 80.0)


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
