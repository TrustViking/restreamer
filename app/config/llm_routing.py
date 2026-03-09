from __future__ import annotations

from app.config.model_resolution import (
    CANONICAL_MODEL_ALIAS_DEFAULTS as DEFAULT_MODEL_ALIASES,
    ResolvedLlmRouting,
    ResolvedLlmTarget,
    build_canonical_model_aliases as build_model_aliases,
    build_llm_target,
    coerce_llm_routing_from_config,
    providers_use_openai,
    resolve_model_alias,
    resolve_model_configuration,
    resolve_provider_for_model,
)


def build_llm_routing(
    *,
    primary_input: str,
    fallback_input: str,
    model_aliases: dict[str, str],
) -> ResolvedLlmRouting:
    resolved = resolve_model_configuration(
        model_aliases=model_aliases,
        main_model_raw=primary_input,
        fallback_model_raw=fallback_input,
    )
    return resolved.routing


def resolve_llm_routing(
    *,
    legacy_provider: str,
    model_aliases: dict[str, str],
    main_model_raw: str,
    fallback_model_raw: str,
    openai_primary_raw: str,
    openai_fallback_raw: str,
    deepseek_model_raw: str,
) -> ResolvedLlmRouting:
    del legacy_provider
    del openai_primary_raw
    del openai_fallback_raw
    del deepseek_model_raw
    resolved = resolve_model_configuration(
        model_aliases=model_aliases,
        main_model_raw=main_model_raw,
        fallback_model_raw=fallback_model_raw,
    )
    return resolved.routing
