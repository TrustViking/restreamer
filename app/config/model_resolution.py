from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


CANONICAL_MODEL_ALIAS_DEFAULTS: dict[str, str] = {
    "OPENAI_MODEL": "gpt-5.1",
    "DEEPSEEK_MODEL": "deepseek-chat",
}


@dataclass(frozen=True)
class ResolvedLlmTarget:
    provider: str
    model: str


@dataclass(frozen=True)
class ResolvedLlmRouting:
    primary: ResolvedLlmTarget
    fallback: ResolvedLlmTarget

    @property
    def providers_used(self) -> tuple[str, ...]:
        ordered_providers: list[str] = []
        for provider_name in (self.primary.provider, self.fallback.provider):
            if provider_name not in ordered_providers:
                ordered_providers.append(provider_name)
        return tuple(ordered_providers)

    @property
    def is_mixed_provider(self) -> bool:
        return self.primary.provider != self.fallback.provider

    def uses_provider(self, provider_name: str) -> bool:
        normalized_provider_name: str = str(provider_name or "").strip().lower()
        return normalized_provider_name in self.providers_used


@dataclass(frozen=True)
class ResolvedModelConfiguration:
    model_aliases: dict[str, str]
    main_model_alias: str
    fallback_model_alias: str
    routing: ResolvedLlmRouting


def build_canonical_model_aliases(
    *,
    explicit_aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    alias_map: dict[str, str] = dict(CANONICAL_MODEL_ALIAS_DEFAULTS)
    if explicit_aliases is None:
        return alias_map
    for alias_name, alias_value in explicit_aliases.items():
        normalized_alias_name: str = str(alias_name or "").strip().upper()
        normalized_alias_value: str = str(alias_value or "").strip()
        if normalized_alias_name and normalized_alias_value:
            alias_map[normalized_alias_name] = normalized_alias_value
    return alias_map


def resolve_provider_for_model(model_name: str) -> str:
    normalized_model_name: str = str(model_name or "").strip().lower()
    if normalized_model_name.startswith("deepseek"):
        return "deepseek"
    if normalized_model_name.startswith("gpt-"):
        return "openai"
    if normalized_model_name.startswith("chatgpt-"):
        return "openai"
    if re.match(r"^o\d", normalized_model_name):
        return "openai"
    if normalized_model_name in {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}:
        return "openai"
    if normalized_model_name:
        return "openai"
    raise RuntimeError("Model name must not be empty when resolving provider.")


def resolve_model_alias(alias_name: str, *, model_aliases: Mapping[str, str]) -> str:
    normalized_alias_name: str = str(alias_name or "").strip().upper()
    if not normalized_alias_name:
        raise RuntimeError("Model alias must not be empty.")
    if normalized_alias_name not in model_aliases:
        supported_aliases: str = ", ".join(sorted(model_aliases))
        raise RuntimeError(
            f"Unsupported model alias {normalized_alias_name!r}. Supported aliases: {supported_aliases}."
        )
    resolved_value: str = str(model_aliases[normalized_alias_name] or "").strip()
    if not resolved_value:
        raise RuntimeError(f"Model alias {normalized_alias_name} is declared but empty.")
    return resolved_value


def build_llm_target(
    *,
    model_alias_name: str,
    model_aliases: Mapping[str, str],
) -> ResolvedLlmTarget:
    canonical_model: str = resolve_model_alias(
        model_alias_name,
        model_aliases=model_aliases,
    )
    return ResolvedLlmTarget(
        provider=resolve_provider_for_model(canonical_model),
        model=canonical_model,
    )


def resolve_model_configuration(
    *,
    model_aliases: Mapping[str, str],
    main_model_raw: str,
    fallback_model_raw: str,
) -> ResolvedModelConfiguration:
    main_model_alias: str = str(main_model_raw or "").strip().upper() or "OPENAI_MODEL"
    fallback_model_alias: str = (
        str(fallback_model_raw or "").strip().upper() or "DEEPSEEK_MODEL"
    )
    routing = ResolvedLlmRouting(
        primary=build_llm_target(
            model_alias_name=main_model_alias,
            model_aliases=model_aliases,
        ),
        fallback=build_llm_target(
            model_alias_name=fallback_model_alias,
            model_aliases=model_aliases,
        ),
    )
    return ResolvedModelConfiguration(
        model_aliases=dict(model_aliases),
        main_model_alias=main_model_alias,
        fallback_model_alias=fallback_model_alias,
        routing=routing,
    )


def coerce_llm_routing_from_config(config: object) -> ResolvedLlmRouting:
    existing_routing: object = getattr(config, "llm_routing", None)
    if isinstance(existing_routing, ResolvedLlmRouting):
        return existing_routing
    explicit_main_model: str = str(getattr(config, "llm_main_model", "") or "").strip()
    explicit_fallback_model: str = str(getattr(config, "llm_fallback_model", "") or "").strip()
    if explicit_main_model and explicit_fallback_model:
        return ResolvedLlmRouting(
            primary=ResolvedLlmTarget(
                provider=resolve_provider_for_model(explicit_main_model),
                model=explicit_main_model,
            ),
            fallback=ResolvedLlmTarget(
                provider=resolve_provider_for_model(explicit_fallback_model),
                model=explicit_fallback_model,
            ),
        )
    model_aliases: dict[str, str] = build_canonical_model_aliases(
        explicit_aliases={
            "OPENAI_MODEL": str(getattr(config, "openai_model_primary", "") or "").strip(),
            "DEEPSEEK_MODEL": str(getattr(config, "deepseek_model", "") or "").strip(),
        }
    )
    legacy_provider: str = str(getattr(config, "llm_provider", "") or "").strip().lower()
    if legacy_provider == "deepseek":
        return ResolvedLlmRouting(
            primary=build_llm_target(
                model_alias_name="DEEPSEEK_MODEL",
                model_aliases=model_aliases,
            ),
            fallback=build_llm_target(
                model_alias_name="DEEPSEEK_MODEL",
                model_aliases=model_aliases,
            ),
        )
    if legacy_provider == "openai":
        return ResolvedLlmRouting(
            primary=build_llm_target(
                model_alias_name="OPENAI_MODEL",
                model_aliases=model_aliases,
            ),
            fallback=build_llm_target(
                model_alias_name="OPENAI_MODEL",
                model_aliases=model_aliases,
            ),
        )
    resolved = resolve_model_configuration(
        model_aliases=model_aliases,
        main_model_raw=str(getattr(config, "configured_main_model_alias", "") or "").strip(),
        fallback_model_raw=str(
            getattr(config, "configured_fallback_model_alias", "") or ""
        ).strip(),
    )
    return resolved.routing


def providers_use_openai(providers: tuple[str, ...]) -> bool:
    return any(str(provider_name or "").strip().lower() == "openai" for provider_name in providers)
