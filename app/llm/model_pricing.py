from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.llm.llm_usage_tracker import OpenAIRequestUsage, RunLocalOpenAIUsageState


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens, Standard tier, short context (developers.openai.com/api/docs/pricing)."""

    input_usd: float
    cached_input_usd: float
    cache_write_usd: float  # models without a cache-write price bill writes as plain input
    output_usd: float


# Snapshot of the pricing page, Sep 2026. Update by hand when the page changes.
MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5.2": ModelPrice(input_usd=1.75, cached_input_usd=0.175, cache_write_usd=1.75, output_usd=14.00),
    "gpt-5.4": ModelPrice(input_usd=2.50, cached_input_usd=0.25, cache_write_usd=2.50, output_usd=15.00),
    "gpt-5.5": ModelPrice(input_usd=5.00, cached_input_usd=0.50, cache_write_usd=5.00, output_usd=30.00),
    "gpt-5.6-sol": ModelPrice(input_usd=4.00, cached_input_usd=0.40, cache_write_usd=5.00, output_usd=20.00),
    "gpt-5.6-terra": ModelPrice(input_usd=2.00, cached_input_usd=0.20, cache_write_usd=2.50, output_usd=12.00),
    "gpt-5.6-luna": ModelPrice(input_usd=0.20, cached_input_usd=0.02, cache_write_usd=0.25, output_usd=1.20),
    "gpt-6-astra": ModelPrice(input_usd=10.00, cached_input_usd=1.00, cache_write_usd=12.50, output_usd=50.00),
}
MODEL_PRICES["gpt-5.6"] = MODEL_PRICES["gpt-5.6-sol"]  # alias routed to Sol

_SNAPSHOT_SUFFIX: re.Pattern[str] = re.compile(r"-\d{4}-\d{2}-\d{2}$")
# Relative to Standard: Flex is half price, Fast (ex-Priority) is double.
SERVICE_TIER_MULTIPLIERS: dict[str, float] = {
    "default": 1.0,
    "auto": 1.0,
    "flex": 0.5,
    "priority": 2.0,
    "fast": 2.0,
}


def price_for_model(model_name: str) -> Optional[ModelPrice]:
    """`gpt-5.4-2026-03-05` (what response.model returns) prices as `gpt-5.4`."""
    cleaned: str = _SNAPSHOT_SUFFIX.sub("", str(model_name or "").strip().lower())
    return MODEL_PRICES.get(cleaned)


def estimate_cost_usd(
    model_name: str,
    usage: OpenAIRequestUsage | RunLocalOpenAIUsageState,
    *,
    service_tier: str = "default",
) -> Optional[float]:
    price: Optional[ModelPrice] = price_for_model(model_name)
    if price is None:
        return None
    multiplier: float = SERVICE_TIER_MULTIPLIERS.get(str(service_tier or "").strip().lower(), 1.0)
    uncached_input: int = max(
        0, usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
    )
    cost: float = (
        uncached_input * price.input_usd
        + usage.cached_input_tokens * price.cached_input_usd
        + usage.cache_write_tokens * price.cache_write_usd
        + usage.output_tokens * price.output_usd
    ) / 1_000_000.0 * multiplier
    return round(cost, 6)
