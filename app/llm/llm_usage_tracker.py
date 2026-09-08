from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Set

from app.llm.llm_rate_limits import _safe_int_or_none


@dataclass(frozen=True)
class OpenAIRequestUsage:
    """Token usage of one Responses API call, as reported by `response.usage`.

    Field names mirror openai>=2.21 `ResponseUsage`: input_tokens,
    input_tokens_details.{cached_tokens, cache_write_tokens}, output_tokens,
    output_tokens_details.reasoning_tokens, total_tokens. cached/cache_write are
    subsets of input_tokens; reasoning is a subset of output_tokens.
    """

    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    response_id: str
    served_model: str
    service_tier: str


@dataclass
class RunLocalOpenAIUsageState:
    requests_sent: int = 0
    repair_calls: int = 0
    structured_calls: int = 0
    fallback_calls: int = 0
    heading_translation_calls: int = 0
    models_used: Set[str] = field(default_factory=set)
    served_models: Set[str] = field(default_factory=set)
    service_tiers: Set[str] = field(default_factory=set)
    usage_reports: int = 0
    estimated_cost_usd: float = 0.0
    cost_known: bool = True
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: OpenAIRequestUsage, *, cost_usd: Optional[float]) -> None:
        """Accumulate one response; cost_usd=None means the served model has no price entry."""
        self.usage_reports += 1
        if cost_usd is None:
            self.cost_known = False
        else:
            self.estimated_cost_usd += cost_usd
        if usage.service_tier:
            self.service_tiers.add(usage.service_tier)
        self.input_tokens += usage.input_tokens
        self.cached_input_tokens += usage.cached_input_tokens
        self.cache_write_tokens += usage.cache_write_tokens
        self.output_tokens += usage.output_tokens
        self.reasoning_tokens += usage.reasoning_tokens
        self.total_tokens += usage.total_tokens
        if usage.served_model:
            self.served_models.add(usage.served_model)

    @property
    def tokens_known(self) -> bool:
        return self.requests_sent > 0 and self.usage_reports == self.requests_sent


_RUN_LOCAL_USAGE_STATE: Optional[RunLocalOpenAIUsageState] = None


def _field(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container.get(name)
    return getattr(container, name, None)


def _int_field(container: Any, name: str) -> int:
    return _safe_int_or_none(_field(container, name)) or 0


def extract_openai_request_usage(response: Any) -> Optional[OpenAIRequestUsage]:
    usage: Any = _field(response, "usage")
    if usage is None:
        return None
    input_tokens: Optional[int] = _safe_int_or_none(_field(usage, "input_tokens"))
    output_tokens: Optional[int] = _safe_int_or_none(_field(usage, "output_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    input_details: Any = _field(usage, "input_tokens_details")
    output_details: Any = _field(usage, "output_tokens_details")
    total_tokens: Optional[int] = _safe_int_or_none(_field(usage, "total_tokens"))
    return OpenAIRequestUsage(
        input_tokens=input_tokens,
        cached_input_tokens=_int_field(input_details, "cached_tokens"),
        cache_write_tokens=_int_field(input_details, "cache_write_tokens"),
        output_tokens=output_tokens,
        reasoning_tokens=_int_field(output_details, "reasoning_tokens"),
        total_tokens=total_tokens if total_tokens is not None else input_tokens + output_tokens,
        response_id=str(_field(response, "id") or ""),
        served_model=str(_field(response, "model") or ""),
        service_tier=str(_field(response, "service_tier") or ""),
    )


def reset_run_local_openai_usage() -> RunLocalOpenAIUsageState:
    global _RUN_LOCAL_USAGE_STATE
    _RUN_LOCAL_USAGE_STATE = RunLocalOpenAIUsageState()
    return _RUN_LOCAL_USAGE_STATE


def get_run_local_openai_usage() -> RunLocalOpenAIUsageState:
    global _RUN_LOCAL_USAGE_STATE
    if _RUN_LOCAL_USAGE_STATE is None:
        _RUN_LOCAL_USAGE_STATE = RunLocalOpenAIUsageState()
    return _RUN_LOCAL_USAGE_STATE


def record_run_local_openai_repair_call() -> None:
    state: RunLocalOpenAIUsageState = get_run_local_openai_usage()
    state.repair_calls += 1
