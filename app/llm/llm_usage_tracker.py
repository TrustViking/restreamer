from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class RunLocalOpenAIUsageState:
    requests_sent: int = 0
    repair_calls: int = 0
    structured_calls: int = 0
    fallback_calls: int = 0
    models_used: Set[str] = field(default_factory=set)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


_RUN_LOCAL_USAGE_STATE: Optional[RunLocalOpenAIUsageState] = None


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
