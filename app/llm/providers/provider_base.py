from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from app.config.settings import AppConfig
from app.llm.llm_client import LlmTraceContext, OpenAITransportResult


@dataclass(frozen=True)
class ProviderModels:
    primary: str
    fallback: str


class LlmProvider(Protocol):
    name: str

    def models(self, *, config: AppConfig) -> ProviderModels:
        ...

    def timeout_sec(self, *, config: AppConfig) -> float:
        ...

    def pre_delay_sec(self, *, config: AppConfig) -> float:
        ...

    def request_merge(
        self,
        *,
        prompt_text: str,
        model_name: str,
        config: AppConfig,
        attempt_label: str,
        max_output_tokens: int,
        structured_schema: Optional[Dict[str, Any]],
        temperature: float,
        trace_context: Optional[LlmTraceContext],
    ) -> OpenAITransportResult:
        ...
