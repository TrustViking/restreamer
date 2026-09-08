from __future__ import annotations

from typing import Any, Dict, Optional

from app.config.settings import AppConfig
from app.llm.models.model_identity import resolve_effective_llm_model
from app.llm.llm_client import LlmTraceContext, OpenAITransportResult
from app.llm.providers.provider_base import LlmProvider, ProviderModels


class OpenAIProvider(LlmProvider):
    name: str = "openai"

    def models(self, *, config: AppConfig) -> ProviderModels:
        effective_model: str = resolve_effective_llm_model(config)
        return ProviderModels(
            primary=effective_model,
            fallback=effective_model,
        )

    def timeout_sec(self, *, config: AppConfig) -> float:
        return float(config.llm.timeout_sec)

    def pre_delay_sec(self, *, config: AppConfig) -> float:
        return float(config.llm.pre_delay_sec)

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
        from app.llm import merge_service

        return merge_service.openai_request_merge(
            prompt_text=prompt_text,
            model_name=model_name,
            timeout_sec=self.timeout_sec(config=config),
            attempt_label=attempt_label,
            max_output_tokens=max_output_tokens,
            reasoning_effort=config.llm.reasoning_effort,
            service_tier=config.llm.service_tier,
            structured_schema=structured_schema,
            temperature=temperature,
            trace_context=trace_context,
        )
