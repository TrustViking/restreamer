from __future__ import annotations

from typing import Any, Dict, Optional

from app.config.settings import AppConfig
from app.llm.openai_client import LlmTraceContext, OpenAITransportResult
from app.llm.providers.base import LlmProvider, ProviderModels


class OpenAIProvider(LlmProvider):
    name: str = "openai"

    def models(self, *, config: AppConfig) -> ProviderModels:
        return ProviderModels(
            primary=str(getattr(config, "openai_model_primary", "") or "").strip() or "gpt-5.1",
            fallback=str(getattr(config, "openai_model_fallback", "") or "").strip() or "gpt-5-mini",
        )

    def timeout_sec(self, *, config: AppConfig) -> float:
        return float(getattr(config, "openai_timeout_sec", 120.0))

    def pre_delay_sec(self, *, config: AppConfig) -> float:
        return float(getattr(config, "openai_pre_delay_sec", 0.0))

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
            structured_schema=structured_schema,
            temperature=temperature,
            trace_context=trace_context,
        )
