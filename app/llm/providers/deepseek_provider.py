from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.llm.merge_parser import extract_json_object_candidates, parse_json_tolerant, strip_json_code_fences
from app.llm.openai_client import (
    LlmTraceContext,
    OpenAITransportResult,
    get_openai_client,
)
from app.llm.providers.base import LlmProvider, ProviderModels

LOGGER = _get_logger_impl(__name__)


def _extract_chat_text(response: Any) -> str:
    choices: Any = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice: Any = choices[0]
    message: Any = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")
    if message is None:
        return ""
    content: Any = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""


def _extract_structured_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    json_text: str = strip_json_code_fences(raw_text)
    if not json_text:
        return None
    candidates = [json_text, *extract_json_object_candidates(json_text)]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed: Any = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return cast(Dict[str, Any], parsed)
    tolerant_payload, _ = parse_json_tolerant(json_text)
    if isinstance(tolerant_payload, dict):
        return cast(Dict[str, Any], tolerant_payload)
    return None


class DeepSeekProvider(LlmProvider):
    name: str = "deepseek"

    def models(self, *, config: AppConfig) -> ProviderModels:
        selected_model: str = str(getattr(config, "deepseek_model", "") or "").strip() or "deepseek-chat"
        return ProviderModels(primary=selected_model, fallback=selected_model)

    def timeout_sec(self, *, config: AppConfig) -> float:
        return float(getattr(config, "deepseek_timeout_sec", 120.0))

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
        del attempt_label
        client: Any = get_openai_client(
            provider_name=self.name,
            api_key_env="DPSK_API_KEY",
            timeout_sec=self.timeout_sec(config=config),
            base_url=str(getattr(config, "deepseek_base_url", "") or "").strip() or "https://api.deepseek.com/v1",
            max_retries=0,
        ).with_options(timeout=self.timeout_sec(config=config), max_retries=0)
        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "timeout": self.timeout_sec(config=config),
            "max_tokens": max_output_tokens,
            "temperature": float(temperature),
        }
        if structured_schema is not None:
            request_kwargs["response_format"] = {"type": "json_object"}
        if trace_context is not None:
            LOGGER.info(
                "llm_request_started provider=%s model=%s attempt=%d request_kind=%s",
                trace_context.provider,
                trace_context.model_name,
                trace_context.attempt_index,
                trace_context.request_kind,
            )
        request_started_at: float = time.perf_counter()
        try:
            response: Any = client.chat.completions.create(**request_kwargs)
            raw_text: str = _extract_chat_text(response)
            if not raw_text:
                raise RuntimeError("deepseek merge returned empty output text")
            if trace_context is not None:
                LOGGER.info(
                    "llm_request_completed provider=%s model=%s success=yes attempt=%d request_kind=%s",
                    trace_context.provider,
                    trace_context.model_name,
                    trace_context.attempt_index,
                    trace_context.request_kind,
                )
            return OpenAITransportResult(
                raw_text=raw_text,
                structured_payload=(
                    _extract_structured_payload(raw_text) if structured_schema is not None else None
                ),
                incomplete_reason="",
                output_item_types=["chat.completion"],
            )
        except Exception as error:
            if trace_context is not None:
                LOGGER.warning(
                    "llm_request_failed provider=%s model=%s error_type=%s attempt=%d request_kind=%s elapsed_ms=%d",
                    trace_context.provider,
                    trace_context.model_name,
                    type(error).__name__,
                    trace_context.attempt_index,
                    trace_context.request_kind,
                    int(round((time.perf_counter() - request_started_at) * 1000.0)),
                )
            raise
