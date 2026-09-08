from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.llm.llm_rate_limits import (
    _extract_openai_headers_from_raw_response,
    _log_openai_rate_limit_snapshot,
)
from app.llm.models.model_compatibility import (
    LlmModelConfigurationError,
    LlmRequestErrorClassification,
    OpenAIRequestCompatibility,
    classify_openai_request_error,
    resolve_openai_request_compatibility,
)
from app.llm.merges.merge_parser import extract_json_object_candidates, parse_json_tolerant, strip_json_code_fences
from app.llm.model_pricing import estimate_cost_usd
from app.llm.models.model_identity import (
    FLEX_RETRY_DELAYS_SEC,
    SERVICE_TIER_DEFAULT,
    SERVICE_TIER_FLEX,
)

# Backward-compatible re-exports — DO NOT REMOVE
from app.llm.llm_usage_tracker import (  # noqa: F401
    OpenAIRequestUsage,
    RunLocalOpenAIUsageState,
    extract_openai_request_usage,
    reset_run_local_openai_usage,
    get_run_local_openai_usage,
    record_run_local_openai_repair_call,
)
from app.llm.llm_rate_limits import (  # noqa: F401
    extract_rate_limit_snapshot,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

LOGGER = _get_logger_impl(__name__)
_OPENAI_CLIENTS: Dict[Tuple[str, str, float, int], Any] = {}


@dataclass(frozen=True)
class OpenAITransportResult:
    raw_text: str
    structured_payload: Optional[Dict[str, Any]]
    incomplete_reason: str
    output_item_types: List[str]


@dataclass(frozen=True)
class LlmTraceContext:
    branch_label: str
    date_key: str
    slot_key: str
    language: str
    provider: str
    model_name: str
    attempt_index: int
    request_kind: str
    source_count: int


def _extract_openai_response_text(response: Any) -> str:
    output_text: Any = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

    output_items: Any = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    if not isinstance(output_items, list):
        return ""
    chunks: List[str] = []
    for output_item in output_items:
        content_items: Any = output_item.get("content") if isinstance(output_item, dict) else getattr(output_item, "content", None)
        if isinstance(content_items, list):
            for content_item in content_items:
                text_value: Any = content_item.get("text") if isinstance(content_item, dict) else getattr(content_item, "text", None)
                item_type: str = str(content_item.get("type", "") if isinstance(content_item, dict) else getattr(content_item, "type", "")).strip().lower()
                if item_type in {"output_text", "text"} and isinstance(text_value, str) and text_value.strip():
                    chunks.append(text_value.strip())
    return "\n".join(chunks).strip()


def _openai_response_output_item_types(response: Any) -> List[str]:
    output_items: Any = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    if not isinstance(output_items, list):
        return []
    item_types: List[str] = []
    for item in output_items:
        item_type: str = str(item.get("type", "") if isinstance(item, dict) else getattr(item, "type", "")).strip().lower()
        if item_type:
            item_types.append(item_type)
    return item_types


def _openai_incomplete_reason(response: Any) -> str:
    incomplete: Any = getattr(response, "incomplete_details", None)
    if incomplete is None and isinstance(response, dict):
        incomplete = response.get("incomplete_details")
    if incomplete is None:
        return ""
    if isinstance(incomplete, dict):
        return str(incomplete.get("reason", "")).strip().lower()
    return str(getattr(incomplete, "reason", "")).strip().lower()


def _usage_log_fields(usage: Optional[OpenAIRequestUsage]) -> str:
    if usage is None:
        return "input_tokens=unknown output_tokens=unknown"
    return (
        f"input_tokens={usage.input_tokens} cached_input_tokens={usage.cached_input_tokens} "
        f"cache_write_tokens={usage.cache_write_tokens} output_tokens={usage.output_tokens} "
        f"reasoning_tokens={usage.reasoning_tokens} total_tokens={usage.total_tokens} "
        f"served_model={usage.served_model or 'unknown'} response_id={usage.response_id or 'unknown'}"
    )


def _log_llm_request_start(
    *,
    trace_context: Optional[LlmTraceContext],
    input_chars: int,
) -> None:
    if trace_context is None:
        return
    LOGGER.info(
        "llm_request_started provider=%s model=%s attempt=%d request_kind=%s",
        trace_context.provider,
        trace_context.model_name,
        trace_context.attempt_index,
        trace_context.request_kind,
    )
    LOGGER.info(
        "llm_request_start branch=%s date_key=%s slot_key=%s lang=%s provider=%s model=%s attempt_index=%d request_kind=%s source_count=%d input_chars=%d estimated_input_tokens=unknown",
        trace_context.branch_label,
        trace_context.date_key,
        trace_context.slot_key,
        trace_context.language,
        trace_context.provider,
        trace_context.model_name,
        trace_context.attempt_index,
        trace_context.request_kind,
        trace_context.source_count,
        input_chars,
    )


def _log_llm_request_finish(
    *,
    trace_context: Optional[LlmTraceContext],
    response: Any,
    success: bool,
    elapsed_ms: int,
    max_output_hit: bool,
) -> None:
    if trace_context is None:
        return
    usage: Optional[OpenAIRequestUsage] = extract_openai_request_usage(response)
    raw_text: str = _extract_openai_response_text(response)
    finish_reason: str = _openai_incomplete_reason(response) or "completed"
    LOGGER.info(
        "llm_request_completed provider=%s model=%s success=%s attempt=%d request_kind=%s",
        trace_context.provider,
        trace_context.model_name,
        "yes" if success else "no",
        trace_context.attempt_index,
        trace_context.request_kind,
    )
    LOGGER.info(
        "llm_request_finish branch=%s date_key=%s slot_key=%s lang=%s provider=%s model=%s attempt_index=%d request_kind=%s success=%s llm_call_ms=%d output_chars=%d %s finish_reason=%s max_output_hit=%s",
        trace_context.branch_label,
        trace_context.date_key,
        trace_context.slot_key,
        trace_context.language,
        trace_context.provider,
        trace_context.model_name,
        trace_context.attempt_index,
        trace_context.request_kind,
        "yes" if success else "no",
        elapsed_ms,
        len(raw_text),
        _usage_log_fields(usage),
        finish_reason,
        "yes" if max_output_hit else "no",
    )


def _log_llm_request_failed(
    *,
    trace_context: Optional[LlmTraceContext],
    error: Exception,
    elapsed_ms: int,
) -> None:
    if trace_context is None:
        return
    LOGGER.warning(
        "llm_request_failed provider=%s model=%s error_type=%s attempt=%d request_kind=%s elapsed_ms=%d",
        trace_context.provider,
        trace_context.model_name,
        type(error).__name__,
        trace_context.attempt_index,
        trace_context.request_kind,
        elapsed_ms,
        extra={"warning_category": "informational"},
    )


def _log_llm_retry_decision(
    *,
    trace_context: Optional[LlmTraceContext],
    reason_code: str,
    retry_index: int,
    retry_kind: str,
    recovered: bool,
) -> None:
    if trace_context is None:
        return
    LOGGER.info(
        "llm_retry_decision branch=%s date_key=%s slot_key=%s lang=%s provider=%s model=%s retry_index=%d retry_kind=%s reason_code=%s recovered=%s",
        trace_context.branch_label,
        trace_context.date_key,
        trace_context.slot_key,
        trace_context.language,
        trace_context.provider,
        trace_context.model_name,
        retry_index,
        retry_kind,
        reason_code,
        "yes" if recovered else "no",
    )


def _record_run_local_openai_request(
    *,
    model_name: str,
    response: Any,
    request_kind: str,
) -> None:
    state: RunLocalOpenAIUsageState = get_run_local_openai_usage()
    cleaned_model_name: str = str(model_name or "").strip() or "unknown"
    state.requests_sent += 1
    state.models_used.add(cleaned_model_name)
    if request_kind == "structured":
        state.structured_calls += 1
    elif request_kind == "heading_translation":
        state.heading_translation_calls += 1
    else:
        state.fallback_calls += 1
    usage: Optional[OpenAIRequestUsage] = extract_openai_request_usage(response)
    if usage is not None:
        state.add(
            usage,
            cost_usd=estimate_cost_usd(
                usage.served_model or cleaned_model_name,
                usage,
                service_tier=usage.service_tier or "default",
            ),
        )


def _is_openai_temperature_unsupported_error(error: Exception) -> bool:
    message: str = str(error or "").lower()
    return "unsupported parameter" in message and "temperature" in message and "invalid_request_error" in message


def _log_openai_request_compatibility(
    *,
    trace_context: Optional[LlmTraceContext],
    compatibility: OpenAIRequestCompatibility,
) -> None:
    request_kind: str = (
        trace_context.request_kind if trace_context is not None else "unknown"
    )
    LOGGER.info(
        "openai_request_compatibility model=%s request_kind=%s model_family=%s reasoning_effort=%s structured_output=%s temperature=%s capability_source=%s",
        compatibility.model_name or "unknown",
        request_kind,
        compatibility.model_family,
        "enabled" if compatibility.reasoning_effort_enabled else "disabled",
        (
            "json_schema"
            if compatibility.structured_output_requested and compatibility.structured_output_supported
            else "disabled"
        ),
        "enabled" if compatibility.temperature_enabled else "disabled",
        compatibility.capability_source,
    )


def _build_openai_responses_request_kwargs(
    *,
    prompt_text: str,
    model_name: str,
    max_tokens: int,
    structured_schema: Optional[Dict[str, Any]],
    temperature: float,
    compatibility: OpenAIRequestCompatibility,
    reasoning_effort: str,
    service_tier: str = "default",
) -> Dict[str, Any]:
    request_kwargs: Dict[str, Any] = {
        "model": model_name,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
        "max_output_tokens": max_tokens,
    }
    if compatibility.reasoning_effort_enabled:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}
    if service_tier and service_tier != "default":
        request_kwargs["service_tier"] = service_tier
    if structured_schema is not None:
        request_kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": structured_schema.get("name", "merge_v1"),
                "strict": True,
                "schema": structured_schema["schema"],
            }
        }
    if compatibility.temperature_enabled:
        request_kwargs["temperature"] = float(temperature)
    return request_kwargs


def _raise_if_fatal_model_configuration_error(
    *,
    provider_name: str,
    model_name: str,
    error: Exception,
) -> None:
    classification: LlmRequestErrorClassification = classify_openai_request_error(error)
    if not classification.fatal_model_configuration:
        return
    raise LlmModelConfigurationError(
        provider_name=provider_name,
        model_name=model_name,
        reason_code=classification.reason_code,
        detail=classification.detail,
        status_code=classification.status_code,
        api_error_code=classification.api_error_code,
        api_error_param=classification.api_error_param,
    ) from error


def get_openai_client(
    *,
    provider_name: str,
    api_key_env: str,
    timeout_sec: float,
    base_url: Optional[str] = None,
    max_retries: int = 0,
) -> Any:
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed.")
    api_key: str = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Env var {api_key_env} is required for {provider_name} calls.")
    normalized_base_url: str = str(base_url or "").strip()
    cache_key: Tuple[str, str, float, int] = (
        provider_name,
        normalized_base_url,
        float(timeout_sec),
        int(max_retries),
    )
    if cache_key in _OPENAI_CLIENTS:
        return _OPENAI_CLIENTS[cache_key]
    client_kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_sec,
        "max_retries": max_retries,
    }
    if normalized_base_url:
        client_kwargs["base_url"] = normalized_base_url
    client: Any = OpenAI(**client_kwargs)
    _OPENAI_CLIENTS[cache_key] = client
    return client


_PROBE_PROMPT_TEXT: str = "Reply with OK."
_PROBE_MAX_OUTPUT_TOKENS: int = 16


def probe_openai_model_access(
    *,
    provider_name: str,
    model_name: str,
    timeout_sec: float,
    reasoning_effort: str,
    api_key_env: str = "GPT_API_KEY",
    base_url: Optional[str] = None,
) -> None:
    """Send one minimal real request through the Responses API.

    A successful HTTP response (even an incomplete one) proves that this project
    can actually use the model with the configured reasoning effort. Fatal
    model/config errors (401/403/404/400) are raised as LlmModelConfigurationError;
    anything else (timeout, 429, 5xx) is re-raised as-is for the caller to treat
    as inconclusive.
    """
    client: Any = get_openai_client(
        provider_name=provider_name,
        api_key_env=api_key_env,
        timeout_sec=timeout_sec,
        base_url=base_url,
        max_retries=0,
    ).with_options(timeout=timeout_sec, max_retries=0)
    compatibility: OpenAIRequestCompatibility = resolve_openai_request_compatibility(
        model_name=model_name,
        structured_output_requested=False,
        temperature_requested=False,
    )
    request_kwargs: Dict[str, Any] = _build_openai_responses_request_kwargs(
        prompt_text=_PROBE_PROMPT_TEXT,
        model_name=model_name,
        max_tokens=_PROBE_MAX_OUTPUT_TOKENS,
        structured_schema=None,
        temperature=0.0,
        compatibility=compatibility,
        reasoning_effort=reasoning_effort,
    )
    try:
        response: Any = client.responses.create(**request_kwargs)
    except Exception as error:
        _raise_if_fatal_model_configuration_error(
            provider_name=provider_name,
            model_name=model_name,
            error=cast(Exception, error),
        )
        raise
    LOGGER.info(
        "openai_model_probe_passed model=%s reasoning_effort=%s probe=responses.create %s",
        model_name,
        reasoning_effort,
        _usage_log_fields(extract_openai_request_usage(response)),
    )


def _extract_structured_payload_or_none(response: Any) -> Optional[Dict[str, Any]]:
    raw_json_text: str = strip_json_code_fences(_extract_openai_response_text(response))
    if not raw_json_text:
        return None
    candidates: List[str] = [raw_json_text]
    candidates.extend(extract_json_object_candidates(raw_json_text))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed: Any = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return cast(Dict[str, Any], parsed)
    tolerant_payload, _ = parse_json_tolerant(raw_json_text)
    if tolerant_payload is not None:
        return cast(Dict[str, Any], tolerant_payload)
    return None


@dataclass(frozen=True)
class OpenAIResponsesRequestSpec:
    """Immutable parameters of one merge request; identical across all fallback attempts."""

    provider_name: str
    model_name: str
    prompt_text: str
    structured_schema: Optional[Dict[str, Any]]
    temperature: float
    reasoning_effort: str
    max_output_tokens: int
    attempt_label: str
    request_kind: str


class OpenAIResponsesTransport:
    """One merge request through the Responses API with an explicit fallback chain.

    Each layer wraps the next one, outermost first::

        send                          max_output_tokens exhausted -> retry once with 2x
        _send_with_flex_fallback      429 on flex -> wait, retry, finally drop to default
        _send_with_temperature_fallback   temperature unsupported -> retry without it
        _create_response              a single API call

    ``_active_service_tier`` is the only mutable state: the flex layer switches it to
    the default tier for good, so a later ``send`` retry no longer asks for flex.
    """

    def __init__(
        self,
        *,
        client: Any,
        spec: OpenAIResponsesRequestSpec,
        compatibility: OpenAIRequestCompatibility,
        service_tier: str = SERVICE_TIER_DEFAULT,
        trace_context: Optional[LlmTraceContext] = None,
    ) -> None:
        self._client: Any = client
        self._spec: OpenAIResponsesRequestSpec = spec
        self._compatibility: OpenAIRequestCompatibility = compatibility
        self._trace_context: Optional[LlmTraceContext] = trace_context
        self._active_service_tier: str = (
            str(service_tier or SERVICE_TIER_DEFAULT).strip().lower() or SERVICE_TIER_DEFAULT
        )

    def send(self) -> OpenAITransportResult:
        used_max_tokens: int = int(self._spec.max_output_tokens)
        response: Any = self._send_with_flex_fallback(used_max_tokens)
        incomplete_reason: str = _openai_incomplete_reason(response)
        if incomplete_reason == "max_output_tokens":
            used_max_tokens = used_max_tokens * 2
            LOGGER.warning(
                "OpenAI response hit max_output_tokens attempt=%s model=%s retrying_once_with_max_output_tokens=%d",
                self._spec.attempt_label,
                self._spec.model_name,
                used_max_tokens,
                extra={"warning_category": "informational"},
            )
            _log_llm_retry_decision(
                trace_context=self._trace_context,
                reason_code="openai_max_output_retry",
                retry_index=1,
                retry_kind="max_output_retry",
                recovered=True,
            )
            response = self._send_with_flex_fallback(used_max_tokens)
            incomplete_reason = _openai_incomplete_reason(response)

        raw_text: str = _extract_openai_response_text(response)
        if not raw_text.strip():
            raise RuntimeError(f"{self._spec.provider_name} merge returned empty output text")
        return OpenAITransportResult(
            raw_text=raw_text,
            structured_payload=(
                _extract_structured_payload_or_none(response)
                if self._spec.structured_schema is not None
                else None
            ),
            incomplete_reason=incomplete_reason,
            output_item_types=_openai_response_output_item_types(response),
        )

    def _send_with_flex_fallback(self, max_tokens: int) -> Any:
        """Flex tier can answer 429 resource_unavailable: wait and retry, then use the default tier."""
        if self._active_service_tier != SERVICE_TIER_FLEX:
            return self._send_with_temperature_fallback(max_tokens)
        for delay_sec in FLEX_RETRY_DELAYS_SEC:
            try:
                return self._send_with_temperature_fallback(max_tokens)
            except Exception as error:
                if classify_openai_request_error(cast(Exception, error)).reason_code != "openai_rate_limit":
                    raise
                LOGGER.warning(
                    "openai_flex_unavailable model=%s retry_in_sec=%.0f detail=%s",
                    self._spec.model_name,
                    delay_sec,
                    str(error)[:200],
                    extra={"warning_category": "informational"},
                )
                time.sleep(delay_sec)
        try:
            return self._send_with_temperature_fallback(max_tokens)
        except Exception as error:
            if classify_openai_request_error(cast(Exception, error)).reason_code != "openai_rate_limit":
                raise
            LOGGER.warning(
                "openai_flex_fallback_to_default model=%s detail=%s",
                self._spec.model_name,
                str(error)[:200],
                extra={"warning_category": "informational"},
            )
            self._active_service_tier = SERVICE_TIER_DEFAULT
            return self._send_with_temperature_fallback(max_tokens)

    def _send_with_temperature_fallback(self, max_tokens: int) -> Any:
        try:
            return self._create_response(max_tokens, self._compatibility)
        except Exception as error:
            if self._compatibility.temperature_enabled and _is_openai_temperature_unsupported_error(cast(Exception, error)):
                compatibility_fallback: OpenAIRequestCompatibility = replace(
                    self._compatibility,
                    temperature_enabled=False,
                )
                LOGGER.warning(
                    "OpenAI model=%s does not support temperature; retrying_without_temperature.",
                    self._spec.model_name,
                )
                _log_llm_retry_decision(
                    trace_context=self._trace_context,
                    reason_code="temperature_unsupported_retry",
                    retry_index=1,
                    retry_kind="retry_without_temperature",
                    recovered=True,
                )
                return self._create_response(max_tokens, compatibility_fallback)
            raise

    def _create_response(
        self,
        max_tokens: int,
        compatibility: OpenAIRequestCompatibility,
    ) -> Any:
        spec: OpenAIResponsesRequestSpec = self._spec
        request_kwargs: Dict[str, Any] = _build_openai_responses_request_kwargs(
            prompt_text=spec.prompt_text,
            model_name=spec.model_name,
            max_tokens=max_tokens,
            structured_schema=spec.structured_schema,
            temperature=spec.temperature,
            compatibility=compatibility,
            reasoning_effort=spec.reasoning_effort,
            service_tier=self._active_service_tier,
        )
        _log_openai_request_compatibility(
            trace_context=self._trace_context,
            compatibility=compatibility,
        )
        _log_llm_request_start(
            trace_context=self._trace_context,
            input_chars=len(spec.prompt_text),
        )
        request_started_at: float = time.perf_counter()
        try:
            raw_response: Any = self._client.responses.with_raw_response.create(**request_kwargs)
            _log_openai_rate_limit_snapshot(response_or_raw=raw_response, model_name=spec.model_name, request_kind="responses.create")
            parse_method: Any = getattr(raw_response, "parse", None)
            parsed_response: Any = parse_method() if callable(parse_method) else raw_response
            _record_run_local_openai_request(
                model_name=spec.model_name,
                response=parsed_response,
                request_kind=spec.request_kind,
            )
            _log_llm_request_finish(
                trace_context=self._trace_context,
                response=parsed_response,
                success=True,
                elapsed_ms=int(round((time.perf_counter() - request_started_at) * 1000.0)),
                max_output_hit=(_openai_incomplete_reason(parsed_response) == "max_output_tokens"),
            )
            return parsed_response
        except AttributeError:
            response: Any = self._client.responses.create(**request_kwargs)
            _log_openai_rate_limit_snapshot(response_or_raw=response, model_name=spec.model_name, request_kind="responses.create")
            _record_run_local_openai_request(model_name=spec.model_name, response=response, request_kind=spec.request_kind)
            _log_llm_request_finish(
                trace_context=self._trace_context,
                response=response,
                success=True,
                elapsed_ms=int(round((time.perf_counter() - request_started_at) * 1000.0)),
                max_output_hit=(_openai_incomplete_reason(response) == "max_output_tokens"),
            )
            return response
        except Exception as error:
            _log_llm_request_failed(
                trace_context=self._trace_context,
                error=cast(Exception, error),
                elapsed_ms=int(round((time.perf_counter() - request_started_at) * 1000.0)),
            )
            if _is_openai_temperature_unsupported_error(cast(Exception, error)):
                raise
            classification: LlmRequestErrorClassification = classify_openai_request_error(
                cast(Exception, error)
            )
            if classification.fatal_model_configuration:
                LOGGER.error(
                    "openai_model_configuration_error provider=%s model=%s reason_code=%s status_code=%s api_error_code=%s api_error_param=%s detail=%s",
                    spec.provider_name,
                    spec.model_name or "unknown",
                    classification.reason_code,
                    str(classification.status_code if classification.status_code is not None else "unknown"),
                    classification.api_error_code or "none",
                    classification.api_error_param or "none",
                    classification.detail,
                    extra={"reason_code": classification.reason_code},
                )
            _raise_if_fatal_model_configuration_error(
                provider_name=spec.provider_name,
                model_name=spec.model_name,
                error=cast(Exception, error),
            )
            raise


def openai_compatible_request_merge(
    *,
    provider_name: str,
    api_key_env: str,
    base_url: Optional[str],
    prompt_text: str,
    model_name: str,
    timeout_sec: float,
    max_retries: int,
    attempt_label: str,
    max_output_tokens: int,
    reasoning_effort: str,
    service_tier: str = SERVICE_TIER_DEFAULT,
    structured_schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    trace_context: Optional[LlmTraceContext] = None,
) -> OpenAITransportResult:
    client: Any = get_openai_client(
        provider_name=provider_name,
        api_key_env=api_key_env,
        timeout_sec=timeout_sec,
        base_url=base_url,
        max_retries=max_retries,
    ).with_options(timeout=timeout_sec, max_retries=max_retries)
    spec: OpenAIResponsesRequestSpec = OpenAIResponsesRequestSpec(
        provider_name=provider_name,
        model_name=model_name,
        prompt_text=prompt_text,
        structured_schema=structured_schema,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        attempt_label=attempt_label,
        request_kind=(
            trace_context.request_kind
            if trace_context is not None
            else ("structured" if structured_schema is not None else "plain")
        ),
    )
    return OpenAIResponsesTransport(
        client=client,
        spec=spec,
        compatibility=resolve_openai_request_compatibility(
            model_name=model_name,
            structured_output_requested=(structured_schema is not None),
            temperature_requested=True,
        ),
        service_tier=service_tier,
        trace_context=trace_context,
    ).send()


def openai_request_merge(
    *,
    prompt_text: str,
    model_name: str,
    timeout_sec: float,
    attempt_label: str,
    max_output_tokens: int,
    reasoning_effort: str,
    service_tier: str = "default",
    structured_schema: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    trace_context: Optional[LlmTraceContext] = None,
) -> OpenAITransportResult:
    return openai_compatible_request_merge(
        provider_name="openai",
        api_key_env="GPT_API_KEY",
        base_url=None,
        prompt_text=prompt_text,
        model_name=model_name,
        timeout_sec=timeout_sec,
        max_retries=0,
        attempt_label=attempt_label,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        service_tier=service_tier,
        structured_schema=structured_schema,
        temperature=temperature,
        trace_context=trace_context,
    )
