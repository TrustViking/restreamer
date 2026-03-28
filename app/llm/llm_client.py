from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.llm.llm_rate_limits import (
    _extract_openai_headers_from_raw_response,
    _log_openai_rate_limit_snapshot,
    _safe_int_or_none,
)
from app.llm.models.model_compatibility import (
    LlmModelConfigurationError,
    LlmRequestErrorClassification,
    OpenAIRequestCompatibility,
    classify_openai_request_error,
    resolve_openai_request_compatibility,
)
from app.llm.merges.merge_parser import extract_json_object_candidates, parse_json_tolerant, strip_json_code_fences

# Backward-compatible re-exports — DO NOT REMOVE
from app.llm.llm_usage_tracker import (  # noqa: F401
    RunLocalOpenAIUsageState,
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


def _extract_openai_usage_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    usage: Any = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return (None, None)
    if isinstance(usage, dict):
        input_tokens: Optional[int] = _safe_int_or_none(
            usage.get("input_tokens") or usage.get("prompt_tokens")
        )
        output_tokens: Optional[int] = _safe_int_or_none(
            usage.get("output_tokens") or usage.get("completion_tokens")
        )
        return (input_tokens, output_tokens)
    input_tokens = _safe_int_or_none(
        getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    )
    output_tokens = _safe_int_or_none(
        getattr(usage, "output_tokens", None)
        or getattr(usage, "completion_tokens", None)
    )
    return (input_tokens, output_tokens)


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
    input_tokens, output_tokens = _extract_openai_usage_tokens(response)
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
        "llm_request_finish branch=%s date_key=%s slot_key=%s lang=%s provider=%s model=%s attempt_index=%d request_kind=%s success=%s llm_call_ms=%d output_chars=%d input_tokens=%s output_tokens=%s finish_reason=%s max_output_hit=%s",
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
        str(input_tokens if input_tokens is not None else "unknown"),
        str(output_tokens if output_tokens is not None else "unknown"),
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
    else:
        state.fallback_calls += 1
    input_tokens, output_tokens = _extract_openai_usage_tokens(response)
    if input_tokens is not None:
        state.input_tokens = (state.input_tokens or 0) + input_tokens
    if output_tokens is not None:
        state.output_tokens = (state.output_tokens or 0) + output_tokens


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
) -> Dict[str, Any]:
    request_kwargs: Dict[str, Any] = {
        "model": model_name,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
        "max_output_tokens": max_tokens,
    }
    if compatibility.reasoning_effort_enabled:
        request_kwargs["reasoning"] = {"effort": "low"}
    if structured_schema is not None:
        request_kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": structured_schema.get("name", "streamertg_merge_v1"),
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


def probe_openai_model_access(
    *,
    provider_name: str,
    model_name: str,
    timeout_sec: float,
    api_key_env: str = "GPT_API_KEY",
    base_url: Optional[str] = None,
) -> None:
    client: Any = get_openai_client(
        provider_name=provider_name,
        api_key_env=api_key_env,
        timeout_sec=timeout_sec,
        base_url=base_url,
        max_retries=0,
    ).with_options(timeout=timeout_sec, max_retries=0)
    try:
        client.models.retrieve(model_name)
    except Exception as error:
        _raise_if_fatal_model_configuration_error(
            provider_name=provider_name,
            model_name=model_name,
            error=cast(Exception, error),
        )
        raise


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
    request_kind: str = (
        trace_context.request_kind
        if trace_context is not None
        else ("structured" if structured_schema is not None else "plain")
    )
    compatibility: OpenAIRequestCompatibility = resolve_openai_request_compatibility(
        model_name=model_name,
        structured_output_requested=(structured_schema is not None),
        temperature_requested=True,
    )

    def _create_response(
        max_tokens: int,
        *,
        compatibility_override: Optional[OpenAIRequestCompatibility] = None,
    ) -> Any:
        active_compatibility: OpenAIRequestCompatibility = (
            compatibility_override if compatibility_override is not None else compatibility
        )
        request_kwargs: Dict[str, Any] = _build_openai_responses_request_kwargs(
            prompt_text=prompt_text,
            model_name=model_name,
            max_tokens=max_tokens,
            structured_schema=structured_schema,
            temperature=temperature,
            compatibility=active_compatibility,
        )
        _log_openai_request_compatibility(
            trace_context=trace_context,
            compatibility=active_compatibility,
        )
        _log_llm_request_start(
            trace_context=trace_context,
            input_chars=len(prompt_text),
        )
        request_started_at: float = time.perf_counter()
        try:
            raw_response: Any = client.responses.with_raw_response.create(**request_kwargs)
            _log_openai_rate_limit_snapshot(response_or_raw=raw_response, model_name=model_name, request_kind="responses.create")
            parse_method: Any = getattr(raw_response, "parse", None)
            parsed_response: Any = parse_method() if callable(parse_method) else raw_response
            _record_run_local_openai_request(
                model_name=model_name,
                response=parsed_response,
                request_kind=request_kind,
            )
            _log_llm_request_finish(
                trace_context=trace_context,
                response=parsed_response,
                success=True,
                elapsed_ms=int(round((time.perf_counter() - request_started_at) * 1000.0)),
                max_output_hit=(_openai_incomplete_reason(parsed_response) == "max_output_tokens"),
            )
            return parsed_response
        except AttributeError:
            response: Any = client.responses.create(**request_kwargs)
            _log_openai_rate_limit_snapshot(response_or_raw=response, model_name=model_name, request_kind="responses.create")
            _record_run_local_openai_request(model_name=model_name, response=response, request_kind=request_kind)
            _log_llm_request_finish(
                trace_context=trace_context,
                response=response,
                success=True,
                elapsed_ms=int(round((time.perf_counter() - request_started_at) * 1000.0)),
                max_output_hit=(_openai_incomplete_reason(response) == "max_output_tokens"),
            )
            return response
        except Exception as error:
            _log_llm_request_failed(
                trace_context=trace_context,
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
                    provider_name,
                    model_name or "unknown",
                    classification.reason_code,
                    str(classification.status_code if classification.status_code is not None else "unknown"),
                    classification.api_error_code or "none",
                    classification.api_error_param or "none",
                    classification.detail,
                    extra={"reason_code": classification.reason_code},
                )
            _raise_if_fatal_model_configuration_error(
                provider_name=provider_name,
                model_name=model_name,
                error=cast(Exception, error),
            )
            raise

    def _create_with_temp_fallback(max_tokens: int) -> Any:
        try:
            return _create_response(max_tokens)
        except Exception as error:
            if compatibility.temperature_enabled and _is_openai_temperature_unsupported_error(cast(Exception, error)):
                compatibility_fallback: OpenAIRequestCompatibility = OpenAIRequestCompatibility(
                    model_name=compatibility.model_name,
                    normalized_model_name=compatibility.normalized_model_name,
                    model_family=compatibility.model_family,
                    reasoning_effort_supported=compatibility.reasoning_effort_supported,
                    reasoning_effort_enabled=compatibility.reasoning_effort_enabled,
                    structured_output_supported=compatibility.structured_output_supported,
                    structured_output_requested=compatibility.structured_output_requested,
                    temperature_supported=compatibility.temperature_supported,
                    temperature_enabled=False,
                    capability_source=compatibility.capability_source,
                )
                LOGGER.warning("OpenAI model=%s does not support temperature; retrying_without_temperature.", model_name)
                _log_llm_retry_decision(
                    trace_context=trace_context,
                    reason_code="temperature_unsupported_retry",
                    retry_index=1,
                    retry_kind="retry_without_temperature",
                    recovered=True,
                )
                return _create_response(
                    max_tokens,
                    compatibility_override=compatibility_fallback,
                )
            raise

    used_max_tokens: int = int(max_output_tokens)
    response: Any = _create_with_temp_fallback(used_max_tokens)
    incomplete_reason: str = _openai_incomplete_reason(response)
    if incomplete_reason == "max_output_tokens":
        used_max_tokens = min(3000, max(1, used_max_tokens * 2))
        LOGGER.warning(
            "OpenAI response hit max_output_tokens attempt=%s model=%s retrying_once_with_max_output_tokens=%d",
            attempt_label,
            model_name,
            used_max_tokens,
        )
        _log_llm_retry_decision(
            trace_context=trace_context,
            reason_code="openai_max_output_retry",
            retry_index=1,
            retry_kind="max_output_retry",
            recovered=True,
        )
        response = _create_with_temp_fallback(used_max_tokens)
        incomplete_reason = _openai_incomplete_reason(response)

    raw_text: str = _extract_openai_response_text(response)
    if not raw_text.strip():
        raise RuntimeError(f"{provider_name} merge returned empty output text")
    return OpenAITransportResult(
        raw_text=raw_text,
        structured_payload=_extract_structured_payload_or_none(response) if structured_schema is not None else None,
        incomplete_reason=incomplete_reason,
        output_item_types=_openai_response_output_item_types(response),
    )


def openai_request_merge(
    *,
    prompt_text: str,
    model_name: str,
    timeout_sec: float,
    attempt_label: str,
    max_output_tokens: int,
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
        structured_schema=structured_schema,
        temperature=temperature,
        trace_context=trace_context,
    )
