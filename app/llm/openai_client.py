from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.llm.merge_parser import extract_json_object_candidates, parse_json_tolerant, strip_json_code_fences

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

LOGGER = _get_logger_impl(__name__)
_OPENAI_CLIENT: Optional[Any] = None


@dataclass(frozen=True)
class OpenAITransportResult:
    raw_text: str
    structured_payload: Optional[Dict[str, Any]]
    incomplete_reason: str
    output_item_types: List[str]


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


def _is_openai_temperature_unsupported_error(error: Exception) -> bool:
    message: str = str(error or "").lower()
    return "unsupported parameter" in message and "temperature" in message and "invalid_request_error" in message


def _openai_should_send_temperature(model_name: str) -> bool:
    normalized_model_name: str = str(model_name or "").strip().lower()
    return not normalized_model_name.startswith("gpt")


def _safe_int_or_none(raw_value: Any) -> Optional[int]:
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    text_value: str = str(raw_value or "").strip().replace(",", "")
    if not text_value or not re.fullmatch(r"-?\d+", text_value):
        return None
    try:
        return int(text_value)
    except Exception:
        return None


def _parse_reset_value_to_seconds(raw_value: str) -> Optional[float]:
    text_value: str = str(raw_value or "").strip().lower()
    if not text_value:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text_value):
        numeric_value: float = float(text_value)
        if numeric_value > 1_000_000_000:
            return max(0.0, numeric_value - time.time())
        return max(0.0, numeric_value)
    total_seconds: float = 0.0
    matched_any: bool = False
    for match in re.finditer(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text_value):
        matched_any = True
        numeric_part: float = float(match.group(1))
        unit_part: str = match.group(2)
        if unit_part == "ms":
            total_seconds += numeric_part / 1000.0
        elif unit_part == "s":
            total_seconds += numeric_part
        elif unit_part == "m":
            total_seconds += numeric_part * 60.0
        elif unit_part == "h":
            total_seconds += numeric_part * 3600.0
    return max(0.0, total_seconds) if matched_any else None


def extract_rate_limit_snapshot(headers: Mapping[str, str]) -> Dict[str, Any]:
    rate_limit_headers: Dict[str, str] = {}
    for header_name, header_value in headers.items():
        name_normalized: str = str(header_name or "").strip().lower()
        if name_normalized.startswith("x-ratelimit-"):
            rate_limit_headers[name_normalized] = str(header_value or "").strip()

    remaining_requests_candidates: List[int] = []
    remaining_tokens_candidates: List[int] = []
    reset_requests_candidates: List[float] = []
    reset_tokens_candidates: List[float] = []
    for header_name, header_value in rate_limit_headers.items():
        if "remaining" in header_name and "request" in header_name:
            value: Optional[int] = _safe_int_or_none(header_value)
            if value is not None:
                remaining_requests_candidates.append(value)
        if "remaining" in header_name and "token" in header_name:
            value = _safe_int_or_none(header_value)
            if value is not None:
                remaining_tokens_candidates.append(value)
        if "reset" in header_name and "request" in header_name:
            value_sec: Optional[float] = _parse_reset_value_to_seconds(header_value)
            if value_sec is not None:
                reset_requests_candidates.append(value_sec)
        if "reset" in header_name and "token" in header_name:
            value_sec = _parse_reset_value_to_seconds(header_value)
            if value_sec is not None:
                reset_tokens_candidates.append(value_sec)

    snapshot: Dict[str, Any] = {}
    if remaining_requests_candidates:
        snapshot["remaining_requests"] = min(remaining_requests_candidates)
    if remaining_tokens_candidates:
        snapshot["remaining_tokens"] = min(remaining_tokens_candidates)
    if reset_requests_candidates:
        snapshot["reset_requests"] = f"{int(round(min(reset_requests_candidates)))}s"
    if reset_tokens_candidates:
        snapshot["reset_tokens"] = f"{int(round(min(reset_tokens_candidates)))}s"
    return snapshot


def _extract_openai_headers_from_raw_response(raw_response: Any) -> Dict[str, str]:
    headers_any: Any = getattr(raw_response, "headers", None)
    if headers_any is None:
        response_any: Any = getattr(raw_response, "response", None)
        headers_any = getattr(response_any, "headers", None)
    if headers_any is None:
        http_response_any: Any = getattr(raw_response, "http_response", None)
        headers_any = getattr(http_response_any, "headers", None)
    if headers_any is None:
        return {}
    if isinstance(headers_any, Mapping):
        return {str(key).strip(): str(value).strip() for key, value in headers_any.items() if str(key).strip()}
    items_attr: Any = getattr(headers_any, "items", None)
    if callable(items_attr):
        try:
            items_result: Any = items_attr()
            if not isinstance(items_result, Iterable):
                return {}
            return {str(key).strip(): str(value).strip() for key, value in cast(Iterable[Tuple[Any, Any]], items_result) if str(key).strip()}
        except Exception:
            return {}
    return {}


def _format_rate_limit_line(snapshot: Dict[str, Any], model_name: str, request_kind: str) -> str:
    parts: List[str] = ["OPENAI RL", f"model={model_name or 'unknown'}", f"kind={request_kind or 'request'}"]
    if isinstance(snapshot.get("remaining_requests"), int):
        parts.append(f"rem_req={snapshot['remaining_requests']}")
    if isinstance(snapshot.get("remaining_tokens"), int):
        parts.append(f"rem_tok={snapshot['remaining_tokens']}")
    if isinstance(snapshot.get("reset_requests"), str):
        parts.append(f"reset_req={snapshot['reset_requests']}")
    if isinstance(snapshot.get("reset_tokens"), str):
        parts.append(f"reset_tok={snapshot['reset_tokens']}")
    return " ".join(parts)


def _log_openai_rate_limit_snapshot(*, response_or_raw: Any, model_name: str, request_kind: str) -> None:
    headers: Dict[str, str] = _extract_openai_headers_from_raw_response(response_or_raw)
    if not headers:
        LOGGER.info("OPENAI RL model=%s kind=%s headers=unavailable", model_name or "unknown", request_kind or "request")
        return
    snapshot: Dict[str, Any] = extract_rate_limit_snapshot(headers)
    if not snapshot:
        LOGGER.info("OPENAI RL model=%s kind=%s", model_name or "unknown", request_kind or "request")
        return
    LOGGER.info("%s", _format_rate_limit_line(snapshot, model_name, request_kind))


def get_openai_client(*, timeout_sec: float) -> Any:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed.")
    api_key: str = os.getenv("GPT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Env var GPT_API_KEY is required for OpenAI calls.")
    _OPENAI_CLIENT = OpenAI(api_key=api_key, timeout=timeout_sec, max_retries=0)
    return _OPENAI_CLIENT


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


def openai_request_merge(*, prompt_text: str, model_name: str, timeout_sec: float, attempt_label: str, max_output_tokens: int, structured_schema: Optional[Dict[str, Any]] = None, temperature: float = 0.0) -> OpenAITransportResult:
    client: Any = get_openai_client(timeout_sec=timeout_sec).with_options(timeout=timeout_sec)
    temperature_enabled: bool = _openai_should_send_temperature(model_name)
    reasoning_effort: str = "low"

    def _create_response(max_tokens: int) -> Any:
        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_tokens,
        }
        if structured_schema is not None:
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": structured_schema.get("name", "streamertg_merge_v1"),
                    "strict": True,
                    "schema": structured_schema["schema"],
                }
            }
        if temperature_enabled:
            request_kwargs["temperature"] = float(temperature)
        try:
            raw_response: Any = client.responses.with_raw_response.create(**request_kwargs)
            _log_openai_rate_limit_snapshot(response_or_raw=raw_response, model_name=model_name, request_kind="responses.create")
            parse_method: Any = getattr(raw_response, "parse", None)
            return parse_method() if callable(parse_method) else raw_response
        except AttributeError:
            response: Any = client.responses.create(**request_kwargs)
            _log_openai_rate_limit_snapshot(response_or_raw=response, model_name=model_name, request_kind="responses.create")
            return response

    def _create_with_temp_fallback(max_tokens: int) -> Any:
        nonlocal temperature_enabled
        try:
            return _create_response(max_tokens)
        except Exception as error:
            if temperature_enabled and _is_openai_temperature_unsupported_error(cast(Exception, error)):
                temperature_enabled = False
                LOGGER.warning("OpenAI model=%s does not support temperature; retrying_without_temperature.", model_name)
                return _create_response(max_tokens)
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
        response = _create_with_temp_fallback(used_max_tokens)
        incomplete_reason = _openai_incomplete_reason(response)

    raw_text: str = _extract_openai_response_text(response)
    if not raw_text.strip():
        raise RuntimeError("openai merge returned empty output text")
    return OpenAITransportResult(
        raw_text=raw_text,
        structured_payload=_extract_structured_payload_or_none(response) if structured_schema is not None else None,
        incomplete_reason=incomplete_reason,
        output_item_types=_openai_response_output_item_types(response),
    )
