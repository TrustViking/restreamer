from __future__ import annotations

import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl

LOGGER = _get_logger_impl(__name__)


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
