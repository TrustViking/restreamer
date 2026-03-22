from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple, cast
from zoneinfo import ZoneInfo

import requests

from app.llm.llm_client import get_run_local_openai_usage
from app.observability.runtime_analytics import log_warning_informational


def _unix_seconds_utc(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(value.timestamp())


def _sum_costs_usd(costs_payload: Dict[str, Any]) -> float:
    total_usd: float = 0.0
    buckets: Any = costs_payload.get("data")
    if not isinstance(buckets, list):
        return 0.0
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        results: Any = bucket.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            amount_obj: Any = result.get("amount")
            if not isinstance(amount_obj, dict):
                continue
            currency: str = str(amount_obj.get("currency", "")).strip().lower()
            value_raw: Any = amount_obj.get("value")
            if currency != "usd" or not isinstance(value_raw, (int, float)):
                continue
            total_usd += float(value_raw)
    return total_usd


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


def _timezone_from_name_or_utc(tz_name: str) -> timezone | ZoneInfo:
    cleaned_tz_name: str = str(tz_name or "").strip() or "UTC"
    try:
        return ZoneInfo(cleaned_tz_name)
    except Exception:
        return timezone.utc


def _sum_usage_payload(
    usage_payload: Dict[str, Any],
) -> Tuple[int, int, int, Dict[str, Dict[str, int]]]:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    per_model: Dict[str, Dict[str, int]] = {}
    data_any: Any = usage_payload.get("data")
    if not isinstance(data_any, list):
        return (0, 0, 0, {})
    for bucket in data_any:
        if not isinstance(bucket, dict):
            continue
        results_any: Any = bucket.get("results")
        if not isinstance(results_any, list):
            continue
        for result_item in results_any:
            if not isinstance(result_item, dict):
                continue
            model_name: str = str(result_item.get("model", "")).strip() or "unknown"
            input_tokens: int = _safe_int_or_none(result_item.get("input_tokens")) or 0
            output_tokens: int = (
                _safe_int_or_none(result_item.get("output_tokens")) or 0
            )
            request_count: int = (
                _safe_int_or_none(result_item.get("num_model_requests")) or 0
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_requests += request_count
            model_entry: Dict[str, int] = per_model.setdefault(
                model_name,
                {"input": 0, "output": 0, "requests": 0, "tokens": 0},
            )
            model_entry["input"] += input_tokens
            model_entry["output"] += output_tokens
            model_entry["requests"] += request_count
            model_entry["tokens"] += input_tokens + output_tokens
    return (total_input_tokens, total_output_tokens, total_requests, per_model)


def _format_top_models_usage(per_model: Dict[str, Dict[str, int]]) -> str:
    ranked_items = sorted(
        per_model.items(),
        key=lambda item: int(item[1].get("tokens", 0)),
        reverse=True,
    )
    if not ranked_items:
        return "none"
    shown_items = ranked_items[:3]
    hidden_items = ranked_items[3:]
    parts = [
        f"{model_name}:{int(values.get('input', 0))}/{int(values.get('output', 0))}/{int(values.get('requests', 0))}"
        for model_name, values in shown_items
    ]
    if hidden_items:
        other_input: int = sum(
            int(values.get("input", 0)) for _, values in hidden_items
        )
        other_output: int = sum(
            int(values.get("output", 0)) for _, values in hidden_items
        )
        other_requests: int = sum(
            int(values.get("requests", 0)) for _, values in hidden_items
        )
        parts.append(f"other:{other_input}/{other_output}/{other_requests}")
    return ", ".join(parts)


def fetch_usage_and_costs_summary(
    admin_key: str,
    tz_name: str,
    now_utc: datetime,
    *,
    summarize_error: Callable[[Exception], str],
) -> Dict[str, Any]:
    cleaned_admin_key: str = str(admin_key or "").strip()
    if not cleaned_admin_key:
        return {"error": "OPENAI_ADMIN_KEY is empty"}
    if now_utc.tzinfo is None:
        return {"error": "now_utc must be timezone-aware"}

    tz_value: timezone | ZoneInfo = _timezone_from_name_or_utc(tz_name)
    now_local: datetime = now_utc.astimezone(tz_value)
    today_start_local: datetime = datetime(
        now_local.year,
        now_local.month,
        now_local.day,
        tzinfo=tz_value,
    )
    month_start_local: datetime = datetime(
        now_local.year,
        now_local.month,
        1,
        tzinfo=tz_value,
    )
    today_start_utc: datetime = today_start_local.astimezone(timezone.utc)
    month_start_utc: datetime = month_start_local.astimezone(timezone.utc)

    project_id: str = str(os.getenv("OPENAI_PROJECT_ID", "") or "").strip()
    timeout_raw: str = str(os.getenv("OPENAI_USAGE_TIMEOUT_SEC", "30.0") or "30.0")
    try:
        timeout_sec: float = max(5.0, float(timeout_raw))
    except Exception:
        timeout_sec = 30.0

    session: requests.Session = requests.Session()
    common_headers: Dict[str, str] = {
        "Authorization": f"Bearer {cleaned_admin_key}",
        "Content-Type": "application/json",
    }
    usage_params: Dict[str, Any] = {
        "start_time": _unix_seconds_utc(today_start_utc),
        "end_time": _unix_seconds_utc(now_utc),
        "bucket_width": "1d",
        "group_by": ["model"],
    }
    costs_params: Dict[str, Any] = {
        "start_time": _unix_seconds_utc(month_start_utc),
        "end_time": _unix_seconds_utc(now_utc),
        "bucket_width": "1d",
    }
    if project_id:
        usage_params["project_ids"] = [project_id]
        costs_params["project_ids"] = [project_id]

    try:
        usage_response: requests.Response = session.get(
            url="https://api.openai.com/v1/organization/usage/completions",
            params=usage_params,
            headers=common_headers,
            timeout=timeout_sec,
        )
        usage_response.raise_for_status()
        usage_payload_any: Any = usage_response.json()
        usage_payload: Dict[str, Any] = (
            cast(Dict[str, Any], usage_payload_any)
            if isinstance(usage_payload_any, dict)
            else {}
        )
        costs_response: requests.Response = session.get(
            url="https://api.openai.com/v1/organization/costs",
            params=costs_params,
            headers=common_headers,
            timeout=timeout_sec,
        )
        costs_response.raise_for_status()
        costs_payload_any: Any = costs_response.json()
        costs_payload: Dict[str, Any] = (
            cast(Dict[str, Any], costs_payload_any)
            if isinstance(costs_payload_any, dict)
            else {}
        )
    except Exception as error:
        return {"error": summarize_error(cast(Exception, error))}

    (
        total_input_tokens,
        total_output_tokens,
        total_requests,
        per_model,
    ) = _sum_usage_payload(usage_payload)
    spent_usd_month: float = _sum_costs_usd(costs_payload)
    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_requests": total_requests,
        "per_model": per_model,
        "spent_usd_month": spent_usd_month,
    }


def log_openai_limits_and_usage(
    logger: logging.Logger,
    *,
    summarize_error: Callable[[Exception], str],
    effective_model: str,
) -> None:
    admin_api_key: str = str(os.getenv("OPENAI_ADMIN_KEY", "") or "").strip()
    if not admin_api_key:
        logger.info("OpenAI summary skipped: OPENAI_ADMIN_KEY is not set.")
        return
    tz_name: str = str(os.getenv("STG_TZ", "UTC") or "UTC").strip() or "UTC"
    summary_payload: Dict[str, Any] = fetch_usage_and_costs_summary(
        admin_key=admin_api_key,
        tz_name=tz_name,
        now_utc=datetime.now(timezone.utc),
        summarize_error=summarize_error,
    )
    error_text: str = str(summary_payload.get("error", "")).strip()
    if error_text:
        log_warning_informational(
            logger,
            "OpenAI summary skipped: %s",
            error_text,
            reason_code="openai_org_usage_unavailable",
        )
        return
    total_input_tokens: int = int(summary_payload.get("total_input_tokens") or 0)
    total_output_tokens: int = int(summary_payload.get("total_output_tokens") or 0)
    total_requests: int = int(summary_payload.get("total_requests") or 0)
    per_model: Dict[str, Dict[str, int]] = cast(
        Dict[str, Dict[str, int]],
        summary_payload.get("per_model") or {},
    )
    logger.info(
        "OPENAI ORG USAGE SNAPSHOT scope=organization_aggregate source_of_truth_for_run=no current_run_effective_model=%s today_input=%d today_output=%d today_req=%d models=%s",
        str(effective_model or "").strip() or "unknown",
        total_input_tokens,
        total_output_tokens,
        total_requests,
        _format_top_models_usage(per_model),
    )
    spent_usd_month: float = float(summary_payload.get("spent_usd_month") or 0.0)
    logger.info(
        "OPENAI ORG COST SNAPSHOT scope=organization_aggregate source_of_truth_for_run=no current_run_effective_model=%s month_spent_usd=%.6f",
        str(effective_model or "").strip() or "unknown",
        spent_usd_month,
    )


def log_run_local_openai_usage(logger: logging.Logger, *, effective_model: str) -> None:
    usage_state = get_run_local_openai_usage()
    model_names: str = ",".join(sorted(usage_state.models_used)) if usage_state.models_used else "none"
    parts: list[str] = [
        "OPENAI RUN USAGE",
        "scope=run_local",
        "source_of_truth_for_run=yes",
        f"effective_model={str(effective_model or '').strip() or 'unknown'}",
        f"requests_sent={usage_state.requests_sent}",
        f"repair_calls={usage_state.repair_calls}",
        f"structured_calls={usage_state.structured_calls}",
        f"fallback_calls={usage_state.fallback_calls}",
        f"models_used={model_names}",
    ]
    tokens_known: bool = (
        usage_state.input_tokens is not None and usage_state.output_tokens is not None
    )
    if usage_state.input_tokens is not None:
        parts.append(f"input_tokens={usage_state.input_tokens}")
    if usage_state.output_tokens is not None:
        parts.append(f"output_tokens={usage_state.output_tokens}")
    parts.append(f"tokens_known={'yes' if tokens_known else 'no'}")
    logger.info("%s", " ".join(parts))
