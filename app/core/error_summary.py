from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, cast


def extract_http_error_reason(error: Exception) -> Optional[str]:
    response_obj: Any = getattr(error, "resp", None)
    status_code: Optional[int] = getattr(response_obj, "status", None)
    raw_content: Any = getattr(error, "content", None)
    if raw_content is None:
        return None
    try:
        content_text: str = (
            raw_content.decode("utf-8", errors="replace")
            if isinstance(raw_content, (bytes, bytearray))
            else str(raw_content)
        )
        payload: Any = json.loads(content_text)
        error_payload: Dict[str, Any] = cast(Dict[str, Any], payload.get("error", {}))
        code: Any = error_payload.get("code", status_code or "")
        status_text: str = str(error_payload.get("status", "")).strip() or "UNKNOWN"
        message: str = str(error_payload.get("message", "")).strip()
        details: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], error_payload.get("details", [])
        )
        reason: str = ""
        service: str = ""
        activation_url: str = ""
        quota_metric: str = ""
        retry_delay: str = ""
        for detail in details:
            detail_type: str = str(detail.get("@type", ""))
            if detail_type.endswith("ErrorInfo"):
                reason = str(detail.get("reason", "")).strip()
                metadata: Dict[str, Any] = cast(
                    Dict[str, Any], detail.get("metadata", {})
                )
                service = str(metadata.get("service", "")).strip()
                activation_url = str(metadata.get("activationUrl", "")).strip()
            elif detail_type.endswith("QuotaFailure"):
                violations: List[Dict[str, Any]] = cast(
                    List[Dict[str, Any]], detail.get("violations", [])
                )
                if violations:
                    quota_metric = str(violations[0].get("quotaMetric", "")).strip()
            elif detail_type.endswith("RetryInfo"):
                retry_delay = str(detail.get("retryDelay", "")).strip()

        parts: List[str] = [f"http={code}", f"status={status_text}"]
        if reason:
            parts.append(f"reason={reason}")
        if service:
            parts.append(f"service={service}")
        if quota_metric:
            parts.append(f"quota_metric={quota_metric}")
        if retry_delay:
            parts.append(f"retry_delay={retry_delay}")
        if activation_url:
            parts.append(f"enable_url={activation_url}")
        if message:
            parts.append(f"message={message[:120]}")
        return " ".join(parts)
    except Exception:
        return None


def extract_telegram_reason(text: str) -> Optional[str]:
    compact: str = re.sub(r"\s+", " ", text).strip()
    if "Telegram API HTTP" not in compact:
        return None
    http_match: Optional[re.Match[str]] = re.search(
        r"Telegram API HTTP\s+(\d+)", compact
    )
    description_match: Optional[re.Match[str]] = re.search(
        r"\"description\"\s*:\s*\"([^\"]+)\"",
        compact,
    )
    parts: List[str] = []
    if http_match:
        parts.append(f"http={http_match.group(1)}")
    if description_match:
        parts.append(f"description={description_match.group(1)}")
    if parts:
        return " ".join(parts)
    return None


def summarize_error(error: Exception, max_len: int = 220) -> str:
    http_reason: Optional[str] = extract_http_error_reason(error)
    if http_reason:
        return http_reason
    raw: str = str(error).replace("\r", " ").replace("\n", " ").strip()
    telegram_reason: Optional[str] = extract_telegram_reason(raw)
    if telegram_reason:
        return telegram_reason
    compact: str = re.sub(r"\s+", " ", raw)
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len].rstrip()}..."
