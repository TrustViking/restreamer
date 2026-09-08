"""Shared retry engine for Google API clients (Docs, Drive).

Both clients previously carried an identical copy of this loop. The engine is a
superset of the two: 429 backoff, transient 5xx translated into a client-specific
error type, immediate raise for everything else, and optional retry of
transport-level connection failures.

Log message formats are built from ``log_prefix`` and ``resource_label`` so the
emitted text stays byte-for-byte identical to the per-client versions.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = Exception  # type: ignore


@dataclass(frozen=True)
class GoogleApiRetryPolicy:
    max_retries: int
    base_delay_sec: float
    max_delay_sec: float
    transient_status_codes: frozenset[int]
    connection_errors: tuple[type[BaseException], ...]
    log_prefix: str
    resource_label: str
    transient_error_factory: Callable[[int, str, Exception], Exception]

    def compute_delay(self, attempt: int) -> float:
        """Exponential backoff: base * 2^(attempt-1), capped at max."""
        delay: float = self.base_delay_sec * (2 ** (attempt - 1))
        return min(delay, self.max_delay_sec)


def execute_with_retry(
    *,
    policy: GoogleApiRetryPolicy,
    logger: logging.Logger,
    operation_name: str,
    resource_id: str,
    request_callable: Callable[[], Any],
    request_count: Optional[int] = None,
) -> Any:
    for attempt in range(1, policy.max_retries + 1):
        try:
            return request_callable()
        except HttpError as error:
            status_code: Optional[int] = getattr(getattr(error, "resp", None), "status", None)
            if status_code == 429 and attempt < policy.max_retries:
                delay_sec: float = policy.compute_delay(attempt)
                if request_count is not None:
                    logger.warning(
                        f"{policy.log_prefix}_batch_update_429 attempt=%d/%d delay_sec=%.1f"
                        f" {policy.resource_label}=%s requests_count=%d",
                        attempt,
                        policy.max_retries,
                        delay_sec,
                        resource_id,
                        request_count,
                        extra={"warning_category": "informational"},
                    )
                else:
                    logger.warning(
                        f"{policy.log_prefix}_request_429 operation=%s attempt=%d/%d"
                        f" delay_sec=%.1f {policy.resource_label}=%s",
                        operation_name,
                        attempt,
                        policy.max_retries,
                        delay_sec,
                        resource_id,
                        extra={"warning_category": "informational"},
                    )
                # Module-attribute call so tests patching <client>.time.sleep also
                # intercept the engine (same time module object).
                time.sleep(delay_sec)
                continue
            if status_code in policy.transient_status_codes:
                raise policy.transient_error_factory(
                    int(status_code), resource_id, error
                ) from error
            raise
        except policy.connection_errors as error:
            if attempt >= policy.max_retries:
                raise
            delay_sec = policy.compute_delay(attempt)
            logger.warning(
                f"{policy.log_prefix}_request_connection_error operation=%s attempt=%d/%d"
                f" delay_sec=%.1f {policy.resource_label}=%s error=%s: %s",
                operation_name,
                attempt,
                policy.max_retries,
                delay_sec,
                resource_id,
                type(error).__name__,
                error,
                extra={"warning_category": "informational"},
            )
            time.sleep(delay_sec)
