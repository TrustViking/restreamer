from __future__ import annotations

import http.client
import time
from typing import Any, Callable, Dict, List, Optional, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = Exception  # type: ignore

LOGGER = _get_logger_impl(__name__)

_DOCS_WRITE_MAX_RETRIES: int = 4
_DOCS_WRITE_BASE_DELAY_SEC: float = 15.0
_DOCS_WRITE_MAX_DELAY_SEC: float = 90.0
_DOCS_TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})
# Transport-level failures below the HTTP layer: connection reset (WinError 10054),
# timeouts, SSL errors, remote disconnect. Retried in place with the same backoff as 429.
_DOCS_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (OSError, http.client.HTTPException)


def _compute_retry_delay(attempt: int) -> float:
    """Exponential backoff: base * 2^(attempt-1), capped at max."""
    delay: float = _DOCS_WRITE_BASE_DELAY_SEC * (2 ** (attempt - 1))
    return min(delay, _DOCS_WRITE_MAX_DELAY_SEC)


class GoogleDocsTransientError(RuntimeError):
    """Raised when Google Docs API returns a transient server-side error."""

    def __init__(self, status_code: int, document_id: str, original: Exception) -> None:
        super().__init__(
            f"Google Docs transient error status={status_code} document_id={document_id}"
        )
        self.status_code = status_code
        self.document_id = document_id
        self.original = original


def _execute_with_retry(
    operation_name: str,
    document_id: str,
    request_callable: Callable[[], Any],
    request_count: Optional[int] = None,
) -> Any:
    for attempt in range(1, _DOCS_WRITE_MAX_RETRIES + 1):
        try:
            return request_callable()
        except HttpError as error:
            status_code: Optional[int] = getattr(getattr(error, "resp", None), "status", None)
            if status_code == 429 and attempt < _DOCS_WRITE_MAX_RETRIES:
                delay_sec: float = _compute_retry_delay(attempt)
                if request_count is not None:
                    LOGGER.warning(
                        "docs_batch_update_429 attempt=%d/%d delay_sec=%.1f document_id=%s requests_count=%d",
                        attempt,
                        _DOCS_WRITE_MAX_RETRIES,
                        delay_sec,
                        document_id,
                        request_count,
                        extra={"warning_category": "informational"},
                    )
                else:
                    LOGGER.warning(
                        "docs_request_429 operation=%s attempt=%d/%d delay_sec=%.1f document_id=%s",
                        operation_name,
                        attempt,
                        _DOCS_WRITE_MAX_RETRIES,
                        delay_sec,
                        document_id,
                        extra={"warning_category": "informational"},
                    )
                time.sleep(delay_sec)
                continue
            if status_code in _DOCS_TRANSIENT_STATUS_CODES:
                raise GoogleDocsTransientError(
                    status_code=int(status_code),
                    document_id=document_id,
                    original=error,
                ) from error
            raise
        except _DOCS_CONNECTION_ERRORS as error:
            if attempt >= _DOCS_WRITE_MAX_RETRIES:
                raise
            delay_sec = _compute_retry_delay(attempt)
            LOGGER.warning(
                "docs_request_connection_error operation=%s attempt=%d/%d delay_sec=%.1f document_id=%s error=%s: %s",
                operation_name,
                attempt,
                _DOCS_WRITE_MAX_RETRIES,
                delay_sec,
                document_id,
                type(error).__name__,
                error,
                extra={"warning_category": "informational"},
            )
            time.sleep(delay_sec)


class GoogleDocsClient:
    def __init__(self, docs_service: Any) -> None:
        self._docs_service: Any = docs_service

    def ping_access(self) -> str:
        # Safe ping without mutating user documents.
        try:
            self._docs_service.documents().get(documentId="pipeline-ping").execute()
            return "probe_document_found(unexpected)"
        except HttpError as error:
            status_code: Optional[int] = getattr(
                getattr(error, "resp", None), "status", None
            )
            if status_code in {400, 404}:
                return f"probe_rejected_as_expected(status={status_code})"
            raise

    def create_document(self, title: str) -> str:
        doc: Dict[str, Any] = cast(
            Dict[str, Any],
            _execute_with_retry(
                operation_name="create_document",
                document_id="<new_document>",
                request_callable=lambda: self._docs_service.documents()
                .create(body={"title": title})
                .execute(),
            ),
        )
        return str(doc["documentId"])

    def get_document(self, document_id: str) -> Dict[str, Any]:
        return cast(
            Dict[str, Any],
            _execute_with_retry(
                operation_name="get_document",
                document_id=document_id,
                request_callable=lambda: self._docs_service.documents()
                .get(documentId=document_id)
                .execute(),
            ),
        )

    def batch_update(
        self, document_id: str, requests_payload: List[Dict[str, Any]]
    ) -> None:
        _execute_with_retry(
            operation_name="batch_update",
            document_id=document_id,
            request_count=len(requests_payload),
            request_callable=lambda: self._docs_service.documents().batchUpdate(
                    documentId=document_id,
                    body={"requests": requests_payload},
                ).execute(),
        )

    def insert_table_at_end(self, document_id: str, rows: int, columns: int) -> None:
        # Вставляем таблицу в конец основного сегмента документа.
        self.batch_update(
            document_id=document_id,
            requests_payload=[
                {
                    "insertTable": {
                        "rows": rows,
                        "columns": columns,
                        "endOfSegmentLocation": {},
                    }
                }
            ],
        )

    def get_document_end_index(self, document_id: str) -> int:
        doc: Dict[str, Any] = self.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc.get("body", {}).get("content", [])
        )
        if not content:
            raise RuntimeError("Google Docs document body.content is empty.")
        end_index_raw: Optional[Any] = content[-1].get("endIndex")
        if end_index_raw is None:
            raise RuntimeError("Google Docs document endIndex is missing.")
        return int(end_index_raw)

    def insert_text_at_index(self, document_id: str, index: int, text: str) -> None:
        self.batch_update(
            document_id=document_id,
            requests_payload=[
                {"insertText": {"location": {"index": int(index)}, "text": text}}
            ],
        )

    def insert_table_at_index(
        self,
        document_id: str,
        rows: int,
        columns: int,
        index: int,
    ) -> None:
        self.batch_update(
            document_id=document_id,
            requests_payload=[
                {
                    "insertTable": {
                        "rows": rows,
                        "columns": columns,
                        "location": {"index": int(index)},
                    }
                }
            ],
        )
