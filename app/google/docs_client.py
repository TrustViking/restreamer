from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, cast

from app.bootstrap.logging_config import get_logger as _get_logger_impl

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = Exception  # type: ignore

LOGGER = _get_logger_impl(__name__)

_DOCS_WRITE_MAX_RETRIES: int = 4
_DOCS_WRITE_BASE_DELAY_SEC: float = 15.0
_DOCS_WRITE_MAX_DELAY_SEC: float = 90.0


def _compute_retry_delay(attempt: int) -> float:
    """Exponential backoff: base * 2^(attempt-1), capped at max."""
    delay: float = _DOCS_WRITE_BASE_DELAY_SEC * (2 ** (attempt - 1))
    return min(delay, _DOCS_WRITE_MAX_DELAY_SEC)


class GoogleDocsClient:
    def __init__(self, docs_service: Any) -> None:
        self._docs_service: Any = docs_service

    def ping_access(self) -> str:
        # Safe ping without mutating user documents.
        try:
            self._docs_service.documents().get(documentId="restreamer-ping").execute()
            return "probe_document_found(unexpected)"
        except HttpError as error:
            status_code: Optional[int] = getattr(
                getattr(error, "resp", None), "status", None
            )
            if status_code in {400, 404}:
                return f"probe_rejected_as_expected(status={status_code})"
            raise

    def create_document(self, title: str) -> str:
        doc: Dict[str, Any] = (
            self._docs_service.documents().create(body={"title": title}).execute()
        )
        return str(doc["documentId"])

    def get_document(self, document_id: str) -> Dict[str, Any]:
        return self._docs_service.documents().get(documentId=document_id).execute()

    def batch_update(
        self, document_id: str, requests_payload: List[Dict[str, Any]]
    ) -> None:
        for attempt in range(1, _DOCS_WRITE_MAX_RETRIES + 1):
            try:
                self._docs_service.documents().batchUpdate(
                    documentId=document_id,
                    body={"requests": requests_payload},
                ).execute()
                return
            except HttpError as error:
                status_code: Optional[int] = getattr(
                    getattr(error, "resp", None), "status", None
                )
                if status_code != 429 or attempt == _DOCS_WRITE_MAX_RETRIES:
                    raise
                delay_sec: float = _compute_retry_delay(attempt)
                LOGGER.warning(
                    "docs_batch_update_429 attempt=%d/%d delay_sec=%.1f document_id=%s requests_count=%d",
                    attempt,
                    _DOCS_WRITE_MAX_RETRIES,
                    delay_sec,
                    document_id,
                    len(requests_payload),
                )
                time.sleep(delay_sec)

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
