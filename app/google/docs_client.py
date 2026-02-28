from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

try:
    from googleapiclient.errors import HttpError
except ImportError:
    HttpError = Exception  # type: ignore


class GoogleDocsClient:
    def __init__(self, docs_service: Any) -> None:
        self._docs_service: Any = docs_service

    def ping_access(self) -> str:
        # Safe ping without mutating user documents.
        try:
            self._docs_service.documents().get(documentId="streamertg-ping").execute()
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
        self._docs_service.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests_payload},
        ).execute()

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
