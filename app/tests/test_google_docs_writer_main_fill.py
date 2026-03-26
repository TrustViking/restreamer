from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.publish.google_docs_writer import GoogleDocsReportWriter


class GoogleDocsWriterMainFillTests(unittest.TestCase):
    class _DocsClient:
        def __init__(self) -> None:
            self._row_start_indices: list[int] = [10, 20]
            self.get_document_calls: int = 0
            self.insert_text_indices: list[int] = []
            self.failed_once: bool = False

        def get_document(self, document_id: str) -> dict[str, object]:
            self.get_document_calls += 1
            if self.get_document_calls >= 2:
                self._row_start_indices = [25, 35]
            return {
                "body": {
                    "content": [
                        {
                            "table": {
                                "tableRows": [
                                    {
                                        "tableCells": [
                                            {
                                                "content": [
                                                    {
                                                        "startIndex": start_index,
                                                        "paragraph": {},
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                    for start_index in self._row_start_indices
                                ]
                            },
                            "startIndex": 10,
                        }
                    ]
                }
            }

        def insert_table_at_end(self, document_id: str, rows: int, columns: int) -> None:
            return None

        def insert_table_at_index(
            self,
            document_id: str,
            rows: int,
            columns: int,
            index: int,
        ) -> None:
            return None

        def batch_update(self, document_id: str, requests_payload: list[dict[str, object]]) -> None:
            if not requests_payload:
                return None
            insert_text_request: dict[str, object] | None = None
            if "insertText" in requests_payload[0]:
                insert_text_request = requests_payload[0]["insertText"]  # type: ignore[index]
            if insert_text_request is None:
                return None
            text: str = str(insert_text_request["text"])  # type: ignore[index]
            location: dict[str, object] = insert_text_request["location"]  # type: ignore[index]
            index: int = int(location["index"])  # type: ignore[index]
            self.insert_text_indices.append(index)
            if text == "Row 2\n" and not self.failed_once:
                self.failed_once = True
                raise RuntimeError(
                    "Invalid requests[0].insertText: The insertion index must be inside the bounds of an existing paragraph"
                )
            return None

    def test_main_fill_refetches_document_and_recomputes_indices_after_bounds_error(
        self,
    ) -> None:
        docs_client = self._DocsClient()
        writer = GoogleDocsReportWriter(
            docs_client=docs_client,
            templates=SimpleNamespace(
                google_doc_bold_line_prefixes_json="[]",
                google_doc_bold_line_prefixes=[],
            ),
        )
        row_values = [
            ("Row 1", True),
            ("Row 2", False),
        ]

        with patch(
            "app.publish.google_docs_writer._build_language_table_rows",
            return_value=row_values,
        ), self.assertLogs(level="DEBUG") as captured:
            writer.write_language_table(
                document_id="doc-main-fill",
                language="uk",
                videos=[],
                merged_content=None,
                merge_attempt=None,
            )

        logs: str = "\n".join(captured.output)
        self.assertEqual([20, 35, 25], docs_client.insert_text_indices)
        self.assertGreaterEqual(docs_client.get_document_calls, 4)
        self.assertIn(
            "language_table_fill_failed doc_id=doc-main-fill lang=uk table_index=1 row_index=1 column_index=0 attempt=initial insert_index=20",
            logs,
        )
        self.assertIn(
            "refetch_doc=yes recompute_cell_indices=yes offset_workaround=no",
            logs,
        )
        self.assertIn(
            "language_table_fill_succeeded doc_id=doc-main-fill lang=uk table_index=1 row_index=1 column_index=0 attempt=retry_recomputed insert_index=35",
            logs,
        )
        self.assertNotIn("insert_index=21", logs)


if __name__ == "__main__":
    unittest.main()
