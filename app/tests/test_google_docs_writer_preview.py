from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.publish.google_docs_writer import GoogleDocsReportWriter


class GoogleDocsWriterPreviewTests(unittest.TestCase):
    class _DocsClient:
        def __init__(self) -> None:
            self.preview_link_attempts: int = 0
            self.preview_link_indices: list[int] = []
            self.preview_image_indices: list[int] = []

        def get_document(self, document_id: str) -> dict[str, object]:
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
                                                        "startIndex": 10 + row_index,
                                                        "paragraph": {},
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                    for row_index in range(7)
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
            if requests_payload and "insertInlineImage" in requests_payload[0]:
                image_request: dict[str, object] = requests_payload[0]["insertInlineImage"]  # type: ignore[index]
                location: dict[str, object] = image_request["location"]  # type: ignore[index]
                self.preview_image_indices.append(int(location["index"]))  # type: ignore[index]
                return None
            payload_text: str = str(requests_payload)
            if "https://youtu.be/abc" in payload_text:
                self.preview_link_attempts += 1
                insert_request: dict[str, object] = requests_payload[0]["insertText"]  # type: ignore[index]
                location = insert_request["location"]  # type: ignore[index]
                self.preview_link_indices.append(int(location["index"]))  # type: ignore[index]
                if self.preview_link_attempts == 1:
                    raise RuntimeError("must be inside the bounds of an existing paragraph")
            return None

    def test_preview_link_recovery_aligns_image_insert_offset(self) -> None:
        docs_client = self._DocsClient()
        writer = GoogleDocsReportWriter(
            docs_client=docs_client,
            templates=SimpleNamespace(google_doc_bold_line_prefixes_json="[]"),
        )
        video = SimpleNamespace(
            row_number=2,
            normalized_link="https://youtu.be/abc",
            original_link="https://youtu.be/abc",
            metadata=SimpleNamespace(
                url="https://youtu.be/abc",
                thumbnail_url="https://example.com/thumb.jpg",
            ),
        )
        row_values = [
            ("UK", True),
            ("Title", True),
            ("Title body", False),
            ("Description", True),
            ("Description body", False),
            ("Preview", True),
            (" ", False),
        ]
        with patch(
            "app.publish.google_docs_writer._build_language_table_rows",
            return_value=row_values,
        ), self.assertLogs(level="INFO") as captured:
            writer.write_language_table(
                document_id="doc",
                language="uk",
                videos=[video],
                merged_content=None,
                merge_attempt=None,
            )
        self.assertEqual(2, docs_client.preview_link_attempts)
        self.assertEqual([17, 18], docs_client.preview_link_indices)
        self.assertEqual([39], docs_client.preview_image_indices)
        self.assertEqual(
            docs_client.preview_link_indices[-1] + len("https://youtu.be/abc\n"),
            docs_client.preview_image_indices[-1],
        )
        self.assertIn(
            "Preview link insert recovered with plus_one=True for row 2 in language uk.",
            "\n".join(captured.output),
        )
        self.assertIn(
            "preview_insert row=2 lang=uk stage=link link_plus_one_used=yes effective_offset=1 paragraph_insert_index=17 link_index=18",
            "\n".join(captured.output),
        )
        self.assertIn(
            "preview_insert row=2 lang=uk stage=image link_plus_one_used=yes effective_offset=1 image_index=39 link_and_image_mode=aligned",
            "\n".join(captured.output),
        )


if __name__ == "__main__":
    unittest.main()
