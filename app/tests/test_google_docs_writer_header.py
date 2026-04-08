from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.core.text_utils import utf16_len
from app.publish.doc_header import DailyDocHeader, HeaderLine
from app.publish.google_docs_writer import GoogleDocsReportWriter


class GoogleDocsWriterHeaderTests(unittest.TestCase):
    class _DocsClient:
        def __init__(self) -> None:
            self.batch_updates: list[list[dict[str, object]]] = []

        def get_document(self, document_id: str) -> dict[str, object]:
            _ = document_id
            return {"body": {"content": [{"endIndex": 101}]}}

        def batch_update(
            self,
            document_id: str,
            requests_payload: list[dict[str, object]],
        ) -> None:
            _ = document_id
            self.batch_updates.append(requests_payload)

    def test_write_header_only_uses_structured_relative_spans(self) -> None:
        docs_client: GoogleDocsWriterHeaderTests._DocsClient = self._DocsClient()
        writer: GoogleDocsReportWriter = GoogleDocsReportWriter(
            docs_client=docs_client,
            templates=SimpleNamespace(
                google_doc_bold_line_prefixes_json='["Regular subtitle"]',
                google_doc_bold_line_prefixes=["Regular subtitle"],
            ),
        )
        warning_heading: str = "DE - 20:00 ⚠ [merge failed — source list]"
        header: DailyDocHeader = DailyDocHeader(
            lines=(
                HeaderLine(text="Main title", is_bold=True),
                HeaderLine(text="Regular subtitle", is_bold=False),
                HeaderLine(text="", is_bold=False),
                HeaderLine(text=warning_heading, is_bold=True),
                HeaderLine(text="Body line", is_bold=False),
            )
        )

        writer.write_header_only(document_id="doc-1", header=header)

        self.assertEqual(1, len(docs_client.batch_updates))
        requests_payload: list[dict[str, object]] = docs_client.batch_updates[0]

        insert_request: dict[str, object] = requests_payload[0]["insertText"]  # type: ignore[index]
        insert_location: dict[str, object] = insert_request["location"]  # type: ignore[index]
        self.assertEqual(100, int(insert_location["index"]))  # type: ignore[index]
        self.assertEqual(header.render_text(), str(insert_request["text"]))  # type: ignore[index]

        style_requests: list[dict[str, object]] = [
            request for request in requests_payload if "updateTextStyle" in request
        ]
        bold_style_requests: list[dict[str, object]] = [
            request
            for request in style_requests
            if bool(request["updateTextStyle"]["textStyle"]["bold"])  # type: ignore[index]
        ]

        relative_spans = header.build_relative_style_spans()
        expected_bold_ranges: list[tuple[int, int]] = [
            (100 + span.start, 100 + span.end) for span in relative_spans
        ]
        actual_bold_ranges: list[tuple[int, int]] = [
            (
                int(request["updateTextStyle"]["range"]["startIndex"]),  # type: ignore[index]
                int(request["updateTextStyle"]["range"]["endIndex"]),  # type: ignore[index]
            )
            for request in bold_style_requests
        ]
        self.assertEqual(expected_bold_ranges, actual_bold_ranges)

        subtitle_start: int = utf16_len("Main title\n")
        subtitle_end: int = subtitle_start + utf16_len("Regular subtitle")
        self.assertNotIn((100 + subtitle_start, 100 + subtitle_end), actual_bold_ranges)


if __name__ == "__main__":
    unittest.main()
