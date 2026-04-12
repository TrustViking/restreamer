from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.publish.docs_request_builder import DocsRequestBuilder


class DocsRequestBuilderRichTests(unittest.TestCase):
    def test_build_insert_and_style_rich_requests_splits_segments(self) -> None:
        segments = [
            ("Производство: ", False),
            ("ALLATRA GRC", True),
            (" — глобальный", False),
        ]
        clean_text = "Производство: ALLATRA GRC — глобальный"
        requests = DocsRequestBuilder.build_insert_and_style_rich_requests(
            index=10,
            text=clean_text,
            base_bold=False,
            segments=segments,
        )
        # 1 insertText + 3 updateTextStyle (one per segment)
        self.assertEqual(4, len(requests))
        self.assertIn("insertText", requests[0])
        # Second segment (ALLATRA GRC) should be bold
        style_requests = [r for r in requests if "updateTextStyle" in r]
        self.assertEqual(3, len(style_requests))
        # First segment: not bold
        self.assertFalse(style_requests[0]["updateTextStyle"]["textStyle"]["bold"])
        # Second segment: bold
        self.assertTrue(style_requests[1]["updateTextStyle"]["textStyle"]["bold"])
        # Third segment: not bold
        self.assertFalse(style_requests[2]["updateTextStyle"]["textStyle"]["bold"])

    def test_build_insert_and_style_rich_requests_respects_base_bold(self) -> None:
        segments = [("Header text", True)]
        clean_text = "Header text"
        requests = DocsRequestBuilder.build_insert_and_style_rich_requests(
            index=10,
            text=clean_text,
            base_bold=True,
            segments=segments,
        )
        style_requests = [r for r in requests if "updateTextStyle" in r]
        self.assertEqual(1, len(style_requests))
        self.assertTrue(style_requests[0]["updateTextStyle"]["textStyle"]["bold"])

    def test_no_markers_uses_original_path(self) -> None:
        """Verify that build_insert_and_style_requests still works for plain text."""
        requests = DocsRequestBuilder.build_insert_and_style_requests(
            index=10,
            text="Plain text\n",
            bold=False,
        )
        self.assertEqual(2, len(requests))
        self.assertIn("insertText", requests[0])
        self.assertIn("updateTextStyle", requests[1])


if __name__ == "__main__":
    unittest.main()
