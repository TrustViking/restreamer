from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.google.sheets_client import GoogleSheetsClient
from app.pipeline.batch_runner import BatchRunner


class SheetLayoutWritebackTests(unittest.TestCase):
    def test_sheets_read_af_layout(self) -> None:
        sheets_service = MagicMock()
        sheets_service.spreadsheets.return_value.values.return_value.get.return_value.execute.return_value = {
            "values": [
                [
                    "L",
                    "Chips",
                    "Links",
                    "Date",
                    "Time",
                    "Preview (Google Drive)",
                ],
                [
                    "uk",
                    "Some title",
                    "https://youtu.be/test123",
                    "29.05.2026",
                    "18:00:00",
                    "https://drive.google.com/uc?export=download&id=file123",
                ],
            ]
        }

        rows = GoogleSheetsClient(sheets_service).read_rows(
            spreadsheet_id="sheet",
            range_name="A:F",
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("https://youtu.be/test123", rows[0].link)
        self.assertEqual("29.05.2026", rows[0].date_raw)
        self.assertEqual("18:00:00", rows[0].time_raw)
        self.assertEqual(2, rows[0].links_column_index)

    def test_writeback_preview_column_f(self) -> None:
        runner = BatchRunner.__new__(BatchRunner)
        runner._logger = logging.getLogger("test-writeback-preview")
        runner._config = SimpleNamespace(google=SimpleNamespace(sheets_id="sheet"))

        services = SimpleNamespace(sheets_client=MagicMock())
        sheet_state = SimpleNamespace(sheet_name_for_writeback="TestSheet")
        prepared_videos = [
            SimpleNamespace(
                row_number=2,
                saved_preview_url="https://drive.google.com/uc?export=download&id=file123",
            )
        ]

        runner._writeback_preview_urls(
            services=services,
            prepared_videos=prepared_videos,
            sheet_state=sheet_state,
            dry_run=False,
        )

        services.sheets_client.update_cell_string.assert_called_once_with(
            spreadsheet_id="sheet",
            cell_a1="'TestSheet'!F2",
            value="https://drive.google.com/uc?export=download&id=file123",
        )

    def test_writeback_language_column_a(self) -> None:
        runner = BatchRunner.__new__(BatchRunner)
        runner._logger = logging.getLogger("test-writeback-language")
        runner._config = SimpleNamespace(google=SimpleNamespace(sheets_id="sheet"))

        services = SimpleNamespace(sheets_client=MagicMock())
        sheet_state = SimpleNamespace(sheet_name_for_writeback="TestSheet")
        prepared_videos = [SimpleNamespace(row_number=2, language="uk")]

        runner._writeback_detected_languages(
            services=services,
            prepared_videos=prepared_videos,
            sheet_state=sheet_state,
            dry_run=False,
        )

        services.sheets_client.update_cell_string.assert_called_once_with(
            spreadsheet_id="sheet",
            cell_a1="'TestSheet'!A2",
            value="uk",
        )


if __name__ == "__main__":
    unittest.main()
