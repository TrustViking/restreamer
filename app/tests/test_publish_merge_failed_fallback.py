from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.core.models import BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE
from app.publish import doc_helpers, telegram_renderer


class MergeFailedPublishFallbackTests(unittest.TestCase):
    def _video(self, *, title: str, description: str) -> SimpleNamespace:
        return SimpleNamespace(
            language="en",
            metadata=SimpleNamespace(
                title=title,
                description=description,
            ),
            normalized_link="https://youtube.com/watch?v=abcdefghijk",
        )

    def _merge_attempt(self) -> SimpleNamespace:
        return SimpleNamespace(
            language="en",
            raw_response_text='{"title":"Internal title","description":"Internal JSON body"}',
            publish_source_label="merge_failed",
            plain_repair_used=False,
            salvaged_title=None,
            rejected_attempts=(),
            block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
        )

    def test_doc_summary_suppresses_failed_merge_raw_json(self) -> None:
        videos = [
            self._video(title="Source 1", description="Source description one."),
            self._video(title="Source 2", description="Source description two."),
        ]
        with self.assertLogs(level="INFO") as captured:
            description_text = doc_helpers._build_descriptions_summary(
                videos=videos,
                templates=None,
                merged_content=None,
                merge_attempt=self._merge_attempt(),
            )
        self.assertNotIn('{"title":"Internal title"', description_text)
        self.assertIn("Source description one.", description_text)
        self.assertIn("Source description two.", description_text)
        logs: str = "\n".join(captured.output)
        self.assertIn("target=doc", logs)
        self.assertIn(
            "block_generation_mode=fallback_after_merge_failure",
            logs,
        )
        self.assertIn("merge_failed=yes", logs)
        self.assertIn("raw_output_suppressed=yes", logs)
        self.assertIn("fallback=source_descriptions", logs)

    def test_telegram_summary_suppresses_failed_merge_raw_json(self) -> None:
        videos = [
            self._video(title="Source 1", description="Source description one."),
            self._video(title="Source 2", description="Source description two."),
        ]
        with self.assertLogs(level="INFO") as captured:
            description_text = telegram_renderer.build_descriptions_summary(
                videos=videos,
                templates=None,
                merged_content=None,
                merge_attempt=self._merge_attempt(),
            )
        self.assertNotIn('{"title":"Internal title"', description_text)
        self.assertIn("Source description one.", description_text)
        self.assertIn("Source description two.", description_text)
        logs: str = "\n".join(captured.output)
        self.assertIn("target=telegram", logs)
        self.assertIn(
            "block_generation_mode=fallback_after_merge_failure",
            logs,
        )
        self.assertIn("merge_failed=yes", logs)
        self.assertIn("raw_output_suppressed=yes", logs)
        self.assertIn("fallback=source_descriptions", logs)

    def test_doc_table_heading_marks_partial_fallback_block(self) -> None:
        rows = doc_helpers._build_language_table_rows(
            language="en",
            videos=[
                self._video(title="Source 1", description="Source description one."),
                self._video(title="Source 2", description="Source description two."),
            ],
            merged_content=None,
            merge_attempt=self._merge_attempt(),
            time_display="18:00",
            templates=SimpleNamespace(
                google_doc_table_labels_json='{"en":["TITLE","DESCRIPTION","PREVIEW"]}',
                google_doc_table_labels={"en": ["TITLE", "DESCRIPTION", "PREVIEW"]},
                google_doc_language_headings_json='{"en":"EN"}',
                google_doc_language_headings={"en": "EN"},
            ),
            artifact_status="partial",
        )
        self.assertEqual("EN - 18:00 ⚠ [merge failed — source list]", rows[0][0])

    def test_doc_table_heading_marks_fallback_only_block_as_rejected_artifact(self) -> None:
        rows = doc_helpers._build_language_table_rows(
            language="en",
            videos=[
                self._video(title="Source 1", description="Source description one."),
                self._video(title="Source 2", description="Source description two."),
            ],
            merged_content=None,
            merge_attempt=self._merge_attempt(),
            time_display="18:00",
            templates=SimpleNamespace(
                google_doc_table_labels_json='{"en":["TITLE","DESCRIPTION","PREVIEW"]}',
                google_doc_table_labels={"en": ["TITLE", "DESCRIPTION", "PREVIEW"]},
                google_doc_language_headings_json='{"en":"EN"}',
                google_doc_language_headings={"en": "EN"},
            ),
            artifact_status="fallback_only",
        )
        self.assertEqual(
            "EN - 18:00 ⚠ [merge failed — source list]",
            rows[0][0],
        )

    def test_doc_table_rows_include_rejected_model_outputs_for_analysis(self) -> None:
        merge_attempt = self._merge_attempt()
        merge_attempt.rejected_attempts = (
            SimpleNamespace(
                attempt_index=1,
                model_name="gpt-5.1",
                reject_reasons=("too_few_expanded_bullets", "weak_source_coverage"),
                title="Rejected title one",
                description="Rejected description one.",
            ),
            SimpleNamespace(
                attempt_index=2,
                model_name="gpt-5.1",
                reject_reasons=("overly_generic_body",),
                title="Rejected title two",
                description="Rejected description two.",
            ),
        )
        rows = doc_helpers._build_language_table_rows(
            language="en",
            videos=[
                self._video(title="Source 1", description="Source description one."),
                self._video(title="Source 2", description="Source description two."),
            ],
            merged_content=None,
            merge_attempt=merge_attempt,
            time_display="18:00",
            templates=SimpleNamespace(
                google_doc_table_labels_json='{"en":["TITLE","DESCRIPTION","PREVIEW"]}',
                google_doc_table_labels={"en": ["TITLE", "DESCRIPTION", "PREVIEW"]},
                google_doc_language_headings_json='{"en":"EN"}',
                google_doc_language_headings={"en": "EN"},
            ),
        )
        flattened_rows = [value for value, _ in rows]
        flattened_text: str = "\n".join(flattened_rows)
        self.assertIn("Source 1", flattened_rows[2])
        self.assertIn("Source description one.", flattened_rows[4])
        self.assertTrue(
            any("MODEL OUTPUTS REJECTED BY VALIDATION." in row for row in flattened_rows)
        )
        self.assertIn(
            "REJECTED TITLE | ATTEMPT 1 | MODEL: gpt-5.1 | REJECT: too_few_expanded_bullets,weak_source_coverage",
            flattened_text,
        )
        self.assertIn("Rejected title one", flattened_text)
        self.assertIn(
            "REJECTED DESCRIPTION | ATTEMPT 2 | MODEL: gpt-5.1 | REJECT: overly_generic_body",
            flattened_text,
        )
        self.assertIn("Rejected description two.", flattened_text)
