from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import (
    BLOCK_GENERATION_MODE_REAL_MERGE,
    MergedLanguageContent,
)
from app.llm.merges.merge_service import (
    attempt_openai_merge_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.merges.merge_validation import MergeAttemptFailure

from app.tests.test_merge_contract_helpers import MergeContractServiceBase


class MergeContractCompactTests(MergeContractServiceBase):
    """Tests for compact contract concerns of merge contract."""

    def test_compact_mode_rejects_1_paragraph_as_underflow(self) -> None:
        compact_response_one_paragraph: SimpleNamespace = self._make_merge_response(
            title="Brussels and Kharkiv: compact update",
            description="Only one paragraph.",
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[
                compact_response_one_paragraph,
                compact_response_one_paragraph,
                compact_response_one_paragraph,
            ],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        validation_reasons: tuple[str, ...] = tuple(attempt.validation_reasons or ())
        self.assertIn("paragraph_underflow", validation_reasons)
        self.assertNotIn("paragraph_overflow", validation_reasons)

    def test_compact_retry_remains_standard_even_with_structured_expanded_reason_codes(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Compact retry should stay standard here.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=("insufficient_expanded_body", "too_few_expanded_bullets"),
        )
        successful_merge = (
            MergedLanguageContent(
                title="Final title",
                description="Paragraph one.\n\nParagraph two.",
                description_selected="Paragraph one.\n\nParagraph two.",
                description_audit="Paragraph one.\n\nParagraph two.",
            ),
            '{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}',
        )
        with patch(
            "app.llm.merges.merge_orchestrator._attempt_merge_once",
            side_effect=[structured_failure, successful_merge],
        ) as attempt_mock:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        retry_profile = attempt_mock.call_args_list[1].kwargs["expanded_retry_profile"]
        self.assertIsNotNone(retry_profile)
        self.assertEqual("standard", retry_profile.retry_mode)
        self.assertEqual((), retry_profile.focus_tags)

    def test_compact_two_source_merge_keeps_existing_behavior(self) -> None:
        compact_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv: focused agenda tonight","description":"Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption and keep the summary tightly factual.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n🔹 practical next steps for viewers following both Brussels and Kharkiv developments\\n\\nThe closing paragraph keeps both source lines grounded without forcing an expanded three-source structure."}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: focused agenda tonight",
                "description": (
                    "Tonight we connect the Brussels vote calendar with the Kharkiv rail disruption and keep the summary tightly factual.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\n"
                    "🔹 practical next steps for viewers following both Brussels and Kharkiv developments\n\n"
                    "The closing paragraph keeps both source lines grounded without forcing an expanded three-source structure."
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[compact_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_style_coverage", joined_logs)
        self.assertNotIn("merge_expanded_quality_gate", joined_logs)

    def test_compact_two_source_merge_does_not_salvage_emoji_overflow(self) -> None:
        compact_emoji_overflow = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv: focused agenda tonight","description":"🔥 Tonight we connect the Brussels vote calendar 🚨 with the Kharkiv rail disruption 🎯 and keep the summary tightly factual 🧭 without missing the live stakes ✨ while covering 💥 all key angles 🔔 from both sources 🌟 for viewers 🎖 tonight 💡.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n🔹 viewer questions and timing watchpoints\\n🔹 next-step logistics for the corridor desk\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv: focused agenda tonight",
                "description": (
                    "🔥 Tonight we connect the Brussels vote calendar 🚨 with the Kharkiv rail disruption 🎯 and keep the summary tightly factual 🧭 without missing the live stakes ✨ while covering 💥 all key angles 🔔 from both sources 🌟 for viewers 🎖 tonight 💡.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes in Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\n"
                    "🔹 viewer questions and timing watchpoints\n"
                    "🔹 next-step logistics for the corridor desk\n\n"
                    "Watch live ✅ and share updates 📣 #briefing"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[compact_emoji_overflow, compact_emoji_overflow],
        ) as request_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos()[:2],
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNone(attempt.merged)
        self.assertEqual(2, request_mock.call_count)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=excessive_emoji_usage", joined_logs)
        self.assertNotIn("merge_llm_validation_salvage", joined_logs)


if __name__ == "__main__":
    unittest.main()
