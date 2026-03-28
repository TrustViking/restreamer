from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    BLOCK_GENERATION_MODE_REAL_MERGE,
    MergedLanguageContent,
)
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.llm.merges.merge_retry import _build_expanded_retry_profile
from app.llm.merges.merge_validation import (
    MergeAttemptFailure,
    _reason_code_from_error,
    _reason_codes_from_error,
)
from app.llm.merges.merge_service import (
    attempt_openai_merge_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.llm.merges.merge_run_summary import MergeRunSummary

from app.tests.test_merge_contract_helpers import MergeContractServiceBase


class MergeContractRetryTests(MergeContractServiceBase):
    """Tests for retry and failure flow concerns of merge contract."""

    def test_expanded_retry_profile_targets_expected_weak_points(self) -> None:
        scenarios = (
            (
                ("insufficient_expanded_body",),
                "targeted",
                "body_depth",
                "post-hook body clearly denser",
            ),
            (
                ("too_few_expanded_bullets",),
                "targeted",
                "bullet_sufficiency",
                "enough distinct, meaningful bullets",
            ),
            (
                ("overly_generic_body",),
                "targeted",
                "source_specificity",
                "source-grounded specifics",
            ),
            (
                ("hook_dominates_body",),
                "targeted",
                "hook_restraint",
                "Keep the hook brief and functional",
            ),
            (
                ("weak_source_coverage",),
                "targeted",
                "source_spread",
                "Restore distinguishable spread across source lines or topic nodes",
            ),
        )
        for reject_signals, expected_mode, expected_focus, expected_fragment in scenarios:
            with self.subTest(reject_signals=reject_signals):
                profile = _build_expanded_retry_profile(
                    source_count=3,
                    reject_signals=reject_signals,
                )
                self.assertEqual(expected_mode, profile.retry_mode)
                self.assertEqual((expected_focus,), profile.focus_tags)
                self.assertIn(expected_fragment, "\n".join(profile.reinforcement_lines))

    def test_three_source_targeted_retry_adds_combined_reinforcement_lines(self) -> None:
        profile = _build_expanded_retry_profile(
            source_count=3,
            reject_signals=(
                "insufficient_expanded_body",
                "too_few_expanded_bullets",
                "overly_generic_body",
                "weak_source_coverage",
            ),
        )
        reinforcement_text: str = "\n".join(profile.reinforcement_lines)
        self.assertIn("2 to 3 short agenda tracks", reinforcement_text)
        self.assertIn("cut generic filler bridges", reinforcement_text)

    def test_four_source_targeted_retry_adds_structured_source_specific_reinforcement(self) -> None:
        profile = _build_expanded_retry_profile(
            source_count=4,
            reject_signals=(
                "insufficient_expanded_body",
                "too_few_expanded_bullets",
                "overly_generic_body",
                "weak_source_coverage",
            ),
        )
        reinforcement_text: str = "\n".join(profile.reinforcement_lines)
        self.assertIn("2 to 3 short agenda tracks", reinforcement_text)
        self.assertIn("cut generic filler bridges", reinforcement_text)
        self.assertIn("do not collapse the post-hook body into one umbrella summary", reinforcement_text)
        self.assertIn("Build 2 to 3 meaningful thematic micro-blocks after the hook", reinforcement_text)
        self.assertIn("source-specific density, not just extra length", reinforcement_text)
        self.assertIn("Make every source leave a recognizable trace in the body", reinforcement_text)

    def test_invalid_primary_response_triggers_retry_then_final_failure(self) -> None:
        responses = [
            SimpleNamespace(raw_text='{"variants":[{"title":"bad"}]}', structured_payload={"variants": [{"title": "bad"}]}),
            SimpleNamespace(raw_text='{"title":"","description":"bad"}', structured_payload={"title": "", "description": "bad"}),
            SimpleNamespace(raw_text='{"title":"Final title","description":"Paragraph one.\\n\\nParagraph two."}', structured_payload={"title": "Final title", "description": "Paragraph one.\n\nParagraph two."}),
        ]

        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=responses) as request_mock:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )

        self.assertEqual(2, request_mock.call_count)
        self.assertIsNone(attempt.merged)
        self.assertEqual("gpt-5.1", attempt.model_name)
        self.assertEqual("merge_failed", attempt.publish_source_label)

    def test_merge_attempt_failure_carries_reason_code(self) -> None:
        failure = MergeAttemptFailure(
            reason_code="missing_keys",
            reason="missing_keys:description",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
            reason_codes=("missing_keys",),
        )
        self.assertEqual("missing_keys", failure.reason_code)
        self.assertEqual('{"title":"x"}', failure.raw_response_text)
        self.assertEqual(("missing_keys",), failure.reason_codes)

    def test_reason_code_from_error_uses_structured_codes_without_text_fallback(self) -> None:
        failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Human-readable text changed and carries no machine-readable list.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
            reason_codes=("overly_generic_body", "weak_source_coverage"),
        )
        with patch(
            "app.llm.merges.merge_service._extract_description_validation_reason_codes"
        ) as parse_mock:
            reason_codes = _reason_codes_from_error(failure)
            primary_reason = _reason_code_from_error(failure)
        self.assertEqual(("overly_generic_body", "weak_source_coverage"), reason_codes)
        self.assertEqual("overly_generic_body", primary_reason)
        parse_mock.assert_not_called()

    def test_reason_code_from_error_uses_single_defensive_text_fallback_when_structured_codes_missing(self) -> None:
        legacy_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="description validation failed: insufficient_expanded_body,weak_source_coverage",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"x"}',
            reason_codes=(),
        )
        with patch(
            "app.llm.merges.merge_validation._extract_description_validation_reason_codes",
            return_value=("insufficient_expanded_body", "weak_source_coverage"),
        ) as parse_mock:
            primary_reason = _reason_code_from_error(legacy_failure)
        self.assertEqual("insufficient_expanded_body", primary_reason)
        parse_mock.assert_called_once_with(str(legacy_failure))

    def test_final_failure_is_honest(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"","description":"Only one paragraph."}',
            structured_payload={"title": "", "description": "Only one paragraph."},
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        self.assertIn("merge_llm_final_failure", "\n".join(captured.output))
        self.assertIn("code=invalid_title", "\n".join(captured.output))
        self.assertEqual(BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE, attempt.block_generation_mode)
        self.assertEqual("fallback_titles", attempt.title_source)
        self.assertEqual("fallback_none", attempt.hook_source)
        self.assertEqual("fallback_source_descriptions", attempt.body_source)

    def test_fatal_model_config_error_does_not_retry_merge_attempt(self) -> None:
        fatal_error = LlmModelConfigurationError(
            provider_name="openai",
            model_name="gpt-5.4",
            reason_code="openai_model_access_denied",
            detail="Project does not have access to model `gpt-5.4`",
            status_code=403,
            api_error_code="access_denied",
            api_error_param="model",
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=fatal_error,
        ) as request_mock:
            with self.assertRaises(LlmModelConfigurationError):
                attempt_openai_merge_with_audit(
                    language="en",
                    videos=self._videos(),
                    config=self._config(),
                    attempt_label="TEST",
                    summarize_error=lambda error: str(error),
                    normalize_youtube_url=lambda url: url,
                    no_description_text="no description",
                )
        self.assertEqual(1, request_mock.call_count)

    def test_successful_merge_attempt_keeps_real_merge_mode(self) -> None:
        good_response = SimpleNamespace(
            raw_text=(
                '{"title":"Final title","description":"Paragraph one with concrete context and key tension.\\n\\n'
                "In this stream you'll see:\\n"
                "🔹 first key point from source one\\n"
                "🔹 second key point from source one\\n"
                "🔹 third key point from source two\\n"
                "🔹 fourth key point from source two\\n"
                '🔹 fifth combined conclusion from both sources"}'
            ),
            structured_payload={
                "title": "Final title",
                "description": (
                    "Paragraph one with concrete context and key tension.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 first key point from source one\n"
                    "🔹 second key point from source one\n"
                    "🔹 third key point from source two\n"
                    "🔹 fourth key point from source two\n"
                    "🔹 fifth combined conclusion from both sources"
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[good_response]):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        self.assertEqual(BLOCK_GENERATION_MODE_REAL_MERGE, attempt.block_generation_mode)
        self.assertEqual(BLOCK_GENERATION_MODE_REAL_MERGE, attempt.merged.block_generation_mode)

    def test_rejected_raw_response_is_preserved_for_final_failure(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"Bad title","description":"Only one paragraph."}',
            structured_payload={"title": "Bad title", "description": "Only one paragraph."},
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        self.assertIn('"title":"Bad title"', attempt.raw_response_text)
        self.assertIn("paragraph count", str(attempt.error_summary))

    def test_final_failed_merge_keeps_structured_rejected_attempt_outputs(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"Bad title","description":"Only one paragraph."}',
            structured_payload={"title": "Bad title", "description": "Only one paragraph."},
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[bad_response, bad_response, bad_response],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        self.assertGreaterEqual(len(attempt.rejected_attempts), 1)
        self.assertEqual(1, attempt.rejected_attempts[0].attempt_index)
        self.assertEqual("gpt-5.1", attempt.rejected_attempts[0].model_name)
        self.assertIn("paragraph_underflow", tuple(attempt.validation_reasons or ()))
        self.assertEqual("Bad title", attempt.rejected_attempts[0].title)
        self.assertEqual("Only one paragraph.", attempt.rejected_attempts[0].description)

    def test_merge_logs_include_stage_reason_code_and_slot_context(self) -> None:
        bad_response = SimpleNamespace(
            raw_text='{"title":"","description":"Only one paragraph."}',
            structured_payload={"title": "", "description": "Only one paragraph."},
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[bad_response, bad_response, bad_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt_openai_merge_with_audit(
                language="en",
                videos=self._videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("branch=merge", joined_logs)
        self.assertIn("date_key=010130", joined_logs)
        self.assertIn("slot_key=010130_1000", joined_logs)
        self.assertIn("language=en", joined_logs)
        self.assertIn("stage=primary", joined_logs)
        self.assertIn("fallback_used=no", joined_logs)
        self.assertIn("code=invalid_title", joined_logs)


if __name__ == "__main__":
    unittest.main()
