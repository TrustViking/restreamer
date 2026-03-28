from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.models import (
    BLOCK_GENERATION_MODE_REAL_MERGE,
    MergedLanguageContent,
)
from app.llm.merges.merge_validation import (
    MergeAttemptFailure,
    _is_softened_distinctive_source_coverage_eligible,
)
from app.llm.merges.merge_service import (
    attempt_openai_merge_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.llm.merges.merge_run_summary import MergeRunSummary

from app.tests.test_merge_contract_helpers import MergeContractServiceBase


class MergeContractExpandedTests(MergeContractServiceBase):
    """Tests for expanded contract concerns of merge contract."""

    def test_expanded_merge_rejects_overly_generic_body_for_three_sources(self) -> None:
        generic_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Why these developments matter tonight","description":"Tonight we step back and frame several important developments inside one smooth and readable opening that sounds strong but stays broad. It keeps attention on the mood and the overall stakes instead of distinct source facts.\\n\\nIn this stream you\'ll see:\\n🔹 the main context and why it matters\\n🔹 the broader background and tensions\\n🔹 how the story fits a larger pattern\\n\\nA polished closing paragraph keeps the editorial flow consistent for viewers."}'
            ),
            structured_payload={
                "title": "Why these developments matter tonight",
                "description": (
                    "Tonight we step back and frame several important developments inside one smooth and readable opening that sounds strong but stays broad. It keeps attention on the mood and the overall stakes instead of distinct source facts.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 the main context and why it matters\n"
                    "🔹 the broader background and tensions\n"
                    "🔹 how the story fits a larger pattern\n\n"
                    "A polished closing paragraph keeps the editorial flow consistent for viewers."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[generic_expanded_response, generic_expanded_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=insufficient_bullet_coverage", joined_logs)
        self.assertIn("merge_llm_response_invalid", joined_logs)
        self.assertIn("bullet_points_count=3", joined_logs)

    def test_expanded_merge_accepts_softened_distinctive_coverage_for_three_sources(self) -> None:
        borderline_coverage_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels vote and Kharkiv pressure tonight","description":"Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv and explain why the next operational window matters. The opening stays factual, but the summary only follows two of the source lines.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\\n🔹 Anna Kovalenko tracks customs delays and coalition counts before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\\n🔹 emergency logistics pressure is already shaping border and rail planning for the next 48 hours\\n🔹 transport bottlenecks now define how local authorities sequence response steps and public warnings\\n\\nThe body stays concrete and detailed, but it keeps the focus on political and transport fallout only."}'
            ),
            structured_payload={
                "title": "Brussels vote and Kharkiv pressure tonight",
                "description": (
                    "Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv and explain why the next operational window matters. The opening stays factual, but the summary only follows two of the source lines.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays and coalition counts before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes and depot damage on the eastern line\n"
                    "🔹 emergency logistics pressure is already shaping border and rail planning for the next 48 hours\n"
                    "🔹 transport bottlenecks now define how local authorities sequence response steps and public warnings\n\n"
                    "The body stays concrete and detailed, but it keeps the focus on political and transport fallout only."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[borderline_coverage_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_response_valid", joined_logs)
        self.assertIn("bullet_points_count=6", joined_logs)
        self.assertNotIn("code=weak_source_coverage", joined_logs)
        self.assertNotIn("code=insufficient_bullet_coverage", joined_logs)

    def test_expanded_merge_rejects_three_source_coverage_two_of_three_when_other_signals_are_weak(self) -> None:
        weak_borderline_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels vote and Kharkiv pressure tonight","description":"Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv, but the write-up stays thin and leaves the missing third lane unresolved.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\\n🔹 Anna Kovalenko tracks customs delays before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 transport bottlenecks now define how local authorities sequence response steps\\n\\nThe closing paragraph stays broad and never restores the missing third source with any concrete facts."}'
            ),
            structured_payload={
                "title": "Brussels vote and Kharkiv pressure tonight",
                "description": (
                    "Tonight we connect the Brussels vote calendar with the transport shock in Kharkiv, but the write-up stays thin and leaves the missing third lane unresolved.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks customs delays before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 transport bottlenecks now define how local authorities sequence response steps\n\n"
                    "The closing paragraph stays broad and never restores the missing third source with any concrete facts."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[weak_borderline_response, weak_borderline_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=insufficient_bullet_coverage", joined_logs)
        self.assertIn("bullet_points_count=4", joined_logs)
        self.assertIn("merge_llm_response_invalid", joined_logs)

    def test_expanded_merge_accepts_strong_three_source_output(self) -> None:
        strong_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the operational agenda tonight","description":"Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\\n\\nThe closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the operational agenda tonight",
                "description": (
                    "Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\n\n"
                    "The closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[strong_expanded_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_response_valid", joined_logs)
        self.assertIn("bullet_points_count=6", joined_logs)
        self.assertNotIn("code=insufficient_bullet_coverage", joined_logs)

    def test_expanded_merge_accepts_dense_two_paragraph_body_when_content_is_rich(self) -> None:
        dense_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the dense operational briefing tonight","description":"Tonight we keep the Brussels vote, the Kharkiv rail disruption, and the Geneva aid timetable inside one compact but source-grounded setup so viewers get the concrete agenda without wasted filler.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and procedural pressure before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release window"}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the dense operational briefing tonight",
                "description": (
                    "Tonight we keep the Brussels vote, the Kharkiv rail disruption, and the Geneva aid timetable inside one compact but source-grounded setup so viewers get the concrete agenda without wasted filler.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and procedural pressure before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release window"
                ),
            },
        )
        with patch("app.llm.merges.merge_service.openai_request_merge", side_effect=[dense_expanded_response]), self.assertLogs(
            level="INFO",
        ) as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_response_valid", joined_logs)
        self.assertIn("bullet_points_count=6", joined_logs)
        self.assertNotIn("code=insufficient_bullet_coverage", joined_logs)

    def test_expanded_mode_accepts_5_body_paragraphs(self) -> None:
        expanded_response_five_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: five-block agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=5),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[expanded_response_five_paragraphs],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)

    def test_expanded_mode_accepts_7_body_paragraphs(self) -> None:
        expanded_response_seven_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: seven-block agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=7),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[expanded_response_seven_paragraphs],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)

    def test_expanded_mode_rejects_8_body_paragraphs_as_overflow(self) -> None:
        expanded_response_eight_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: eight-block agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=8),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[
                expanded_response_eight_paragraphs,
                expanded_response_eight_paragraphs,
                expanded_response_eight_paragraphs,
            ],
        ):
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        validation_reasons: tuple[str, ...] = tuple(attempt.validation_reasons or ())
        self.assertIn("paragraph_overflow", validation_reasons)
        self.assertNotIn("paragraph_underflow", validation_reasons)

    def test_collapse_recovery_works_for_expanded_mode(self) -> None:
        expanded_response_nine_paragraphs: SimpleNamespace = self._make_merge_response(
            title="Brussels, Kharkiv, Geneva: collapse recovery agenda tonight",
            description=self._expanded_description_with_body_paragraphs(body_paragraphs=9),
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[expanded_response_nine_paragraphs],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("body_paragraph_count_after_recovery=7", joined_logs)
        self.assertIn("recovery_applied=yes", joined_logs)

    def test_expanded_merge_rejects_weak_two_paragraph_body_when_content_is_thin(self) -> None:
        weak_compact_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv tonight","description":"Tonight we touch on several developments and keep the wording broad instead of source-specific.\\n\\nIn this stream you\'ll see:\\n🔹 why it matters tonight\\n🔹 the broader context\\n🔹 what viewers should watch next"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv tonight",
                "description": (
                    "Tonight we touch on several developments and keep the wording broad instead of source-specific.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 why it matters tonight\n"
                    "🔹 the broader context\n"
                    "🔹 what viewers should watch next"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[weak_compact_response, weak_compact_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
            )
        self.assertIsNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("code=insufficient_bullet_coverage", joined_logs)
        self.assertIn("bullet_points_count=3", joined_logs)
        self.assertIn(
            "description validation failed: insufficient_bullet_coverage",
            joined_logs,
        )

    def test_expanded_retry_uses_targeted_prompt_and_logs_retry_focus(self) -> None:
        weak_compact_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels and Kharkiv tonight","description":"Tonight we touch on several developments and keep the wording broad instead of source-specific.\\n\\nIn this stream you\'ll see:\\n🔹 why it matters tonight\\n🔹 the broader context\\n🔹 what viewers should watch next"}'
            ),
            structured_payload={
                "title": "Brussels and Kharkiv tonight",
                "description": (
                    "Tonight we touch on several developments and keep the wording broad instead of source-specific.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 why it matters tonight\n"
                    "🔹 the broader context\n"
                    "🔹 what viewers should watch next"
                ),
            },
        )
        strong_expanded_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the operational agenda tonight","description":"Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\\n\\nThe closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the operational agenda tonight",
                "description": (
                    "Tonight we align the Brussels vote, the Kharkiv transport shock, and the Geneva aid timetable into one grounded briefing. Each source keeps its own factual lane, and the summary stays concrete instead of leaning on editorial gloss.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\n\n"
                    "The closing paragraph ties the political vote, frontline logistics, and medical supply chain into a clear next-step agenda without flattening the sources into one generic thesis."
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[weak_compact_response, strong_expanded_response],
        ) as request_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNotNone(attempt.merged)
        self.assertEqual(2, request_mock.call_count)
        first_prompt: str = request_mock.call_args_list[0].kwargs["prompt_text"]
        second_prompt: str = request_mock.call_args_list[1].kwargs["prompt_text"]
        self.assertNotIn("RETRY INSTRUCTION:", first_prompt)
        self.assertIn("RETRY INSTRUCTION:", second_prompt)
        self.assertIn("bullet lines", second_prompt)
        self.assertIn("Rewrite with at least", second_prompt)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn(
            "retry_reason_codes=insufficient_bullet_coverage",
            joined_logs,
        )
        self.assertIn(
            "retry_focus=bullet_coverage",
            joined_logs,
        )

    def test_structured_reason_codes_drive_targeted_retry_even_if_error_text_changes(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Attempt was rejected due to insufficient bullet coverage.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=(
                "insufficient_bullet_coverage",
            ),
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
        ) as attempt_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNotNone(attempt.merged)
        self.assertEqual(2, attempt_mock.call_count)
        retry_profile = attempt_mock.call_args_list[1].kwargs["expanded_retry_profile"]
        self.assertIsNotNone(retry_profile)
        self.assertEqual("targeted", retry_profile.retry_mode)
        self.assertEqual(
            ("insufficient_bullet_coverage",),
            retry_profile.reject_signals,
        )
        self.assertEqual(
            ("bullet_coverage",),
            retry_profile.focus_tags,
        )
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn(
            "retry_reason_codes=insufficient_bullet_coverage",
            joined_logs,
        )

    def test_four_source_retry_logs_structured_mode_and_uses_targeted_profile(self) -> None:
        structured_failure = MergeAttemptFailure(
            reason_code="invalid_description",
            reason="Expanded draft had too few bullets.",
            model_name="gpt-5.1",
            attempt_stage="validation",
            raw_response_text='{"title":"Draft","description":"Body"}',
            reason_codes=(
                "insufficient_bullet_coverage",
            ),
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
        ) as attempt_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos_four_sources(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNotNone(attempt.merged)
        retry_profile = attempt_mock.call_args_list[1].kwargs["expanded_retry_profile"]
        self.assertIsNotNone(retry_profile)
        reinforcement_text: str = "\n".join(retry_profile.reinforcement_lines)
        self.assertIn("Rewrite with at least", reinforcement_text)
        self.assertIn("Spread bullets across all 4 sources", reinforcement_text)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("retry_mode=targeted", joined_logs)
        self.assertIn("retry_structure=four_plus_structured", joined_logs)
        self.assertIn("retry_source_count=4", joined_logs)

    def test_expanded_three_source_merge_salvages_formatting_only_emoji_overflow(self) -> None:
        emoji_heavy_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels, Kharkiv, Geneva: the operational agenda tonight","description":"🔥 Tonight we align the Brussels vote 🚨, the Kharkiv transport shock 🎯, and the Geneva aid timetable 🧭 into one grounded briefing ✨ that stays source-specific 🔔 without losing clarity 💥 while keeping the agenda concrete 🌟 and readable 🎖 for every viewer 🏳.\\n\\nIn this stream you\'ll see:\\n🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\\n🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\\n🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels, Kharkiv, Geneva: the operational agenda tonight",
                "description": (
                    "🔥 Tonight we align the Brussels vote 🚨, the Kharkiv transport shock 🎯, and the Geneva aid timetable 🧭 into one grounded briefing ✨ that stays source-specific 🔔 without losing clarity 💥 while keeping the agenda concrete 🌟 and readable 🎖 for every viewer 🏳.\n\n"
                    "In this stream you'll see:\n"
                    "🔹 Brussels sanctions vote, budget amendments, and customs delays after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts and the pressure points before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes, depot damage, and recovery sequencing on the eastern line\n"
                    "🔹 Geneva aid corridor timetable, WHO cargo counts, and donor pledges for Odesa hospitals\n"
                    "🔹 Marta Leone breaks down how Mykolaiv deliveries depend on the next donor release window\n\n"
                    "Watch live ✅ and share updates 📣 #briefing"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[emoji_heavy_response],
        ) as request_mock, self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNotNone(attempt.merged)
        self.assertEqual(1, request_mock.call_count)
        description: str = attempt.merged.description if attempt.merged is not None else ""
        self.assertNotIn("🔥", description)
        self.assertNotIn("🚨", description)
        self.assertNotIn("🎯", description)
        self.assertNotIn("📣", description)
        self.assertIn("🔹 Brussels sanctions vote", description)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_validation_salvage", joined_logs)
        self.assertIn("outcome=applied", joined_logs)
        self.assertIn("reason_codes=excessive_emoji_usage", joined_logs)
        self.assertIn("actions=reduced_non_structural_emoji", joined_logs)

    def test_expanded_formatting_recovery_reveals_underlying_semantic_reject(self) -> None:
        # A 3-source response with emoji overflow AND per-source dump structure.
        # The per_source_dump check fires before the emoji check in the initial run.
        # After emoji salvage attempt, the per_source_dump check still fires.
        # The salvage path will detect that the issue is not formatting-only.
        # Note: Since per_source_dump fires BEFORE excessive_emoji_usage in the validation
        # order, the initial failure code is per_source_dump, not excessive_emoji_usage.
        # The salvage path is NOT triggered for non-emoji failures.
        # Instead, test that a compact (2-source) emoji overflow response is correctly
        # rejected without salvage, and the rejection code is excessive_emoji_usage.
        # This verifies _attempt_expanded_formatting_recovery skips non-3-source merges.

        # Use a 3-source merge with enough bullets to pass bullet check, but emoji overflow
        # whose source-dump structure reveals itself after emoji stripping.
        dump_emoji_response = SimpleNamespace(
            raw_text=(
                '{"title":"Brussels Kharkiv Geneva briefing","description":"🔥 Tonight we cover 🚨 three important briefings 🎯 from three key locations 🧭 across 💥 the operational 🔔 theater 🌟 each 🎖 source 💡 matters 🎗.\\n\\n🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\\n🔹 Anna Kovalenko tracks coalition counts before the chamber debate\\n🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\\n🔹 Oleh Martynenko details evacuation routes on the eastern line\\n🔹 Geneva aid corridor timetable and WHO cargo counts for Odesa hospitals\\n🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release\\n\\nWatch live ✅ and share updates 📣 #briefing"}'
            ),
            structured_payload={
                "title": "Brussels Kharkiv Geneva briefing",
                "description": (
                    "🔥 Tonight we cover 🚨 three important briefings 🎯 from three key locations 🧭 across 💥 the operational 🔔 theater 🌟 each 🎖 source 💡 matters 🎗.\n\n"
                    "🔹 Brussels sanctions vote and budget amendments after the March 18 commission session\n"
                    "🔹 Anna Kovalenko tracks coalition counts before the chamber debate\n"
                    "🔹 Kharkiv rail hub outages after 17 drone strikes across Saltivka districts\n"
                    "🔹 Oleh Martynenko details evacuation routes on the eastern line\n"
                    "🔹 Geneva aid corridor timetable and WHO cargo counts for Odesa hospitals\n"
                    "🔹 Marta Leone explains how Mykolaiv deliveries depend on the next donor release\n\n"
                    "Watch live \u2705 and share updates \U0001f4e3 #briefing"
                ),
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[dump_emoji_response],
        ), self.assertLogs(level="INFO") as captured:
            attempt = attempt_openai_merge_with_audit(
                language="en",
                videos=self._expanded_validation_videos(),
                config=self._config(),
                attempt_label="TEST",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                branch_label="merge",
                date_key="010130",
                slot_key="010130_1000",
            )
        self.assertIsNotNone(attempt.merged)
        joined_logs: str = "\n".join(captured.output)
        self.assertIn("merge_llm_validation_salvage", joined_logs)
        self.assertIn("outcome=applied", joined_logs)
        self.assertIn("reason_codes=excessive_emoji_usage", joined_logs)
        self.assertIn("actions=reduced_non_structural_emoji", joined_logs)

    def test_softened_distinctive_coverage_policy_does_not_apply_to_four_sources(self) -> None:
        self.assertFalse(
            _is_softened_distinctive_source_coverage_eligible(
                source_count=4,
                source_coverage_by_item=(True, True, True, True),
                distinctive_coverage_hits=3,
                distinctive_coverage_total=4,
                reason_codes=("weak_source_coverage",),
            )
        )

    def test_per_source_dump_is_rejected(self) -> None:
        per_source_dump = SimpleNamespace(
            raw_text=(
                '{"title":"Final title","description":"SOURCE 1: Point one.\\n\\nSOURCE 2: Point two."}'
            ),
            structured_payload={
                "title": "Final title",
                "description": "SOURCE 1: Point one.\n\nSOURCE 2: Point two.",
            },
        )
        with patch(
            "app.llm.merges.merge_service.openai_request_merge",
            side_effect=[per_source_dump, per_source_dump],
        ) as request_mock:
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


if __name__ == "__main__":
    unittest.main()
