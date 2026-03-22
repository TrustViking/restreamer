from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from typing import Optional

from app.core.models import MergedLanguageContent
from app.llm.merges.merge_parser import parse_merge_response_or_raise
from app.llm.merges.merge_quality import MergeQualityDiagnostics, normalize_merge_description
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.merges.merge_links import (
    OfficialLinksFillResult,
    OfficialLinksSelection,
)
from app.llm.merges.merge_prompt import MergeContractMode
from app.llm.merges.merge_validation import (
    _has_cta_in_opening_lines_before_hook_or_bullet,
    _looks_like_bad_hook_paragraph,
    _validate_coverage_preserving_merge_or_raise,
)
from app.llm.merges.merge_service import attempt_openai_merge_with_audit
from app.publish import doc_helpers
from app.publish.post_llm_sanitation import (
    MergedPublicationPayload as SanitizedMergedPublicationPayload,
    _final_description_has_opener_cta,
)


class AuditRunRegressionTests(unittest.TestCase):
    def _video(self, *, description: str) -> SimpleNamespace:
        return SimpleNamespace(
            language="ru",
            forced_block_language=None,
            date_display="21.03.2026",
            scheduled_at_kiev=datetime(2026, 3, 21, 20, 0, 0),
            metadata=SimpleNamespace(
                title="Источник",
                description=description,
                url="https://youtube.com/watch?v=abcdefghijk",
            ),
            normalized_link="https://youtube.com/watch?v=abcdefghijk",
            original_link="https://youtube.com/watch?v=abcdefghijk",
        )

    def test_hook_echo_repair_propagates_to_description_audit(self) -> None:
        first_paragraph: str = (
            "Февральский эфир показал, как один тезис многократно повторяется в разных формулировках, "
            "и это требует аккуратной проверки фактов перед выводами о последствиях."
        )
        second_paragraph: str = (
            "Февральский эфир показал, как один тезис многократно повторяется в разных формулировках, "
            "но ниже идут новые подтверждения из источников без рекламного тона.\n"
            "🔹 Первый подтвержденный факт с датой и участниками обсуждения.\n"
            "🔹 Второй подтвержденный факт с последствиями для повестки."
        )
        third_paragraph: str = (
            "Третий абзац добавляет отдельный контекст и не повторяет открывающий тезис."
        )
        original_description: str = (
            f"{first_paragraph}\n\n{second_paragraph}\n\n{third_paragraph}"
        )
        merged_content: MergedLanguageContent = MergedLanguageContent(
            title="Итоговый заголовок",
            description=original_description,
            description_selected=original_description,
            description_audit=original_description,
        )
        merge_quality_diagnostics: MergeQualityDiagnostics = normalize_merge_description(
            description=original_description,
            language="ru",
            source_texts=(),
        ).diagnostics
        official_links_selection: OfficialLinksSelection = OfficialLinksSelection(
            found_in_sources=0,
            kept_links=(),
        )
        official_links_fill: OfficialLinksFillResult = OfficialLinksFillResult(
            description=original_description,
            links_in_output=0,
            fill_applied=False,
        )

        repaired_diagnostics, repaired_content = _validate_coverage_preserving_merge_or_raise(
            merged_content=merged_content,
            videos=[],
            official_links_selection=official_links_selection,
            official_links_fill=official_links_fill,
            merge_quality=merge_quality_diagnostics,
        )
        _ = repaired_diagnostics

        self.assertNotEqual(original_description, repaired_content.description)
        self.assertEqual(repaired_content.description, repaired_content.description_audit)
        second_repaired_paragraph: str = repaired_content.description.split("\n\n")[1]
        self.assertTrue(second_repaired_paragraph.startswith("🔹"))

    def test_soft_cta_opener_is_rejected_by_merge_stage_validator(self) -> None:
        soft_cta_line: str = (
            "Если вы смотрели стрим, напишите, какие эпизоды февраля 2026 года "
            "показались вам самыми показательными."
        )
        opening_paragraphs: list[str] = [
            soft_cta_line,
            "🔹 Первый факт из эфира с проверяемым источником.",
        ]
        merge_stage_detected: bool = _has_cta_in_opening_lines_before_hook_or_bullet(
            opening_paragraphs
        )
        bad_hook_detected: bool = _looks_like_bad_hook_paragraph(soft_cta_line)
        self.assertTrue(merge_stage_detected or bad_hook_detected)

    def test_soft_cta_opener_is_caught_by_publish_stage_check(self) -> None:
        description_text: str = (
            "Если вы смотрели стрим, напишите, какие эпизоды февраля 2026 года "
            "показались вам самыми показательными.\n\n"
            "🔹 Первый факт с подтверждением из источника.\n"
            "🔹 Второй факт с временной привязкой."
        )
        self.assertTrue(_final_description_has_opener_cta(description_text))

    def test_publish_payload_with_duplicate_is_blocked_by_publish_gate(self) -> None:
        blocked_payload: SanitizedMergedPublicationPayload = SanitizedMergedPublicationPayload(
            title_text="Merged title",
            description_text="Merged description",
            block_generation_mode="real_merge",
            has_publish_stage_duplicate=True,
            has_publish_stage_opener_cta=False,
        )
        merged_content: MergedLanguageContent = MergedLanguageContent(
            title="Merged title",
            description="Merged description",
            description_selected="Merged description",
            description_audit="Merged description",
        )
        videos: list[SimpleNamespace] = [self._video(description="Source description.")]

        self.assertTrue(doc_helpers._is_merge_payload_blocked(blocked_payload))
        with patch(
            "app.publish.doc_helpers.build_sanitized_merged_publication_payload",
            return_value=blocked_payload,
        ):
            guarded_payload: Optional[SanitizedMergedPublicationPayload] = (
                doc_helpers._build_guarded_merged_payload(
                target="doc",
                videos=videos,
                merged_content=merged_content,
                merge_attempt=None,
                use_audit_text=True,
            )
            )
        self.assertIsNone(guarded_payload)

    def test_tail_separation_recovery_increments_paragraph_recovery_used(self) -> None:
        recovered_raw_text: str = json.dumps(
            {
                "title": "Recovered title",
                "description": (
                    "Paragraph one explains the key conflict with concrete evidence.\n\n"
                    "Paragraph two keeps the timeline and actors explicit.\n\n"
                "Paragraph three adds downstream effects and risks.\n\n"
                "Paragraph four records practical implications for the audience.\n\n"
                "Paragraph five captures additional verified details from the same broadcast."
            ),
        },
            ensure_ascii=False,
        )
        parsed_content, _ = parse_merge_response_or_raise(
            provider_name="openai",
            model_name="gpt-5.1",
            raw_text=recovered_raw_text,
            max_body_paragraphs=4,
        )
        self.assertTrue(parsed_content.tail_recovery_applied)

        merge_summary: MergeRunSummary = MergeRunSummary()
        config: SimpleNamespace = SimpleNamespace(templates=SimpleNamespace())
        videos: list[SimpleNamespace] = [
            self._video(description="Описание первого источника."),
            self._video(description="Описание второго источника."),
        ]
        contract_mode: MergeContractMode = MergeContractMode(
            mode_label="compact",
            source_count=2,
            bullet_range_label="4-7",
            bullet_range_min=4,
            bullet_range_max=7,
            expanded_structure_enabled=False,
            contract_block="compact contract",
            max_body_paragraphs=4,
        )

        with patch(
            "app.llm.merges.merge_service.resolve_effective_llm_model",
            return_value="gpt-5.1",
        ), patch(
            "app.llm.merges.merge_service.get_llm_provider",
            return_value=SimpleNamespace(name="openai"),
        ), patch(
            "app.llm.merges.merge_prompt._select_merge_contract_mode",
            return_value=contract_mode,
        ), patch(
            "app.llm.merges.merge_service._attempt_merge_once",
            return_value=(parsed_content, recovered_raw_text),
        ):
            attempt = attempt_openai_merge_with_audit(
                language="ru",
                videos=videos,
                config=config,
                attempt_label="AUDIT_REGRESSION",
                summarize_error=lambda error: str(error),
                normalize_youtube_url=lambda url: url,
                no_description_text="no description",
                merge_run_summary=merge_summary,
            )

        self.assertIsNotNone(attempt.merged)
        self.assertEqual(1, merge_summary.paragraph_recovery_used)


if __name__ == "__main__":
    unittest.main()
