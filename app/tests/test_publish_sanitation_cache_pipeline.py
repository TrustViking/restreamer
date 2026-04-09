from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from app.core.models import MergedLanguageContent, SanitizedPublishBlock
from app.pipeline.slot_processing import SlotProcessResult


class TestSanitationCalledOncePerLanguage:
    """Verify build_sanitized_merged_publication_payload is called
    exactly once per merge-language inside process_slot."""

    def _make_video(self, *, language: str, row: int) -> MagicMock:
        from datetime import datetime

        video: MagicMock = MagicMock()
        video.metadata = MagicMock()
        video.metadata.description = f"Description for row {row}"
        video.metadata.title = f"Title {row}"
        video.metadata.language = language
        video.language = language
        video.forced_block_language = None
        video.normalized_link = f"https://youtu.be/VIDEO{row:03d}AAAA"
        video.row_number = row
        video.date_display = "10.04.2026"
        dt: datetime = datetime(2026, 4, 10, 18, 0, tzinfo=ZoneInfo("Europe/Kyiv"))
        video.scheduled_at_kiev = dt
        video.local_thumbnail_path = None
        video.thumbnail = MagicMock()
        return video

    def _make_merged_content(self, *, language: str) -> MergedLanguageContent:
        return MergedLanguageContent(
            title=f"Merged title {language}",
            description=f"Merged description {language}",
            title_audit=f"Merged title {language}",
            description_audit=f"Merged description {language}",
        )

    def _make_payload_stub(self, *, language: str) -> SimpleNamespace:
        return SimpleNamespace(
            title_text=f"Sanitized title {language}",
            description_text=f"Sanitized description {language}",
            block_generation_mode="real_merge",
            has_publish_stage_duplicate=False,
            has_publish_stage_opener_cta=False,
        )

    def test_sanitation_called_once_per_language_in_process_slot(self) -> None:
        """Mock the heavy merge step so process_slot only runs the
        sanitation caching loop. Verify call count = 1 per language."""
        video_ru_1: MagicMock = self._make_video(language="ru", row=5)
        video_ru_2: MagicMock = self._make_video(language="ru", row=6)

        merged_ru: MergedLanguageContent = self._make_merged_content(language="ru")

        call_log: list[str] = []

        def _tracking_side_effect(**kwargs: object) -> SimpleNamespace:
            lang: str = str(kwargs.get("language", "?"))
            call_log.append(lang)
            return self._make_payload_stub(language=lang)

        from app.llm.merges.merge_run_summary import MergeRunSummary

        summary: MergeRunSummary = MergeRunSummary()

        config: MagicMock = MagicMock()
        config.llm.source_desc_max_chars = 2000
        config.llm.provider = "openai"
        config.google.form_url = "https://example.com"
        config.google.contacts = "test"
        config.templates = MagicMock()

        with (
            patch(
                "app.pipeline.slot_processing.attempt_llm_merge_with_audit",
            ) as mock_merge,
            patch(
                "app.pipeline.slot_processing.build_sanitized_merged_publication_payload",
                side_effect=_tracking_side_effect,
            ) as mock_sanitation,
            patch(
                "app.pipeline.slot_processing.enforce_openai_merged_paragraphs",
                side_effect=lambda language, merged_content, videos, config, no_description_text, merge_run_summary, branch_label, date_key, slot_key: merged_content,
            ),
            patch(
                "app.pipeline.slot_processing.build_header_context",
                return_value={
                    "time_cet": "17:00",
                    "time_kiev": "18:00",
                    "time_gmt": "15:00",
                    "date": "10.04.2026",
                    "form_url": "https://example.com",
                    "contacts": "test",
                },
            ),
            patch(
                "app.pipeline.slot_processing.resolve_effective_llm_model",
                return_value="gpt-5.2",
            ),
            patch(
                "app.pipeline.slot_processing.record_branch_model_used",
            ),
            patch(
                "app.pipeline.slot_processing.record_slot_total_ms",
            ),
            patch(
                "app.pipeline.slot_processing.log_stage_timing",
            ),
        ):
            merge_attempt_mock: MagicMock = MagicMock()
            merge_attempt_mock.merged = merged_ru
            merge_attempt_mock.model_name = "gpt-5.2"
            merge_attempt_mock.used_model_names = ("gpt-5.2",)
            merge_attempt_mock.error_summary = None
            merge_attempt_mock.language = "ru"
            merge_attempt_mock.title_source = "openai_main"
            merge_attempt_mock.hook_source = "openai_main"
            merge_attempt_mock.hashtags_source = "openai_main"
            merge_attempt_mock.body_source = "main_merge"
            merge_attempt_mock.block_generation_mode = "real_merge"
            merge_attempt_mock.salvaged_title = None
            mock_merge.return_value = merge_attempt_mock

            from app.pipeline.slot_processing import process_slot

            result: SlotProcessResult = process_slot(
                logger=logging.getLogger("test"),
                config=config,
                videos=[video_ru_1, video_ru_2],
                date_key="100426",
                slot_time_key="1800",
                llm_merge_enabled=True,
                cet_tz=ZoneInfo("Europe/Berlin"),
                merge_run_summary=summary,
                branch_label="merge",
            )

        assert mock_sanitation.call_count == 1, (
            f"Expected build_sanitized_merged_publication_payload called once, "
            f"got {mock_sanitation.call_count}"
        )
        assert call_log == ["ru"], f"Expected ['ru'], got {call_log}"
        assert "ru" in result.sanitized_blocks
        assert result.sanitized_blocks["ru"].title_text == "Sanitized title ru"
        assert result.sanitized_blocks["ru"].is_blocked is False


class TestDocHelpersCacheHitSkipsSanitation:
    """Verify _build_language_table_rows does NOT call
    build_sanitized_merged_publication_payload when sanitized_block is provided."""

    def test_cache_hit_skips_sanitation(self) -> None:
        from app.publish.doc_helpers import _build_language_table_rows

        cached: SanitizedPublishBlock = SanitizedPublishBlock(
            language="ru",
            title_text="Cached title",
            description_text="Cached description",
            block_generation_mode="real_merge",
            is_blocked=False,
        )

        templates_mock: MagicMock = MagicMock()
        templates_mock.google_doc_table_labels = {
            "ru": ["НАЗВАНИЕ", "ОПИСАНИЕ", "ПРЕВЬЮ"],
        }

        with patch(
            "app.publish.doc_helpers.build_sanitized_merged_publication_payload",
        ) as mock_sanitation:
            rows: list[tuple[str, bool]] = _build_language_table_rows(
                language="ru",
                videos=[],
                merged_content=None,
                merge_attempt=None,
                time_display="18:00",
                templates=templates_mock,
                artifact_status="full",
                sanitized_block=cached,
            )

        mock_sanitation.assert_not_called()
        row_texts: list[str] = [str(row[0]) for row in rows]
        assert any("Cached title" in text for text in row_texts), (
            f"Cached title not found in rows: {row_texts}"
        )
        assert any("Cached description" in text for text in row_texts), (
            f"Cached description not found in rows: {row_texts}"
        )
