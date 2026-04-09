from __future__ import annotations

from app.core.models import SanitizedPublishBlock
from app.pipeline.slot_processing import SlotProcessResult


class TestSlotProcessResultSanitizedBlocks:
    """Verify that SlotProcessResult carries sanitized_blocks field."""

    def test_default_empty(self) -> None:
        result: SlotProcessResult = SlotProcessResult(
            slot_key="100426_1800_ru",
            slot_time_key="1800",
            header_context={},
            day_videos=[],
            language_groups={},
            merged_content_by_language={},
            merge_audit_by_language={},
            real_merge_blocks=0,
        )
        assert result.sanitized_blocks == {}

    def test_carries_cached_block(self) -> None:
        block: SanitizedPublishBlock = SanitizedPublishBlock(
            language="ru",
            title_text="Заголовок",
            description_text="Описание",
            is_blocked=False,
        )
        result: SlotProcessResult = SlotProcessResult(
            slot_key="100426_1800_ru",
            slot_time_key="1800",
            header_context={},
            day_videos=[],
            language_groups={},
            merged_content_by_language={},
            merge_audit_by_language={},
            real_merge_blocks=0,
            sanitized_blocks={"ru": block},
        )
        assert result.sanitized_blocks["ru"].title_text == "Заголовок"
        assert result.sanitized_blocks["ru"].is_blocked is False
