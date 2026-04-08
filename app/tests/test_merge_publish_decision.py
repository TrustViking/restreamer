from __future__ import annotations

from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.pipeline.branch_executor import _resolve_merge_publish_decision
from app.pipeline.slot_processing import SlotProcessResult


def _make_minimal_slot(
    *,
    merge_skipped_languages: tuple[str, ...] = (),
    has_items: bool = True,
    merge_artifact_status: str = "none",
) -> SlotProcessResult:
    language_groups = {
        "uk": [object()] if has_items else [],
        "en": [],
        "ru": [],
        "other": [],
    }
    return SlotProcessResult(
        slot_key="test_1400",
        slot_time_key="1400",
        header_context={},
        day_videos=[],
        language_groups=language_groups,
        merged_content_by_language={},
        merge_audit_by_language={},
        real_merge_blocks=0,
        merge_candidate_blocks=0,
        fallback_merge_blocks=0,
        merge_artifact_status=merge_artifact_status,
        fallback_merge_targets=(),
        merge_skipped_languages=merge_skipped_languages,
    )


def test_nomerge_branch_always_publishes() -> None:
    decision = _resolve_merge_publish_decision(
        branch_name=BRANCH_NOMERGE,
        merge_artifact_status="none",
        slot_results=[],
        date_key="020426",
        fallback_merge_targets=[],
        merge_audit_by_language={},
    )
    assert decision.create_doc is True
    assert decision.publish_telegram is True
    assert decision.send_info_message is False


def test_full_merge_publishes() -> None:
    decision = _resolve_merge_publish_decision(
        branch_name=BRANCH_MERGE,
        merge_artifact_status="full",
        slot_results=[],
        date_key="020426",
        fallback_merge_targets=[],
        merge_audit_by_language={},
    )
    assert decision.create_doc is True
    assert decision.publish_telegram is True


def test_fallback_only_blocks_doc_and_telegram_sends_info() -> None:
    decision = _resolve_merge_publish_decision(
        branch_name=BRANCH_MERGE,
        merge_artifact_status="fallback_only",
        slot_results=[],
        date_key="020426",
        fallback_merge_targets=["020426_1400:uk"],
        merge_audit_by_language={},
    )
    assert decision.create_doc is False
    assert decision.publish_telegram is False
    assert decision.send_info_message is True
    assert "не удалось" in decision.info_message_text


def test_insufficient_descriptions_publishes_as_nomerge_with_info() -> None:
    slot = _make_minimal_slot(
        merge_skipped_languages=("uk",),
        has_items=True,
    )
    decision = _resolve_merge_publish_decision(
        branch_name=BRANCH_MERGE,
        merge_artifact_status="none",
        slot_results=[slot],
        date_key="020426",
        fallback_merge_targets=[],
        merge_audit_by_language={},
    )
    assert decision.create_doc is True
    assert decision.publish_telegram is True
    assert decision.send_info_message is True
    assert "объединение не требуется" in decision.info_message_text


def test_no_data_blocks_everything() -> None:
    slot = _make_minimal_slot(has_items=False)
    decision = _resolve_merge_publish_decision(
        branch_name=BRANCH_MERGE,
        merge_artifact_status="none",
        slot_results=[slot],
        date_key="020426",
        fallback_merge_targets=[],
        merge_audit_by_language={},
    )
    assert decision.create_doc is False
    assert decision.publish_telegram is False
    assert decision.send_info_message is False
