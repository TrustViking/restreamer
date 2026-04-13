from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Tuple
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE
from app.core.models import LanguageMergeAttempt, PlannedVideo
from app.core.text_utils import format_date_key_for_display
from app.llm.merges.merge_run_summary import MergeRunSummary
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.observability.runtime_analytics import (
    log_date_started,
    log_docs_publish_summary,
    log_merge_summary,
    log_stage_timing,
    log_telegram_publish_summary,
    record_branch_completed,
    record_branch_failed,
    record_branch_model_used,
    record_branch_started,
    record_date_branch_execution,
    record_docs_created,
    record_docs_failed,
    record_stage_duration,
    record_telegram_failed,
    record_telegram_sent,
    record_telegram_skipped,
)
from app.observability.startup_health import log_section
from app.paths.name_builder import NamePathBuilder
from app.planning import (
    language_sort_key,
    planned_video_block_language,
    planned_video_time_key,
)
from app.planning.slots import format_time_key_for_display
from app.telegram.bot_client import TelegramBotClient

from .daily_doc_publish import publish_daily_document
from .daily_telegram_publish import publish_daily_telegram
from .debug_artifacts import DebugArtifactWriter
from .operator_notifier import OperatorNotifier
from .runtime_services import BatchServices
from .slot_processing import SlotProcessResult, process_slot, resolve_merge_artifact_status

if TYPE_CHECKING:
    from .batch_runner import AuditBranch


@dataclass(frozen=True)
class MergePublishDecision:
    create_doc: bool
    publish_telegram: bool
    send_info_message: bool
    info_message_text: str
    decision_reason: str


def _resolve_merge_publish_decision(
    *,
    branch_name: str,
    merge_artifact_status: str,
    slot_results: List[SlotProcessResult],
    date_key: str,
    fallback_merge_targets: List[str],
    merge_audit_by_language: Dict[str, LanguageMergeAttempt],
) -> MergePublishDecision:
    if branch_name != BRANCH_MERGE:
        return MergePublishDecision(
            create_doc=True,
            publish_telegram=True,
            send_info_message=False,
            info_message_text="",
            decision_reason="nomerge_branch",
        )

    if merge_artifact_status in {"full", "partial"}:
        return MergePublishDecision(
            create_doc=True,
            publish_telegram=True,
            send_info_message=False,
            info_message_text="",
            decision_reason=f"{merge_artifact_status}_merge_success",
        )

    if merge_artifact_status == "fallback_only":
        failure_details: List[str] = []
        for target in fallback_merge_targets:
            parts: List[str] = target.split(":", maxsplit=1)
            slot_key: str = parts[0] if parts else target
            language: str = parts[1] if len(parts) > 1 else "unknown"
            attempt: LanguageMergeAttempt | None = merge_audit_by_language.get(language)
            reasons_text: str = "unknown"
            attempts_text: str = ""
            if attempt is not None:
                reasons_list: List[str] = list(attempt.validation_reasons or [])
                if reasons_list:
                    reasons_text = ", ".join(reasons_list)
                total_attempts: int = 1 + len(attempt.rejected_attempts)
                attempts_text = f", {total_attempts} попыток"
            failure_details.append(
                f"• {language} ({slot_key}) - забраковано: {reasons_text}{attempts_text}"
            )
        details_text: str = "\n".join(failure_details) if failure_details else "• детали недоступны"
        info_text: str = (
            "⚠️ Merge: объединение не удалось\n\n"
            f"{details_text}\n\n"
            "Документ не создан."
        )
        return MergePublishDecision(
            create_doc=False,
            publish_telegram=False,
            send_info_message=True,
            info_message_text=info_text,
            decision_reason="merge_fallback_only_rejected",
        )

    all_merge_skipped_languages: List[str] = []
    has_any_items: bool = False
    for slot in slot_results:
        all_merge_skipped_languages.extend(slot.merge_skipped_languages)
        if slot.day_videos or any(slot.language_groups.values()):
            has_any_items = True
    if has_any_items and all_merge_skipped_languages:
        info_text = (
            "ℹ️ Режим: merge. Для этой даты объединение не требуется. "
            "Ниже — вывод без объединения 👇"
        )
        return MergePublishDecision(
            create_doc=True,
            publish_telegram=True,
            send_info_message=True,
            info_message_text=info_text,
            decision_reason="merge_insufficient_publish_as_nomerge",
        )

    return MergePublishDecision(
        create_doc=False,
        publish_telegram=False,
        send_info_message=False,
        info_message_text="",
        decision_reason="no_data",
    )


class BranchExecutor:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        config: AppConfig,
        telegram_client: TelegramBotClient,
        name_builder: NamePathBuilder,
        kiev_tz: ZoneInfo,
        cet_tz: ZoneInfo,
        debug_artifact_writer: DebugArtifactWriter,
        notifier: OperatorNotifier,
    ) -> None:
        self._logger = logger
        self._config = config
        self._telegram_client = telegram_client
        self._name_builder = name_builder
        self._kiev_tz = kiev_tz
        self._cet_tz = cet_tz
        self._debug_artifact_writer = debug_artifact_writer
        self._notifier: OperatorNotifier = notifier

    def execute(
        self,
        *,
        services: BatchServices,
        branch: AuditBranch,
        date_key: str,
        date_videos: List[PlannedVideo],
        dry_run: bool,
        merge_run_summary: MergeRunSummary,
        run_id: str = "",
        stage_index: int = 1,
        stage_count: int = 1,
    ) -> None:
        slot_processing_started_at: float = time.perf_counter()
        slots_by_time_and_language: Dict[Tuple[str, str], List[PlannedVideo]] = (
            self._group_date_videos_by_time_and_language(
            date_videos=date_videos
        )
        )
        if stage_count > 1 and stage_index > 1:
            self._notifier.emit(
                f"⏭ Переход к этапу {branch.name}",
                to_telegram=not dry_run,
            )
        self._notifier.emit(
            f"▶️ Этап {stage_index}/{stage_count}: {branch.name}, дата {format_date_key_for_display(date_key)}",
            to_telegram=not dry_run,
        )
        self._logger.info(
            "[%s] Date branch started: %s slots=%d items=%d",
            branch.name,
            date_key,
            len(slots_by_time_and_language),
            len(date_videos),
        )
        record_branch_started(branch_label=branch.name)
        record_date_branch_execution(date_key=date_key, branch_label=branch.name)
        log_date_started(
            logger=self._logger,
            date_key=f"{date_key}/{branch.name}",
            slot_count=len(slots_by_time_and_language),
            item_count=len(date_videos),
        )

        merge_snapshot_before: tuple[int, int, int, int, int, int, int, int, int] = (
            merge_run_summary.merge_success,
            merge_run_summary.validation_rejected,
            merge_run_summary.retry_used,
            merge_run_summary.final_failure,
            merge_run_summary.paragraph_recovery_used,
            merge_run_summary.real_merge_blocks,
            merge_run_summary.merge_candidate_blocks,
            merge_run_summary.fallback_merge_blocks,
            merge_run_summary.partial_merge_artifacts,
        )
        slot_results: List[SlotProcessResult] = []
        ordered_slot_keys: List[Tuple[str, str]] = sorted(
            slots_by_time_and_language.keys(),
            key=lambda key: (key[0], language_sort_key(key[1])),
        )
        slot_total: int = len(ordered_slot_keys)
        for slot_index, (slot_time_key, slot_language) in enumerate(
            ordered_slot_keys,
            start=1,
        ):
            slot_video_count: int = len(
                slots_by_time_and_language[(slot_time_key, slot_language)]
            )
            self._notifier.emit(
                f"⏳ Слот {slot_index}/{slot_total} {format_time_key_for_display(slot_time_key)} {slot_language}: обработка {slot_video_count} видео",
                to_telegram=not dry_run,
            )
            try:
                processed_slot_result = process_slot(
                    logger=self._logger,
                    config=self._config,
                    videos=slots_by_time_and_language[(slot_time_key, slot_language)],
                    date_key=date_key,
                    slot_time_key=slot_time_key,
                    llm_merge_enabled=branch.llm_merge_enabled,
                    cet_tz=self._cet_tz,
                    merge_run_summary=merge_run_summary,
                    branch_label=branch.name,
                )
            except LlmModelConfigurationError as error:
                self._logger.error(
                    "[%s] slot_process_fatal_model_config date_key=%s slot_key=%s reason_code=%s model=%s provider=%s reason=%s",
                    branch.name,
                    date_key,
                    f"{date_key}_{slot_time_key}_{slot_language}",
                    error.reason_code,
                    error.model_name or "unknown",
                    error.provider_name or "unknown",
                    error.detail,
                )
                raise
            for merge_attempt in processed_slot_result.merge_audit_by_language.values():
                if str(merge_attempt.model_name or "").strip():
                    record_branch_model_used(
                        date_key=date_key,
                        branch_label=branch.name,
                        model_name=merge_attempt.model_name,
                    )
            slot_results.append(processed_slot_result)
            self._notifier.emit(
                f"✅ Слот {slot_index}/{slot_total} {format_time_key_for_display(slot_time_key)} {slot_language}: готов",
                to_telegram=not dry_run,
            )
        slot_processing_ms: int = int(round((time.perf_counter() - slot_processing_started_at) * 1000.0))
        record_stage_duration(stage_name="slot_processing", elapsed_ms=slot_processing_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="slot_processing",
            elapsed_ms=slot_processing_ms,
            scope="branch",
            branch_label=branch.name,
            date_key=date_key,
        )

        real_merge_blocks: int = sum(slot.real_merge_blocks for slot in slot_results)
        merge_candidate_blocks: int = sum(slot.merge_candidate_blocks for slot in slot_results)
        fallback_merge_blocks: int = sum(slot.fallback_merge_blocks for slot in slot_results)
        merge_artifact_status: str = resolve_merge_artifact_status(
            merge_candidate_blocks=merge_candidate_blocks,
            real_merge_blocks=real_merge_blocks,
            fallback_merge_blocks=fallback_merge_blocks,
        )
        fallback_merge_targets: List[str] = [
            target
            for slot in slot_results
            for target in slot.fallback_merge_targets
        ]
        if branch.name == BRANCH_MERGE:
            if merge_artifact_status == "full":
                merge_run_summary.record_full_merge_artifact()
            elif merge_artifact_status == "partial":
                merge_run_summary.record_partial_merge_artifact()
        merge_snapshot_after: tuple[int, int, int, int, int, int, int, int, int] = (
            merge_run_summary.merge_success,
            merge_run_summary.validation_rejected,
            merge_run_summary.retry_used,
            merge_run_summary.final_failure,
            merge_run_summary.paragraph_recovery_used,
            merge_run_summary.real_merge_blocks,
            merge_run_summary.merge_candidate_blocks,
            merge_run_summary.fallback_merge_blocks,
            merge_run_summary.partial_merge_artifacts,
        )
        log_merge_summary(
            logger=self._logger,
            groups=len(slot_results),
            merge_success=merge_snapshot_after[0] - merge_snapshot_before[0],
            validation_rejected=merge_snapshot_after[1] - merge_snapshot_before[1],
            retry_used=merge_snapshot_after[2] - merge_snapshot_before[2],
            final_failure=merge_snapshot_after[3] - merge_snapshot_before[3],
            paragraph_recovery_used=merge_snapshot_after[4] - merge_snapshot_before[4],
            real_merge_blocks=merge_snapshot_after[5] - merge_snapshot_before[5],
            merge_candidate_blocks=merge_snapshot_after[6] - merge_snapshot_before[6],
            fallback_merge_blocks=merge_snapshot_after[7] - merge_snapshot_before[7],
            partial_merge_artifacts=merge_snapshot_after[8] - merge_snapshot_before[8],
        )
        self._logger.info(
            "[%s] merge_artifact_observability date_key=%s merge_artifact_status=%s merged_blocks=%d fallback_blocks=%d merge_candidate_blocks=%d fallback_targets=%s",
            branch.name,
            date_key,
            merge_artifact_status,
            real_merge_blocks,
            fallback_merge_blocks,
            merge_candidate_blocks,
            ",".join(fallback_merge_targets) or "none",
        )
        self._debug_artifact_writer.write_merge_reject_artifacts(
            slot_results=slot_results,
            date_key=date_key,
            processing_mode=branch.processing_mode,
            branch_label=branch.name,
            run_id=run_id,
        )
        combined_merge_audit: Dict[str, LanguageMergeAttempt] = {}
        for slot in slot_results:
            for language, attempt in slot.merge_audit_by_language.items():
                combined_merge_audit.setdefault(language, attempt)
        publish_decision: MergePublishDecision = _resolve_merge_publish_decision(
            branch_name=branch.name,
            merge_artifact_status=merge_artifact_status,
            slot_results=slot_results,
            date_key=date_key,
            fallback_merge_targets=fallback_merge_targets,
            merge_audit_by_language=combined_merge_audit,
        )
        self._logger.info(
            "[%s] merge_publish_decision date_key=%s create_doc=%s publish_telegram=%s send_info=%s reason=%s",
            branch.name,
            date_key,
            "yes" if publish_decision.create_doc else "no",
            "yes" if publish_decision.publish_telegram else "no",
            "yes" if publish_decision.send_info_message else "no",
            publish_decision.decision_reason,
        )
        merge_status_text: str = ""
        if branch.name == BRANCH_MERGE:
            if publish_decision.decision_reason in {"full_merge_success", "partial_merge_success"}:
                merge_status_text = f"✅ Этап merge, дата {format_date_key_for_display(date_key)}: объединение выполнено."
            elif publish_decision.decision_reason == "no_data":
                merge_status_text = f"ℹ️ Этап merge, дата {format_date_key_for_display(date_key)}: нет данных для объединения."
        if publish_decision.send_info_message and publish_decision.info_message_text:
            self._notifier.emit(
                publish_decision.info_message_text,
                to_telegram=not dry_run,
            )
            self._logger.info(
                "[%s] merge_info_message_emitted date_key=%s reason=%s to_telegram=%s",
                branch.name,
                date_key,
                publish_decision.decision_reason,
                "yes" if not dry_run else "no",
            )
        if not publish_decision.create_doc:
            self._logger.info(
                "[%s] merge_doc_skipped date_key=%s reason=%s",
                branch.name,
                date_key,
                publish_decision.decision_reason,
            )
            log_docs_publish_summary(logger=self._logger, created=0, failed=0)
            log_telegram_publish_summary(logger=self._logger, sent=0, failed=0, skipped=1)
            record_telegram_skipped(count=1, date_key=date_key, branch_label=branch.name)
            if merge_status_text:
                self._notifier.emit(merge_status_text, to_telegram=not dry_run)
            record_branch_completed(branch_label=branch.name)
            return

        log_section(logger=self._logger, title=f"Publish Daily Docs [{branch.name}]")
        doc_publish_started_at: float = time.perf_counter()
        try:
            doc_publish_result = publish_daily_document(
                logger=self._logger,
                config=self._config,
                docs_client=services.docs_client,
                drive_client=services.drive_client,
                report_writer=services.report_writer,
                name_builder=self._name_builder,
                slot_results=slot_results,
                date_key=date_key,
                dry_run=dry_run,
                processing_mode=branch.processing_mode,
                kiev_tz=self._kiev_tz,
                branch_label=branch.name,
            )
        except Exception:
            record_branch_failed(branch_label=branch.name)
            record_docs_failed(count=1, date_key=date_key, branch_label=branch.name)
            log_docs_publish_summary(logger=self._logger, created=0, failed=1)
            raise
        doc_publish_ms: int = int(round((time.perf_counter() - doc_publish_started_at) * 1000.0))
        record_stage_duration(stage_name="doc_publish", elapsed_ms=doc_publish_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="doc_publish",
            elapsed_ms=doc_publish_ms,
            scope="branch",
            branch_label=branch.name,
            date_key=date_key,
        )
        docs_created_count: int = 0 if dry_run else 1
        self._logger.info(
            "[%s] merge_doc_result date_key=%s merge_doc_created=%s reason=%s",
            branch.name,
            date_key,
            "yes" if doc_publish_result.google_doc_created else "no",
            "published" if doc_publish_result.google_doc_created else ("dry_run" if dry_run else "not_created"),
        )
        if doc_publish_result.google_doc_created:
            self._notifier.emit("✅ Док создан", to_telegram=not dry_run)
        elif dry_run:
            self._notifier.emit("✅ Док подготовлен (dry run)", to_telegram=False)
        record_docs_created(count=docs_created_count, date_key=date_key, branch_label=branch.name)
        log_docs_publish_summary(logger=self._logger, created=docs_created_count, failed=0)
        if not publish_decision.publish_telegram:
            self._logger.info(
                "[%s] merge_telegram_skipped date_key=%s reason=%s",
                branch.name,
                date_key,
                publish_decision.decision_reason,
            )
            log_telegram_publish_summary(
                logger=self._logger,
                sent=0,
                failed=0,
                skipped=1,
            )
            record_telegram_skipped(
                count=1,
                date_key=date_key,
                branch_label=branch.name,
            )
            if merge_status_text:
                self._notifier.emit(merge_status_text, to_telegram=not dry_run)
            record_branch_completed(branch_label=branch.name)
            return

        log_section(logger=self._logger, title=f"Publish Daily Telegram [{branch.name}]")
        effective_processing_mode: str = branch.processing_mode
        if publish_decision.decision_reason == "merge_insufficient_publish_as_nomerge":
            effective_processing_mode = "nomerge"
        telegram_publish_started_at: float = time.perf_counter()
        try:
            telegram_result = publish_daily_telegram(
                logger=self._logger,
                config=self._config,
                telegram_client=self._telegram_client,
                templates=self._config.templates,
                slot_results=slot_results,
                date_key=date_key,
                doc_url=doc_publish_result.doc_url,
                header_context=doc_publish_result.header_context,
                dry_run=dry_run,
                processing_mode=effective_processing_mode,
                branch_label=branch.name,
            )
        except Exception:
            record_branch_failed(branch_label=branch.name)
            record_telegram_failed(count=1, date_key=date_key, branch_label=branch.name)
            log_telegram_publish_summary(logger=self._logger, sent=0, failed=1, skipped=0)
            raise
        telegram_publish_ms: int = int(round((time.perf_counter() - telegram_publish_started_at) * 1000.0))
        record_stage_duration(stage_name="telegram_publish", elapsed_ms=telegram_publish_ms)
        log_stage_timing(
            logger=self._logger,
            stage_name="telegram_publish",
            elapsed_ms=telegram_publish_ms,
            scope="branch",
            branch_label=branch.name,
            date_key=date_key,
        )
        record_telegram_sent(count=telegram_result.sent_count, date_key=date_key, branch_label=branch.name)
        record_telegram_failed(count=telegram_result.failed_count, date_key=date_key, branch_label=branch.name)
        record_telegram_skipped(count=telegram_result.skipped_count, date_key=date_key, branch_label=branch.name)
        log_telegram_publish_summary(
            logger=self._logger,
            sent=telegram_result.sent_count,
            failed=telegram_result.failed_count,
            skipped=telegram_result.skipped_count,
        )
        self._notifier.emit(
            f"✅ Telegram-пакет: отправлено {telegram_result.sent_count}",
            to_telegram=False,
        )
        if merge_status_text:
            self._notifier.emit(merge_status_text, to_telegram=not dry_run)
        record_branch_completed(branch_label=branch.name)

    def _group_date_videos_by_time_and_language(
        self,
        *,
        date_videos: List[PlannedVideo],
    ) -> Dict[Tuple[str, str], List[PlannedVideo]]:
        slots_by_time_and_language: Dict[Tuple[str, str], List[PlannedVideo]] = {}
        for item in date_videos:
            key: Tuple[str, str] = (
                planned_video_time_key(item),
                planned_video_block_language(item),
            )
            slots_by_time_and_language.setdefault(key, []).append(item)
        return slots_by_time_and_language
