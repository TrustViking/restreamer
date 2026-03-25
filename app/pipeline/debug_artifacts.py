from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from app.core.branching import BRANCH_MERGE, BRANCH_NOMERGE
from app.core.models import LanguageMergeAttempt
from app.observability.runtime_analytics import get_branch_date_summary
from app.paths.name_builder import NamePathBuilder

from .slot_processing import SlotProcessResult


class DebugArtifactWriter:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        name_builder: NamePathBuilder,
    ) -> None:
        self._logger = logger
        self._name_builder = name_builder

    def write_merge_reject_artifacts(
        self,
        *,
        slot_results: List[SlotProcessResult],
        date_key: str,
        processing_mode: str,
        branch_label: str,
    ) -> None:
        if branch_label != BRANCH_MERGE:
            return
        for slot_result in slot_results:
            for language, merge_attempt in slot_result.merge_audit_by_language.items():
                if not self._should_write_merge_reject_debug_artifact(
                    merge_attempt=merge_attempt
                ):
                    continue
                source_count: int = len(slot_result.language_groups.get(language, ()))
                json_path = self._name_builder.build_merge_reject_debug_json_path(
                    date_key=date_key,
                    slot_key=slot_result.slot_key,
                    language=language,
                    processing_mode=processing_mode,
                    source_count=source_count,
                )
                if json_path is None:
                    continue
                artifact_payload: Dict[str, Any] = {
                    "case_metadata": {
                        "date_key": date_key,
                        "slot_key": slot_result.slot_key,
                        "language": language,
                        "source_count": source_count,
                        "merge_mode": "expanded" if source_count >= 3 else "compact",
                        "processing_mode": processing_mode,
                        "publish_source_label": str(
                            merge_attempt.publish_source_label or ""
                        ).strip(),
                    },
                    "attempts": [
                        {
                            "attempt_index": rejected_attempt.attempt_index,
                            "model": rejected_attempt.model_name,
                            "reject_reasons": list(rejected_attempt.reject_reasons),
                            "title": rejected_attempt.title,
                            "description": rejected_attempt.description,
                            "raw_response_text": rejected_attempt.raw_response_text,
                        }
                        for rejected_attempt in merge_attempt.rejected_attempts
                    ],
                }
                json_path.parent.mkdir(parents=True, exist_ok=True)
                json_path.write_text(
                    json.dumps(artifact_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._logger.info(
                    '[%s] merge_reject_debug_json_written date_key=%s slot_key=%s language=%s attempts=%d path="%s"',
                    branch_label,
                    date_key,
                    slot_result.slot_key,
                    language,
                    len(merge_attempt.rejected_attempts),
                    str(json_path),
                )

    def log_audit_branch_compare(self, *, date_key: str) -> None:
        merge_state = get_branch_date_summary(date_key=date_key, branch_label=BRANCH_MERGE)
        nomerge_state = get_branch_date_summary(date_key=date_key, branch_label=BRANCH_NOMERGE)
        if merge_state is None and nomerge_state is None:
            return
        comparison_status: str = "complete" if merge_state is not None and nomerge_state is not None else "incomplete"
        log_method = self._logger.info if comparison_status == "complete" else self._logger.debug
        log_method(
            "audit_branch_compare date_key=%s merge_executed=%s nomerge_executed=%s merge_doc_created=%s nomerge_doc_created=%s merge_telegram_sent=%s nomerge_telegram_sent=%s merge_contract_failures=%d nomerge_contract_failures=%d merge_models_used=%s nomerge_models_used=%s comparison_status=%s",
            date_key,
            "yes" if merge_state is not None else "no",
            "yes" if nomerge_state is not None else "no",
            "yes" if merge_state is not None and merge_state.docs_created > 0 else "no",
            "yes" if nomerge_state is not None and nomerge_state.docs_created > 0 else "no",
            "yes" if merge_state is not None and merge_state.telegram_sent > 0 else "no",
            "yes" if nomerge_state is not None and nomerge_state.telegram_sent > 0 else "no",
            merge_state.contract_failures if merge_state is not None else 0,
            nomerge_state.contract_failures if nomerge_state is not None else 0,
            ",".join(sorted(merge_state.models_used)) if merge_state is not None and merge_state.models_used else "none",
            ",".join(sorted(nomerge_state.models_used)) if nomerge_state is not None and nomerge_state.models_used else "none",
            comparison_status,
        )

    def _should_write_merge_reject_debug_artifact(
        self,
        *,
        merge_attempt: LanguageMergeAttempt,
    ) -> bool:
        return (
            str(merge_attempt.publish_source_label or "").strip() == "merge_failed"
            and bool(merge_attempt.rejected_attempts)
        )
