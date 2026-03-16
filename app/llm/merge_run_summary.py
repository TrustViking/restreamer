from __future__ import annotations

from dataclasses import dataclass
import logging


@dataclass
class MergeRunSummary:
    merge_success: int = 0
    validation_rejected: int = 0
    retry_used: int = 0
    final_failure: int = 0
    paragraph_recovery_used: int = 0
    real_merge_blocks: int = 0
    merge_candidate_blocks: int = 0
    fallback_merge_blocks: int = 0
    full_merge_artifacts: int = 0
    partial_merge_artifacts: int = 0

    @property
    def structured_ok(self) -> int:
        return self.merge_success

    @property
    def structured_failed(self) -> int:
        return self.validation_rejected

    @property
    def repair_used(self) -> int:
        return self.retry_used

    def record_merge_success(self) -> None:
        self.merge_success += 1

    def record_validation_rejected(self) -> None:
        self.validation_rejected += 1

    def record_retry_used(self) -> None:
        self.retry_used += 1

    def record_final_failure(self) -> None:
        self.final_failure += 1

    def record_paragraph_recovery_used(self) -> None:
        self.paragraph_recovery_used += 1

    def record_real_merge_block(self) -> None:
        self.real_merge_blocks += 1

    def record_merge_candidate_block(self) -> None:
        self.merge_candidate_blocks += 1

    def record_fallback_merge_block(self) -> None:
        self.fallback_merge_blocks += 1

    def record_full_merge_artifact(self) -> None:
        self.full_merge_artifacts += 1

    def record_partial_merge_artifact(self) -> None:
        self.partial_merge_artifacts += 1

    @property
    def had_real_merge_blocks(self) -> bool:
        return self.real_merge_blocks > 0

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info(
            "merge_run_summary merge_success=%d validation_rejected=%d retry_used=%d final_failure=%d paragraph_recovery_used=%d real_merge_blocks=%d merge_candidate_blocks=%d fallback_merge_blocks=%d full_merge_artifacts=%d partial_merge_artifacts=%d had_real_merge_blocks=%s",
            self.merge_success,
            self.validation_rejected,
            self.retry_used,
            self.final_failure,
            self.paragraph_recovery_used,
            self.real_merge_blocks,
            self.merge_candidate_blocks,
            self.fallback_merge_blocks,
            self.full_merge_artifacts,
            self.partial_merge_artifacts,
            "yes" if self.had_real_merge_blocks else "no",
        )
