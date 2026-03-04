from __future__ import annotations

from dataclasses import dataclass
import logging


@dataclass
class MergeRunSummary:
    primary_success: int = 0
    validation_rejected: int = 0
    primary_retry_used: int = 0
    fallback_success: int = 0
    final_failure: int = 0
    paragraph_recovery_used: int = 0

    @property
    def structured_ok(self) -> int:
        return self.primary_success

    @property
    def structured_failed(self) -> int:
        return self.validation_rejected

    @property
    def plain_fallback_ok(self) -> int:
        return self.fallback_success

    @property
    def repair_used(self) -> int:
        return self.primary_retry_used

    def record_primary_success(self) -> None:
        self.primary_success += 1

    def record_validation_rejected(self) -> None:
        self.validation_rejected += 1

    def record_primary_retry_used(self) -> None:
        self.primary_retry_used += 1

    def record_fallback_success(self) -> None:
        self.fallback_success += 1

    def record_final_failure(self) -> None:
        self.final_failure += 1

    def record_paragraph_recovery_used(self) -> None:
        self.paragraph_recovery_used += 1

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info(
            "merge_run_summary primary_success=%d validation_rejected=%d primary_retry_used=%d fallback_success=%d final_failure=%d paragraph_recovery_used=%d",
            self.primary_success,
            self.validation_rejected,
            self.primary_retry_used,
            self.fallback_success,
            self.final_failure,
            self.paragraph_recovery_used,
        )
