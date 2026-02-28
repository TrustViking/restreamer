from __future__ import annotations

from dataclasses import dataclass
import logging


@dataclass
class MergeRunSummary:
    structured_ok: int = 0
    structured_failed: int = 0
    plain_fallback_ok: int = 0
    repair_used: int = 0
    paragraph_recovery_used: int = 0

    def record_structured_ok(self) -> None:
        self.structured_ok += 1

    def record_structured_failed(self) -> None:
        self.structured_failed += 1

    def record_plain_fallback_ok(self) -> None:
        self.plain_fallback_ok += 1

    def record_repair_used(self) -> None:
        self.repair_used += 1

    def record_paragraph_recovery_used(self) -> None:
        self.paragraph_recovery_used += 1

    def log_summary(self, logger: logging.Logger) -> None:
        logger.info(
            "merge_run_summary structured_ok=%d structured_failed=%d plain_fallback_ok=%d repair_used=%d paragraph_recovery_used=%d",
            self.structured_ok,
            self.structured_failed,
            self.plain_fallback_ok,
            self.repair_used,
            self.paragraph_recovery_used,
        )
