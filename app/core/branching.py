from __future__ import annotations

BRANCH_NOMERGE: str = "nomerge"
BRANCH_MERGE_MAIN: str = "merge_main"
BRANCH_MERGE_MAIN_FALLBACK_PACKAGING: str = "merge_main_fallback_packaging"


def audit_branch_labels(*, audit_mode: str) -> list[str]:
    normalized_audit_mode: str = str(audit_mode or "").strip().lower()
    if normalized_audit_mode == "unite":
        return [
            BRANCH_NOMERGE,
            BRANCH_MERGE_MAIN,
            BRANCH_MERGE_MAIN_FALLBACK_PACKAGING,
        ]
    if normalized_audit_mode == "merge":
        return [BRANCH_MERGE_MAIN, BRANCH_MERGE_MAIN_FALLBACK_PACKAGING]
    return [BRANCH_NOMERGE]
