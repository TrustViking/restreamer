from __future__ import annotations

BRANCH_NOMERGE: str = "nomerge"
BRANCH_MERGE: str = "merge"


def audit_branch_labels(*, audit_mode: str) -> list[str]:
    normalized_audit_mode: str = str(audit_mode or "").strip().lower()
    if normalized_audit_mode == "audit":
        return [
            BRANCH_NOMERGE,
            BRANCH_MERGE,
        ]
    if normalized_audit_mode == "merge":
        return [BRANCH_MERGE]
    return [BRANCH_NOMERGE]
