from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

WARNING_CATEGORY_INFORMATIONAL: str = "informational"
WARNING_CATEGORY_OPERATIONAL: str = "operational"


@dataclass
class BranchAnalyticsState:
    started: bool = False
    completed: bool = False
    failed: bool = False


@dataclass
class BranchDateAnalyticsState:
    docs_created: int = 0
    docs_failed: int = 0
    telegram_sent: int = 0
    telegram_failed: int = 0
    telegram_skipped: int = 0
    contract_failures: int = 0
    contract_recovered: int = 0
    contract_unrecovered: int = 0
    branch_total_ms: int = 0
    models_used: Set[str] = field(default_factory=set)


@dataclass
class RuntimeAnalyticsState:
    debug_enabled: bool
    run_started_at: float = field(default_factory=time.perf_counter)
    warnings_operational: int = 0
    warnings_informational: int = 0
    errors: int = 0
    warning_reason_counts: Dict[str, int] = field(default_factory=dict)
    error_reason_counts: Dict[str, int] = field(default_factory=dict)
    rows_loaded: int = 0
    planned_items: int = 0
    rows_skipped: int = 0
    unique_dates_processed: Set[str] = field(default_factory=set)
    date_branch_executions: int = 0
    docs_created: int = 0
    docs_failed: int = 0
    telegram_sent: int = 0
    telegram_failed: int = 0
    telegram_skipped: int = 0
    malformed_tail_url_fragments_dropped: int = 0
    merge_final_failure: int = 0
    merge_validation_rejected: int = 0
    publish_gate_blocked_count: int = 0
    publish_gate_blocked_languages: Set[str] = field(default_factory=set)
    malformed_tail_cleanup_keys: Set[str] | None = None
    branch_results: Dict[str, BranchAnalyticsState] = field(default_factory=dict)
    branch_date_results: Dict[str, BranchDateAnalyticsState] = field(default_factory=dict)
    stage_durations_ms: Dict[str, int] = field(default_factory=dict)
    slot_total_ms_by_key: Dict[str, int] = field(default_factory=dict)
    degraded_recovered_count: int = 0
    degraded_unrecovered_count: int = 0
    content_contract_failures: int = 0
    content_contract_recovered: int = 0
    content_contract_unrecovered: int = 0
