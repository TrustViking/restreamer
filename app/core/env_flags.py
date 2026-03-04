from __future__ import annotations

import os
import re
from typing import Optional


def load_bool_env(name: str, default: bool) -> bool:
    raw_value: str = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default


def load_int_env(name: str, default: int, min_value: int = 0) -> int:
    raw_value: str = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed: int = int(raw_value)
    except Exception:
        return default
    return max(min_value, parsed)


def load_float_env(name: str, default: float, min_value: float = 0.0) -> float:
    raw_value: str = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed: float = float(raw_value)
    except Exception:
        return default
    return max(min_value, parsed)


def sheets_link_writeback_enabled_from_env() -> bool:
    return load_bool_env("STG_SHEETS_LINK_WRITEBACK", True)


def strip_chapter_timestamps_enabled_from_env() -> bool:
    return load_bool_env("STG_STRIP_CHAPTER_TIMESTAMPS", True)


def sheets_link_normalize_report_limit_from_env() -> int:
    return load_int_env("STG_SHEETS_LINK_NORMALIZE_REPORT_LIMIT", 20, min_value=1)


def stg_debug_prompt_to_file_from_env() -> bool:
    raw_value: str = os.getenv("STG_DEBUG_PROMPT_TO_FILE", "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def stg_debug_prompt_dir_from_env() -> str:
    return os.getenv("STG_DEBUG_PROMPT_DIR", "_debug_prompts").strip() or "_debug_prompts"


def llm_allow_in_dry_run_from_env() -> bool:
    return load_bool_env("STG_LLM_ALLOW_IN_DRY_RUN", False)


def sheets_autoexpand_range_from_env() -> bool:
    return load_bool_env("STG_SHEETS_AUTOEXPAND_RANGE", False)


def strip_chapter_timestamps(text: str) -> str:
    raw_text: str = str(text or "")
    if not raw_text:
        return raw_text
    chapter_pattern: re.Pattern[str] = re.compile(
        r"^\s*(?:\d{1,2}\s*:\s*)?\d{1,2}\s*:\s*\d{2}\s+\S.*$"
    )
    cleaned_lines: list[str] = []
    for line in raw_text.splitlines(keepends=True):
        stripped_line: str = line.strip()
        if stripped_line and chapter_pattern.match(stripped_line):
            continue
        cleaned_lines.append(line)
    return "".join(cleaned_lines)


def optional_env(name: str) -> Optional[str]:
    value: str = os.getenv(name, "").strip()
    return value or None
