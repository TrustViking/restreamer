from __future__ import annotations

import re
from typing import Optional

from app.config.env_reader import EnvReader


def load_bool_env(name: str, default: bool) -> bool:
    return EnvReader.bool(name, default, strict=False)


def load_int_env(name: str, default: int, min_value: int = 0) -> int:
    return EnvReader.int(name, default, min_value=min_value, strict=False)


def load_float_env(name: str, default: float, min_value: float = 0.0) -> float:
    return EnvReader.float(name, default, min_value=min_value, strict=False)


def sheets_link_writeback_enabled_from_env() -> bool:
    return load_bool_env("STG_SHEETS_LINK_WRITEBACK", True)


def strip_chapter_timestamps_enabled_from_env() -> bool:
    return load_bool_env("STG_STRIP_CHAPTER_TIMESTAMPS", True)


def sheets_link_normalize_report_limit_from_env() -> int:
    return load_int_env("STG_SHEETS_LINK_NORMALIZE_REPORT_LIMIT", 20, min_value=1)


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
    return EnvReader.str_optional(name)
