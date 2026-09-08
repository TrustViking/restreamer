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


_STRICT_TIMESTAMP_HEADING_RE = re.compile(
    r"^\s*("
    r"Тайм[\-]?код[ыиі]?"
    r"|Timestamps?"
    r"|Timecodes?"
    r"|Chapters?(?:\s+list)?"
    r"|Video\s+chapters"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

_BROAD_TIMESTAMP_HEADING_RE = re.compile(
    r"^\s*("
    r"Зміст(?:\s+відео)?"
    r"|Содержание"
    r"|Оглавление"
    r"|Главы"
    r"|Розділи"
    r"|Нав[іи]гац[іи]я"
    r"|Contents"
    r"|Timeline"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

_CHAPTER_LINE_RE = re.compile(
    r"^\s*(?:\d{1,2}\s*:\s*)?\d{1,2}\s*:\s*\d{2}\s+\S.*$"
)

_CHAPTER_ZERO_START_RE = re.compile(
    r"^\s*0{1,2}\s*:\s*0{2}(?:\s*:\s*0{2})?\s+\S"
)

_TIME_TOKEN = r"(?:\d{1,2}\s*:\s*)?\d{1,2}\s*:\s*\d{2}"
_RANGE_SEP = r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]"
_CHAPTER_RANGE_LINE_RE = re.compile(
    rf"^\s*{_TIME_TOKEN}\s*{_RANGE_SEP}\s*{_TIME_TOKEN}\s*\S.*$"
)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-•—–*·]|\d{1,3}[.)])\s+")
_HH_MM_SS_ANYWHERE_RE = re.compile(r"\d{1,2}\s*:\s*\d{1,2}\s*:\s*\d{2}")
_MIN_RANGE_BLOCK_SIZE: int = 3

_HH_MM_SS_STRICT = r"\d{1,2}\s*:\s*\d{1,2}\s*:\s*\d{2}"
_TAIL_HH_MM_SS_LINE_RE = re.compile(
    rf"^\s*{_HH_MM_SS_STRICT}(?:\s+\S.*)?$"
)
_TAIL_HH_MM_SS_RANGE_RE = re.compile(
    rf"^\s*(?:"
    rf"{_HH_MM_SS_STRICT}\s*{_RANGE_SEP}\s*{_TIME_TOKEN}"
    rf"|{_TIME_TOKEN}\s*{_RANGE_SEP}\s*{_HH_MM_SS_STRICT}"
    rf")(?:\s+\S.*)?$"
)


def _is_timestamp_heading(stripped_line: str, *, strict_only: bool = False) -> bool:
    if _STRICT_TIMESTAMP_HEADING_RE.match(stripped_line):
        return True
    if not strict_only and _BROAD_TIMESTAMP_HEADING_RE.match(stripped_line):
        return True
    return False


def _strip_bullet_prefix(line: str) -> str:
    return _BULLET_PREFIX_RE.sub("", line, count=1)


def _is_chapter_or_range_line(line_stripped: str) -> bool:
    """Строка матчит либо single timestamp-line, либо range-line."""
    line_without_bullet: str = _strip_bullet_prefix(line_stripped)
    return bool(
        _CHAPTER_LINE_RE.match(line_without_bullet)
        or _CHAPTER_RANGE_LINE_RE.match(line_without_bullet)
    )


def _is_tail_orphan_candidate(line_no_newline: str) -> bool:
    """Строка подходит под tail-orphan cleanup: одиночная HH:MM:SS или range с HH:MM:SS."""
    line_without_bullet: str = _strip_bullet_prefix(line_no_newline)
    return bool(
        _TAIL_HH_MM_SS_LINE_RE.match(line_without_bullet)
        or _TAIL_HH_MM_SS_RANGE_RE.match(line_without_bullet)
    )


def _strip_tail_orphan_hms(lines: list[str]) -> list[str]:
    """
    Удалить хвостовые одиночные HH:MM:SS timestamp-строки (возможно подряд несколько).

    Условия:
    - Одна или несколько подряд HH:MM:SS строк в самом конце текста
    - Пустые строки между ними и в хвосте допускаются
    - Перед удаляемым блоком должна быть хотя бы одна содержательная не-HH:MM:SS строка
      (защита: если весь текст это только timestamp-строка, не трогаем)
    """
    tail_start: int = len(lines)
    k: int = len(lines) - 1
    seen_tail_timestamp: bool = False
    while k >= 0:
        stripped: str = lines[k].rstrip("\n\r").strip()
        if not stripped:
            k -= 1
            continue
        if _is_tail_orphan_candidate(lines[k].rstrip("\n\r")):
            seen_tail_timestamp = True
            tail_start = k
            k -= 1
            continue
        break

    if not seen_tail_timestamp:
        return lines

    has_content_before: bool = any(lines[idx].strip() for idx in range(tail_start))
    if not has_content_before:
        return lines

    result_end: int = tail_start
    while result_end > 0 and not lines[result_end - 1].strip():
        result_end -= 1

    return lines[:result_end]

def strip_chapter_timestamps(text: str) -> str:
    raw_text: str = str(text or "")
    if not raw_text:
        return raw_text

    lines = raw_text.splitlines(keepends=True)
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- Режим A.1: строгий заголовок (удаляем даже без таймкодов) ---
        if _STRICT_TIMESTAMP_HEADING_RE.match(stripped):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and _is_chapter_or_range_line(lines[i].rstrip("\n\r")):
                i += 1
            continue

        # --- Режим A.2: широкий заголовок (lookahead — удаляем только если есть таймкоды) ---
        if _BROAD_TIMESTAMP_HEADING_RE.match(stripped):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and _is_chapter_or_range_line(lines[j].rstrip("\n\r")):
                i = j
                while i < len(lines) and _is_chapter_or_range_line(lines[i].rstrip("\n\r")):
                    i += 1
                continue
            result.append(line)
            i += 1
            continue

        # --- Режим B: без заголовка, блок начинается с 0:00 ---
        line_no_newline: str = line.rstrip("\n\r")
        line_without_bullet: str = _strip_bullet_prefix(line_no_newline)
        if _CHAPTER_ZERO_START_RE.match(line_without_bullet):
            j = i
            while j < len(lines) and _is_chapter_or_range_line(lines[j].rstrip("\n\r")):
                j += 1
            block_size = j - i
            if block_size >= 2:
                i = j
                continue
            result.append(line)
            i += 1
            continue

        # --- Режим C: headless range-блок, порог 3+, хотя бы одна строка с hh:mm:ss ---
        if _CHAPTER_RANGE_LINE_RE.match(line_without_bullet):
            j = i
            while (
                j < len(lines)
                and _CHAPTER_RANGE_LINE_RE.match(
                    _strip_bullet_prefix(lines[j].rstrip("\n\r"))
                )
            ):
                j += 1
            block_size = j - i
            if block_size >= _MIN_RANGE_BLOCK_SIZE:
                has_hms = any(_HH_MM_SS_ANYWHERE_RE.search(lines[k]) for k in range(i, j))
                if has_hms:
                    i = j
                    continue
            result.append(line)
            i += 1
            continue

        result.append(line)
        i += 1

    result = _strip_tail_orphan_hms(result)
    joined = "".join(result)

    # post-filter: orphan strict heading only (broad headings must survive)
    cleaned: list[str] = [
        ln for ln in joined.splitlines(keepends=True)
        if not _STRICT_TIMESTAMP_HEADING_RE.match(ln.strip())
    ]
    joined = "".join(cleaned)

    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined


def optional_env(name: str) -> Optional[str]:
    return EnvReader.str_optional(name)
