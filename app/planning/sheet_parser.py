from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple
from zoneinfo import ZoneInfo


def parse_sheet_datetime(date_raw: str, time_raw: str, tz: ZoneInfo) -> datetime:
    date_clean: str = date_raw.strip()
    time_clean: str = time_raw.strip()
    date_formats: Tuple[str, ...] = (
        "%d.%m.%Y",
        "%d.%m.%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%d%m%y",
        "%d%m%Y",
    )
    time_formats: Tuple[str, ...] = (
        "%H:%M",
        "%H.%M",
        "%H%M",
        "%H:%M:%S",
        "%I:%M %p",
        "%I %p",
    )

    parsed_date: Optional[datetime] = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_clean, fmt)
            break
        except ValueError:
            continue
    if parsed_date is None:
        raise ValueError(f"Unsupported Date format: {date_raw!r}")

    parsed_time: Optional[datetime] = None
    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(time_clean, fmt)
            break
        except ValueError:
            continue
    if parsed_time is None:
        raise ValueError(f"Unsupported Time format: {time_raw!r}")

    return datetime(
        year=parsed_date.year,
        month=parsed_date.month,
        day=parsed_date.day,
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        tzinfo=tz,
    )


def column_letters_to_index(column_letters: str) -> Optional[int]:
    cleaned: str = str(column_letters or "").strip().upper()
    if not cleaned or not re.fullmatch(r"[A-Z]+", cleaned):
        return None
    index_value: int = 0
    for symbol in cleaned:
        index_value = (index_value * 26) + (ord(symbol) - ord("A") + 1)
    return index_value


def extract_end_column_letters_from_sheet_range(range_name: str) -> Optional[str]:
    cleaned_range: str = str(range_name or "").strip()
    if not cleaned_range:
        return None
    range_without_sheet: str = cleaned_range.split("!", 1)[-1]
    right_part: str = range_without_sheet.split(":", 1)[-1]
    match: Optional[re.Match[str]] = re.search(r"([A-Za-z]+)\d*$", right_part.strip())
    if match is None:
        return None
    return match.group(1).upper()


def sheet_range_includes_merge_column(range_name: str) -> bool:
    end_column_letters: Optional[str] = extract_end_column_letters_from_sheet_range(
        range_name
    )
    end_column_index: Optional[int] = column_letters_to_index(end_column_letters or "")
    merge_column_index: Optional[int] = column_letters_to_index("E")
    if end_column_index is None or merge_column_index is None:
        return True
    return end_column_index >= merge_column_index


def expand_sheet_range_to_af(range_name: str) -> str:
    cleaned_range: str = str(range_name or "").strip()
    if not cleaned_range:
        return "A:F"
    if "!" in cleaned_range:
        sheet_prefix: str = cleaned_range.split("!", 1)[0].strip()
        if sheet_prefix:
            return f"{sheet_prefix}!A:F"
    return "A:F"
