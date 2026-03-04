from __future__ import annotations

from dataclasses import dataclass
import re


_WORD_CHAR_RE: re.Pattern[str] = re.compile(r"\w", re.UNICODE)
_SENTENCE_END_CHARS: str = ".!?…"
_BOUNDARY_CHARS: str = " \t\r\n,;:)]}\"'»"


@dataclass(frozen=True)
class SafeTrimResult:
    text: str
    trimmed: bool
    reason: str
    original_length: int
    trimmed_length: int


def safe_trim_right(text: str, *, max_length: int) -> SafeTrimResult:
    normalized_text: str = str(text or "")
    original_length: int = len(normalized_text)
    if max_length <= 0:
        return SafeTrimResult(
            text="",
            trimmed=bool(normalized_text),
            reason="empty_limit",
            original_length=original_length,
            trimmed_length=0,
        )
    if original_length <= max_length:
        return SafeTrimResult(
            text=normalized_text,
            trimmed=False,
            reason="not_trimmed",
            original_length=original_length,
            trimmed_length=original_length,
        )

    sentence_index: int = _find_sentence_boundary(normalized_text, max_length)
    if sentence_index > 0:
        trimmed_text: str = normalized_text[:sentence_index].rstrip()
        if trimmed_text:
            return _build_result(
                original_text=normalized_text,
                trimmed_text=trimmed_text,
                reason="sentence_boundary",
            )

    word_index: int = _find_word_boundary(normalized_text, max_length)
    if word_index > 0:
        trimmed_text = normalized_text[:word_index].rstrip()
        if trimmed_text:
            return _build_result(
                original_text=normalized_text,
                trimmed_text=trimmed_text,
                reason="word_boundary",
            )

    symbol_index: int = _find_symbol_boundary(normalized_text, max_length)
    if symbol_index > 0:
        trimmed_text = normalized_text[:symbol_index].rstrip()
        if trimmed_text:
            return _build_result(
                original_text=normalized_text,
                trimmed_text=trimmed_text,
                reason="symbol_boundary",
            )

    return SafeTrimResult(
        text="",
        trimmed=True,
        reason="drop_long_token",
        original_length=original_length,
        trimmed_length=0,
    )


def align_trimmed_suffix(text: str, *, max_length: int) -> str:
    normalized_text: str = str(text or "")
    if max_length <= 0:
        return ""
    if len(normalized_text) <= max_length:
        return normalized_text
    start_index: int = len(normalized_text) - max_length
    if start_index <= 0:
        return normalized_text
    if _cuts_word(normalized_text, start_index):
        while start_index < len(normalized_text) and _is_word_char(
            normalized_text[start_index]
        ):
            start_index += 1
        while start_index < len(normalized_text) and normalized_text[start_index].isspace():
            start_index += 1
    return normalized_text[start_index:].lstrip()


def _build_result(
    *, original_text: str, trimmed_text: str, reason: str
) -> SafeTrimResult:
    return SafeTrimResult(
        text=trimmed_text,
        trimmed=trimmed_text != original_text,
        reason=reason,
        original_length=len(original_text),
        trimmed_length=len(trimmed_text),
    )


def _find_sentence_boundary(text: str, max_length: int) -> int:
    minimum_index: int = max(1, int(max_length * 0.6))
    for index in range(min(max_length, len(text)), 0, -1):
        if text[index - 1] not in _SENTENCE_END_CHARS:
            continue
        if index < minimum_index:
            continue
        return index
    return 0


def _find_word_boundary(text: str, max_length: int) -> int:
    minimum_index: int = max(1, int(max_length * 0.5))
    for index in range(min(max_length, len(text)), 0, -1):
        current_char: str = text[index - 1]
        if current_char.isspace() or current_char in _BOUNDARY_CHARS:
            if index < minimum_index:
                continue
            return index
    return 0


def _find_symbol_boundary(text: str, max_length: int) -> int:
    for index in range(min(max_length, len(text)), 0, -1):
        if not _is_word_char(text[index - 1]):
            return index
    return 0


def _cuts_word(text: str, start_index: int) -> bool:
    if start_index <= 0 or start_index >= len(text):
        return False
    return _is_word_char(text[start_index - 1]) and _is_word_char(text[start_index])


def _is_word_char(value: str) -> bool:
    return bool(value and _WORD_CHAR_RE.fullmatch(value))
