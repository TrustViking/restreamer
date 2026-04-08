from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from app.core.text_utils import utf16_len


@dataclass(frozen=True)
class HeaderLine:
    text: str
    is_bold: bool


@dataclass(frozen=True)
class TextStyleSpan:
    start: int
    end: int
    is_bold: bool


@dataclass(frozen=True)
class DailyDocHeader:
    lines: Tuple[HeaderLine, ...]

    def render_text(self) -> str:
        line_texts: Tuple[str, ...] = tuple(line.text for line in self.lines)
        return "\n".join(line_texts)

    def build_relative_style_spans(self) -> Tuple[TextStyleSpan, ...]:
        spans: list[TextStyleSpan] = []
        relative_cursor: int = 0
        line_index: int
        line: HeaderLine
        for line_index, line in enumerate(self.lines):
            line_text: str = line.text
            line_length_utf16: int = utf16_len(line_text)
            if line.is_bold and line_text != "":
                spans.append(
                    TextStyleSpan(
                        start=relative_cursor,
                        end=relative_cursor + line_length_utf16,
                        is_bold=True,
                    )
                )
            relative_cursor += line_length_utf16
            if line_index < len(self.lines) - 1:
                relative_cursor += utf16_len("\n")
        return tuple(spans)
