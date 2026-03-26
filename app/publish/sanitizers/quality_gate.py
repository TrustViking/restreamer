"""Publication quality gate: checks for structural issues in final descriptions."""
from __future__ import annotations

import re
from typing import List

from app.core.constants import CTA_FIRST_PARAGRAPH_PREFIXES
from app.core.text_utils import has_duplicate_paragraphs
from app.llm.merges.merge_validation_helpers import looks_like_bad_hook_paragraph


class PublishQualityGate:
    """Final quality checks on the composed publication description."""

    @staticmethod
    def has_duplicate_paragraphs(description_text: str) -> bool:
        """Delegates to text_utils.has_duplicate_paragraphs."""
        return has_duplicate_paragraphs(description_text)

    @staticmethod
    def has_opener_cta(description_text: str) -> bool:
        """Was `_final_description_has_opener_cta`."""
        paragraphs: List[str] = [
            part.strip()
            for part in re.split(r"\n\s*\n", str(description_text or "").strip())
            if part.strip()
        ]
        if not paragraphs:
            return False
        first_paragraph: str = paragraphs[0]
        first_non_empty_line: str = ""
        for raw_line in first_paragraph.split("\n"):
            normalized_line: str = str(raw_line or "").strip()
            if normalized_line:
                first_non_empty_line = normalized_line
                break
        if not first_non_empty_line:
            return False
        if looks_like_bad_hook_paragraph(first_non_empty_line) or looks_like_bad_hook_paragraph(first_paragraph):
            return True
        normalized_first_line: str = first_non_empty_line.lower()
        for raw_prefix in CTA_FIRST_PARAGRAPH_PREFIXES:
            normalized_prefix: str = str(raw_prefix or "").strip().lower()
            if normalized_prefix and normalized_first_line.startswith(normalized_prefix):
                return True
        return False
