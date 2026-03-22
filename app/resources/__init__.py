from __future__ import annotations

from app.resources.text_catalog import (
    canonical_service_lines,
    merge_agenda_headings,
    merge_service_hints,
    promotional_opener_phrases,
    resolve_official_links_heading,
    resolve_recommended_materials_heading,
)

__all__: tuple[str, ...] = (
    "canonical_service_lines",
    "merge_agenda_headings",
    "merge_service_hints",
    "promotional_opener_phrases",
    "resolve_official_links_heading",
    "resolve_recommended_materials_heading",
)
