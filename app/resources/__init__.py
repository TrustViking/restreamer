from __future__ import annotations

from typing import TYPE_CHECKING

from app.resources.text_catalog import (
    canonical_service_lines,
    merge_agenda_headings,
    merge_service_hints,
    promotional_opener_phrases,
)

if TYPE_CHECKING:
    from app.config.settings import AppConfig


def init_heading_resolver(*, config: AppConfig) -> None:
    from app.resources.heading_resolver import init_heading_resolver as _impl

    _impl(config=config)


def resolve_official_links_heading(language: str) -> str:
    from app.resources.heading_resolver import resolve_official_links_heading as _impl

    return _impl(language)


def resolve_recommended_materials_heading(language: str) -> str:
    from app.resources.heading_resolver import (
        resolve_recommended_materials_heading as _impl,
    )

    return _impl(language)


__all__: tuple[str, ...] = (
    "canonical_service_lines",
    "init_heading_resolver",
    "merge_agenda_headings",
    "merge_service_hints",
    "promotional_opener_phrases",
    "resolve_official_links_heading",
    "resolve_recommended_materials_heading",
)
