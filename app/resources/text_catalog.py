from __future__ import annotations

from functools import lru_cache
from typing import Dict

from app.resources.resource_loader import load_json_resource, load_lines_resource


@lru_cache(maxsize=1)
def canonical_service_lines() -> Dict[str, Dict[str, str]]:
    raw_payload: Dict[str, object] = load_json_resource("canonical_service_lines.json")
    normalized_payload: Dict[str, Dict[str, str]] = {}
    for language_key, language_payload in raw_payload.items():
        if not isinstance(language_payload, dict):
            continue
        normalized_payload[str(language_key)] = {
            str(key): str(value)
            for key, value in language_payload.items()
        }
    return normalized_payload


@lru_cache(maxsize=1)
def official_links_headings() -> Dict[str, str]:
    raw_payload: Dict[str, object] = load_json_resource("official_links_headings.json")
    normalized_payload: Dict[str, str] = {
        str(key): str(value)
        for key, value in raw_payload.items()
    }
    return normalized_payload


@lru_cache(maxsize=1)
def recommended_materials_headings() -> Dict[str, str]:
    raw_payload: Dict[str, object] = load_json_resource("recommended_materials_headings.json")
    normalized_payload: Dict[str, str] = {
        str(key): str(value)
        for key, value in raw_payload.items()
    }
    return normalized_payload


def resolve_official_links_heading(language: str) -> str:
    normalized_language: str = str(language or "").strip().lower()
    headings: Dict[str, str] = official_links_headings()
    return headings.get(normalized_language, headings.get("other", ""))


def resolve_recommended_materials_heading(language: str) -> str:
    normalized_language: str = str(language or "").strip().lower()
    headings: Dict[str, str] = recommended_materials_headings()
    return headings.get(normalized_language, headings.get("other", ""))


@lru_cache(maxsize=1)
def promotional_opener_phrases() -> tuple[str, ...]:
    phrases: tuple[str, ...] = load_lines_resource("promotional_opener_phrases.txt")
    return phrases


@lru_cache(maxsize=1)
def merge_service_hints() -> tuple[str, ...]:
    hints: tuple[str, ...] = load_lines_resource("merge_service_hints.txt")
    return hints


@lru_cache(maxsize=1)
def merge_agenda_headings() -> tuple[str, ...]:
    headings: tuple[str, ...] = load_lines_resource("merge_agenda_headings.txt")
    return headings
