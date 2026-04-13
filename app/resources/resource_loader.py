from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


def _resource_dir() -> Path:
    import sys as _sys
    if getattr(_sys, "frozen", False):
        base: Path = Path(_sys._MEIPASS)  # type: ignore[attr-defined]
        return base / "app" / "resources" / "text"
    return Path(__file__).resolve().parent / "text"


def _resource_path(resource_name: str) -> Path:
    resource_path: Path = _resource_dir() / str(resource_name or "").strip()
    return resource_path


@lru_cache(maxsize=None)
def load_text_resource(resource_name: str) -> str:
    resource_path: Path = _resource_path(resource_name)
    resource_text: str = resource_path.read_text(encoding="utf-8")
    return resource_text


@lru_cache(maxsize=None)
def load_lines_resource(resource_name: str) -> tuple[str, ...]:
    raw_text: str = load_text_resource(resource_name)
    lines: list[str] = [line.strip() for line in raw_text.splitlines()]
    cleaned_lines: tuple[str, ...] = tuple(
        line for line in lines if line and not line.startswith("#")
    )
    return cleaned_lines


@lru_cache(maxsize=None)
def load_json_resource(resource_name: str) -> Dict[str, Any]:
    raw_text: str = load_text_resource(resource_name)
    parsed: Any = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Resource {resource_name} must contain a JSON object")
    normalized: Dict[str, Any] = {str(key): value for key, value in parsed.items()}
    return normalized
