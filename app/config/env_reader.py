from __future__ import annotations

import os
from typing import Optional


class EnvReader:
    """Centralised reader for environment variables with typed parsing."""

    _TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})
    _FALSY: frozenset[str] = frozenset({"0", "false", "no", "off"})

    @staticmethod
    def bool(name: str, default: bool, *, strict: bool = True) -> bool:
        """Read env as bool with configurable error handling."""
        raw_value: str = os.getenv(name, "").strip().lower()
        if not raw_value:
            return default
        if raw_value in EnvReader._TRUTHY:
            return True
        if raw_value in EnvReader._FALSY:
            return False
        if strict:
            raise RuntimeError(
                f"Invalid boolean for {name}: {raw_value!r}. "
                "Allowed: 1/0, true/false, yes/no, on/off."
            )
        return default

    @staticmethod
    def int(name: str, default: int, *, min_value: int = 0, strict: bool = True) -> int:
        """Read env as int with optional minimum bound."""
        raw_value: str = os.getenv(name, "").strip()
        if not raw_value:
            return default
        try:
            parsed: int = int(raw_value)
        except (ValueError, TypeError) as error:
            if strict:
                raise RuntimeError(f"Invalid integer for {name}: {raw_value!r}") from error
            return default
        if parsed < min_value:
            if strict:
                raise RuntimeError(f"{name} must be >= {min_value}, got {parsed}")
            return max(min_value, parsed)
        return parsed

    @staticmethod
    def float(name: str, default: float, *, min_value: float = 0.0, strict: bool = True) -> float:
        """Read env as float with optional minimum bound."""
        raw_value: str = os.getenv(name, "").strip()
        if not raw_value:
            return default
        try:
            parsed: float = float(raw_value)
        except (ValueError, TypeError) as error:
            if strict:
                raise RuntimeError(f"Invalid float for {name}: {raw_value!r}") from error
            return default
        if parsed < min_value:
            if strict:
                raise RuntimeError(f"{name} must be >= {min_value}, got {parsed}")
            return max(min_value, parsed)
        return parsed

    @staticmethod
    def str_required(name: str) -> str:
        """Read required env var and fail if missing or empty."""
        value: Optional[str] = os.getenv(name)
        if value is None or not value.strip():
            raise RuntimeError(f"Env var {name} is required.")
        return value.strip()

    @staticmethod
    def str_optional(name: str) -> Optional[str]:
        """Read optional env var, returning None if absent or empty."""
        value: str = os.getenv(name, "").strip()
        return value or None
