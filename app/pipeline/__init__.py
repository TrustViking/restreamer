from __future__ import annotations

from typing import Any

__all__ = ["BatchRunner"]


def __getattr__(name: str) -> Any:
    if name == "BatchRunner":
        from .batch_runner import BatchRunner

        return BatchRunner
    raise AttributeError(name)
