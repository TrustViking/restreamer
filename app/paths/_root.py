"""Resolve project root for both dev and frozen (PyInstaller onedir) modes."""
from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    """Return the project root directory.

    Dev mode: _root.py lives at app/paths/_root.py -> parents[2] is project root.
    Frozen mode: sys.executable is the .exe in the portable directory root.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = _resolve_project_root()
