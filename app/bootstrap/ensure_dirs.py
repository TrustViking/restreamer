"""Ensure portable directory layout exists next to the executable."""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

from app.paths.project_paths import ProjectPaths


def ensure_portable_dirs(
    *,
    project_paths: ProjectPaths,
    logger: logging.Logger,
) -> None:
    """Create logs/, state/, config/, secrets/, image/, docs/ next to the exe/project root."""
    project_root: Path = project_paths.project_root
    for subdir in ("logs", "state", "config", "secrets", "image", "docs"):
        target: Path = project_root / subdir
        target.mkdir(parents=True, exist_ok=True)
        logger.debug("ensure_dir: %s", target)


def ensure_config_files(
    *,
    project_paths: ProjectPaths,
    logger: logging.Logger,
) -> None:
    """In frozen mode, copy bundled config/templates to user-facing config/ if missing."""
    frozen: bool = bool(getattr(sys, "frozen", False))
    if not frozen:
        return

    runtime_config: Path = project_paths.runtime_config_path
    if not runtime_config.exists():
        bundled: Path = project_paths.bundled_config_path
        if bundled.exists():
            runtime_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled, runtime_config)
            logger.info("config_bootstrap: copied %s -> %s", bundled, runtime_config)

    runtime_templates: Path = project_paths.templates_path
    if not runtime_templates.exists():
        bundled_templates: Path = project_paths.bundled_templates_path
        if bundled_templates.exists():
            runtime_templates.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled_templates, runtime_templates)
            logger.info("config_bootstrap: copied %s -> %s", bundled_templates, runtime_templates)
