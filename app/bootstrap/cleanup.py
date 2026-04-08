"""Daily cleanup of stale files in logs/, docs/, image/ directories."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

_MARKER_FILENAME: str = ".last_cleanup"
_PROTECTED_LOG_FILENAMES: frozenset[str] = frozenset({"bot_known_groups.json"})


def _should_run_cleanup(marker_path: Path) -> bool:
    """Return True if cleanup hasn't run today (marker file older than 24h or absent)."""
    if not marker_path.exists():
        return True
    try:
        age_seconds: float = time.time() - marker_path.stat().st_mtime
        return age_seconds > 86400
    except OSError:
        return True


def _touch_marker(marker_path: Path) -> None:
    """Create or update the cleanup marker file."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(
        f"Last cleanup: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )


def _resolve_template_base(template: str, *, project_root: Path) -> Optional[Path]:
    """Extract base directory from a path template like './docs/{date}'."""
    if not template:
        return None
    before_placeholder: str = template.split("{")[0].rstrip("/\\").strip()
    if not before_placeholder:
        return None
    candidate: Path = Path(before_placeholder)
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    if candidate == project_root.resolve():
        return None
    return candidate


def _remove_stale_files(
    directory: Path,
    *,
    max_age_seconds: float,
    logger: logging.Logger,
    protected_filenames: frozenset[str] | None = None,
) -> int:
    """Remove files older than max_age_seconds. Returns count of removed files."""
    if not directory.is_dir():
        return 0

    removed_count: int = 0
    now_seconds: float = time.time()
    item: Path
    for item in directory.rglob("*"):
        if not item.is_file():
            continue
        if item.name.startswith("."):
            continue
        if protected_filenames is not None and item.name in protected_filenames:
            logger.debug("cleanup: keep protected file %s", item)
            continue
        try:
            file_age_seconds: float = now_seconds - item.stat().st_mtime
            if file_age_seconds <= max_age_seconds:
                continue
            item.unlink()
            removed_count += 1
            logger.debug(
                "cleanup: removed %s (age=%.0fh)",
                item,
                file_age_seconds / 3600.0,
            )
        except OSError as error:
            logger.warning("cleanup: failed to remove %s: %s", item, error)

    reverse_item: Path
    for reverse_item in sorted(directory.rglob("*"), reverse=True):
        if not reverse_item.is_dir():
            continue
        try:
            reverse_item.rmdir()
        except OSError:
            continue

    return removed_count


def run_daily_cleanup(
    *,
    logger: logging.Logger,
    project_root: Path,
    max_age_days: int,
    local_image_dir_template: str,
    local_doc_dir_template: Optional[str],
) -> None:
    """Run cleanup of stale files, at most once per 24 hours."""
    marker_path: Path = project_root / "logs" / _MARKER_FILENAME
    if not _should_run_cleanup(marker_path):
        logger.debug("cleanup: skipped (already ran within 24h)")
        return

    logger.info("cleanup: starting daily cleanup (max_age_days=%d)", max_age_days)
    max_age_seconds: float = float(max_age_days) * 86400.0
    total_removed_count: int = 0

    logs_dir: Path = project_root / "logs"
    total_removed_count += _remove_stale_files(
        logs_dir,
        max_age_seconds=max_age_seconds,
        logger=logger,
        protected_filenames=_PROTECTED_LOG_FILENAMES,
    )

    doc_base: Optional[Path] = _resolve_template_base(
        local_doc_dir_template or "",
        project_root=project_root,
    )
    if doc_base is not None:
        total_removed_count += _remove_stale_files(
            doc_base,
            max_age_seconds=max_age_seconds,
            logger=logger,
        )

    image_base: Optional[Path] = _resolve_template_base(
        local_image_dir_template or "",
        project_root=project_root,
    )
    if image_base is not None:
        total_removed_count += _remove_stale_files(
            image_base,
            max_age_seconds=max_age_seconds,
            logger=logger,
        )

    _touch_marker(marker_path)
    logger.info("cleanup: done, removed %d stale file(s)", total_removed_count)
