"""Tests for app.bootstrap.cleanup module."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from app.bootstrap.cleanup import (
    _resolve_template_base,
    _should_run_cleanup,
    run_daily_cleanup,
)


@pytest.fixture
def cleanup_root(tmp_path: Path) -> Path:
    """Create a temporary project root with logs/, docs/, image/ dirs."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "image").mkdir()
    return tmp_path


def _create_stale_file(path: Path, age_hours: float) -> Path:
    """Create a file and set its mtime to `age_hours` hours ago."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test content", encoding="utf-8")
    old_time: float = time.time() - (age_hours * 3600.0)
    os.utime(path, (old_time, old_time))
    return path


class TestShouldRunCleanup:
    def test_no_marker_returns_true(self, tmp_path: Path) -> None:
        marker: Path = tmp_path / ".last_cleanup"
        assert _should_run_cleanup(marker) is True

    def test_fresh_marker_returns_false(self, tmp_path: Path) -> None:
        marker: Path = tmp_path / ".last_cleanup"
        marker.write_text("recent", encoding="utf-8")
        assert _should_run_cleanup(marker) is False

    def test_old_marker_returns_true(self, tmp_path: Path) -> None:
        marker: Path = tmp_path / ".last_cleanup"
        marker.write_text("old", encoding="utf-8")
        old_time: float = time.time() - 90000.0
        os.utime(marker, (old_time, old_time))
        assert _should_run_cleanup(marker) is True


class TestResolveTemplateBase:
    def test_relative_path_resolved_from_root(self, tmp_path: Path) -> None:
        result: Path | None = _resolve_template_base("./docs/{date}", project_root=tmp_path)
        assert result is not None
        assert result == (tmp_path / "docs").resolve()

    def test_empty_template_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_template_base("", project_root=tmp_path) is None

    def test_project_root_itself_returns_none(self, tmp_path: Path) -> None:
        """Safety check: don't return project_root as a cleanup target."""
        assert _resolve_template_base("./", project_root=tmp_path) is None


class TestRunDailyCleanup:
    def test_fresh_marker_skips_cleanup(self, cleanup_root: Path) -> None:
        stale: Path = _create_stale_file(cleanup_root / "logs" / "old.log", age_hours=100.0)
        marker: Path = cleanup_root / "logs" / ".last_cleanup"
        marker.write_text("recent", encoding="utf-8")
        run_daily_cleanup(
            logger=logging.getLogger("test"),
            project_root=cleanup_root,
            max_age_days=3,
            local_image_dir_template=str(cleanup_root / "image" / "{date}"),
            local_doc_dir_template=str(cleanup_root / "docs" / "{date}"),
        )
        assert stale.exists(), "File should NOT be removed when marker is fresh"

    def test_stale_files_removed_fresh_kept(self, cleanup_root: Path) -> None:
        stale: Path = _create_stale_file(cleanup_root / "logs" / "old.log", age_hours=100.0)
        fresh: Path = cleanup_root / "logs" / "new.log"
        fresh.write_text("fresh content", encoding="utf-8")
        run_daily_cleanup(
            logger=logging.getLogger("test"),
            project_root=cleanup_root,
            max_age_days=3,
            local_image_dir_template=str(cleanup_root / "image" / "{date}"),
            local_doc_dir_template=str(cleanup_root / "docs" / "{date}"),
        )
        assert not stale.exists(), "Stale file should be removed"
        assert fresh.exists(), "Fresh file should be kept"

    def test_docs_and_image_cleaned(self, cleanup_root: Path) -> None:
        stale_doc: Path = _create_stale_file(
            cleanup_root / "docs" / "2026-03-28" / "report.docx",
            age_hours=100.0,
        )
        stale_img: Path = _create_stale_file(
            cleanup_root / "image" / "2026-03-28" / "uk" / "pic.jpg",
            age_hours=100.0,
        )
        run_daily_cleanup(
            logger=logging.getLogger("test"),
            project_root=cleanup_root,
            max_age_days=3,
            local_image_dir_template=str(cleanup_root / "image" / "{date}" / "{language}"),
            local_doc_dir_template=str(cleanup_root / "docs" / "{date}"),
        )
        assert not stale_doc.exists(), "Stale doc should be removed"
        assert not stale_img.exists(), "Stale image should be removed"

    def test_marker_created_after_cleanup(self, cleanup_root: Path) -> None:
        marker: Path = cleanup_root / "logs" / ".last_cleanup"
        assert not marker.exists()
        run_daily_cleanup(
            logger=logging.getLogger("test"),
            project_root=cleanup_root,
            max_age_days=3,
            local_image_dir_template=str(cleanup_root / "image" / "{date}"),
            local_doc_dir_template=str(cleanup_root / "docs" / "{date}"),
        )
        assert marker.exists(), "Marker file should be created after cleanup"
