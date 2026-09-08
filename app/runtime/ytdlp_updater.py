"""Тихое автообновление yt-dlp раз в N дней."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


STATE_FILENAME = "ytdlp_last_check.json"
UPDATE_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class UpdateStatus:
    """Результат попытки обновления для отображения при старте."""

    current_version: Optional[str]
    last_check_days_ago: Optional[int]
    attempted: bool
    succeeded: bool
    message: str


def maybe_update_ytdlp(
    *,
    ytdlp_path: Path,
    state_dir: Path,
    enabled: bool,
    interval_days: int,
    logger: logging.Logger,
) -> UpdateStatus:
    """Если прошло >= interval_days с последней проверки - запустить yt-dlp -U.

    Правила:
    - Если enabled=False - только читаем текущую версию, обновление не запускаем.
    - Если интервал не истёк - только читаем версию и дни с прошлой проверки.
    - Ошибки обновления никогда не блокируют работу, только логируются.
    - Файл yt-dlp.exe сохраняет свой путь - `-U` перезаписывает сам себя.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / STATE_FILENAME

    current_version = _read_current_version(ytdlp_path, logger)
    last_check_ts = _read_last_check(state_file)
    days_ago = _days_since(last_check_ts) if last_check_ts is not None else None

    if not enabled:
        return UpdateStatus(
            current_version=current_version,
            last_check_days_ago=days_ago,
            attempted=False,
            succeeded=False,
            message="auto-update disabled",
        )

    if last_check_ts is not None and days_ago is not None and days_ago < interval_days:
        return UpdateStatus(
            current_version=current_version,
            last_check_days_ago=days_ago,
            attempted=False,
            succeeded=False,
            message=f"check skipped: {days_ago}d since last check (interval {interval_days}d)",
        )

    logger.info("yt-dlp auto-update: запуск `%s -U`", ytdlp_path)
    succeeded = False
    message = ""
    try:
        result = subprocess.run(
            [str(ytdlp_path), "-U"],
            capture_output=True,
            text=True,
            timeout=UPDATE_TIMEOUT_SECONDS,
            encoding="utf-8",
            errors="replace",
        )
        succeeded = result.returncode == 0
        message_lines = (result.stdout or "").strip().splitlines()[-1:] or [""]
        message = message_lines[0] if message_lines else ""
        if not succeeded:
            logger.warning(
                "yt-dlp -U завершился с кодом %d: stdout=%r stderr=%r",
                result.returncode,
                result.stdout,
                result.stderr,
            )
        else:
            logger.info("yt-dlp -U успешно: %s", message)
    except subprocess.TimeoutExpired:
        message = f"timeout {UPDATE_TIMEOUT_SECONDS}s"
        logger.warning("yt-dlp -U превысил таймаут %d сек", UPDATE_TIMEOUT_SECONDS)
    except Exception as exc:
        message = f"error: {exc}"
        logger.warning("yt-dlp -U не удалось: %s", exc)

    _write_last_check(state_file, int(time.time()))
    new_version = _read_current_version(ytdlp_path, logger)

    return UpdateStatus(
        current_version=new_version,
        last_check_days_ago=0,
        attempted=True,
        succeeded=succeeded,
        message=message,
    )


def _read_current_version(ytdlp_path: Path, logger: logging.Logger) -> Optional[str]:
    if not ytdlp_path.exists():
        return None
    try:
        result = subprocess.run(
            [str(ytdlp_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return (result.stdout or "").strip() or None
    except Exception as exc:
        logger.debug("yt-dlp --version failed: %s", exc)
    return None


def _read_last_check(state_file: Path) -> Optional[int]:
    if not state_file.exists():
        return None
    try:
        payload = json.loads(state_file.read_text("utf-8"))
        ts = payload.get("last_check_ts")
        return int(ts) if ts is not None else None
    except Exception:
        return None


def _write_last_check(state_file: Path, ts: int) -> None:
    try:
        state_file.write_text(
            json.dumps({"last_check_ts": ts}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _days_since(ts: int) -> int:
    now = int(time.time())
    return max(0, (now - ts) // 86400)
