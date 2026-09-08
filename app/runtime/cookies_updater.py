"""Проверка состояния файла YouTube cookies."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CookiesUpdateStatus:
    cookies_file: Optional[Path]   # путь к файлу (None если не настроен)
    file_exists: bool
    file_age_days: Optional[int]   # возраст файла в днях по mtime
    message: str
    format_valid: Optional[bool] = None   # None если файл отсутствует; True/False иначе
    account_name: Optional[str] = None   # имя YouTube-аккаунта, если удалось определить


def validate_cookies_format(path: Path) -> tuple[bool, str]:
    """Проверить, что cookies.txt начинается с Netscape-заголовка.

    Returns:
        (True, "ok") если формат валидный.
        (False, <reason>) если файл пустой или формат неверный.
    """
    text: str = path.read_bytes().decode("utf-8-sig", errors="replace")
    first_line: str | None = None
    for line in text.splitlines():
        stripped: str = line.strip()
        if stripped:
            first_line = stripped
            break

    if first_line is None:
        return False, "пустой файл"

    if first_line.startswith("# Netscape HTTP Cookie File"):
        return True, "ok"

    return False, f"первая строка не Netscape-заголовок: {first_line[:80]}"


def check_cookies(
    *,
    cookies_file: Optional[Path],
    warn_age_days: int,
    logger: logging.Logger,
    ytdlp_path: Optional[Path] = None,
) -> CookiesUpdateStatus:
    """Проверить наличие и возраст файла cookies.

    Правила:
    - Если cookies_file is None — INFO (cookies опциональны), статус «не настроены».
    - Если файл не существует — INFO (cookies опциональны), статус «файл не найден».
    - Если файл существует — DEBUG возраст; WARNING если возраст > warn_age_days.
    """
    if cookies_file is None:
        logger.info(
            "cookies: путь не настроен — приватные/age-gated видео могут быть недоступны; "
            "при необходимости положите cookies-файл в secrets/cookies.txt"
        )
        return CookiesUpdateStatus(
            cookies_file=None,
            file_exists=False,
            file_age_days=None,
            account_name=None,
            format_valid=None,
            message="не настроены",
        )

    if not cookies_file.exists():
        logger.info(
            "cookies: файл не найден: %s — приватные/age-gated видео могут быть недоступны; "
            "при необходимости экспортируйте cookies через расширение браузера и положите "
            "файл в secrets/cookies.txt",
            cookies_file,
        )
        return CookiesUpdateStatus(
            cookies_file=cookies_file,
            file_exists=False,
            file_age_days=None,
            account_name=None,
            format_valid=None,
            message="файл не найден",
        )

    format_ok, format_reason = validate_cookies_format(cookies_file)
    if not format_ok:
        logger.error("cookies: невалидный формат файла %s: %s", cookies_file, format_reason)
        return CookiesUpdateStatus(
            cookies_file=cookies_file,
            file_exists=True,
            file_age_days=None,
            account_name=None,
            format_valid=False,
            message=f"невалидный формат: {format_reason}",
        )

    age = _file_age_days(cookies_file)
    account_name: Optional[str] = None
    if ytdlp_path is not None and ytdlp_path.exists():
        account_name = _get_youtube_account_name(
            cookies_file=cookies_file,
            ytdlp_path=ytdlp_path,
            logger=logger,
        )

    if age > warn_age_days:
        logger.warning(
            "cookies: файл устарел (%d дн. > %d дн.): %s — "
            "рекомендуется обновить cookies через расширение браузера",
            age,
            warn_age_days,
            cookies_file,
        )
    else:
        logger.debug(
            "cookies: файл актуален, возраст %d дн.%s: %s",
            age,
            f" | аккаунт: {account_name}" if account_name else "",
            cookies_file,
        )

    return CookiesUpdateStatus(
        cookies_file=cookies_file,
        file_exists=True,
        file_age_days=age,
        account_name=account_name,
        format_valid=True,
        message="актуальны" if age <= warn_age_days else f"устарели ({age} дн.)",
    )


def _get_youtube_account_name(
    cookies_file: Path,
    ytdlp_path: Path,
    logger: logging.Logger,
) -> Optional[str]:
    """Получить имя YouTube-аккаунта (канала) через yt-dlp."""
    probe_url = "https://www.youtube.com/feed/library"
    try:
        result = subprocess.run(
            [
                str(ytdlp_path),
                "--cookies",
                str(cookies_file),
                "--no-warnings",
                "--flat-playlist",
                "--playlist-items",
                "0",
                "--print",
                "channel",
                probe_url,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=20.0,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        logger.debug("cookies: не удалось определить аккаунт: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug(
            "cookies: не удалось определить аккаунт, yt-dlp вернул код %s",
            result.returncode,
        )
        return None

    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not line:
        return None

    try:
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            name = parsed.get("channel")
            if isinstance(name, str) and name.strip():
                return name.strip()
    except json.JSONDecodeError:
        pass
    return line.strip() or None


def _file_age_days(path: Path) -> int:
    """Возраст файла в полных днях по mtime."""
    mtime = path.stat().st_mtime
    age_seconds = time.time() - mtime
    return max(0, int(age_seconds // 86400))
