"""Operator-facing startup banner: yt-dlp / deno / cookies status header."""
from __future__ import annotations

import logging
from typing import Optional

from app.runtime.cookies_updater import CookiesUpdateStatus
from app.runtime.deno_updater import DenoUpdateStatus
from app.runtime.ytdlp_updater import UpdateStatus


_BANNER_WIDTH: int = 59
_LABEL_WIDTH: int = 9


def print_runtime_banner(
    *,
    ytdlp_status: UpdateStatus,
    deno_status: DenoUpdateStatus,
    cookies_status: CookiesUpdateStatus,
    console_logger: logging.Logger,
) -> None:
    """Print the four-to-five line runtime header to the operator console.

    Output shape:

        ═══════════════════════════════════════════════════════════
          BROADCASTER — генерация публикаций из Google Sheet
        ───────────────────────────────────────────────────────────
          yt-dlp   <version> (<status>)
          deno     <version> (<status>)
          cookies  <name> (<возраст>)
          аккаунт  <youtube channel>           ← only if known
        ═══════════════════════════════════════════════════════════

    Each line is emitted via console_logger.info(...) so it appears on stdout
    and in the operator (screen) log file with no timestamp/level prefix.
    """
    bar_thick: str = "═" * _BANNER_WIDTH
    bar_thin: str = "─" * _BANNER_WIDTH

    console_logger.info(bar_thick)
    console_logger.info("  BROADCASTER — генерация публикаций из Google Sheet")
    console_logger.info(bar_thin)
    console_logger.info("  %s%s", _label("yt-dlp"), _build_ytdlp_info(ytdlp_status))
    console_logger.info("  %s%s", _label("deno"), _build_deno_info(deno_status))
    console_logger.info("  %s%s", _label("cookies"), _build_cookies_info(cookies_status))
    account_line: Optional[str] = _build_account_info(cookies_status)
    if account_line is not None:
        console_logger.info("  %s%s", _label("аккаунт"), account_line)
    console_logger.info(bar_thick)
    console_logger.info("")


def _label(text: str) -> str:
    return f"{text:<{_LABEL_WIDTH}}"


def _build_ytdlp_info(status: UpdateStatus) -> str:
    if status.current_version is None:
        return "не найден"
    version: str = status.current_version
    suffix: Optional[str] = _format_update_suffix(
        attempted=status.attempted,
        succeeded=status.succeeded,
        last_check_days_ago=status.last_check_days_ago,
    )
    if suffix is None:
        return version
    return f"{version} ({suffix})"


def _build_deno_info(status: DenoUpdateStatus) -> str:
    if status.current_version is None:
        # message может быть "не настроен" или другой пояснительный текст
        return status.message or "не настроен"
    # `deno --version` возвращает первой строкой "deno 2.7.14 (stable, release, x86_64-pc-windows-msvc)"
    # Берём только номер версии (вторая токен).
    raw: str = status.current_version or ""
    parts: list[str] = raw.split()
    version: str = parts[1] if len(parts) >= 2 else raw
    suffix: Optional[str] = _format_update_suffix(
        attempted=status.attempted,
        succeeded=status.succeeded,
        last_check_days_ago=status.last_check_days_ago,
    )
    if suffix is None:
        return version
    return f"{version} ({suffix})"


def _build_cookies_info(status: CookiesUpdateStatus) -> str:
    if status.cookies_file is None:
        return "не настроены"
    name: str = status.cookies_file.name
    if not status.file_exists:
        return f"⚠ файл не найден ({name})"
    age_days: Optional[int] = status.file_age_days
    age_text: str = f"{age_days} дн." if age_days is not None else "?"
    if "устарели" in (status.message or ""):
        return f"{name} ⚠ устарели ({age_text}) — обновите вручную"
    return f"{name} (возраст {age_text})"


def _build_account_info(status: CookiesUpdateStatus) -> Optional[str]:
    name: Optional[str] = status.account_name
    if name is None:
        return None
    name = name.strip()
    if not name:
        return None
    return name


def _format_update_suffix(
    *,
    attempted: bool,
    succeeded: bool,
    last_check_days_ago: Optional[int],
) -> Optional[str]:
    if attempted and succeeded:
        return "обновлено сейчас"
    if attempted:
        return "обновление не удалось"
    if last_check_days_ago is not None:
        return f"проверено {last_check_days_ago} дн. назад"
    return None
