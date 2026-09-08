from __future__ import annotations

import unittest
from pathlib import Path
from typing import List

from app.runtime.cookies_updater import CookiesUpdateStatus
from app.runtime.deno_updater import DenoUpdateStatus
from app.runtime.startup_banner import print_runtime_banner as _emit_banner
from app.runtime.ytdlp_updater import UpdateStatus


class _CapturingLogger:
    """Минимальный stand-in для logger.info(...)."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def info(self, msg: str, *args: object) -> None:
        if args:
            try:
                self.lines.append(msg % args)
            except Exception:
                self.lines.append(msg)
        else:
            self.lines.append(msg)


class StartupBannerTests(unittest.TestCase):
    def _ytdlp(self, **overrides) -> UpdateStatus:
        defaults = dict(
            current_version="2025.09.01",
            last_check_days_ago=0,
            attempted=True,
            succeeded=True,
            message="updated",
        )
        defaults.update(overrides)
        return UpdateStatus(**defaults)

    def _deno(self, **overrides) -> DenoUpdateStatus:
        defaults = dict(
            current_version="deno 2.7.14 (stable, release, x86_64-pc-windows-msvc)",
            last_check_days_ago=0,
            attempted=True,
            succeeded=True,
            message="v2.7.14",
        )
        defaults.update(overrides)
        return DenoUpdateStatus(**defaults)

    def _cookies(self, **overrides) -> CookiesUpdateStatus:
        defaults = dict(
            cookies_file=Path("secrets/cookies.txt"),
            file_exists=True,
            file_age_days=0,
            message="актуальны",
            format_valid=True,
            account_name=None,
        )
        defaults.update(overrides)
        return CookiesUpdateStatus(**defaults)

    def test_banner_emits_header_separators_and_three_status_lines(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(),
            deno_status=self._deno(),
            cookies_status=self._cookies(),
            console_logger=sink,
        )
        # 1 thick + title + thin + 3 status + thick + blank = 8 lines
        self.assertEqual(len(sink.lines), 8)
        self.assertTrue(sink.lines[0].startswith("═"))
        self.assertIn("BROADCASTER", sink.lines[1])
        self.assertTrue(sink.lines[2].startswith("─"))
        self.assertIn("yt-dlp", sink.lines[3])
        self.assertIn("2025.09.01", sink.lines[3])
        self.assertIn("deno", sink.lines[4])
        self.assertIn("2.7.14", sink.lines[4])  # extracted from "deno 2.7.14 (...)"
        self.assertIn("cookies", sink.lines[5])
        self.assertIn("cookies.txt", sink.lines[5])
        self.assertTrue(sink.lines[6].startswith("═"))
        self.assertEqual(sink.lines[7], "")

    def test_banner_includes_account_line_when_account_name_present(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(),
            deno_status=self._deno(),
            cookies_status=self._cookies(account_name="My YouTube Channel"),
            console_logger=sink,
        )
        # 8 base lines + 1 account line = 9 lines
        self.assertEqual(len(sink.lines), 9)
        # account line must appear between cookies (idx 5) and closing thick bar
        account_line_index: int = 6
        self.assertIn("аккаунт", sink.lines[account_line_index])
        self.assertIn("My YouTube Channel", sink.lines[account_line_index])
        self.assertTrue(sink.lines[7].startswith("═"))

    def test_banner_omits_account_line_when_account_name_blank(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(),
            deno_status=self._deno(),
            cookies_status=self._cookies(account_name="   "),
            console_logger=sink,
        )
        self.assertEqual(len(sink.lines), 8)
        self.assertFalse(any("аккаунт" in line for line in sink.lines))

    def test_ytdlp_line_shows_check_age_when_not_attempted(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(attempted=False, succeeded=False, last_check_days_ago=3),
            deno_status=self._deno(),
            cookies_status=self._cookies(),
            console_logger=sink,
        )
        self.assertIn("проверено 3 дн. назад", sink.lines[3])

    def test_ytdlp_line_shows_failure_when_attempted_failed(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(attempted=True, succeeded=False),
            deno_status=self._deno(),
            cookies_status=self._cookies(),
            console_logger=sink,
        )
        self.assertIn("обновление не удалось", sink.lines[3])

    def test_ytdlp_line_shows_not_found_when_version_none(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(current_version=None),
            deno_status=self._deno(),
            cookies_status=self._cookies(),
            console_logger=sink,
        )
        self.assertIn("не найден", sink.lines[3])

    def test_deno_line_falls_back_to_message_when_no_version(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(),
            deno_status=self._deno(current_version=None, message="не настроен"),
            cookies_status=self._cookies(),
            console_logger=sink,
        )
        self.assertIn("не настроен", sink.lines[4])

    def test_cookies_line_shows_not_configured_when_no_path(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(),
            deno_status=self._deno(),
            cookies_status=self._cookies(
                cookies_file=None,
                file_exists=False,
                file_age_days=None,
                message="не настроены",
            ),
            console_logger=sink,
        )
        self.assertIn("не настроены", sink.lines[5])

    def test_cookies_line_warns_on_missing_file(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(),
            deno_status=self._deno(),
            cookies_status=self._cookies(
                cookies_file=Path("secrets/cookies.txt"),
                file_exists=False,
                file_age_days=None,
                message="файл не найден",
            ),
            console_logger=sink,
        )
        self.assertIn("⚠", sink.lines[5])
        self.assertIn("файл не найден", sink.lines[5])

    def test_cookies_line_warns_on_stale(self) -> None:
        sink = _CapturingLogger()
        _emit_banner(
            ytdlp_status=self._ytdlp(),
            deno_status=self._deno(),
            cookies_status=self._cookies(
                file_age_days=120,
                message="устарели (120 дн.)",
            ),
            console_logger=sink,
        )
        self.assertIn("устарели", sink.lines[5])
        self.assertIn("120", sink.lines[5])


if __name__ == "__main__":
    unittest.main()
