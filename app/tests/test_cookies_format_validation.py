from pathlib import Path
import logging

from app.runtime.cookies_updater import check_cookies, validate_cookies_format


def test_valid_netscape_header(tmp_path: Path) -> None:
    f = tmp_path / "cookies.txt"
    f.write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tFALSE\t0\tname\tvalue\n", encoding="utf-8")
    ok, reason = validate_cookies_format(f)
    assert ok is True
    assert reason == "ok"


def test_valid_netscape_header_with_bom(tmp_path: Path) -> None:
    f = tmp_path / "cookies.txt"
    f.write_bytes("\ufeff# Netscape HTTP Cookie File\n".encode("utf-8"))
    ok, reason = validate_cookies_format(f)
    assert ok is True


def test_empty_file(tmp_path: Path) -> None:
    f = tmp_path / "cookies.txt"
    f.write_text("", encoding="utf-8")
    ok, reason = validate_cookies_format(f)
    assert ok is False
    assert "пустой" in reason


def test_only_whitespace(tmp_path: Path) -> None:
    f = tmp_path / "cookies.txt"
    f.write_text("\n\n   \n\t\n", encoding="utf-8")
    ok, reason = validate_cookies_format(f)
    assert ok is False
    assert "пустой" in reason


def test_json_content(tmp_path: Path) -> None:
    f = tmp_path / "cookies.txt"
    f.write_text('{"key": "value"}\n', encoding="utf-8")
    ok, reason = validate_cookies_format(f)
    assert ok is False
    assert "Netscape" in reason


def test_garbage_content(tmp_path: Path) -> None:
    f = tmp_path / "cookies.txt"
    f.write_text("это не cookies, а просто текст\n", encoding="utf-8")
    ok, reason = validate_cookies_format(f)
    assert ok is False


def test_check_cookies_returns_invalid_status_for_empty_file(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("", encoding="utf-8")

    status = check_cookies(
        cookies_file=cookies,
        warn_age_days=30,
        logger=logging.getLogger("test_cookies_empty"),
        ytdlp_path=None,
    )

    assert status.file_exists is True
    assert status.format_valid is False
    assert status.file_age_days is None
    assert status.account_name is None
    assert "невалидный формат" in status.message


def test_check_cookies_returns_invalid_status_for_garbage(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text('{"json": "not netscape"}\n', encoding="utf-8")

    status = check_cookies(
        cookies_file=cookies,
        warn_age_days=30,
        logger=logging.getLogger("test_cookies_garbage"),
        ytdlp_path=None,
    )

    assert status.file_exists is True
    assert status.format_valid is False
    assert "невалидный формат" in status.message


def test_check_cookies_returns_valid_status_for_correct_file(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.txt"
    cookies.write_text(
        "# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tFALSE\t0\tx\ty\n",
        encoding="utf-8",
    )

    status = check_cookies(
        cookies_file=cookies,
        warn_age_days=30,
        logger=logging.getLogger("test_cookies_valid"),
        ytdlp_path=None,
    )

    assert status.file_exists is True
    assert status.format_valid is True
    assert status.file_age_days is not None  # возраст посчитан


def test_check_cookies_returns_none_format_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.txt"

    status = check_cookies(
        cookies_file=missing,
        warn_age_days=30,
        logger=logging.getLogger("test_cookies_missing"),
        ytdlp_path=None,
    )

    assert status.file_exists is False
    assert status.format_valid is None  # None = файла нет, формат не применим
