from __future__ import annotations

from urllib.parse import SplitResult, urlsplit, urlunsplit


def normalize_display_url(url: str) -> str:
    raw_url: str = str(url or "")
    if not raw_url:
        return raw_url
    try:
        parsed: SplitResult = urlsplit(raw_url)
    except Exception:
        return raw_url
    scheme: str = str(parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return raw_url
    if not parsed.netloc:
        return raw_url
    if parsed.path not in ("", "/"):
        return raw_url
    if parsed.query or parsed.fragment:
        return raw_url
    try:
        normalized_url: str = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    except Exception:
        return raw_url
    return normalized_url
