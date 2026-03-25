"""Shared URL utility constants for the restreamer pipeline."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_LANG_ONLY_PATH_RE: re.Pattern[str] = re.compile(
    r"^/[a-z]{2}(?:-[a-z]{2})?/?$",
    re.IGNORECASE,
)


def _canonical_domain_key(url: str) -> str:
    """Return a canonical key for URL deduplication.

    Lowercases and strips ``www.`` from the domain, strips language-code-only
    path suffixes (e.g., ``/uk``, ``/en``, ``/fr-fr``), strips query and
    fragment, and strips trailing slashes from non-trivial paths.
    """
    parsed = urlsplit(str(url or "").strip())
    scheme: str = parsed.scheme.lower()
    domain: str = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    path: str = parsed.path
    if _LANG_ONLY_PATH_RE.match(path):
        path = ""
    else:
        path = path.rstrip("/")
    return f"{scheme}://{domain}{path}"


TRACKING_QUERY_KEYS: tuple[str, ...] = (
    "si",
    "feature",
    "pp",
    "fbclid",
    "gclid",
    "igsh",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref_src",
    "ref_url",
    "spm",
)

SOCIAL_PLATFORM_HOSTS: frozenset[str] = frozenset(
    {
        "x.com",
        "twitter.com",
        "t.me",
        "telegram.me",
        "facebook.com",
        "fb.com",
        "instagram.com",
        "threads.net",
        "linkedin.com",
        "tiktok.com",
        "reddit.com",
        "vk.com",
    }
)


def is_social_platform_host(host: str) -> bool:
    normalized_host: str = str(host or "").strip().lower()
    if normalized_host.startswith("www."):
        normalized_host = normalized_host[4:]
    return normalized_host in SOCIAL_PLATFORM_HOSTS


def strip_tracking_params(url: str) -> str:
    raw_url: str = str(url or "").strip()
    if not raw_url:
        return raw_url
    try:
        parts = urlsplit(raw_url)
    except Exception:
        return raw_url
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return raw_url
    filtered_query_items: list[tuple[str, str]] = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        normalized_key: str = key.lower().strip()
        if normalized_key.startswith("utm_") or normalized_key in TRACKING_QUERY_KEYS:
            continue
        filtered_query_items.append((key, value))
    sanitized_query: str = urlencode(filtered_query_items, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, sanitized_query, parts.fragment))


def normalize_official_link_display(url: str) -> str:
    raw_url: str = str(url or "").strip()
    if not raw_url:
        return raw_url
    try:
        parts = urlsplit(raw_url)
    except Exception:
        return raw_url
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return raw_url
    host: str = parts.netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return raw_url
    if is_social_platform_host(host):
        return strip_tracking_params(raw_url)
    return f"{parts.scheme}://{host}"
