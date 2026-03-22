"""Shared URL utility constants for the restreamer pipeline."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

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
