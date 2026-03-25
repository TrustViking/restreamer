from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from app.core.models import PlannedVideo
from app.core.official_links import is_official_links_heading
from app.core.url_normalizer import normalize_display_url
from app.core.url_utils import _canonical_domain_key, normalize_official_link_display, strip_tracking_params
from app.resources.resource_loader import load_lines_resource
from app.llm.merges.merge_constants import URL_PATTERN
from app.llm.merges.merge_text_utils import _extract_description_paragraphs_raw, _is_official_links_heading_line
from app.llm.merges.merge_youtube import _is_youtube_host
from app.resources import resolve_official_links_heading

_OFFICIAL_LINK_CONTEXT_HINTS: tuple[str, ...] = load_lines_resource("lexicon_official_link_hints.txt")

@dataclass(frozen=True)
class OfficialLinksSelection:
    found_in_sources: int
    kept_links: tuple[str, ...]

@dataclass(frozen=True)
class OfficialLinksFillResult:
    description: str
    links_in_output: int
    fill_applied: bool

def _normalize_link_candidate(url: str) -> Optional[str]:
    raw_url: str = str(url or "").strip().strip("<>()[]{}").rstrip(".,;")
    if not raw_url:
        return None
    parts = urlsplit(raw_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    stripped_tracking_url: str = strip_tracking_params(raw_url)
    stripped_parts = urlsplit(stripped_tracking_url)
    result_url: str = urlunsplit(
        (stripped_parts.scheme, stripped_parts.netloc, stripped_parts.path, stripped_parts.query, "")
    )
    return normalize_display_url(result_url)

def _official_links_heading(language: str) -> str:
    return resolve_official_links_heading(language)

def _canonical_link_key(url: str) -> str:
    return _canonical_domain_key(url)

def _line_has_official_context(line_text: str) -> bool:
    normalized_line: str = str(line_text or "").strip().lower()
    if not normalized_line:
        return False
    return any(hint in normalized_line for hint in _OFFICIAL_LINK_CONTEXT_HINTS)

def _extract_official_links_from_sources(videos: Sequence[PlannedVideo]) -> OfficialLinksSelection:
    raw_candidates: List[tuple[str, str]] = []
    domain_counts: dict[str, int] = {}
    for video in videos:
        for line in str(video.metadata.description or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            urls: List[str] = [match.group(0) for match in URL_PATTERN.finditer(line)]
            if not urls:
                continue
            for url in urls:
                normalized_url: Optional[str] = _normalize_link_candidate(url)
                if normalized_url is None:
                    continue
                parts = urlsplit(normalized_url)
                if _is_youtube_host(parts.netloc):
                    continue
                raw_candidates.append((normalized_url, line))
                domain: str = parts.netloc.lower().strip()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
    if not raw_candidates:
        return OfficialLinksSelection(found_in_sources=0, kept_links=())

    scored_candidates: List[tuple[int, int, str]] = []
    for index, (url, line_text) in enumerate(raw_candidates):
        parts = urlsplit(url)
        domain: str = parts.netloc.lower().strip()
        score: int = 0
        if parts.scheme == "https":
            score += 20
        if _line_has_official_context(line_text):
            score += 20
        score += min(20, domain_counts.get(domain, 1) * 5)
        if parts.query:
            score -= 5
        score += max(0, 15 - (len(url) // 15))
        scored_candidates.append((score, -index, url))

    scored_candidates.sort(reverse=True)
    selected_links: List[str] = []
    seen_keys: set[str] = set()
    for _, _, url in scored_candidates:
        canonical_key: str = _canonical_link_key(url)
        if canonical_key in seen_keys:
            continue
        seen_keys.add(canonical_key)
        selected_links.append(url)
        if len(selected_links) >= 3:
            break
    return OfficialLinksSelection(
        found_in_sources=len(raw_candidates),
        kept_links=tuple(selected_links),
    )

def _description_has_official_links_block(description: str) -> bool:
    normalized: str = str(description or "")
    if not normalized:
        return False
    heading_present: bool = any(
        is_official_links_heading(str(line or "").strip())
        for line in normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    if not heading_present:
        return False
    return any(
        not _is_youtube_host(urlsplit(match.group(0)).netloc)
        for match in URL_PATTERN.finditer(normalized)
    )

def _count_output_official_links(description: str) -> int:
    seen_keys: set[str] = set()
    for match in URL_PATTERN.finditer(str(description or "")):
        normalized_url: Optional[str] = _normalize_link_candidate(match.group(0))
        if normalized_url is None:
            continue
        if _is_youtube_host(urlsplit(normalized_url).netloc):
            continue
        seen_keys.add(_canonical_link_key(normalized_url))
    return len(seen_keys)

def _looks_like_close_paragraph(text: str) -> bool:
    normalized: str = str(text or "").strip().lower()
    if not normalized:
        return False
    if "#" in normalized:
        return True
    return any(hint in normalized for hint in ("subscribe", "join", "watch", "follow", "диві", "долуч", "смотрите", "подпис"))

def _inject_official_links_block_if_missing(
    *,
    description: str,
    language: str,
    official_links: Sequence[str],
) -> OfficialLinksFillResult:
    if not official_links:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    if _description_has_official_links_block(description):
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    paragraphs: List[str] = _extract_description_paragraphs_raw(description)
    if not paragraphs:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=0,
            fill_applied=False,
        )
    normalized_official_links: List[str] = [
        normalize_official_link_display(str(url or "").strip())
        for url in official_links
        if str(url or "").strip()
    ]
    links_block: str = "\n".join([_official_links_heading(language), *normalized_official_links]).strip()
    updated_paragraphs: List[str] = list(paragraphs)
    if len(updated_paragraphs) <= 3:
        if _looks_like_close_paragraph(updated_paragraphs[-1]):
            updated_paragraphs.insert(-1, links_block)
        else:
            updated_paragraphs.append(links_block)
    elif _looks_like_close_paragraph(updated_paragraphs[-1]):
        updated_paragraphs[-1] = f"{links_block}\n{updated_paragraphs[-1].strip()}".strip()
    else:
        return OfficialLinksFillResult(
            description=description,
            links_in_output=_count_output_official_links(description),
            fill_applied=False,
        )
    updated_description: str = "\n\n".join(
        paragraph for paragraph in updated_paragraphs if paragraph.strip()
    ).strip()
    return OfficialLinksFillResult(
        description=updated_description,
        links_in_output=_count_output_official_links(updated_description),
        fill_applied=True,
    )
