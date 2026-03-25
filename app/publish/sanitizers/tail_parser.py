from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from app.core.cta_detection import CTA_HINTS, looks_like_cta_line as _looks_like_cta_line_shared
from app.core.text_utils import split_paragraphs
from app.llm.merges.merge_constants import ALLOWED_BULLET_MARKERS, URL_LINE_PATTERN
from app.publish.sanitizers.url_selector import _sanitize_source_url, _dedupe_nonempty


_HASHTAG_TOKEN_RE: re.Pattern[str] = re.compile(r"^#[^\s#]+$")
_DOUBLE_BULLET_MARKER_ALT: str = "|".join(re.escape(marker) for marker in ALLOWED_BULLET_MARKERS)
_DOUBLE_BULLET_RE: re.Pattern[str] = re.compile(
    rf"^(\s*)({_DOUBLE_BULLET_MARKER_ALT})((?:\s+(?:{_DOUBLE_BULLET_MARKER_ALT}))+)\s*",
    re.MULTILINE,
)
@dataclass(frozen=True)
class TailParts:
    body_end_index: int
    cta_lines: List[str]
    hashtag_lines: List[str]
    hashtags_split_from_cta: bool
    source_urls: List[str]
    source_url_change_count: int
    malformed_source_urls_dropped: int


@dataclass(frozen=True)
class EmbeddedTailParts:
    body_text: str
    cta_lines: List[str]
    hashtag_lines: List[str]
    hashtags_split_from_cta: bool
    source_urls: List[str]
    source_url_change_count: int
    malformed_source_urls_dropped: int


def _dedupe_cta_lines(lines: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned_line: str = re.sub(r"\s+", " ", str(line or "")).strip()
        if not cleaned_line:
            continue
        normalized_key: str = cleaned_line.lower()
        if normalized_key in seen:
            continue
        seen.add(normalized_key)
        result.append(cleaned_line)
    return result


class TailParser:
    """Parses CTA, hashtag, and source URL 'tail' fragments from LLM description text."""

    @staticmethod
    def is_source_url_line(line: str) -> bool:
        candidate: str = str(line or "").strip().strip("<>()[]{}").rstrip(".,;")
        return bool(candidate and URL_LINE_PATTERN.fullmatch(candidate))

    @staticmethod
    def is_hashtags_line(line: str) -> bool:
        tokens: List[str] = [item for item in str(line or "").split() if item]
        return bool(tokens) and all(_HASHTAG_TOKEN_RE.fullmatch(item) for item in tokens)

    @staticmethod
    def looks_like_cta_line(line: str) -> bool:
        return _looks_like_cta_line_shared(line)

    @staticmethod
    def is_standalone_cta_line(line: str) -> bool:
        normalized_line: str = re.sub(r"\s+", " ", str(line or "")).strip().lower()
        if not normalized_line:
            return False
        normalized_line = normalized_line.lstrip("-*•> ")
        return any(normalized_line.startswith(hint) for hint in CTA_HINTS)

    @staticmethod
    def merge_hashtag_lines(lines: Sequence[str]) -> str:
        deduped_tokens: List[str] = []
        seen_tokens: set[str] = set()
        for line in lines:
            for token in str(line or "").split():
                cleaned_token: str = token.strip()
                if not cleaned_token or not _HASHTAG_TOKEN_RE.fullmatch(cleaned_token):
                    continue
                normalized_key: str = cleaned_token.lower()
                if normalized_key in seen_tokens:
                    continue
                seen_tokens.add(normalized_key)
                deduped_tokens.append(cleaned_token)
        return " ".join(deduped_tokens)

    @staticmethod
    def dedupe_cta_lines(lines: Sequence[str]) -> List[str]:
        return _dedupe_cta_lines(lines)

    @staticmethod
    def clean_double_bullet_markers(text: str) -> str:
        """Collapse doubled or mixed bullet markers at the start of a line to the first marker only."""
        return _DOUBLE_BULLET_RE.sub(r"\1\2 ", text)

    @staticmethod
    def extract_url_tail(paragraph: str) -> tuple[str, List[str], int, int]:
        normalized_paragraph: str = str(paragraph or "").strip()
        if not normalized_paragraph:
            return ("", [], 0, 0)
        match: Optional[re.Match[str]] = re.search(
            r"(?is)(?:^|[\s\(\[])(https?://\S+(?:\s+https?://\S+)*)\s*$",
            normalized_paragraph,
        )
        if match is None:
            return (normalized_paragraph, [], 0, 0)
        prefix_text: str = normalized_paragraph[: match.start(1)].rstrip()
        raw_urls: List[str] = [item for item in match.group(1).split() if item]
        sanitized_urls: List[str] = []
        change_count: int = 0
        malformed_url_count: int = 0
        for raw_url in raw_urls:
            sanitized_url: Optional[str] = _sanitize_source_url(raw_url)
            if sanitized_url is None:
                malformed_url_count += 1
                continue
            if sanitized_url != raw_url:
                change_count += 1
            sanitized_urls.append(sanitized_url)
        return (
            prefix_text,
            _dedupe_nonempty(sanitized_urls),
            change_count,
            malformed_url_count,
        )

    @staticmethod
    def extract_hashtag_tail(paragraph: str) -> tuple[str, str]:
        normalized_paragraph: str = str(paragraph or "").strip()
        if not normalized_paragraph:
            return ("", "")
        match: Optional[re.Match[str]] = re.search(
            r"(?is)(#[^\s#]+(?:\s+#[^\s#]+)*)\s*$",
            normalized_paragraph,
        )
        if match is None:
            return (normalized_paragraph, "")
        hashtag_candidate: str = match.group(1).strip()
        if not TailParser.is_hashtags_line(hashtag_candidate):
            return (normalized_paragraph, "")
        prefix_text: str = normalized_paragraph[: match.start(1)].rstrip(" ,;")
        return (prefix_text, hashtag_candidate)

    @staticmethod
    def extract_cta_tail(paragraph: str) -> tuple[str, str]:
        normalized_paragraph: str = str(paragraph or "").strip()
        if not normalized_paragraph:
            return ("", "")
        sentence_parts: List[str] = re.split(r"(?<=[.!?…])\s+", normalized_paragraph)
        if len(sentence_parts) < 2:
            if TailParser.looks_like_cta_line(normalized_paragraph):
                return ("", normalized_paragraph)
            return (normalized_paragraph, "")
        cta_candidate: str = sentence_parts[-1].strip()
        if not TailParser.looks_like_cta_line(cta_candidate):
            return (normalized_paragraph, "")
        body_candidate: str = " ".join(sentence_parts[:-1]).strip()
        if not body_candidate:
            return (normalized_paragraph, "")
        return (body_candidate, cta_candidate)

    @staticmethod
    def split_tail(lines: Sequence[str]) -> TailParts:
        body_end_index: int = len(lines)
        source_urls: List[str] = []
        source_url_change_count: int = 0
        malformed_source_urls_dropped: int = 0
        while body_end_index > 0:
            candidate: str = lines[body_end_index - 1].strip()
            if not candidate:
                body_end_index -= 1
                continue
            if not TailParser.is_source_url_line(candidate):
                break
            sanitized_url: Optional[str] = _sanitize_source_url(candidate)
            if sanitized_url is None:
                malformed_source_urls_dropped += 1
                body_end_index -= 1
                continue
            if sanitized_url != candidate:
                source_url_change_count += 1
            source_urls.insert(0, sanitized_url)
            body_end_index -= 1

        hashtag_lines: List[str] = []
        hashtags_split_from_cta: bool = False
        while body_end_index > 0:
            candidate = lines[body_end_index - 1].strip()
            if not candidate:
                body_end_index -= 1
                continue
            if not TailParser.is_hashtags_line(candidate):
                break
            hashtag_lines.insert(0, candidate)
            body_end_index -= 1

        cta_lines: List[str] = []
        while body_end_index > 0:
            candidate = lines[body_end_index - 1].strip()
            if not candidate:
                body_end_index -= 1
                continue
            candidate_without_hashtags: str
            extracted_hashtags: str
            candidate_without_hashtags, extracted_hashtags = TailParser.extract_hashtag_tail(candidate)
            if extracted_hashtags and TailParser.is_standalone_cta_line(candidate_without_hashtags):
                hashtag_lines.insert(0, extracted_hashtags)
                hashtags_split_from_cta = True
                candidate = candidate_without_hashtags
            if not candidate:
                body_end_index -= 1
                continue
            if not TailParser.is_standalone_cta_line(candidate):
                break
            cta_lines.insert(0, candidate)
            body_end_index -= 1

        return TailParts(
            body_end_index=body_end_index,
            cta_lines=cta_lines,
            hashtag_lines=hashtag_lines,
            hashtags_split_from_cta=hashtags_split_from_cta,
            source_urls=source_urls,
            source_url_change_count=source_url_change_count,
            malformed_source_urls_dropped=malformed_source_urls_dropped,
        )

    @staticmethod
    def extract_embedded(text: str) -> EmbeddedTailParts:
        paragraphs: List[str] = split_paragraphs(text)
        cleaned_paragraphs: List[str] = []
        cta_lines: List[str] = []
        hashtag_lines: List[str] = []
        hashtags_split_from_cta: bool = False
        source_urls: List[str] = []
        source_url_change_count: int = 0
        malformed_source_urls_dropped: int = 0

        for paragraph in paragraphs:
            cleaned_paragraph: str = paragraph.strip()
            cleaned_paragraph, extracted_urls, url_change_count, malformed_url_count = (
                TailParser.extract_url_tail(cleaned_paragraph)
            )
            cleaned_paragraph, extracted_hashtags = TailParser.extract_hashtag_tail(cleaned_paragraph)
            cleaned_paragraph, extracted_cta = TailParser.extract_cta_tail(cleaned_paragraph)
            if extracted_hashtags and extracted_cta:
                hashtags_split_from_cta = True
            if cleaned_paragraph:
                cleaned_paragraphs.append(cleaned_paragraph)
            source_urls.extend(extracted_urls)
            source_url_change_count += url_change_count
            malformed_source_urls_dropped += malformed_url_count
            if extracted_hashtags:
                hashtag_lines.append(extracted_hashtags)
            if extracted_cta:
                cta_lines.append(extracted_cta)

        return EmbeddedTailParts(
            body_text="\n\n".join(cleaned_paragraphs).strip(),
            cta_lines=cta_lines,
            hashtag_lines=hashtag_lines,
            hashtags_split_from_cta=hashtags_split_from_cta,
            source_urls=_dedupe_nonempty(source_urls),
            source_url_change_count=source_url_change_count,
            malformed_source_urls_dropped=malformed_source_urls_dropped,
        )
