from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class VideoMetadata:
    url: str
    title: str
    description: str
    thumbnail_url: str
    youtube_language: Optional[str]


@dataclass(frozen=True)
class NormalizedImage:
    bytes_data: bytes
    extension: str
    mime_type: str


@dataclass(frozen=True)
class SheetRow:
    row_number: int
    link: str
    date_raw: str
    time_raw: str
    merge_raw: str
    merge_languages: List[str]
    links_column_index: int


@dataclass(frozen=True)
class RowVideoCharacteristics:
    row_index: int
    raw_link: str
    normalized_link: str
    date: str
    time: str
    detected_source_language: str
    merge_languages: List[str]
    base_block_language: str


@dataclass(frozen=True)
class LinkNormalizationCandidate:
    row_index: int
    column_ref: str
    old_value: str
    new_value: str


@dataclass(frozen=True)
class PlannedVideo:
    row_number: int
    original_link: str
    normalized_link: str
    scheduled_at_kiev: datetime
    date_key: str
    date_display: str
    language: str
    metadata: VideoMetadata
    thumbnail: NormalizedImage
    local_thumbnail_path: Optional[Path]
    forced_block_language: Optional[str] = None
    row_characteristics: Optional[RowVideoCharacteristics] = None


@dataclass(frozen=True)
class PreparedVideo:
    row_number: int
    original_link: str
    normalized_link: str
    date_raw: str
    time_raw: str
    scheduled_at_kiev: datetime
    date_key: str
    date_display: str
    language: str
    metadata: VideoMetadata
    thumbnail: NormalizedImage
    local_thumbnail_path: Optional[Path]
    merge_raw: str
    merge_languages: List[str]


@dataclass(frozen=True)
class MergedLanguageContent:
    title: str
    description: str
    cta_text: Optional[str] = None
    hashtags_line: Optional[str] = None
    links: Tuple[str, ...] = ()
    title_selected: Optional[str] = None
    description_selected: Optional[str] = None
    title_audit: Optional[str] = None
    description_audit: Optional[str] = None
    llm_model: Optional[str] = None


@dataclass(frozen=True)
class LanguageMergeAttempt:
    language: str
    model_name: str
    raw_response_text: str
    merged: Optional[MergedLanguageContent]
    error_summary: Optional[str]
    salvaged_title: Optional[str] = None
    publish_source_label: Optional[str] = None
    plain_repair_used: Optional[bool] = None
    plain_repair_fallback_model: Optional[str] = None
    validation_reasons: Optional[List[str]] = None


@dataclass(frozen=True)
class MergedPublicationPayload:
    title_text: str
    description_text: str
