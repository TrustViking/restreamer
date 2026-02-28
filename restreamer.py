from __future__ import annotations

import dataclasses
import io
import json
import logging
import os
import re
import secrets
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Set,
    cast,
)
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dotenv import load_dotenv

from yt_dlp import YoutubeDL
import yaml
from app.bootstrap.cli import build_cli_parser
from app.bootstrap.logging_config import (
    get_logger,
    resolve_logger_name_meta,
    setup_logging,
)
from app.config.app_config_loader import (
    load_config_from_env as _load_config_from_env_impl,
)
from app.config.settings import AppConfig, AppTemplates
from app.config.validators import (
    describe_google_doc_share_mode,
    normalize_processing_mode,
    now_filter_timezone,
)
from app.google import (
    GoogleDocsClient,
    GoogleDriveClient,
    GoogleServicesFactory,
    GoogleSheetsClient,
)
from app.planning import (
    deduplicate_planned_videos_within_date_language,
    deduplicate_planned_videos_within_slot_language,
    format_time_key_for_display,
    language_index,
    merge_semantics_from_env,
    parse_merge_languages,
    planned_video_block_language,
    planned_video_slot_key,
    planned_video_time_key,
)
from app.publish import GoogleDocsReportWriter
from app.publish.doc_helpers import _no_description_text as _publish_no_description_text
from app.publish.telegram_renderer import (
    build_titles_summary,
    build_single_mode_message,
    build_telegram_header_text,
    build_telegram_key_form_reminder,
    build_telegram_language_block,
    build_telegram_language_digest_block,
    build_telegram_language_merged_block,
    build_telegram_language_nomerge_block,
    build_telegram_post_header_text,
)
from app.llm import (
    MergeRunSummary,
    attempt_openai_merge_with_audit,
    attempt_openai_single_source_translate_with_audit,
    enforce_openai_merged_paragraphs,
)
from app.core.constants import (
    LOGGER_NAME_ENV_VAR,
    MERGED_DESCRIPTION_HARD_CEILING,
    URL_PATTERN,
    _SOURCE_URL_LINE_RE,
)
from app.core.models import (
    LanguageMergeAttempt,
    LinkNormalizationCandidate,
    MergedLanguageContent,
    NormalizedImage,
    PlannedVideo,
    PlannedVideoItem,
    RowVideoCharacteristics,
    SheetRow,
    VideoMetadata,
)

# Google API (опционально)
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

LOGGER = get_logger(__name__)
_OPENAI_CLIENT: Optional[Any] = None
_ACTIVE_TEMPLATES: Optional["AppTemplates"] = None


# ----------------------------
# Получение метаданных YouTube
# ----------------------------


class YouTubeMetadataFetcher:
    def fetch(self, video_url: str) -> VideoMetadata:
        raise NotImplementedError


class YtDlpYouTubeMetadataFetcher(YouTubeMetadataFetcher):
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._timeout_seconds: float = timeout_seconds

    def fetch(self, video_url: str) -> VideoMetadata:
        ydl_options: Dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "socket_timeout": self._timeout_seconds,
            "extract_flat": False,
            "noplaylist": True,
        }
        with YoutubeDL(cast(Any, ydl_options)) as ydl:
            ydl_any: Any = ydl
            info: Dict[str, Any] = cast(
                Dict[str, Any],
                ydl_any.extract_info(video_url, download=False),
            )
        LOGGER.debug(
            "yt-dlp diagnostics: _type=%s id=%s webpage_url=%s",
            info.get("_type"),
            info.get("id"),
            info.get("webpage_url"),
        )
        title: str = str(info.get("title") or "").strip()
        description: str = str(info.get("description") or "").strip()
        thumbnail_url: str = str(info.get("thumbnail") or "").strip()
        video_id: str = str(info.get("id") or "").strip()
        youtube_language: Optional[str] = (
            str(info.get("language") or info.get("channel_language") or "").strip()
            or None
        )

        if not title:
            raise ValueError("Не удалось получить title (yt-dlp вернул пусто).")
        if not thumbnail_url:
            fallback_id: Optional[str] = None
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                fallback_id = video_id
            else:
                fallback_match: Optional[re.Match[str]] = re.search(
                    r"(?:youtu\.be/|v=)([A-Za-z0-9_-]{11})",
                    video_url,
                )
                if fallback_match:
                    fallback_id = fallback_match.group(1)

            if fallback_id:
                thumbnail_url = f"https://i.ytimg.com/vi/{fallback_id}/hqdefault.jpg"
                LOGGER.warning(
                    "yt-dlp returned empty thumbnail_url; using fallback video id %s: %s",
                    fallback_id,
                    thumbnail_url,
                )
            else:
                raise ValueError(
                    "Не удалось получить thumbnail_url (yt-dlp вернул пусто)."
                )
        if not description:
            LOGGER.warning(
                "yt-dlp returned empty description for url=%s (id=%s).",
                video_url,
                video_id or "unknown",
            )

        return VideoMetadata(
            url=video_url,
            title=title,
            description=description,
            thumbnail_url=thumbnail_url,
            youtube_language=youtube_language,
        )


def _extract_youtube_video_id(raw_value: str) -> Optional[str]:
    text: str = str(raw_value or "").strip()
    if not text:
        return None
    context_patterns: Tuple[Tuple[str, str], ...] = (
        (
            "context",
            r"(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/watch\?[^#\s]*?(?:[?&]v=|&v=)([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
    )
    first_context_match: Optional[Tuple[int, str, str]] = None
    for method_name, pattern in context_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate: str = str(match.group(1) or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                continue
            candidate_pos: int = match.start(1)
            if first_context_match is None or candidate_pos < first_context_match[0]:
                first_context_match = (candidate_pos, method_name, candidate)
    if first_context_match is not None:
        _, method_name, candidate = first_context_match
        LOGGER.debug("youtube_id_extracted method=%s id=%s", method_name, candidate)
        return candidate

    for match in re.finditer(r"([A-Za-z0-9_-]{11})", text):
        candidate: str = str(match.group(1) or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            continue
        start_index: int = match.start(1)
        end_index: int = match.end(1)
        char_before: str = text[start_index - 1] if start_index > 0 else ""
        char_after: str = text[end_index] if end_index < len(text) else ""
        if char_before and re.fullmatch(r"[A-Za-z0-9_-]", char_before):
            continue
        if char_after and re.fullmatch(r"[A-Za-z0-9_-]", char_after):
            continue
        LOGGER.debug("youtube_id_extracted method=raw id=%s", candidate)
        return candidate
    return None


def _normalize_youtube_link(raw: str) -> Optional[str]:
    video_id: Optional[str] = _extract_youtube_video_id(raw)
    if not video_id:
        return None
    return f"https://youtu.be/{video_id}"


def _normalize_youtube_video_url(video_url: str) -> str:
    normalized: Optional[str] = _normalize_youtube_link(video_url)
    if not normalized:
        raise ValueError(f"Не удалось извлечь YouTube video id из URL: {video_url}")
    return normalized


# ----------------------------
# Telegram
# ----------------------------


class TelegramBotClient:
    def __init__(
        self, bot_token: str, chat_id: str, timeout_seconds: float = 20.0
    ) -> None:
        self._bot_token: str = bot_token
        self._chat_id: str = chat_id
        self._timeout_seconds: float = timeout_seconds

    def send_text(self, text: str) -> None:
        # В Telegram есть ограничение на длину сообщения; режем на части безопасно.
        chunks: List[str] = _split_text_for_telegram(text=text, max_chunk_size=3500)
        self._send_text_chunks(chunks=chunks)

    def get_me(self) -> Dict[str, Any]:
        url: str = self._make_api_url("getMe")
        response: requests.Response = requests.get(url, timeout=self._timeout_seconds)
        _raise_for_telegram_response(response=response)
        payload: Dict[str, Any] = response.json()
        return cast(Dict[str, Any], payload.get("result", {}))

    def _send_text_chunks(self, chunks: Iterable[str]) -> None:
        for chunk in chunks:
            self._post_json(
                method="sendMessage",
                payload={
                    "chat_id": self._chat_id,
                    "text": chunk,
                    "disable_web_page_preview": False,
                },
            )

    def send_photo_as_file_bytes(
        self,
        photo_bytes: bytes,
        filename: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> None:
        # Use sendDocument to avoid Telegram image compression.
        url: str = self._make_api_url("sendDocument")
        files: Dict[str, Tuple[str, bytes, str]] = {
            "document": (filename, photo_bytes, mime_type),
        }
        data: Dict[str, str] = {"chat_id": self._chat_id}
        if caption:
            data["caption"] = caption

        response: requests.Response = requests.post(
            url, data=data, files=files, timeout=self._timeout_seconds
        )
        _raise_for_telegram_response(response=response)

    def _post_json(self, method: str, payload: Dict[str, Any]) -> None:
        url: str = self._make_api_url(method)
        response: requests.Response = requests.post(
            url, json=payload, timeout=self._timeout_seconds
        )
        _raise_for_telegram_response(response=response)

    def _make_api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self._bot_token}/{method}"


def _raise_for_telegram_response(response: requests.Response) -> None:
    if response.status_code != 200:
        raise RuntimeError(f"Telegram API HTTP {response.status_code}: {response.text}")
    payload: Dict[str, Any] = response.json()
    if not payload.get("ok", False):
        raise RuntimeError(f"Telegram API error: {response.text}")


def _split_text_for_telegram(text: str, max_chunk_size: int) -> List[str]:
    normalized: str = text.replace("\r\n", "\n")
    if len(normalized) <= max_chunk_size:
        return [normalized]

    chunks: List[str] = []
    start_index: int = 0
    while start_index < len(normalized):
        end_index: int = min(start_index + max_chunk_size, len(normalized))
        # Пытаемся разрезать по переводу строки.
        newline_index: int = normalized.rfind("\n", start_index, end_index)
        if newline_index > start_index + 200:
            end_index = newline_index + 1

        chunks.append(normalized[start_index:end_index])
        start_index = end_index
    return chunks

# ----------------------------
# HTTP-вспомогательные функции
# ----------------------------


class HttpClient:
    def __init__(self, timeout_seconds: float = 20.0) -> None:
        self._timeout_seconds: float = timeout_seconds

    def get_bytes(self, url: str) -> bytes:
        response: requests.Response = requests.get(url, timeout=self._timeout_seconds)
        response.raise_for_status()
        return response.content

    def post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response: requests.Response = requests.post(
            url,
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())


# ----------------------------
# Google Docs/Drive (инфраструктура вынесена в app.google)
# ----------------------------


def _build_safe_entity_name(video_title: str) -> str:
    transliterated: str = _transliterate_cyrillic_to_latin(video_title)
    normalized_title: str = re.sub(r"\s+", "_", transliterated.strip().lower())
    clean_title: str = re.sub(r"[^a-z0-9_]+", "_", normalized_title)
    clean_title = re.sub(r"_+", "_", clean_title).strip("_")
    if not clean_title:
        clean_title = "video"
    return clean_title[:120]


def _transliterate_cyrillic_to_latin(value: str) -> str:
    mapping: Dict[str, str] = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "h",
        "ґ": "g",
        "д": "d",
        "е": "e",
        "є": "ie",
        "ж": "zh",
        "з": "z",
        "и": "y",
        "і": "i",
        "ї": "yi",
        "й": "i",
        "к": "k",
        "л": "l",
        "м": "m",
        "н": "n",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s",
        "т": "t",
        "у": "u",
        "ф": "f",
        "х": "kh",
        "ц": "ts",
        "ч": "ch",
        "ш": "sh",
        "щ": "shch",
        "ь": "",
        "ю": "iu",
        "я": "ia",
        "ё": "yo",
        "э": "e",
        "ъ": "",
    }
    output: List[str] = []
    for char in value.lower():
        if char in mapping:
            output.append(mapping[char])
            continue
        if ("a" <= char <= "z") or ("0" <= char <= "9"):
            output.append(char)
            continue
        if char.isspace() or char in {"-", "_"}:
            output.append("_")
            continue
        output.append("_")
    return "".join(output)


def _display_language_code(language: str) -> str:
    if language == "uk":
        return "ua"
    return language


def _build_drive_preview_path_segments(
    *,
    template: str,
    language: str,
    date_key: str,
) -> List[str]:
    raw_template: str = str(template or "").strip()
    if not raw_template:
        return []
    rendered_path: str = raw_template.format(
        streamertg="streamertg",
        preview="preview",
        language=_display_language_code(language),
        date=date_key,
    )
    return [
        segment.strip()
        for segment in re.split(r"[\\/]+", rendered_path)
        if segment.strip()
    ]


def _guess_image_extension_and_mime(image_bytes: bytes) -> Tuple[str, str]:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return ".gif", "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp", "image/webp"
    return ".jpg", "image/jpeg"


def _normalize_thumbnail(image_bytes: bytes) -> NormalizedImage:
    # Единая нормализация thumbnail для Telegram и Drive/Docs.
    # Предпочитаем JPEG для предсказуемой совместимости.
    if Image is None:
        ext, mime = _guess_image_extension_and_mime(image_bytes)
        LOGGER.warning(
            "Pillow is not installed; keeping original thumbnail format (%s).",
            mime,
        )
        return NormalizedImage(
            bytes_data=image_bytes,
            extension=ext,
            mime_type=mime,
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
            output = io.BytesIO()
            rgb_image.save(output, format="JPEG", quality=90)
            return NormalizedImage(
                bytes_data=output.getvalue(),
                extension=".jpg",
                mime_type="image/jpeg",
            )
    except Exception as conversion_error:
        ext, mime = _guess_image_extension_and_mime(image_bytes)
        LOGGER.warning(
            "Thumbnail conversion to JPEG failed; keeping original format (%s). Error: %s",
            mime,
            conversion_error,
        )
        return NormalizedImage(
            bytes_data=image_bytes,
            extension=ext,
            mime_type=mime,
        )


def _value_from_row(row_values: List[str], index: int) -> str:
    if index >= len(row_values):
        return ""
    return str(row_values[index]).strip()


def _normalize_header_name(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", value.strip().lower(), flags=re.IGNORECASE)


def _find_header_index(
    normalized_header: List[str],
    aliases: Tuple[str, ...],
) -> Optional[int]:
    normalized_aliases: Tuple[str, ...] = tuple(
        _normalize_header_name(alias) for alias in aliases
    )
    for index, name in enumerate(normalized_header):
        if name in normalized_aliases:
            return index
    for index, name in enumerate(normalized_header):
        if any(alias in name for alias in normalized_aliases if alias):
            return index
    return None


def _parse_sheet_datetime(date_raw: str, time_raw: str, tz: ZoneInfo) -> datetime:
    date_clean: str = date_raw.strip()
    time_clean: str = time_raw.strip()
    date_formats: Tuple[str, ...] = (
        "%d.%m.%Y",
        "%d.%m.%y",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%d",
        "%d%m%y",
        "%d%m%Y",
    )
    time_formats: Tuple[str, ...] = (
        "%H:%M",
        "%H.%M",
        "%H%M",
        "%H:%M:%S",
        "%I:%M %p",
        "%I %p",
    )

    parsed_date: Optional[datetime] = None
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_clean, fmt)
            break
        except ValueError:
            continue
    if parsed_date is None:
        raise ValueError(f"Unsupported Date format: {date_raw!r}")

    parsed_time: Optional[datetime] = None
    for fmt in time_formats:
        try:
            parsed_time = datetime.strptime(time_clean, fmt)
            break
        except ValueError:
            continue
    if parsed_time is None:
        raise ValueError(f"Unsupported Time format: {time_raw!r}")

    return datetime(
        year=parsed_date.year,
        month=parsed_date.month,
        day=parsed_date.day,
        hour=parsed_time.hour,
        minute=parsed_time.minute,
        tzinfo=tz,
    )

def _column_letters_to_index(column_letters: str) -> Optional[int]:
    cleaned: str = str(column_letters or "").strip().upper()
    if not cleaned or not re.fullmatch(r"[A-Z]+", cleaned):
        return None
    index_value: int = 0
    for symbol in cleaned:
        index_value = (index_value * 26) + (ord(symbol) - ord("A") + 1)
    return index_value


def _column_index_to_letters(column_index_zero_based: int) -> str:
    if column_index_zero_based < 0:
        raise ValueError("column index must be >= 0")
    value: int = column_index_zero_based + 1
    symbols: List[str] = []
    while value > 0:
        value, rem = divmod(value - 1, 26)
        symbols.append(chr(ord("A") + rem))
    return "".join(reversed(symbols))


def _sheet_name_from_range(range_name: str) -> Optional[str]:
    cleaned: str = str(range_name or "").strip()
    if "!" not in cleaned:
        return None
    raw_sheet_name: str = cleaned.split("!", 1)[0].strip()
    if not raw_sheet_name:
        return None
    return raw_sheet_name.strip("'")


def _build_sheet_cell_a1(
    *,
    sheet_name: Optional[str],
    row_index: int,
    col_index_zero_based: int,
) -> str:
    column_letters: str = _column_index_to_letters(col_index_zero_based)
    if row_index < 1:
        raise ValueError("row index must be >= 1")
    if not sheet_name:
        return f"{column_letters}{row_index}"
    escaped_sheet_name: str = str(sheet_name).replace("'", "''")
    return f"'{escaped_sheet_name}'!{column_letters}{row_index}"


def _handle_normalized_link_writeback(
    *,
    sheets_client: GoogleSheetsClient,
    spreadsheet_id: str,
    sheet_name_for_writeback: Optional[str],
    row_number: int,
    links_column_index: int,
    old_link: str,
    normalized_link: str,
    writeback_enabled: bool,
    normalization_candidates: Optional[List[LinkNormalizationCandidate]] = None,
    links_column_label: Optional[str] = None,
) -> bool:
    old_link_value: str = str(old_link or "").strip()
    new_link_value: str = str(normalized_link or "").strip()
    if not new_link_value:
        return False
    if new_link_value == old_link_value:
        LOGGER.debug(
            'Row %d: link_validated changed=false url="%s"',
            row_number,
            new_link_value,
        )
        return False
    if normalization_candidates is not None:
        normalization_candidates.append(
            LinkNormalizationCandidate(
                row_index=row_number,
                column_ref=str(links_column_label or "").strip()
                or _column_index_to_letters(links_column_index),
                old_value=old_link_value,
                new_value=new_link_value,
            )
        )
    if not writeback_enabled:
        LOGGER.info(
            'Row %d: link_changed writeback=disabled old="%s" new="%s"',
            row_number,
            old_link_value,
            new_link_value,
        )
        return True
    try:
        link_cell_a1: str = _build_sheet_cell_a1(
            sheet_name=sheet_name_for_writeback,
            row_index=row_number,
            col_index_zero_based=links_column_index,
        )
        sheets_client.update_cell_string(
            spreadsheet_id=spreadsheet_id,
            cell_a1=link_cell_a1,
            value=new_link_value,
        )
        LOGGER.info(
            'Row %d: link_changed writeback=applied old="%s" new="%s"',
            row_number,
            old_link_value,
            new_link_value,
        )
        return True
    except Exception as writeback_error:
        LOGGER.warning(
            'Row %d: link_changed writeback=failed old="%s" new="%s" reason=%s',
            row_number,
            old_link_value,
            new_link_value,
            _summarize_error(writeback_error),
        )
        return True


def _extract_end_column_letters_from_sheet_range(range_name: str) -> Optional[str]:
    cleaned_range: str = str(range_name or "").strip()
    if not cleaned_range:
        return None
    range_without_sheet: str = cleaned_range.split("!", 1)[-1]
    right_part: str = range_without_sheet.split(":", 1)[-1]
    match: Optional[re.Match[str]] = re.search(r"([A-Za-z]+)\d*$", right_part.strip())
    if match is None:
        return None
    return match.group(1).upper()


def _sheet_range_includes_merge_column(range_name: str) -> bool:
    end_column_letters: Optional[str] = _extract_end_column_letters_from_sheet_range(
        range_name
    )
    end_column_index: Optional[int] = _column_letters_to_index(end_column_letters or "")
    merge_column_index: Optional[int] = _column_letters_to_index("E")
    if end_column_index is None or merge_column_index is None:
        return True
    return end_column_index >= merge_column_index


def _expand_sheet_range_to_af(range_name: str) -> str:
    cleaned_range: str = str(range_name or "").strip()
    if not cleaned_range:
        return "A:F"
    if "!" in cleaned_range:
        sheet_prefix: str = cleaned_range.split("!", 1)[0].strip()
        if sheet_prefix:
            return f"{sheet_prefix}!A:F"
    return "A:F"

def _normalize_language(raw_language: Optional[str]) -> Optional[str]:
    if not raw_language:
        return None
    normalized: str = raw_language.strip().lower()
    if normalized.startswith(("uk", "ua")):
        return "uk"
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("ru"):
        return "ru"
    return None


def _detect_language_from_text(text: str) -> str:
    low: str = text.lower()
    if not low:
        return "other"
    if any(ch in low for ch in "іїєґ"):
        return "uk"

    cyrillic_count: int = len(re.findall(r"[а-яё]", low))
    latin_count: int = len(re.findall(r"[a-z]", low))
    if cyrillic_count > 0 and latin_count == 0:
        return "ru"
    if latin_count >= cyrillic_count and latin_count > 0:
        return "en"
    if cyrillic_count > latin_count:
        return "ru"
    return "other"


def _detect_language(metadata: VideoMetadata) -> str:
    from_youtube: Optional[str] = _normalize_language(metadata.youtube_language)
    if from_youtube:
        return from_youtube
    return _detect_language_from_text(
        f"{metadata.title}\n{metadata.description}".strip()
    )

def _language_heading(language: str) -> str:
    templates: Optional[AppTemplates] = globals().get("_ACTIVE_TEMPLATES")
    if templates is None:
        return "OTHER"
    try:
        payload: Any = json.loads(templates.google_doc_language_headings_json)
        if isinstance(payload, dict):
            return str(payload.get(language, payload.get("other", "OTHER")))
    except Exception:
        pass
    return "OTHER"


def _render_template(template: str, values: Dict[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as error:
        raise RuntimeError(f"Template render failed, missing key: {error}") from error


def _active_templates() -> Optional[AppTemplates]:
    return cast(Optional[AppTemplates], globals().get("_ACTIVE_TEMPLATES"))


def _build_doc_header_text(
    header_context: Dict[str, str],
    templates: AppTemplates,
) -> str:
    context: Dict[str, str] = dict(header_context)
    context.setdefault("language_time_titles", "")
    return _render_template(templates.google_doc_header, context)


def _debug_run_merge_column_self_test() -> None:
    assert parse_merge_languages("") == []
    assert parse_merge_languages("  ") == []
    assert parse_merge_languages("uk") == ["uk"]
    assert parse_merge_languages("ua,ru") == ["uk", "ru"]
    assert parse_merge_languages("ua|ru") == ["uk", "ru"]
    assert parse_merge_languages("ua;ru") == ["uk", "ru"]
    assert parse_merge_languages("RU,ua,en,ru") == ["uk", "en", "ru"]

    scheduled_at: datetime = datetime(2026, 2, 22, 12, 0, tzinfo=timezone.utc)
    base_video: PlannedVideo = PlannedVideo(
        row_number=10,
        original_link="https://youtu.be/AAAAAAAAAAA",
        normalized_link="https://youtu.be/AAAAAAAAAAA",
        scheduled_at_kiev=scheduled_at,
        date_key="220226",
        date_display="22.02.2026",
        language="en",
        metadata=VideoMetadata(
            url="https://youtu.be/AAAAAAAAAAA",
            title="Base title",
            description="Base description",
            thumbnail_url="https://i.ytimg.com/vi/AAAAAAAAAAA/hqdefault.jpg",
            youtube_language="en",
        ),
        thumbnail=NormalizedImage(
            bytes_data=b"x",
            extension="jpg",
            mime_type="image/jpeg",
        ),
        local_thumbnail_path=None,
    )

    unchanged: List[PlannedVideo] = [base_video] + [
        dataclasses.replace(base_video, forced_block_language=language_code)
        for language_code in parse_merge_languages("")
    ]
    assert len(unchanged) == 1
    assert [planned_video_block_language(item) for item in unchanged] == ["en"]

    cloned: List[PlannedVideo] = [base_video] + [
        dataclasses.replace(base_video, forced_block_language=language_code)
        for language_code in parse_merge_languages("ua,ru")
    ]
    assert len(cloned) == 3
    assert [planned_video_block_language(item) for item in cloned] == [
        "en",
        "uk",
        "ru",
    ]

    base_block_lang: str = planned_video_block_language(base_video)
    clones_with_skip: List[PlannedVideo] = []
    for merge_language in parse_merge_languages("uk"):
        if merge_language == base_block_lang:
            continue
        clones_with_skip.append(
            dataclasses.replace(base_video, forced_block_language=merge_language)
        )
    assert len(clones_with_skip) == 1
    assert [planned_video_block_language(item) for item in clones_with_skip] == ["uk"]

    uk_base_video: PlannedVideo = dataclasses.replace(base_video, language="uk")
    uk_base_block_lang: str = planned_video_block_language(uk_base_video)
    uk_clones_with_skip: List[PlannedVideo] = []
    for merge_language in parse_merge_languages("uk"):
        if merge_language == uk_base_block_lang:
            continue
        uk_clones_with_skip.append(
            dataclasses.replace(uk_base_video, forced_block_language=merge_language)
        )
    assert len(uk_clones_with_skip) == 0


def _debug_run_merge_range_semantics_and_dedup_self_test() -> None:
    assert _sheet_range_includes_merge_column("A:F") is True
    assert _sheet_range_includes_merge_column("A:D") is False
    assert _sheet_range_includes_merge_column("Sheet1!A:D") is False
    assert _expand_sheet_range_to_af("A:D") == "A:F"
    assert _expand_sheet_range_to_af("Sheet1!A:D") == "Sheet1!A:F"

    scheduled_at: datetime = datetime(2026, 2, 24, 14, 0, tzinfo=timezone.utc)
    base_video: PlannedVideo = PlannedVideo(
        row_number=2,
        original_link="https://youtu.be/xg7W6wrOYBA",
        normalized_link="https://youtu.be/xg7W6wrOYBA",
        scheduled_at_kiev=scheduled_at,
        date_key="240226",
        date_display="24.02.2026",
        language="en",
        metadata=VideoMetadata(
            url="https://youtu.be/xg7W6wrOYBA",
            title="Base EN",
            description="Desc EN",
            thumbnail_url="https://i.ytimg.com/vi/xg7W6wrOYBA/hqdefault.jpg",
            youtube_language="en",
        ),
        thumbnail=NormalizedImage(
            bytes_data=b"x",
            extension="jpg",
            mime_type="image/jpeg",
        ),
        local_thumbnail_path=None,
    )
    merge_languages: List[str] = parse_merge_languages("ua,ru")

    override_items: List[PlannedVideo] = [base_video]
    base_block_lang: str = planned_video_block_language(base_video)
    for merge_language in merge_languages:
        if merge_language == base_block_lang:
            continue
        override_items.append(
            dataclasses.replace(base_video, forced_block_language=merge_language)
        )
    if (
        merge_languages
        and "override" == "override"
        and base_block_lang not in merge_languages
    ):
        override_items.pop(0)
    assert [planned_video_block_language(item) for item in override_items] == [
        "uk",
        "ru",
    ]

    add_items: List[PlannedVideo] = [base_video]
    for merge_language in merge_languages:
        if merge_language == base_block_lang:
            continue
        add_items.append(
            dataclasses.replace(base_video, forced_block_language=merge_language)
        )
    assert [planned_video_block_language(item) for item in add_items] == [
        "en",
        "uk",
        "ru",
    ]

    duplicate_row_video: PlannedVideo = dataclasses.replace(base_video, row_number=8)
    deduped_items: List[PlannedVideo] = (
        deduplicate_planned_videos_within_date_language(
            [base_video, duplicate_row_video]
        )
    )
    assert len(deduped_items) == 1
    assert deduped_items[0].row_number == 2


def _safe_console_text(text: str) -> str:
    stdout_encoding: str = getattr(sys.stdout, "encoding", None) or "utf-8"
    normalized: str = str(text)
    try:
        normalized.encode(stdout_encoding, errors="strict")
        return normalized
    except Exception:
        try:
            return normalized.encode(stdout_encoding, errors="backslashreplace").decode(
                stdout_encoding, errors="strict"
            )
        except Exception:
            return normalized.encode("utf-8", errors="backslashreplace").decode(
                "utf-8", errors="strict"
            )


def _safe_console_print(text: str) -> None:
    print(_safe_console_text(text))


def _log_section(title: str) -> None:
    LOGGER.info("")
    LOGGER.info("=== %s ===", title)


def _extract_http_error_reason(error: Exception) -> Optional[str]:
    response_obj: Any = getattr(error, "resp", None)
    status_code: Optional[int] = getattr(response_obj, "status", None)
    raw_content: Any = getattr(error, "content", None)
    if raw_content is None:
        return None
    try:
        content_text: str = (
            raw_content.decode("utf-8", errors="replace")
            if isinstance(raw_content, (bytes, bytearray))
            else str(raw_content)
        )
        payload: Any = json.loads(content_text)
        error_payload: Dict[str, Any] = cast(Dict[str, Any], payload.get("error", {}))
        code: Any = error_payload.get("code", status_code or "")
        status_text: str = str(error_payload.get("status", "")).strip() or "UNKNOWN"
        message: str = str(error_payload.get("message", "")).strip()
        details: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], error_payload.get("details", [])
        )
        reason: str = ""
        service: str = ""
        activation_url: str = ""
        quota_metric: str = ""
        retry_delay: str = ""
        for detail in details:
            detail_type: str = str(detail.get("@type", ""))
            if detail_type.endswith("ErrorInfo"):
                reason = str(detail.get("reason", "")).strip()
                metadata: Dict[str, Any] = cast(
                    Dict[str, Any], detail.get("metadata", {})
                )
                service = str(metadata.get("service", "")).strip()
                activation_url = str(metadata.get("activationUrl", "")).strip()
            elif detail_type.endswith("QuotaFailure"):
                violations: List[Dict[str, Any]] = cast(
                    List[Dict[str, Any]], detail.get("violations", [])
                )
                if violations:
                    quota_metric = str(violations[0].get("quotaMetric", "")).strip()
            elif detail_type.endswith("RetryInfo"):
                retry_delay = str(detail.get("retryDelay", "")).strip()

        parts: List[str] = [f"http={code}", f"status={status_text}"]
        if reason:
            parts.append(f"reason={reason}")
        if service:
            parts.append(f"service={service}")
        if quota_metric:
            parts.append(f"quota_metric={quota_metric}")
        if retry_delay:
            parts.append(f"retry_delay={retry_delay}")
        if activation_url:
            parts.append(f"enable_url={activation_url}")
        if message:
            parts.append(f"message={message[:120]}")
        return " ".join(parts)
    except Exception:
        return None


def _extract_telegram_reason(text: str) -> Optional[str]:
    compact: str = re.sub(r"\s+", " ", text).strip()
    if "Telegram API HTTP" not in compact:
        return None
    http_match: Optional[re.Match[str]] = re.search(
        r"Telegram API HTTP\s+(\d+)", compact
    )
    description_match: Optional[re.Match[str]] = re.search(
        r"\"description\"\s*:\s*\"([^\"]+)\"",
        compact,
    )
    parts: List[str] = []
    if http_match:
        parts.append(f"http={http_match.group(1)}")
    if description_match:
        parts.append(f"description={description_match.group(1)}")
    if parts:
        return " ".join(parts)
    return None


def _summarize_error(error: Exception, max_len: int = 220) -> str:
    http_reason: Optional[str] = _extract_http_error_reason(error)
    if http_reason:
        return http_reason
    raw: str = str(error).replace("\r", " ").replace("\n", " ").strip()
    telegram_reason: Optional[str] = _extract_telegram_reason(raw)
    if telegram_reason:
        return telegram_reason
    compact: str = re.sub(r"\s+", " ", raw)
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len].rstrip()}..."


def _compact_single_line(value: object, *, max_len: int = 300) -> str:
    compact: str = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len].rstrip()}..."


def _sheets_link_writeback_enabled_from_env() -> bool:
    return _load_bool_env("STG_SHEETS_LINK_WRITEBACK", True)


def _strip_chapter_timestamps_enabled_from_env() -> bool:
    return _load_bool_env("STG_STRIP_CHAPTER_TIMESTAMPS", True)


def _sheets_link_normalize_report_limit_from_env() -> int:
    return _load_int_env("STG_SHEETS_LINK_NORMALIZE_REPORT_LIMIT", 20, min_value=1)


def strip_chapter_timestamps(text: str) -> str:
    raw_text: str = str(text or "")
    if not raw_text:
        return raw_text
    chapter_pattern: re.Pattern[str] = re.compile(
        r"^\s*(?:\d{1,2}\s*:\s*)?\d{1,2}\s*:\s*\d{2}\s+\S.*$"
    )
    cleaned_lines: List[str] = []
    for line in raw_text.splitlines(keepends=True):
        stripped_line: str = line.strip()
        if stripped_line and chapter_pattern.match(stripped_line):
            continue
        cleaned_lines.append(line)
    return "".join(cleaned_lines)


def _log_link_normalization_report(
    *,
    normalization_candidates: List[LinkNormalizationCandidate],
    writeback_enabled: bool,
) -> None:
    LOGGER.info("links_normalized_total=%d", len(normalization_candidates))
    LOGGER.info("links_writeback=%s", "enabled" if writeback_enabled else "disabled")
    if not normalization_candidates:
        return
    if writeback_enabled:
        LOGGER.info(
            "links_normalized_report_details=omitted total=%d",
            len(normalization_candidates),
        )
        return
    report_limit: int = _sheets_link_normalize_report_limit_from_env()
    for item in normalization_candidates[:report_limit]:
        LOGGER.info(
            "row=%d col=%s old=%s -> new=%s",
            item.row_index,
            item.column_ref,
            _compact_single_line(item.old_value, max_len=220),
            _compact_single_line(item.new_value, max_len=220),
        )
    if len(normalization_candidates) > report_limit:
        LOGGER.info(
            "links_normalized_report_truncated=%d",
            len(normalization_candidates) - report_limit,
        )


def _unix_seconds_utc(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(value.timestamp())


def _sum_costs_usd(costs_payload: Dict[str, Any]) -> float:
    total_usd: float = 0.0
    buckets: Any = costs_payload.get("data")
    if not isinstance(buckets, list):
        return 0.0
    for bucket in buckets:
        if not isinstance(bucket, dict):
            continue
        results: Any = bucket.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            amount_obj: Any = result.get("amount")
            if not isinstance(amount_obj, dict):
                continue
            currency: str = str(amount_obj.get("currency", "")).strip().lower()
            value_raw: Any = amount_obj.get("value")
            if currency != "usd" or not isinstance(value_raw, (int, float)):
                continue
            total_usd += float(value_raw)
    return total_usd


def _safe_int_or_none(raw_value: Any) -> Optional[int]:
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    text_value: str = str(raw_value or "").strip().replace(",", "")
    if not text_value or not re.fullmatch(r"-?\d+", text_value):
        return None
    try:
        return int(text_value)
    except Exception:
        return None


def _timezone_from_name_or_utc(tz_name: str) -> timezone | ZoneInfo:
    cleaned_tz_name: str = str(tz_name or "").strip() or "UTC"
    try:
        return ZoneInfo(cleaned_tz_name)
    except Exception:
        return timezone.utc


def _sum_usage_payload(
    usage_payload: Dict[str, Any],
) -> Tuple[int, int, int, Dict[str, Dict[str, int]]]:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    per_model: Dict[str, Dict[str, int]] = {}
    data_any: Any = usage_payload.get("data")
    if not isinstance(data_any, list):
        return (0, 0, 0, {})
    for bucket in data_any:
        if not isinstance(bucket, dict):
            continue
        results_any: Any = bucket.get("results")
        if not isinstance(results_any, list):
            continue
        for result_item in results_any:
            if not isinstance(result_item, dict):
                continue
            model_name: str = str(result_item.get("model", "")).strip() or "unknown"
            input_tokens: int = _safe_int_or_none(result_item.get("input_tokens")) or 0
            output_tokens: int = (
                _safe_int_or_none(result_item.get("output_tokens")) or 0
            )
            request_count: int = (
                _safe_int_or_none(result_item.get("num_model_requests")) or 0
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_requests += request_count
            model_entry: Dict[str, int] = per_model.setdefault(
                model_name,
                {"input": 0, "output": 0, "requests": 0, "tokens": 0},
            )
            model_entry["input"] += input_tokens
            model_entry["output"] += output_tokens
            model_entry["requests"] += request_count
            model_entry["tokens"] += input_tokens + output_tokens
    return (total_input_tokens, total_output_tokens, total_requests, per_model)


def _format_top_models_usage(per_model: Dict[str, Dict[str, int]]) -> str:
    ranked_items: List[Tuple[str, Dict[str, int]]] = sorted(
        per_model.items(),
        key=lambda item: int(item[1].get("tokens", 0)),
        reverse=True,
    )
    if not ranked_items:
        return "none"
    shown_items: List[Tuple[str, Dict[str, int]]] = ranked_items[:3]
    hidden_items: List[Tuple[str, Dict[str, int]]] = ranked_items[3:]
    parts: List[str] = [
        f"{model_name}:{int(values.get('input', 0))}/{int(values.get('output', 0))}/{int(values.get('requests', 0))}"
        for model_name, values in shown_items
    ]
    if hidden_items:
        other_input: int = sum(
            int(values.get("input", 0)) for _, values in hidden_items
        )
        other_output: int = sum(
            int(values.get("output", 0)) for _, values in hidden_items
        )
        other_requests: int = sum(
            int(values.get("requests", 0)) for _, values in hidden_items
        )
        parts.append(f"other:{other_input}/{other_output}/{other_requests}")
    return ", ".join(parts)


def fetch_usage_and_costs_summary(
    admin_key: str,
    tz_name: str,
    now_utc: datetime,
) -> Dict[str, Any]:
    cleaned_admin_key: str = str(admin_key or "").strip()
    if not cleaned_admin_key:
        return {"error": "OPENAI_ADMIN_KEY is empty"}
    if now_utc.tzinfo is None:
        return {"error": "now_utc must be timezone-aware"}

    tz_value: timezone | ZoneInfo = _timezone_from_name_or_utc(tz_name)
    now_local: datetime = now_utc.astimezone(tz_value)
    today_start_local: datetime = datetime(
        now_local.year,
        now_local.month,
        now_local.day,
        tzinfo=tz_value,
    )
    month_start_local: datetime = datetime(
        now_local.year,
        now_local.month,
        1,
        tzinfo=tz_value,
    )
    today_start_utc: datetime = today_start_local.astimezone(timezone.utc)
    month_start_utc: datetime = month_start_local.astimezone(timezone.utc)

    project_id: str = str(os.getenv("OPENAI_PROJECT_ID", "") or "").strip()
    timeout_raw: str = str(os.getenv("OPENAI_USAGE_TIMEOUT_SEC", "30.0") or "30.0")
    try:
        timeout_sec: float = max(5.0, float(timeout_raw))
    except Exception:
        timeout_sec = 30.0

    session: requests.Session = requests.Session()
    common_headers: Dict[str, str] = {
        "Authorization": f"Bearer {cleaned_admin_key}",
        "Content-Type": "application/json",
    }
    usage_params: Dict[str, Any] = {
        "start_time": _unix_seconds_utc(today_start_utc),
        "end_time": _unix_seconds_utc(now_utc),
        "bucket_width": "1d",
        "group_by": ["model"],
    }
    costs_params: Dict[str, Any] = {
        "start_time": _unix_seconds_utc(month_start_utc),
        "end_time": _unix_seconds_utc(now_utc),
        "bucket_width": "1d",
    }
    if project_id:
        usage_params["project_ids"] = [project_id]
        costs_params["project_ids"] = [project_id]

    try:
        usage_response: requests.Response = session.get(
            url="https://api.openai.com/v1/organization/usage/completions",
            params=usage_params,
            headers=common_headers,
            timeout=timeout_sec,
        )
        usage_response.raise_for_status()
        usage_payload_any: Any = usage_response.json()
        usage_payload: Dict[str, Any] = (
            cast(Dict[str, Any], usage_payload_any)
            if isinstance(usage_payload_any, dict)
            else {}
        )
        costs_response: requests.Response = session.get(
            url="https://api.openai.com/v1/organization/costs",
            params=costs_params,
            headers=common_headers,
            timeout=timeout_sec,
        )
        costs_response.raise_for_status()
        costs_payload_any: Any = costs_response.json()
        costs_payload: Dict[str, Any] = (
            cast(Dict[str, Any], costs_payload_any)
            if isinstance(costs_payload_any, dict)
            else {}
        )
    except Exception as error:
        return {"error": _summarize_error(cast(Exception, error))}

    (
        total_input_tokens,
        total_output_tokens,
        total_requests,
        per_model,
    ) = _sum_usage_payload(usage_payload)
    spent_usd_month: float = _sum_costs_usd(costs_payload)
    return {
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_requests": total_requests,
        "per_model": per_model,
        "spent_usd_month": spent_usd_month,
    }


def _log_openai_limits_and_usage(logger: logging.Logger) -> None:
    admin_api_key: str = str(os.getenv("OPENAI_ADMIN_KEY", "") or "").strip()
    if not admin_api_key:
        logger.info("OpenAI summary skipped: OPENAI_ADMIN_KEY is not set.")
        return
    tz_name: str = str(os.getenv("STG_TZ", "UTC") or "UTC").strip() or "UTC"
    summary_payload: Dict[str, Any] = fetch_usage_and_costs_summary(
        admin_key=admin_api_key,
        tz_name=tz_name,
        now_utc=datetime.now(timezone.utc),
    )
    error_text: str = str(summary_payload.get("error", "")).strip()
    if error_text:
        logger.warning("OpenAI summary skipped: %s", error_text)
        return
    total_input_tokens: int = int(summary_payload.get("total_input_tokens") or 0)
    total_output_tokens: int = int(summary_payload.get("total_output_tokens") or 0)
    total_requests: int = int(summary_payload.get("total_requests") or 0)
    per_model: Dict[str, Dict[str, int]] = cast(
        Dict[str, Dict[str, int]],
        summary_payload.get("per_model") or {},
    )
    logger.info(
        "OPENAI USAGE today input=%d output=%d req=%d models=%s",
        total_input_tokens,
        total_output_tokens,
        total_requests,
        _format_top_models_usage(per_model),
    )
    spent_usd_month: float = float(summary_payload.get("spent_usd_month") or 0.0)
    budget_raw: str = str(os.getenv("OPENAI_MONTHLY_BUDGET_USD", "") or "").strip()
    budget_usd: Optional[float] = None
    if budget_raw:
        try:
            budget_usd = float(budget_raw)
        except Exception:
            budget_usd = None
    if budget_usd is not None:
        remaining_usd: float = float(budget_usd) - spent_usd_month
        logger.info(
            "OPENAI COST month spent_usd=%.6f remaining_usd=%.6f budget_usd=%.6f",
            spent_usd_month,
            remaining_usd,
            float(budget_usd),
        )
    else:
        logger.info("OPENAI COST month spent_usd=%.6f", spent_usd_month)


def _log_config_summary(
    config: "AppConfig",
    *,
    mode_label: str,
    resolved_processing_mode: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    LOGGER.info("run_id=%s Config loaded successfully for mode=%s.", run_id, mode_label)
    LOGGER.info(
        "run_id=%s Config summary: google=%s telegram=%s templates=%s",
        run_id,
        "enabled" if config.google_enabled else "disabled",
        "enabled" if config.telegram_enabled else "disabled",
        str(config.stg_templates_path),
    )
    LOGGER.info(
        "run_id=%s Config summary: sheets=%s range=%s",
        run_id,
        config.google_sheets_id,
        config.google_sheets_range,
    )
    LOGGER.info(
        "run_id=%s Config summary: config_processing_mode=%s resolved_processing_mode=%s now_tz_mode=%s llm_provider=%s openai_primary=%s openai_fallback=%s openai_timeout_sec=%.1f openai_max_output_tokens=%d llm_source_desc_max_chars=%d llm_run_if_single_source=%s openai_pre_delay_sec=%.1f",
        run_id,
        config.processing_mode,
        str(resolved_processing_mode or config.processing_mode),
        config.now_tz_mode,
        config.llm_provider,
        config.openai_model_primary,
        config.openai_model_fallback,
        config.openai_timeout_sec,
        config.openai_max_output_tokens,
        config.llm_source_desc_max_chars,
        config.llm_run_if_single_source,
        config.openai_pre_delay_sec,
    )
    LOGGER.info(
        "run_id=%s sheets_link_writeback=%s",
        run_id,
        "enabled" if _sheets_link_writeback_enabled_from_env() else "disabled",
    )
    LOGGER.info(
        "run_id=%s strip_chapter_timestamps=%s",
        run_id,
        "enabled" if _strip_chapter_timestamps_enabled_from_env() else "disabled",
    )
    LOGGER.info(
        "run_id=%s local_doc_export_enabled=%s",
        run_id,
        "true" if bool(str(config.local_doc_dir_template or "").strip()) else "false",
    )


def _safe_debug_filename_part(value: str) -> str:
    cleaned: str = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    if not cleaned:
        cleaned = "value"
    return cleaned[:80]


def _load_stg_debug_prompt_to_file_from_env() -> bool:
    raw_value: str = os.getenv("STG_DEBUG_PROMPT_TO_FILE", "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _load_stg_debug_prompt_dir_from_env() -> str:
    return (
        os.getenv("STG_DEBUG_PROMPT_DIR", "_debug_prompts").strip() or "_debug_prompts"
    )


def _load_bool_env(name: str, default: bool) -> bool:
    raw_value: str = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default


def _load_int_env(name: str, default: int, min_value: int = 0) -> int:
    raw_value: str = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed: int = int(raw_value)
    except Exception:
        return default
    return max(min_value, parsed)


def _load_float_env(name: str, default: float, min_value: float = 0.0) -> float:
    raw_value: str = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        parsed: float = float(raw_value)
    except Exception:
        return default
    return max(min_value, parsed)


def _build_header_context(
    videos: List[PlannedVideo],
    form_url: str,
    contacts: str,
    cet_tz: ZoneInfo,
) -> Dict[str, str]:
    if not videos:
        raise ValueError("videos must not be empty")

    first_stream: PlannedVideo = min(
        videos,
        key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
    )
    stream_dt_kiev: datetime = first_stream.scheduled_at_kiev
    stream_dt_cet: datetime = stream_dt_kiev.astimezone(cet_tz)
    stream_dt_gmt: datetime = stream_dt_kiev.astimezone(timezone.utc)
    kiev_minus_1: datetime = stream_dt_kiev - timedelta(hours=1)
    gmt_minus_1: datetime = stream_dt_gmt - timedelta(hours=1)

    times_by_language: Dict[str, str] = {"uk": "", "en": "", "ru": ""}
    for language in ("uk", "en", "ru"):
        same_language: List[PlannedVideo] = [
            item for item in videos if planned_video_block_language(item) == language
        ]
        if not same_language:
            continue
        unique_times: List[str] = sorted(
            {item.scheduled_at_kiev.strftime("%H:%M") for item in same_language}
        )
        times_by_language[language] = ", ".join(unique_times)

    return {
        "date": stream_dt_kiev.strftime("%d.%m.%Y"),
        "time_cet": stream_dt_cet.strftime("%H:%M"),
        "time_kiev": stream_dt_kiev.strftime("%H:%M"),
        "time_gmt": stream_dt_gmt.strftime("%H:%M"),
        "time_kiev_minus_1": kiev_minus_1.strftime("%H:%M"),
        "time_gmt_minus_1": gmt_minus_1.strftime("%H:%M"),
        "time_ukr": times_by_language["uk"],
        "time_eng": times_by_language["en"],
        "time_ru": times_by_language["ru"],
        "form_url": form_url,
        "contacts": contacts,
    }


# ----------------------------
# Оркестратор (приложение)
# ----------------------------


class NamePathBuilder:
    def __init__(
        self,
        local_image_dir_template: str,
        local_doc_dir_template: Optional[str],
        preview_name_template: str,
        doc_title_template: str,
        language_codes_json: str,
        max_filename_stem: int,
    ) -> None:
        self._local_image_dir_template: str = local_image_dir_template
        self._local_doc_dir_template: Optional[str] = (
            str(local_doc_dir_template or "").strip() or None
        )
        self._preview_name_template: str = preview_name_template
        self._doc_title_template: str = doc_title_template
        self._max_filename_stem: int = max(16, int(max_filename_stem))
        self._language_codes: Dict[str, str] = {}
        try:
            payload: Any = json.loads(language_codes_json)
            if isinstance(payload, dict):
                for key, value in payload.items():
                    self._language_codes[str(key)] = str(value).upper()
        except Exception as error:
            raise RuntimeError(
                f"Invalid template files.language_codes: {error}"
            ) from error

    def build_doc_title(
        self,
        date_key: str,
        created_at: datetime,
        processing_mode: str,
    ) -> str:
        creation_stamp: str = created_at.strftime("%H%M_%d%m%y")
        return _render_template(
            self._doc_title_template,
            {
                "date": date_key,
                "creation_stamp": creation_stamp,
                "processing_mode": processing_mode,
            },
        )

    def build_docx_path(self, date_key: str, doc_title: str) -> Optional[Path]:
        if self._local_doc_dir_template is None:
            return None
        base_dir: str = self._local_doc_dir_template.format(date=date_key)
        safe_stem: str = re.sub(r"[^A-Za-z0-9._-]+", "_", str(doc_title or "").strip())
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("._-")
        safe_stem = safe_stem[: self._max_filename_stem].rstrip("._-") or "document"
        return Path(base_dir) / f"{safe_stem}.docx"

    def build_image_path(
        self,
        language: str,
        date_key: str,
        language_position: int,
        title: str,
        extension: str,
    ) -> Path:
        display_language: str = _display_language_code(language)
        base_dir: str = self._local_image_dir_template.format(
            language=display_language,
            date=date_key,
        )
        file_language_code: str = self._language_codes.get(language, "OT")
        safe_title: str = _build_safe_entity_name(title)
        raw_stem: str = _render_template(
            self._preview_name_template,
            {
                "index": language_position,
                "language": file_language_code,
                "title": safe_title,
                "date": date_key,
                "display_language": display_language.upper(),
            },
        )
        safe_stem: str = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_stem)
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("._-")
        safe_stem = safe_stem[: self._max_filename_stem].rstrip("._-") or "video"
        return Path(base_dir) / f"{safe_stem}{extension}"


class StreamPipelineApp:
    def __init__(
        self,
        config: AppConfig,
        metadata_fetcher: YouTubeMetadataFetcher,
        http_client: HttpClient,
        telegram_client: TelegramBotClient,
    ) -> None:
        self._config = config
        self._metadata_fetcher = metadata_fetcher
        self._http_client = http_client
        self._telegram_client = telegram_client
        self._last_merge_run_summary: Optional[MergeRunSummary] = None
        self._kiev_tz = _load_zoneinfo(config.timezone_kiev)
        self._cet_tz = _load_zoneinfo(config.timezone_cet)
        self._name_builder = NamePathBuilder(
            local_image_dir_template=config.local_image_dir_template,
            local_doc_dir_template=config.local_doc_dir_template,
            preview_name_template=config.templates.files_preview_name_template,
            doc_title_template=config.templates.files_doc_title_template,
            language_codes_json=config.templates.files_language_codes_json,
            max_filename_stem=config.preview_filename_max_stem,
        )

    def run_batch(self, dry_run: bool, *, processing_mode: str, run_id: str) -> None:
        if not self._config.google_enabled:
            raise RuntimeError("Для batch режима GOOGLE_ENABLED должен быть включен.")
        merge_run_summary: MergeRunSummary = MergeRunSummary()
        self._last_merge_run_summary = merge_run_summary
        resolved_processing_mode: str = normalize_processing_mode(
            processing_mode,
            source="processing_mode",
        )
        services_factory: GoogleServicesFactory = GoogleServicesFactory()
        sheets_client: GoogleSheetsClient = GoogleSheetsClient(
            sheets_service=services_factory.create_sheets_service()
        )
        docs_client: GoogleDocsClient = GoogleDocsClient(
            docs_service=services_factory.create_docs_service()
        )
        drive_client: GoogleDriveClient = GoogleDriveClient(
            drive_service=services_factory.create_drive_service()
        )
        report_writer: GoogleDocsReportWriter = GoogleDocsReportWriter(
            docs_client=docs_client,
            templates=self._config.templates,
        )

        startup_errors: List[str] = []
        _log_section("Startup Health Check")
        LOGGER.info("Startup identity check: begin.")
        LOGGER.info("Logger name in use: %s", LOGGER.name)
        logger_name_resolved, logger_name_source, logger_env_present = (
            resolve_logger_name_meta()
        )
        LOGGER.info(
            "Logger env probe: var=%s env_present=%s source=%s resolved=%s",
            LOGGER_NAME_ENV_VAR,
            logger_env_present,
            logger_name_source,
            logger_name_resolved,
        )

        _log_section("Environment")
        google_auth_mode: str = "unknown"
        try:
            google_auth_mode = services_factory.get_auth_mode()
            LOGGER.info("Google auth mode: %s", google_auth_mode)
            if google_auth_mode == "oauth":
                oauth_credentials_path, oauth_token_path = (
                    services_factory.get_oauth_paths()
                )
                LOGGER.info(
                    "Google OAuth paths: credentials=%s token=%s",
                    oauth_credentials_path,
                    oauth_token_path,
                )
                LOGGER.info(
                    "Google auth hint: default GOOGLE_AUTH_MODE=oauth; set GOOGLE_AUTH_MODE=service_account to force service account mode."
                )
        except Exception as error:
            issue: str = f"Google auth mode check failed: {_summarize_error(error)}"
            startup_errors.append(issue)
            LOGGER.error(issue)

        try:
            google_runtime_principal: str = (
                services_factory.get_runtime_principal_email()
            )
            LOGGER.info(
                "Google runtime account (script): %s",
                google_runtime_principal,
            )
        except Exception as error:
            issue = f"Google runtime account lookup failed: {_summarize_error(error)}"
            startup_errors.append(issue)
            LOGGER.error(issue)

        try:
            google_project_id, google_project_name = (
                services_factory.get_google_project_info(strict=True)
            )
            LOGGER.info(
                "Google project resolved via API: name=%s id=%s",
                google_project_name,
                google_project_id,
            )
        except Exception as error:
            # Non-blocking diagnostic check: does not affect pipeline execution.
            LOGGER.warning(
                "Google project API lookup skipped: %s",
                _summarize_error(error),
            )

        _log_section("Google APIs")
        try:
            sheets_owner_info: str = drive_client.get_file_owner_info(
                file_id=self._config.google_sheets_id
            )
            LOGGER.info(
                "Google owner account for Sheets file: %s",
                sheets_owner_info,
            )
        except Exception as error:
            issue = f"Google Sheets owner lookup failed: {_summarize_error(error)}"
            startup_errors.append(issue)
            LOGGER.error(issue)

        try:
            sheets_id_resolved, sheets_title = sheets_client.ping_access(
                spreadsheet_id=self._config.google_sheets_id
            )
            LOGGER.info(
                "Google Sheets API OK. Server response: spreadsheet_id=%s, title=%s",
                sheets_id_resolved,
                sheets_title,
            )
        except Exception as error:
            issue = f"Google Sheets connection failed: {_summarize_error(error)}"
            startup_errors.append(issue)
            LOGGER.error(issue)

        try:
            drive_user_name, drive_user_email = drive_client.ping_access()
            LOGGER.info(
                "Google Drive API OK. Server response: user=%s <%s>",
                drive_user_name,
                drive_user_email,
            )
        except Exception as error:
            issue = f"Google Drive connection failed: {_summarize_error(error)}"
            startup_errors.append(issue)
            LOGGER.error(issue)

        try:
            docs_probe_result: str = docs_client.ping_access()
            LOGGER.info(
                "Google Docs API OK. Server response: %s",
                docs_probe_result,
            )
        except Exception as error:
            issue = f"Google Docs connection failed: {_summarize_error(error)}"
            startup_errors.append(issue)
            LOGGER.error(issue)

        _log_section("LLM API")
        LOGGER.info(
            "run_id=%s Processing mode selected: %s",
            run_id,
            resolved_processing_mode,
        )
        LOGGER.info(
            "run_id=%s LLM provider selected: %s",
            run_id,
            self._config.llm_provider,
        )
        LOGGER.info(
            "OpenAI request model.primary=%s model.fallback=%s timeout_sec=%.1f max_output_tokens=%d max_retries=%d source_desc_max_chars=%d pre_delay_sec=%.1f",
            self._config.openai_model_primary,
            self._config.openai_model_fallback,
            self._config.openai_timeout_sec,
            self._config.openai_max_output_tokens,
            0,
            self._config.llm_source_desc_max_chars,
            self._config.openai_pre_delay_sec,
        )
        merge_config: AppConfig = self._config
        gpt_api_key_present: bool = bool(os.getenv("GPT_API_KEY", "").strip())
        llm_allow_in_dry_run: bool = _load_bool_env("STG_LLM_ALLOW_IN_DRY_RUN", False)
        llm_merge_enabled: bool = False
        merge_mode_enabled: bool = resolved_processing_mode == "merge"
        if merge_mode_enabled:
            llm_merge_enabled = gpt_api_key_present
            if dry_run and not llm_allow_in_dry_run:
                llm_merge_enabled = False
                LOGGER.info(
                    "LLM selection: disabled in dry-run by STG_LLM_ALLOW_IN_DRY_RUN=0."
                )
        if llm_merge_enabled:
            LOGGER.info(
                "LLM selection: provider=%s enabled for merge stage.",
                self._config.llm_provider,
            )
        else:
            if not merge_mode_enabled:
                LOGGER.info("llm_merge=skipped reason=processing_mode=nomerge")
            elif dry_run and not llm_allow_in_dry_run:
                LOGGER.info(
                    "LLM selection: dry-run mode -> merge disabled, append fallback will be used."
                )
            else:
                LOGGER.warning(
                    "LLM selection: disabled (missing provider prerequisites). Continue without merge stage."
                )

        _log_section("Telegram API")
        try:
            if not self._config.telegram_enabled:
                raise RuntimeError("telegram.enabled=false")
            telegram_me: Dict[str, Any] = self._telegram_client.get_me()
            telegram_bot_name: str = str(
                telegram_me.get("username")
                or telegram_me.get("first_name")
                or telegram_me.get("id")
                or "unknown"
            ).strip()
            LOGGER.info("Telegram bot identity: %s", telegram_bot_name)
            LOGGER.info(
                "Telegram API OK. Server response: bot=%s, id=%s",
                telegram_bot_name,
                str(telegram_me.get("id", "unknown")),
            )
        except Exception as error:
            issue = f"Telegram connection failed: {_summarize_error(error)}"
            startup_errors.append(issue)
            LOGGER.error(issue)

        if startup_errors:
            LOGGER.warning(
                "Startup health-check decision: CONTINUE_WITH_WARNINGS. Failed checks: %d",
                len(startup_errors),
            )
            for issue in startup_errors:
                LOGGER.warning("Startup check failure detail: %s", issue)
        else:
            LOGGER.info("Startup health-check decision: CONTINUE.")
        configured_range: str = self._config.google_sheets_range
        effective_range: str = configured_range
        autoexpand_range: bool = _load_bool_env("STG_SHEETS_AUTOEXPAND_RANGE", False)
        if not _sheet_range_includes_merge_column(configured_range):
            LOGGER.warning(
                "Sheets range %r does not include Merge column (E); merge settings will be ignored. Use A:F.",
                configured_range,
            )
            if autoexpand_range:
                effective_range = _expand_sheet_range_to_af(configured_range)
                LOGGER.warning(
                    "STG_SHEETS_AUTOEXPAND_RANGE=1 -> using expanded range %r",
                    effective_range,
                )
        LOGGER.info(
            "run_id=%s Reading Google Sheets: spreadsheet=%s range=%s",
            run_id,
            self._config.google_sheets_id,
            effective_range,
        )
        rows: List[SheetRow] = sheets_client.read_rows(
            spreadsheet_id=self._config.google_sheets_id,
            range_name=effective_range,
        )
        LOGGER.info("Rows loaded from sheet: %d", len(rows))
        sheet_name_for_writeback: Optional[str] = _sheet_name_from_range(
            effective_range
        )
        sheets_link_writeback_enabled: bool = _sheets_link_writeback_enabled_from_env()
        link_normalization_candidates: List[LinkNormalizationCandidate] = []

        now_filter_tz: timezone | ZoneInfo = now_filter_timezone(
            self._config.now_tz_mode,
            self._kiev_tz,
        )
        now_for_filter: datetime = datetime.now(now_filter_tz)
        LOGGER.info(
            "now_tz=%s now=%s",
            self._config.now_tz_mode,
            now_for_filter.isoformat(),
        )
        merge_semantics: str = merge_semantics_from_env()
        LOGGER.info("merge_semantics=%s", merge_semantics)
        planned_items: List[PlannedVideoItem] = []
        for row in rows:
            LOGGER.info("Row %d: read", row.row_number)
            if not row.link:
                LOGGER.warning("Row %d: skipped, empty Links", row.row_number)
                continue
            if not row.date_raw or not row.time_raw:
                LOGGER.warning("Row %d: skipped, missing Date/Time", row.row_number)
                continue

            try:
                scheduled_at: datetime = _parse_sheet_datetime(
                    date_raw=row.date_raw,
                    time_raw=row.time_raw,
                    tz=self._kiev_tz,
                )
                if scheduled_at < now_for_filter:
                    LOGGER.info(
                        "Row %d: skipped, already in the past (%s)",
                        row.row_number,
                        scheduled_at.isoformat(),
                    )
                    continue

                normalized_link: Optional[str] = _normalize_youtube_link(row.link)
                if not normalized_link:
                    LOGGER.warning(
                        "Row %d: skipped, cannot extract YouTube video id from Links=%r",
                        row.row_number,
                        row.link,
                    )
                    continue
                _handle_normalized_link_writeback(
                    sheets_client=sheets_client,
                    spreadsheet_id=self._config.google_sheets_id,
                    sheet_name_for_writeback=sheet_name_for_writeback,
                    row_number=row.row_number,
                    links_column_index=row.links_column_index,
                    old_link=row.link,
                    normalized_link=normalized_link,
                    writeback_enabled=sheets_link_writeback_enabled,
                    normalization_candidates=link_normalization_candidates,
                    links_column_label=f"Links/{_column_index_to_letters(row.links_column_index)}",
                )
                metadata: VideoMetadata = self._metadata_fetcher.fetch(
                    video_url=normalized_link
                )
                LOGGER.info("Row %d: metadata fetched", row.row_number)
                language: str = _detect_language(metadata)
                LOGGER.info("Row %d: language=%s", row.row_number, language)

                thumbnail_bytes: bytes = self._http_client.get_bytes(
                    metadata.thumbnail_url
                )
                normalized_thumbnail: NormalizedImage = _normalize_thumbnail(
                    thumbnail_bytes
                )
                date_key: str = scheduled_at.strftime("%d%m%y")

                base_video: PlannedVideo = PlannedVideo(
                    row_number=row.row_number,
                    original_link=row.link,
                    normalized_link=normalized_link,
                    scheduled_at_kiev=scheduled_at,
                    date_key=date_key,
                    date_display=scheduled_at.strftime("%d.%m.%Y"),
                    language=language,
                    metadata=metadata,
                    thumbnail=normalized_thumbnail,
                    local_thumbnail_path=None,
                )
                base_block_lang: str = planned_video_block_language(base_video)
                row_merge_languages: List[str] = (
                    list(row.merge_languages)
                    if resolved_processing_mode == "merge"
                    else []
                )
                if resolved_processing_mode == "nomerge" and (
                    bool(row.merge_languages) or bool(str(row.merge_raw or "").strip())
                ):
                    LOGGER.info(
                        'Row %d: merge column ignored due to processing_mode=nomerge (merge_langs_raw="%s")',
                        row.row_number,
                        row.merge_raw,
                    )
                row_characteristics: RowVideoCharacteristics = RowVideoCharacteristics(
                    row_index=row.row_number,
                    raw_link=row.link,
                    normalized_link=normalized_link,
                    date=row.date_raw,
                    time=row.time_raw,
                    detected_source_language=language,
                    merge_languages=row_merge_languages,
                    base_block_language=base_block_lang,
                )
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "Row %d metadata: raw_link=%r normalized_link=%r date=%r time=%r detected_source_language=%s merge_languages=%s base_block_language=%s",
                        row.row_number,
                        row_characteristics.raw_link,
                        row_characteristics.normalized_link,
                        row_characteristics.date,
                        row_characteristics.time,
                        row_characteristics.detected_source_language,
                        row_characteristics.merge_languages,
                        row_characteristics.base_block_language,
                    )
                base_video = dataclasses.replace(
                    base_video,
                    row_characteristics=row_characteristics,
                )
                time_key: str = scheduled_at.strftime("%H%M")
                slot_key: str = f"{date_key}_{time_key}"
                planned_items.append(
                    PlannedVideoItem(
                        row_number=row.row_number,
                        normalized_url=normalized_link,
                        date_key=date_key,
                        time_key=time_key,
                        slot_key=slot_key,
                        base_lang=base_block_lang,
                        merge_langs=row_merge_languages,
                        title="",
                        description="",
                        preview="",
                        base_video=base_video,
                    )
                )
            except Exception as error:
                LOGGER.exception(
                    "Row %d: processing failed. reason=%s",
                    row.row_number,
                    error,
                )

        LOGGER.info("Planning stage: items=%d", len(planned_items))
        processed: List[PlannedVideo] = []
        nomerge_reset_logged: bool = False
        for item in planned_items:
            base_video = item.base_video
            if resolved_processing_mode == "nomerge":
                video_for_nomerge = base_video
                if hasattr(base_video, "forced_block_language"):
                    forced_before: Any = getattr(
                        base_video, "forced_block_language", None
                    )
                    if dataclasses.is_dataclass(base_video):
                        video_for_nomerge = dataclasses.replace(
                            base_video,
                            forced_block_language=None,
                        )
                    else:
                        try:
                            setattr(base_video, "forced_block_language", None)
                        except Exception:
                            pass
                    if not nomerge_reset_logged:
                        LOGGER.debug("nomerge: forced_block_language reset enabled")
                        nomerge_reset_logged = True
                    if forced_before:
                        LOGGER.debug(
                            "Row %d: nomerge forced_block_language reset (was=%r)",
                            item.row_number,
                            forced_before,
                        )
                processed.append(video_for_nomerge)
                continue
            if item.merge_langs and merge_semantics == "override":
                if item.base_lang in item.merge_langs:
                    processed.append(base_video)
                for merge_language in item.merge_langs:
                    if merge_language == item.base_lang:
                        continue
                    processed.append(
                        dataclasses.replace(
                            base_video,
                            forced_block_language=merge_language,
                        )
                    )
                continue
            processed.append(base_video)
            for merge_language in item.merge_langs:
                if merge_language == item.base_lang:
                    LOGGER.warning(
                        "Row %d: merge token %r matches base block %r; skipping clone",
                        item.row_number,
                        merge_language,
                        item.base_lang,
                    )
                    continue
                processed.append(
                    dataclasses.replace(
                        base_video,
                        forced_block_language=merge_language,
                    )
                )
        processed = deduplicate_planned_videos_within_slot_language(processed)
        if not processed:
            _log_link_normalization_report(
                normalization_candidates=link_normalization_candidates,
                writeback_enabled=sheets_link_writeback_enabled,
            )
            LOGGER.warning("No videos to process after filtering.")
            return

        processed_by_slot: Dict[str, List[PlannedVideo]] = {}
        for item in processed:
            processed_by_slot.setdefault(planned_video_slot_key(item), []).append(item)
        for slot_key in sorted(processed_by_slot.keys()):
            slot_items: List[PlannedVideo] = processed_by_slot[slot_key]
            date_key_for_slot: str = slot_items[0].date_key
            slot_lang_counts: Dict[str, int] = {
                language: sum(
                    1
                    for slot_item in slot_items
                    if planned_video_block_language(slot_item) == language
                )
                for language in ("uk", "en", "ru", "other")
            }
            LOGGER.info(
                "After dedup: date=%s slot=%s counts_by_language=uk:%d en:%d ru:%d other:%d",
                date_key_for_slot,
                slot_key,
                slot_lang_counts["uk"],
                slot_lang_counts["en"],
                slot_lang_counts["ru"],
                slot_lang_counts["other"],
            )

        videos_by_date: Dict[str, List[PlannedVideo]] = {}
        for item in processed:
            videos_by_date.setdefault(item.date_key, []).append(item)

        for date_key in sorted(videos_by_date.keys()):
            date_videos_all: List[PlannedVideo] = videos_by_date[date_key]
            slots_by_time: Dict[str, List[PlannedVideo]] = {}
            for item in date_videos_all:
                slots_by_time.setdefault(planned_video_time_key(item), []).append(item)

            slot_results: List[Dict[str, Any]] = []
            for slot_time_key in sorted(slots_by_time.keys()):
                slot_key: str = f"{date_key}_{slot_time_key}"
                day_videos: List[PlannedVideo] = sorted(
                    slots_by_time[slot_time_key],
                    key=lambda item: (
                        language_index(planned_video_block_language(item)),
                        item.scheduled_at_kiev.time(),
                        item.row_number,
                    ),
                )
                slot_lang_counts: Dict[str, int] = {
                    language: sum(
                        1
                        for item in day_videos
                        if planned_video_block_language(item) == language
                    )
                    for language in ("uk", "en", "ru", "other")
                }
                LOGGER.info(
                    "Processing slot=%s date=%s time=%s counts_by_language=uk:%d en:%d ru:%d other:%d",
                    slot_key,
                    date_key,
                    slot_time_key,
                    slot_lang_counts["uk"],
                    slot_lang_counts["en"],
                    slot_lang_counts["ru"],
                    slot_lang_counts["other"],
                )

                preview_root_folder_id: Optional[str] = (
                    self._config.google_drive_preview_folder_id
                    or self._config.google_drive_folder_id
                )
                language_groups: Dict[str, List[PlannedVideo]] = {
                    "uk": [],
                    "en": [],
                    "ru": [],
                    "other": [],
                }
                updated_day_videos: List[PlannedVideo] = []
                for language in ("uk", "en", "ru", "other"):
                    language_items: List[PlannedVideo] = [
                        item
                        for item in day_videos
                        if planned_video_block_language(item) == language
                    ]
                    preview_target_folder_id: Optional[str] = preview_root_folder_id
                    if not dry_run and preview_root_folder_id and language_items:
                        try:
                            preview_path_parts: List[str] = (
                                _build_drive_preview_path_segments(
                                    template=self._config.google_drive_preview_path_template,
                                    language=language,
                                    date_key=date_key,
                                )
                            )
                            if preview_path_parts:
                                preview_target_folder_id = (
                                    drive_client.ensure_folder_path(
                                        parent_folder_id=preview_root_folder_id,
                                        path_parts=preview_path_parts,
                                    )
                                )
                        except Exception as error:
                            LOGGER.warning(
                                "Date %s language=%s: failed to ensure Drive preview path under %s. "
                                "template=%r. Fallback to parent folder. reason=%s",
                                slot_key,
                                language,
                                preview_root_folder_id,
                                self._config.google_drive_preview_path_template,
                                _summarize_error(error),
                            )
                    for language_position, video in enumerate(language_items, start=1):
                        local_image_path: Path = self._name_builder.build_image_path(
                            language=language,
                            date_key=date_key,
                            language_position=language_position,
                            title=video.metadata.title,
                            extension=video.thumbnail.extension,
                        )
                        local_image_path.parent.mkdir(parents=True, exist_ok=True)
                        local_image_path.write_bytes(video.thumbnail.bytes_data)
                        LOGGER.info(
                            "Row %d: thumbnail saved to %s",
                            video.row_number,
                            local_image_path,
                        )
                        if (
                            not dry_run
                            and preview_target_folder_id
                            and local_image_path.exists()
                        ):
                            try:
                                drive_client.upload_image_and_make_public(
                                    image_path=local_image_path,
                                    folder_id=preview_target_folder_id,
                                    mime_type=video.thumbnail.mime_type,
                                )
                                LOGGER.info(
                                    "Row %d: thumbnail uploaded to Google Drive (%s)",
                                    video.row_number,
                                    local_image_path.name,
                                )
                            except Exception as error:
                                LOGGER.warning(
                                    "Row %d: thumbnail upload to Google Drive failed. reason=%s",
                                    video.row_number,
                                    error,
                                )
                        updated_video: PlannedVideo = dataclasses.replace(
                            video,
                            local_thumbnail_path=local_image_path,
                        )
                        language_groups[language].append(updated_video)
                        updated_day_videos.append(updated_video)

                day_videos = updated_day_videos
                merged_content_by_language: Dict[str, MergedLanguageContent] = {}
                merge_audit_by_language: Dict[str, LanguageMergeAttempt] = {}
                if llm_merge_enabled:
                    for language in ("uk", "en", "ru", "other"):
                        language_items_for_merge: List[PlannedVideo] = sorted(
                            language_groups[language],
                            key=lambda item: (
                                item.scheduled_at_kiev.time(),
                                item.row_number,
                            ),
                        )
                        non_empty_descriptions_count: int = sum(
                            1
                            for item in language_items_for_merge
                            if item.metadata.description.strip()
                        )
                        run_classic_merge: bool = non_empty_descriptions_count >= 2
                        run_single_source_translate: bool = False
                        if (
                            self._config.llm_run_if_single_source
                            and non_empty_descriptions_count == 1
                            and len(language_items_for_merge) == 1
                        ):
                            single_item: PlannedVideo = language_items_for_merge[0]
                            row_meta: Optional[RowVideoCharacteristics] = (
                                single_item.row_characteristics
                            )
                            if (
                                row_meta is not None
                                and bool(row_meta.merge_languages)
                                and row_meta.detected_source_language != language
                            ):
                                run_single_source_translate = True
                                LOGGER.info(
                                    "Single-source LLM translate enabled: src_lang=%s -> target_lang=%s row=%d",
                                    row_meta.detected_source_language,
                                    language,
                                    row_meta.row_index,
                                )
                        if not run_classic_merge and not run_single_source_translate:
                            LOGGER.info(
                                "LLM merge skipped for language=%s: non-empty descriptions=%d.",
                                language,
                                non_empty_descriptions_count,
                            )
                            continue
                        merge_attempt: LanguageMergeAttempt
                        if run_single_source_translate:
                            merge_attempt = attempt_openai_single_source_translate_with_audit(
                                language=language,
                                videos=language_items_for_merge,
                                config=merge_config,
                                attempt_label=f"OPENAI_TRANSLATE_{language.upper()}",
                                summarize_error=_summarize_error,
                                no_description_text=_publish_no_description_text(
                                    _active_templates()
                                ),
                                merge_run_summary=merge_run_summary,
                            )
                        else:
                            merge_attempt = attempt_openai_merge_with_audit(
                                language=language,
                                videos=language_items_for_merge,
                                config=merge_config,
                                attempt_label=f"OPENAI_MERGE_{language.upper()}",
                                summarize_error=_summarize_error,
                                normalize_youtube_url=_normalize_youtube_video_url,
                                no_description_text=_publish_no_description_text(
                                    _active_templates()
                                ),
                                merge_run_summary=merge_run_summary,
                            )
                        LOGGER.info(
                            "LLM merge result language=%s provider=%s model=%s success=%s",
                            language,
                            self._config.llm_provider,
                            merge_attempt.model_name,
                            "yes" if merge_attempt.merged is not None else "no",
                        )
                        merge_audit_by_language[language] = merge_attempt
                        if merge_attempt.merged is not None:
                            merged_content_value: MergedLanguageContent = (
                                merge_attempt.merged
                            )
                            if len(language_items_for_merge) > 1:
                                merged_content_value = enforce_openai_merged_paragraphs(
                                    language=language,
                                    merged_content=merged_content_value,
                                    videos=language_items_for_merge,
                                    config=self._config,
                                    no_description_text=_publish_no_description_text(
                                        _active_templates()
                                    ),
                                    merge_run_summary=merge_run_summary,
                                )
                            merged_content_by_language[language] = merged_content_value
                        else:
                            LOGGER.warning(
                                "LLM merge failed for language=%s provider=%s, using append fallback. reason=%s",
                                language,
                                self._config.llm_provider,
                                merge_attempt.error_summary or "unknown",
                            )
                else:
                    LOGGER.info(
                        "slot=%s merge skipped (LLM unavailable). Using append mode.",
                        slot_key,
                    )

                header_context: Dict[str, str] = _build_header_context(
                    videos=day_videos,
                    form_url=self._config.google_form_url,
                    contacts=self._config.google_contacts,
                    cet_tz=self._cet_tz,
                )
                slot_results.append(
                    {
                        "slot_key": slot_key,
                        "slot_time_key": slot_time_key,
                        "header_context": header_context,
                        "day_videos": day_videos,
                        "language_groups": language_groups,
                        "merged_content_by_language": merged_content_by_language,
                        "merge_audit_by_language": merge_audit_by_language,
                    }
                )

            if not slot_results:
                continue
            for slot in slot_results:
                slot_key_value: str = cast(str, slot["slot_key"])
                language_groups_for_log: Dict[str, List[PlannedVideo]] = cast(
                    Dict[str, List[PlannedVideo]], slot["language_groups"]
                )
                LOGGER.info(
                    "Render plan: slot=%s date=%s uk=%d en=%d ru=%d other=%d",
                    slot_key_value,
                    date_key,
                    len(language_groups_for_log["uk"]),
                    len(language_groups_for_log["en"]),
                    len(language_groups_for_log["ru"]),
                    len(language_groups_for_log["other"]),
                )
                first_urls: Dict[str, str] = {}
                for language in ("uk", "en", "ru", "other"):
                    items: List[PlannedVideo] = language_groups_for_log[language]
                    if items:
                        first_urls[language] = items[0].normalized_link
                if first_urls:
                    LOGGER.debug(
                        "Render plan first_urls: slot=%s urls=%s",
                        slot_key_value,
                        first_urls,
                    )
            # Build a dynamic header list (only present language/time blocks).
            header_blocks: List[str] = []
            for slot in slot_results:
                time_display: str = format_time_key_for_display(
                    cast(str, slot["slot_time_key"])
                )
                for language in ("uk", "en", "ru", "other"):
                    slot_language_items: List[PlannedVideo] = cast(
                        List[PlannedVideo],
                        cast(Dict[str, List[PlannedVideo]], slot["language_groups"])[
                            language
                        ],
                    )
                    if not slot_language_items:
                        continue
                    merged_title: str = ""
                    if resolved_processing_mode == "nomerge":
                        merged_title = build_titles_summary(
                            videos=slot_language_items
                        ).strip()
                    else:
                        merged_content: Optional[MergedLanguageContent] = cast(
                            Dict[str, MergedLanguageContent],
                            slot["merged_content_by_language"],
                        ).get(language)
                        if merged_content is not None and merged_content.title.strip():
                            merged_title = merged_content.title.strip()
                        elif len(slot_language_items) == 1:
                            merged_title = slot_language_items[0].metadata.title.strip()
                        else:
                            merged_title = " / ".join(
                                item.metadata.title.strip()
                                for item in slot_language_items[:3]
                                if item.metadata.title.strip()
                            ).strip()
                    if not merged_title:
                        continue
                    header_blocks.append(
                        f"{_language_heading(language)} - {time_display}\n{merged_title}"
                    )
            language_time_titles: str = "\n\n".join(header_blocks).strip()

            doc_title: str = self._name_builder.build_doc_title(
                date_key=date_key,
                created_at=datetime.now(self._kiev_tz),
                processing_mode=resolved_processing_mode,
            )
            first_header_context: Dict[str, str] = cast(
                Dict[str, str], slot_results[0]["header_context"]
            )
            first_header_context = dict(first_header_context)
            first_header_context["language_time_titles"] = language_time_titles
            doc_url: str = "DRY_RUN_DOC_URL"
            if not dry_run:
                try:
                    document_id = docs_client.create_document(title=doc_title)
                except Exception as error:
                    raise RuntimeError(
                        "Google Docs create_document failed. "
                        f"date={date_key} title={doc_title!r} "
                        f"reason={_summarize_error(error)}"
                    ) from error
                if self._config.google_doc_share_mode != "private":
                    role_by_mode: Dict[str, str] = {
                        "anyone_reader": "reader",
                        "anyone_commenter": "commenter",
                        "anyone_writer": "writer",
                    }
                    role: str = role_by_mode[self._config.google_doc_share_mode]
                    try:
                        drive_client.set_anyone_permission(
                            file_id=document_id,
                            role=role,
                        )
                    except Exception as error:
                        raise RuntimeError(
                            "Google Drive set_anyone_permission failed. "
                            f"document_id={document_id} "
                            f"share_mode={self._config.google_doc_share_mode} role={role} "
                            f"reason={_summarize_error(error)}"
                        ) from error
                if self._config.google_drive_folder_id:
                    try:
                        drive_client.move_file_to_folder(
                            file_id=document_id,
                            folder_id=self._config.google_drive_folder_id,
                        )
                    except Exception as error:
                        raise RuntimeError(
                            "Google Drive move_file_to_folder failed. "
                            f"document_id={document_id} "
                            f"folder_id={self._config.google_drive_folder_id} "
                            f"reason={_summarize_error(error)}"
                        ) from error
                try:
                    report_writer.write_header_only(
                        document_id=document_id,
                        header_text=_build_doc_header_text(
                            header_context=first_header_context,
                            templates=self._config.templates,
                        ),
                    )
                    guard_index: int = (
                        docs_client.get_document_end_index(document_id=document_id) - 1
                    )
                    docs_client.insert_text_at_index(
                        document_id=document_id,
                        index=guard_index,
                        text="\n",
                    )
                    LOGGER.info(
                        "docs_formatting_guard date_key=%s inserted_blank_paragraph_before_first_table index=%d",
                        date_key,
                        guard_index,
                    )
                    first_table_insert_index: int = (
                        docs_client.get_document_end_index(document_id=document_id) - 1
                    )
                    first_table: bool = True
                    for slot in slot_results:
                        time_display: str = format_time_key_for_display(
                            cast(str, slot["slot_time_key"])
                        )
                        language_groups = cast(
                            Dict[str, List[PlannedVideo]], slot["language_groups"]
                        )
                        merged_content_by_language = cast(
                            Dict[str, MergedLanguageContent],
                            slot["merged_content_by_language"],
                        )
                        merge_audit_by_language = cast(
                            Dict[str, LanguageMergeAttempt],
                            slot["merge_audit_by_language"],
                        )
                        for language in ("uk", "en", "ru", "other"):
                            if not language_groups[language]:
                                continue
                            if not first_table:
                                report_writer.insert_page_break(document_id=document_id)
                            report_writer.write_language_table(
                                document_id=document_id,
                                language=language,
                                videos=language_groups[language],
                                merged_content=merged_content_by_language.get(language),
                                merge_attempt=merge_audit_by_language.get(language),
                                time_display=time_display,
                                table_insert_index=(
                                    first_table_insert_index if first_table else None
                                ),
                            )
                            first_table = False
                except Exception as error:
                    raise RuntimeError(
                        "Google Docs write_daily_document failed. "
                        f"document_id={document_id} date={date_key} "
                        f"reason={_summarize_error(error)}"
                    ) from error
                self._export_document_to_local_docx(
                    drive_client=drive_client,
                    document_id=document_id,
                    date_key=date_key,
                    doc_title=doc_title,
                )
                doc_url = f"https://docs.google.com/document/d/{document_id}/edit"
            LOGGER.info(
                "date=%s: Google Doc created: %s (slots=%d)",
                date_key,
                doc_url,
                len(slot_results),
            )
            slot_keys_for_date: List[str] = [
                cast(str, slot["slot_key"]) for slot in slot_results
            ]
            slots_list_text: str = f"[{','.join(slot_keys_for_date)}]"
            LOGGER.info(
                "date_key=%s doc_url=%s slots=%s doc_grouping=date telegram_send_mode=per_date",
                date_key,
                doc_url,
                slots_list_text,
            )
            combined_day_videos: List[PlannedVideo] = []
            combined_merged_content_by_language: Dict[str, MergedLanguageContent] = {}
            combined_merge_audit_by_language: Dict[str, LanguageMergeAttempt] = {}
            merge_language_collisions: Set[str] = set()
            for slot in slot_results:
                combined_day_videos.extend(cast(List[PlannedVideo], slot["day_videos"]))
                slot_merged_map: Dict[str, MergedLanguageContent] = cast(
                    Dict[str, MergedLanguageContent], slot["merged_content_by_language"]
                )
                for language, merged_content in slot_merged_map.items():
                    if (
                        language in merge_language_collisions
                        or language in combined_merged_content_by_language
                    ):
                        merge_language_collisions.add(language)
                        combined_merged_content_by_language.pop(language, None)
                        combined_merge_audit_by_language.pop(language, None)
                        continue
                    combined_merged_content_by_language[language] = merged_content
                slot_merge_audit_map: Dict[str, LanguageMergeAttempt] = cast(
                    Dict[str, LanguageMergeAttempt], slot["merge_audit_by_language"]
                )
                for language, merge_attempt in slot_merge_audit_map.items():
                    if language in merge_language_collisions:
                        continue
                    if language in combined_merge_audit_by_language:
                        merge_language_collisions.add(language)
                        combined_merged_content_by_language.pop(language, None)
                        combined_merge_audit_by_language.pop(language, None)
                        continue
                    combined_merge_audit_by_language[language] = merge_attempt
            if merge_language_collisions:
                LOGGER.info(
                    "telegram merge map collision: date_key=%s disabled_languages=%s",
                    date_key,
                    sorted(merge_language_collisions),
                )
            LOGGER.info(
                "telegram_batch_start date_key=%s doc_url=%s video_count=%d slots=%s send_mode=per_date",
                date_key,
                doc_url,
                len(combined_day_videos),
                slots_list_text,
            )
            self._send_telegram_for_date(
                day_videos=combined_day_videos,
                header_context=first_header_context,
                doc_url=doc_url,
                merged_content_by_language=combined_merged_content_by_language,
                merge_audit_by_language=combined_merge_audit_by_language,
                dry_run=dry_run,
                slot_key=f"date:{date_key}",
                processing_mode=resolved_processing_mode,
            )
        _log_link_normalization_report(
            normalization_candidates=link_normalization_candidates,
            writeback_enabled=sheets_link_writeback_enabled,
        )

    def log_last_merge_run_summary(self) -> None:
        if self._last_merge_run_summary is None:
            return
        self._last_merge_run_summary.log_summary(LOGGER)

    def _export_document_to_local_docx(
        self,
        *,
        drive_client: GoogleDriveClient,
        document_id: str,
        date_key: str,
        doc_title: str,
    ) -> None:
        local_doc_path: Optional[Path] = self._name_builder.build_docx_path(
            date_key=date_key,
            doc_title=doc_title,
        )
        if local_doc_path is None:
            return
        try:
            docx_bytes: bytes = drive_client.export_google_doc_as_docx(document_id)
            local_doc_path.parent.mkdir(parents=True, exist_ok=True)
            local_doc_path.write_bytes(docx_bytes)
            LOGGER.info('doc_local_exported path="%s"', local_doc_path)
        except Exception as error:
            LOGGER.warning(
                'doc_local_export_failed reason="%s"',
                _summarize_error(error),
            )

    def _send_telegram_for_date(
        self,
        day_videos: List[PlannedVideo],
        header_context: Dict[str, str],
        doc_url: str,
        merged_content_by_language: Optional[Dict[str, MergedLanguageContent]],
        merge_audit_by_language: Optional[Dict[str, LanguageMergeAttempt]],
        dry_run: bool,
        slot_key: str,
        processing_mode: str,
    ) -> None:
        if not self._config.telegram_enabled:
            LOGGER.info("Telegram disabled by TELEGRAM_ENABLED=0.")
            return
        LOGGER.info("Telegram send for slot=%s", slot_key)

        date_separator: str = self._config.telegram_symbol_separator * max(
            1, int(self._config.telegram_separator_repeat_count)
        )
        header_message: str = build_telegram_header_text(
            context=header_context,
            generated_doc_url=doc_url,
            config=self._config,
        )
        if dry_run:
            LOGGER.info("DRY RUN Telegram date separator start:\n%s", date_separator)
            LOGGER.info("DRY RUN Telegram header:\n%s", header_message)
        else:
            self._telegram_client.send_text(date_separator)
            self._telegram_client.send_text(header_message)

        grouped: Dict[str, List[PlannedVideo]] = {
            "uk": [],
            "en": [],
            "ru": [],
            "other": [],
        }
        merged_map: Dict[str, MergedLanguageContent] = merged_content_by_language or {}
        merge_attempt_map: Dict[str, LanguageMergeAttempt] = (
            merge_audit_by_language or {}
        )
        for video in day_videos:
            grouped[planned_video_block_language(video)].append(video)
        LOGGER.info(
            "telegram_send batch_key=%s items_uk=%d en=%d ru=%d other=%d processing_mode=%s",
            slot_key,
            len(grouped["uk"]),
            len(grouped["en"]),
            len(grouped["ru"]),
            len(grouped["other"]),
            processing_mode,
        )

        for language in ("uk", "en", "ru", "other"):
            items: List[PlannedVideo] = sorted(
                grouped[language],
                key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
            )
            merged_for_language: Optional[MergedLanguageContent] = merged_map.get(
                language
            )
            if len(items) > 1 and merged_for_language is not None:
                merged_block_text: str = build_telegram_language_merged_block(
                    language=language,
                    videos=items,
                    merged_content=merged_for_language,
                    merge_attempt=merge_attempt_map.get(language),
                    config=self._config,
                    templates=_active_templates(),
                )
                if dry_run:
                    LOGGER.info(
                        "DRY RUN Telegram merged block (%s):\n%s",
                        language,
                        merged_block_text,
                    )
                    for item in items:
                        if item.local_thumbnail_path is None:
                            raise RuntimeError(
                                f"Row {item.row_number}: local thumbnail path is not set."
                            )
                        LOGGER.info(
                            "DRY RUN Telegram merged preview file row %d: %s",
                            item.row_number,
                            item.local_thumbnail_path.name,
                        )
                else:
                    self._telegram_client.send_text(merged_block_text)
                    for item in items:
                        if item.local_thumbnail_path is None:
                            raise RuntimeError(
                                f"Row {item.row_number}: local thumbnail path is not set."
                            )
                        self._telegram_client.send_photo_as_file_bytes(
                            photo_bytes=item.thumbnail.bytes_data,
                            filename=item.local_thumbnail_path.name,
                            mime_type=item.thumbnail.mime_type,
                        )
                continue
            if len(items) > 1 and processing_mode == "nomerge":
                nomerge_block_text: str = build_telegram_language_nomerge_block(
                    language=language,
                    videos=items,
                    config=self._config,
                    templates=_active_templates(),
                )
                if dry_run:
                    LOGGER.info(
                        "DRY RUN Telegram nomerge block (%s):\n%s",
                        language,
                        nomerge_block_text,
                    )
                    for item in items:
                        if item.local_thumbnail_path is None:
                            raise RuntimeError(
                                f"Row {item.row_number}: local thumbnail path is not set."
                            )
                        LOGGER.info(
                            "DRY RUN Telegram nomerge preview file row %d: %s",
                            item.row_number,
                            item.local_thumbnail_path.name,
                        )
                else:
                    self._telegram_client.send_text(nomerge_block_text)
                    for item in items:
                        if item.local_thumbnail_path is None:
                            raise RuntimeError(
                                f"Row {item.row_number}: local thumbnail path is not set."
                            )
                        self._telegram_client.send_photo_as_file_bytes(
                            photo_bytes=item.thumbnail.bytes_data,
                            filename=item.local_thumbnail_path.name,
                            mime_type=item.thumbnail.mime_type,
                        )
                continue
            for item in items:
                block_text = build_telegram_language_block(
                    item,
                    config=self._config,
                    templates=_active_templates(),
                )
                if dry_run:
                    LOGGER.info(
                        "DRY RUN Telegram block for row %d:\n%s",
                        item.row_number,
                        block_text,
                    )
                    continue
                if item.local_thumbnail_path is None:
                    raise RuntimeError(
                        f"Row {item.row_number}: local thumbnail path is not set."
                    )
                self._telegram_client.send_text(block_text)
                self._telegram_client.send_photo_as_file_bytes(
                    photo_bytes=item.thumbnail.bytes_data,
                    filename=item.local_thumbnail_path.name,
                    mime_type=item.thumbnail.mime_type,
                )

        key_form_reminder: str = build_telegram_key_form_reminder(
            header_context, config=self._config
        )
        if dry_run:
            LOGGER.info(
                "DRY RUN Telegram key form reminder block:\n%s",
                key_form_reminder,
            )
        else:
            self._telegram_client.send_text(key_form_reminder)

        sparkle_separator_message: str = (
            self._config.templates.telegram_sparkle_separator
        )
        if dry_run:
            LOGGER.info(
                "DRY RUN Telegram sparkle separator block:\n%s",
                sparkle_separator_message,
            )
        else:
            self._telegram_client.send_text(sparkle_separator_message)

        post_header_message: str = build_telegram_post_header_text(
            header_context=header_context,
            config=self._config,
        )
        if dry_run:
            LOGGER.info(
                "DRY RUN Telegram post-date header block:\n%s",
                post_header_message,
            )
        else:
            self._telegram_client.send_text(post_header_message)

        for language in ("uk", "en", "ru", "other"):
            items = sorted(
                grouped[language],
                key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
            )
            if not items:
                continue
            digest_text: str = build_telegram_language_digest_block(
                language=language,
                videos=items,
                context=header_context,
                config=self._config,
            )
            if dry_run:
                LOGGER.info(
                    "DRY RUN Telegram language digest block (%s):\n%s",
                    language,
                    digest_text,
                )
                continue
            self._telegram_client.send_text(digest_text)
            for item in items:
                if item.local_thumbnail_path is None:
                    raise RuntimeError(
                        f"Row {item.row_number}: local thumbnail path is not set."
                    )
                self._telegram_client.send_photo_as_file_bytes(
                    photo_bytes=item.thumbnail.bytes_data,
                    filename=item.local_thumbnail_path.name,
                    mime_type=item.thumbnail.mime_type,
                )

        if dry_run:
            LOGGER.info("DRY RUN Telegram date separator end:\n%s", date_separator)
        else:
            self._telegram_client.send_text(date_separator)

    def run_single(self, video_url: str, dry_run: bool) -> None:
        if not re.match(r"^https?://", video_url):
            raise ValueError("Ожидался URL с http:// или https://")
        normalized_video_url: str = _normalize_youtube_video_url(video_url)
        metadata: VideoMetadata = self._metadata_fetcher.fetch(
            video_url=normalized_video_url
        )
        language: str = _detect_language(metadata)
        thumbnail_bytes: bytes = self._http_client.get_bytes(metadata.thumbnail_url)
        normalized_thumbnail: NormalizedImage = _normalize_thumbnail(thumbnail_bytes)
        message_text: str = build_single_mode_message(
            metadata=metadata,
            language=language,
            config=self._config,
            templates=_active_templates(),
        )
        if dry_run:
            LOGGER.info("DRY RUN single mode message:\n%s", message_text)
            return
        self._telegram_client.send_text(text=message_text)
        self._telegram_client.send_photo_as_file_bytes(
            photo_bytes=normalized_thumbnail.bytes_data,
            filename=f"{_build_safe_entity_name(metadata.title)}{normalized_thumbnail.extension}",
            mime_type=normalized_thumbnail.mime_type,
            caption=metadata.title[:900] if metadata.title else None,
        )


# ----------------------------
# CLI / конфиг
# ----------------------------


def _load_config_from_env() -> AppConfig:
    config: AppConfig = _load_config_from_env_impl(
        logger=LOGGER,
        summarize_error=_summarize_error,
    )
    globals()["_ACTIVE_TEMPLATES"] = config.templates
    return config


def _load_zoneinfo(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        raise RuntimeError(
            "Timezone database is unavailable for this Python environment. "
            "Install tzdata in the active venv: "
            r"'.venv_streamertg\Scripts\python.exe -m pip install tzdata'. "
            f"Missing zone: {name}"
        ) from error


def main(argv: Sequence[str]) -> int:
    load_dotenv()
    globals()["LOGGER"] = get_logger(__name__)
    argv_list: List[str] = list(argv)
    run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)

    parser = build_cli_parser()
    args = parser.parse_args(argv_list)

    setup_logging(debug=bool(args.debug))
    if bool(args.debug):
        _debug_run_merge_column_self_test()
        _debug_run_merge_range_semantics_and_dedup_self_test()

    config: AppConfig = _load_config_from_env()
    config_processing_mode_raw: str = str(config.processing_mode or "").strip()
    processing_mode: str = config_processing_mode_raw or "merge"
    mode_source: str = "config" if config_processing_mode_raw else "default"
    if bool(args.merge):
        processing_mode = "merge"
        mode_source = "cli"
    elif bool(args.nomerge):
        processing_mode = "nomerge"
        mode_source = "cli"
    processing_mode = normalize_processing_mode(
        processing_mode,
        source="runtime processing mode",
    )

    env_stg_multi_mode: str = os.getenv("STG_MULTI_MODE", "").strip() or "<unset>"
    env_stg_processing_mode: str = (
        os.getenv("STG_PROCESSING_MODE", "").strip() or "<unset>"
    )
    env_stg_merge_semantics: str = (
        os.getenv("STG_MERGE_SEMANTICS", "").strip() or "<unset>"
    )
    config_processing_mode_for_banner: str = (
        config_processing_mode_raw if config_processing_mode_raw else "missing"
    )
    merge_verdict: str = (
        "MERGE=ENABLED" if processing_mode == "merge" else "MERGE=DISABLED"
    )
    LOGGER.info("=== STARTUP BANNER BEGIN ===")
    LOGGER.info("run_id=%s", run_id)
    LOGGER.info("argv=%s", repr(list(sys.argv)))
    LOGGER.info("cwd=%s", os.getcwd())
    LOGGER.info("python=%s", sys.executable)
    LOGGER.info(
        "args.merge=%s args.nomerge=%s args.debug=%s args.dry_run=%s",
        bool(args.merge),
        bool(args.nomerge),
        bool(args.debug),
        bool(args.dry_run),
    )
    LOGGER.info("config_processing_mode=%s", config_processing_mode_for_banner)
    LOGGER.info("env.STG_MULTI_MODE=%s", env_stg_multi_mode)
    LOGGER.info("env.STG_PROCESSING_MODE=%s", env_stg_processing_mode)
    LOGGER.info("env.mode_key_used=%s", "<unused>")
    LOGGER.info("env.STG_MERGE_SEMANTICS=%s", env_stg_merge_semantics)
    LOGGER.info(
        "resolved_processing_mode=%s source=%s",
        processing_mode,
        mode_source,
    )
    LOGGER.info("%s", merge_verdict)
    LOGGER.info("=== STARTUP BANNER END ===")

    def _log_run_startup_line(text: str) -> None:
        LOGGER.info(text)
        _safe_console_print(text)

    _log_run_startup_line("Run startup dump begin")
    _log_run_startup_line(f"Run argv(sys): {repr(list(sys.argv))}")
    _log_run_startup_line(f"Run argv(main): {repr(argv_list)}")
    _log_run_startup_line(
        "Run args: "
        f"args.merge={bool(args.merge)} "
        f"args.nomerge={bool(args.nomerge)} "
        f"args.debug={bool(args.debug)} "
        f"args.dry_run={bool(args.dry_run)}"
    )
    _log_run_startup_line(
        f"Run resolved: resolved_processing_mode={processing_mode} (source={mode_source})"
    )
    _log_run_startup_line(f"cwd={os.getcwd()}")
    _log_run_startup_line(f"script_path={str(Path(__file__).resolve())}")
    _log_run_startup_line(
        f"config_processing_mode={config_processing_mode_raw or '<empty>'}"
    )
    _log_run_startup_line("Run startup dump end")

    mode_label: str = f"batch:{processing_mode}"
    _log_config_summary(
        config,
        mode_label=mode_label,
        resolved_processing_mode=processing_mode,
        run_id=run_id,
    )
    LOGGER.debug(
        "Google Doc share mode resolved: %s (%s)",
        config.google_doc_share_mode,
        describe_google_doc_share_mode(config.google_doc_share_mode),
    )

    metadata_fetcher: YouTubeMetadataFetcher = YtDlpYouTubeMetadataFetcher()
    http_client: HttpClient = HttpClient()
    telegram_client: TelegramBotClient = TelegramBotClient(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )

    app: StreamPipelineApp = StreamPipelineApp(
        config=config,
        metadata_fetcher=metadata_fetcher,
        http_client=http_client,
        telegram_client=telegram_client,
    )

    exit_code: int = 0
    try:
        app.run_batch(
            dry_run=bool(args.dry_run),
            processing_mode=processing_mode,
            run_id=run_id,
        )
        exit_code = 0
    except Exception as error:
        LOGGER.error("FATAL: %s", _summarize_error(error))
        exit_code = 1
    finally:
        try:
            _log_openai_limits_and_usage(LOGGER)
        except Exception:
            LOGGER.exception("OpenAI usage report failed")
        try:
            app.log_last_merge_run_summary()
        except Exception:
            LOGGER.exception("Merge run summary report failed")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
