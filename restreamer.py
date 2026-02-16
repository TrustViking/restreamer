from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from dotenv import load_dotenv
from yt_dlp import YoutubeDL

# Google API (опционально)
try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    GoogleAuthRequest = None  # type: ignore
    Credentials = None  # type: ignore
    InstalledAppFlow = None  # type: ignore
    build = None  # type: ignore
    MediaFileUpload = None  # type: ignore
    HttpError = Exception  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


LOGGER = logging.getLogger("restreamer")


# ----------------------------
# Модели
# ----------------------------


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


def _normalize_youtube_video_url(video_url: str) -> str:
    parsed = urlparse(video_url.strip())
    host = parsed.netloc.lower()
    path = parsed.path
    query = parse_qs(parsed.query)

    video_id: Optional[str] = None
    if "youtu.be" in host:
        candidate: str = path.strip("/").split("/")[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            video_id = candidate
    elif "youtube.com" in host or "m.youtube.com" in host:
        if path == "/watch":
            candidate = (query.get("v") or [""])[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                video_id = candidate
        else:
            parts: List[str] = [part for part in path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
                candidate = parts[1]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                    video_id = candidate

    if not video_id:
        fallback_match: Optional[re.Match[str]] = re.search(
            r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
            video_url,
        )
        if fallback_match:
            video_id = fallback_match.group(1)

    if not video_id:
        raise ValueError(f"Не удалось извлечь YouTube video id из URL: {video_url}")

    return f"https://youtu.be/{video_id}"


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


def _telegram_safe_time(value: str) -> str:
    # Prevent Telegram from turning HH:MM into a clickable timestamp.
    return value.replace(":", ":\u2060")


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


# ----------------------------
# Google Docs/Drive (опционально)
# ----------------------------


class GoogleServicesFactory:
    """
    Создает аутентифицированные клиенты Google Docs + Drive через OAuth.
    """

    _SCOPES: Tuple[str, ...] = (
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    )

    def __init__(self, credentials_path: Path, token_path: Path) -> None:
        self._credentials_path: Path = credentials_path
        self._token_path: Path = token_path

    def create_docs_service(self) -> Any:
        if build is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )
        creds: Any = self._get_credentials()
        return build("docs", "v1", credentials=creds)

    def create_drive_service(self) -> Any:
        if build is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )
        creds: Any = self._get_credentials()
        return build("drive", "v3", credentials=creds)

    def create_sheets_service(self) -> Any:
        if build is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )
        creds: Any = self._get_credentials()
        return build("sheets", "v4", credentials=creds)

    def _get_credentials(self) -> Any:
        if Credentials is None or GoogleAuthRequest is None or InstalledAppFlow is None:
            raise RuntimeError(
                "Google библиотеки не установлены. "
                "Установи google-api-python-client и auth пакеты."
            )

        creds: Optional[Any] = None
        if self._token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(self._token_path), scopes=list(self._SCOPES)
            )
            if creds is not None and not creds.has_scopes(list(self._SCOPES)):
                LOGGER.warning(
                    "Existing token has insufficient scopes. Re-auth is required."
                )
                creds = None

        if creds is not None and getattr(creds, "valid", False):
            return creds

        if (
            creds is not None
            and getattr(creds, "expired", False)
            and getattr(creds, "refresh_token", None)
        ):
            try:
                creds.refresh(GoogleAuthRequest())
                creds_any: Any = creds
                self._token_path.write_text(str(creds_any.to_json()), encoding="utf-8")
                return creds
            except Exception as refresh_error:
                LOGGER.warning(
                    "Token refresh failed (%s). Re-auth flow will be started.",
                    refresh_error,
                )
                creds = None

        flow: Any = InstalledAppFlow.from_client_secrets_file(
            str(self._credentials_path), scopes=list(self._SCOPES)
        )
        creds = flow.run_local_server(port=0)
        creds_any: Any = creds
        self._token_path.write_text(str(creds_any.to_json()), encoding="utf-8")
        return creds


class GoogleSheetsClient:
    def __init__(self, sheets_service: Any) -> None:
        self._sheets_service: Any = sheets_service

    def read_rows(self, spreadsheet_id: str, range_name: str) -> List[SheetRow]:
        try:
            response: Dict[str, Any] = (
                self._sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )
        except HttpError as error:
            error_text: str = str(error)
            if "Unable to parse range" not in error_text:
                raise
            LOGGER.warning(
                "Range %s is invalid for spreadsheet %s; fallback to A:C on first sheet.",
                range_name,
                spreadsheet_id,
            )
            response = (
                self._sheets_service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range="A:C")
                .execute()
            )
        values: List[List[str]] = cast(List[List[str]], response.get("values", []))
        if not values:
            LOGGER.warning("Google Sheets range is empty: %s", range_name)
            return []

        header: List[str] = [str(value).strip() for value in values[0]]
        normalized_header: List[str] = [_normalize_header_name(item) for item in header]

        links_index: Optional[int] = _find_header_index(
            normalized_header=normalized_header,
            aliases=("links", "link", "url", "video", "youtube"),
        )
        date_index: Optional[int] = _find_header_index(
            normalized_header=normalized_header,
            aliases=("date", "дата", "day"),
        )
        time_index: Optional[int] = _find_header_index(
            normalized_header=normalized_header,
            aliases=("time", "время", "hour"),
        )
        if links_index is None or date_index is None or time_index is None:
            raise RuntimeError(
                "Google Sheets header не распознан. "
                f"Found columns: {header!r}. "
                "Нужны колонки для Links/Date/Time."
            )

        rows: List[SheetRow] = []
        for row_number, row_values in enumerate(values[1:], start=2):
            rows.append(
                SheetRow(
                    row_number=row_number,
                    link=_value_from_row(row_values=row_values, index=links_index),
                    date_raw=_value_from_row(row_values=row_values, index=date_index),
                    time_raw=_value_from_row(row_values=row_values, index=time_index),
                )
            )
        return rows


class GoogleDriveClient:
    def __init__(self, drive_service: Any) -> None:
        self._drive_service: Any = drive_service

    def upload_image_and_make_public(
        self,
        image_path: Path,
        folder_id: Optional[str],
        mime_type: str = "image/jpeg",
    ) -> Tuple[str, str]:
        """
        Загружает файл в Drive, открывает публичный доступ по ссылке,
        возвращает (file_id, public_direct_url).
        """
        file_metadata: Dict[str, Any] = {"name": image_path.name}
        if folder_id:
            file_metadata["parents"] = [folder_id]

        if MediaFileUpload is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )

        media: Any = MediaFileUpload(
            str(image_path), mimetype=mime_type, resumable=False
        )
        created: Dict[str, Any] = (
            self._drive_service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )
        file_id: str = str(created["id"])

        # Делаем файл публичным: доступен любому, у кого есть ссылка.
        self._drive_service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
        ).execute()

        # Более надежный direct URL для insertInlineImage.
        public_url: str = f"https://drive.google.com/uc?export=download&id={file_id}"
        for attempt in range(1, 6):
            try:
                response: requests.Response = requests.get(
                    public_url,
                    timeout=10,
                    allow_redirects=True,
                )
                content_type: str = str(
                    response.headers.get("Content-Type", "")
                ).lower()
                if response.status_code == 200 and content_type.startswith("image/"):
                    LOGGER.debug(
                        "Drive image URL is ready on attempt %d: %s",
                        attempt,
                        public_url,
                    )
                    break
            except requests.RequestException:
                pass
            time.sleep(1.0)
        else:
            LOGGER.warning(
                "Drive image URL may still be unavailable after retries: %s",
                public_url,
            )
        return file_id, public_url

    def move_file_to_folder(self, file_id: str, folder_id: str) -> None:
        file_info: Dict[str, Any] = (
            self._drive_service.files().get(fileId=file_id, fields="parents").execute()
        )
        previous_parents: str = ",".join(file_info.get("parents", []))
        self._drive_service.files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        ).execute()

    def set_anyone_permission(self, file_id: str, role: str) -> None:
        if role not in {"reader", "commenter", "writer"}:
            raise ValueError(f"Unsupported Google Drive anyone role: {role}")
        self._drive_service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": role},
            fields="id",
        ).execute()


class GoogleDocsClient:
    def __init__(self, docs_service: Any) -> None:
        self._docs_service: Any = docs_service

    def create_document(self, title: str) -> str:
        doc: Dict[str, Any] = (
            self._docs_service.documents().create(body={"title": title}).execute()
        )
        return str(doc["documentId"])

    def get_document(self, document_id: str) -> Dict[str, Any]:
        return self._docs_service.documents().get(documentId=document_id).execute()

    def batch_update(
        self, document_id: str, requests_payload: List[Dict[str, Any]]
    ) -> None:
        self._docs_service.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests_payload},
        ).execute()

    def insert_table_at_end(self, document_id: str, rows: int, columns: int) -> None:
        # Вставляем таблицу в конец основного сегмента документа.
        self.batch_update(
            document_id=document_id,
            requests_payload=[
                {
                    "insertTable": {
                        "rows": rows,
                        "columns": columns,
                        "endOfSegmentLocation": {},
                    }
                }
            ],
        )


class GoogleDocsReportWriter:
    """
    Создает таблицу в документе и заполняет ее.
    Вставляет изображение в ячейку preview через insertInlineImage
    (нужен публичный URL изображения).
    """

    def __init__(self, docs_client: GoogleDocsClient) -> None:
        self._docs_client: GoogleDocsClient = docs_client

    def write_daily_document(
        self,
        document_id: str,
        header_text: str,
        language_groups: Dict[str, List[PlannedVideo]],
    ) -> None:
        self._insert_header_text(
            document_id=document_id,
            text=f"{header_text}\n\n",
        )

        non_empty_languages: List[str] = [
            language
            for language in ("uk", "en", "ru", "other")
            if language_groups.get(language)
        ]

        for index, language in enumerate(non_empty_languages):
            self.write_language_table(
                document_id=document_id,
                language=language,
                videos=language_groups[language],
            )
            if index < len(non_empty_languages) - 1:
                self._docs_client.batch_update(
                    document_id=document_id,
                    requests_payload=[
                        {"insertPageBreak": {"endOfSegmentLocation": {}}}
                    ],
                )

    def _insert_styled_text(self, document_id: str, text: str, bold: bool) -> None:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc["body"]["content"]
        )
        start_index: int = int(content[-1]["endIndex"]) - 1
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=[
                {"insertText": {"location": {"index": start_index}, "text": text}},
                {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": start_index,
                            "endIndex": start_index + len(text),
                        },
                        "textStyle": {
                            "weightedFontFamily": {"fontFamily": "Arial"},
                            "fontSize": {"magnitude": 13, "unit": "PT"},
                            "bold": bold,
                        },
                        "fields": "weightedFontFamily,fontSize,bold",
                    }
                },
            ],
        )

    def _insert_header_text(self, document_id: str, text: str) -> None:
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc["body"]["content"]
        )
        start_index: int = int(content[-1]["endIndex"]) - 1

        requests_payload: List[Dict[str, Any]] = [
            {"insertText": {"location": {"index": start_index}, "text": text}},
            {
                "updateTextStyle": {
                    "range": {
                        "startIndex": start_index,
                        "endIndex": start_index + len(text),
                    },
                    "textStyle": {
                        "weightedFontFamily": {"fontFamily": "Arial"},
                        "fontSize": {"magnitude": 13, "unit": "PT"},
                        "bold": False,
                    },
                    "fields": "weightedFontFamily,fontSize,bold",
                }
            },
        ]

        # Header emphasis required by template.
        bold_line_prefixes: Tuple[str, ...] = (
            "Ежедневные стримы / Everyday streams",
            "❇️ Эфир",
            "❇️ Форма для ключей",
            "При технических проблемах / In case of technical problems",
            "❇️ Опис / Description / Описание",
            "UA - ",
            "ENG - ",
            "RU - ",
        )
        cursor: int = 0
        for line in text.splitlines(keepends=True):
            line_start: int = start_index + cursor
            line_end: int = line_start + len(line)
            if any(line.startswith(prefix) for prefix in bold_line_prefixes):
                requests_payload.append(
                    {
                        "updateTextStyle": {
                            "range": {
                                "startIndex": line_start,
                                "endIndex": line_end,
                            },
                            "textStyle": {"bold": True},
                            "fields": "bold",
                        }
                    }
                )
            cursor += len(line)

        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=requests_payload,
        )

    def write_language_table(
        self,
        document_id: str,
        language: str,
        videos: List[PlannedVideo],
    ) -> None:
        row_values: List[Tuple[str, bool]] = _build_language_table_rows(
            language=language,
            videos=videos,
        )
        rows: int = len(row_values)
        columns: int = 1
        self._docs_client.insert_table_at_end(
            document_id=document_id,
            rows=rows,
            columns=columns,
        )
        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )

        def _cell_start_index(row_index: int, use_plus_one: bool) -> int:
            idx: int = _cell_index(row=row_index, col=0, columns=columns)
            return cell_start_indices[idx] + (1 if use_plus_one else 0)

        def _build_requests(use_plus_one: bool) -> List[Dict[str, Any]]:
            requests_payload: List[Dict[str, Any]] = []
            for row_index in range(rows - 1, -1, -1):
                text, is_bold = row_values[row_index]
                start_index: int = _cell_start_index(
                    row_index=row_index,
                    use_plus_one=use_plus_one,
                )
                text_to_insert: str = f"{text}\n"
                requests_payload.append(
                    {
                        "insertText": {
                            "location": {"index": start_index},
                            "text": text_to_insert,
                        }
                    }
                )
                requests_payload.append(
                    {
                        "updateTextStyle": {
                            "range": {
                                "startIndex": start_index,
                                "endIndex": start_index + len(text_to_insert),
                            },
                            "textStyle": {
                                "weightedFontFamily": {"fontFamily": "Arial"},
                                "fontSize": {"magnitude": 13, "unit": "PT"},
                                "bold": is_bold,
                            },
                            "fields": "weightedFontFamily,fontSize,bold",
                        }
                    }
                )
            return requests_payload

        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_requests(use_plus_one=False),
            )
        except Exception:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_requests(use_plus_one=True),
            )

        self._apply_language_table_visual_style(
            document_id=document_id,
            rows=rows,
            columns=columns,
            row_values=row_values,
        )
        self._apply_language_table_content_style(
            document_id=document_id,
            rows=rows,
            columns=columns,
            row_values=row_values,
        )

        preview_label_row_index: int = 5
        preview_first_item_row_index: int = preview_label_row_index + 1

        def _refresh_row_start_index(row_index: int) -> int:
            updated_doc: Dict[str, Any] = self._docs_client.get_document(
                document_id=document_id
            )
            updated_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
                doc=updated_doc,
                rows=rows,
                columns=columns,
            )
            idx: int = _cell_index(row=row_index, col=0, columns=columns)
            return updated_indices[idx] + 1

        indexed_videos: List[Tuple[int, PlannedVideo]] = list(
            enumerate(videos, start=1)
        )
        for preview_index, video in reversed(indexed_videos):
            row_index: int = preview_first_item_row_index + (preview_index - 1)
            row_start_index: int = _refresh_row_start_index(row_index)
            link_text: str = f"{video.normalized_link}\n"
            try:
                self._docs_client.batch_update(
                    document_id=document_id,
                    requests_payload=[
                        {
                            "insertText": {
                                "location": {"index": row_start_index},
                                "text": link_text,
                            }
                        }
                    ],
                )
            except Exception:
                LOGGER.warning(
                    "Preview link insert failed for row %d in language %s.",
                    video.row_number,
                    language,
                )
                continue

            inserted: bool = False
            for image_uri in _thumbnail_candidates(video):
                try:
                    row_start_index = _refresh_row_start_index(row_index)
                    self._docs_client.batch_update(
                        document_id=document_id,
                        requests_payload=[
                            {
                                "insertInlineImage": {
                                    "location": {
                                        "index": row_start_index + len(link_text)
                                    },
                                    "uri": image_uri,
                                    "objectSize": {
                                        "height": {"magnitude": 120, "unit": "PT"},
                                        "width": {"magnitude": 210, "unit": "PT"},
                                    },
                                }
                            },
                        ],
                    )
                    inserted = True
                    break
                except Exception:
                    continue
            if not inserted:
                LOGGER.warning(
                    "Preview image insert skipped for row %d in language %s.",
                    video.row_number,
                    language,
                )
 
    def _apply_language_table_visual_style(
        self,
        document_id: str,
        rows: int,
        columns: int,
        row_values: List[Tuple[str, bool]],
    ) -> None:
        # Calm palette for header rows in each language table.
        row_to_rgb: Dict[int, Tuple[float, float, float]] = {
            0: (0.78, 0.84, 0.94),  # language row: muted blue
            1: (0.85, 0.91, 0.83),  # title row: soft green
            3: (0.81, 0.86, 0.78),  # description row: sage
            5: (0.87, 0.83, 0.76),  # preview row: warm sand
        }

        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        table_start_index: int = _find_last_table_start_index(doc=doc)
        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )

        requests_payload: List[Dict[str, Any]] = []
        for row_index, (red, green, blue) in row_to_rgb.items():
            if row_index >= rows:
                continue

            requests_payload.append(
                {
                    "updateTableCellStyle": {
                        "tableRange": {
                            "tableCellLocation": {
                                "tableStartLocation": {"index": table_start_index},
                                "rowIndex": row_index,
                                "columnIndex": 0,
                            },
                            "rowSpan": 1,
                            "columnSpan": 1,
                        },
                        "tableCellStyle": {
                            "backgroundColor": {
                                "color": {
                                    "rgbColor": {
                                        "red": red,
                                        "green": green,
                                        "blue": blue,
                                    }
                                }
                            }
                        },
                        "fields": "backgroundColor",
                    }
                }
            )

            cell_idx: int = _cell_index(row=row_index, col=0, columns=columns)
            paragraph_start: int = cell_start_indices[cell_idx] + 1
            paragraph_end: int = paragraph_start + len(row_values[row_index][0]) + 1
            requests_payload.append(
                {
                    "updateParagraphStyle": {
                        "range": {
                            "startIndex": paragraph_start,
                            "endIndex": paragraph_end,
                        },
                        "paragraphStyle": {"alignment": "CENTER"},
                        "fields": "alignment",
                    }
                }
            )

        if requests_payload:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=requests_payload,
            )

    def _apply_language_table_content_style(
        self,
        document_id: str,
        rows: int,
        columns: int,
        row_values: List[Tuple[str, bool]],
    ) -> None:
        # Content rows: titles and descriptions.
        title_text_row: int = 2
        description_text_row: int = 4
        if rows <= description_text_row:
            return

        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)
        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )

        def _row_range(row_index: int) -> Tuple[int, int]:
            cell_idx: int = _cell_index(row=row_index, col=0, columns=columns)
            # Text can be inserted either at paragraph boundary or +1 fallback.
            # Start from paragraph boundary to avoid losing the first character style.
            start_index: int = cell_start_indices[cell_idx]
            end_index: int = start_index + len(row_values[row_index][0]) + 1
            return start_index, end_index

        title_start, title_end = _row_range(title_text_row)
        desc_start, desc_end = _row_range(description_text_row)

        requests_payload: List[Dict[str, Any]] = [
            {
                "updateTextStyle": {
                    "range": {"startIndex": title_start, "endIndex": title_end},
                    "textStyle": {"bold": True},
                    "fields": "bold",
                }
            },
            {
                "updateTextStyle": {
                    "range": {"startIndex": desc_start, "endIndex": desc_end},
                    "textStyle": {"bold": False},
                    "fields": "bold",
                }
            },
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": title_start, "endIndex": title_end},
                    "paragraphStyle": {"alignment": "JUSTIFIED"},
                    "fields": "alignment",
                }
            },
            {
                "updateParagraphStyle": {
                    "range": {"startIndex": desc_start, "endIndex": desc_end},
                    "paragraphStyle": {"alignment": "START"},
                    "fields": "alignment",
                }
            },
        ]
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=requests_payload,
        )

    def write_video_table(
        self,
        document_id: str,
        video: VideoMetadata,
        docs_thumbnail_url: str,
        fallback_external_thumbnail_url: str,
    ) -> None:
        rows: int = 4
        columns: int = 1

        self._docs_client.insert_table_at_end(
            document_id=document_id,
            rows=rows,
            columns=columns,
        )

        doc: Dict[str, Any] = self._docs_client.get_document(document_id=document_id)

        cell_start_indices: List[int] = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )
        LOGGER.debug(
            "Google table raw paragraph start indices: %s",
            cell_start_indices,
        )

        def _cell_start_index(
            row_index: int, col_index: int, use_plus_one: bool
        ) -> int:
            idx: int = _cell_index(row=row_index, col=col_index, columns=columns)
            if use_plus_one:
                # Вставляем внутрь paragraph ячейки, а не в его структурную границу.
                return cell_start_indices[idx] + 1
            return cell_start_indices[idx]

        def _build_text_requests_payload(use_plus_one: bool) -> List[Dict[str, Any]]:
            effective_cell_indices: List[int] = [
                _cell_start_index(
                    row_index=row_idx,
                    col_index=0,
                    use_plus_one=use_plus_one,
                )
                for row_idx in range(rows)
            ]
            LOGGER.debug(
                "Google table effective insert indices (row0..row%d), plus_one=%s: %s",
                rows - 1,
                use_plus_one,
                effective_cell_indices,
            )

            requests_payload: List[Dict[str, Any]] = []
            description_text: str = video.description.strip() or "(no description)"

            def _build_format_request(
                start_index: int, end_index: int
            ) -> Dict[str, Any]:
                return {
                    "updateTextStyle": {
                        "range": {
                            "startIndex": start_index,
                            "endIndex": end_index,
                        },
                        "textStyle": {
                            "weightedFontFamily": {"fontFamily": "Arial"},
                            "fontSize": {"magnitude": 13, "unit": "PT"},
                        },
                        "fields": "weightedFontFamily,fontSize",
                    }
                }

            # Важно: requests идут снизу вверх, чтобы сдвиг индексов не ломал следующие вставки.
            url_start_index: int = _cell_start_index(
                row_index=2,
                col_index=0,
                use_plus_one=use_plus_one,
            )
            url_text: str = f"{video.url}\n"
            requests_payload.append(
                {
                    "insertText": {
                        "location": {"index": url_start_index},
                        "text": url_text,
                    }
                }
            )
            requests_payload.append(
                _build_format_request(
                    start_index=url_start_index,
                    end_index=url_start_index + len(url_text),
                )
            )

            description_start_index: int = _cell_start_index(
                row_index=1,
                col_index=0,
                use_plus_one=use_plus_one,
            )
            description_full_text: str = f"{description_text}\n"
            requests_payload.append(
                {
                    "insertText": {
                        "location": {"index": description_start_index},
                        "text": description_full_text,
                    }
                }
            )
            requests_payload.append(
                _build_format_request(
                    start_index=description_start_index,
                    end_index=description_start_index + len(description_full_text),
                )
            )

            title_start_index: int = _cell_start_index(
                row_index=0,
                col_index=0,
                use_plus_one=use_plus_one,
            )
            title_text: str = f"{video.title}\n"
            requests_payload.append(
                {
                    "insertText": {
                        "location": {"index": title_start_index},
                        "text": title_text,
                    }
                }
            )
            requests_payload.append(
                _build_format_request(
                    start_index=title_start_index,
                    end_index=title_start_index + len(title_text),
                )
            )
            LOGGER.debug(
                "Google Docs text batchUpdate requests order: url(row2), description(row1), title(row0)"
            )
            return requests_payload

        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_text_requests_payload(use_plus_one=False),
            )
            LOGGER.info("Google Docs table text fill succeeded with plus_one=False.")
        except Exception as base_error:
            LOGGER.warning(
                "Google Docs table text fill failed with plus_one=False. Retrying with plus_one=True. Error: %s",
                base_error,
            )
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=_build_text_requests_payload(use_plus_one=True),
            )
            LOGGER.info(
                "Google Docs table text fill succeeded with plus_one=True (fallback)."
            )

        # Важно: после вставки текста структура документа изменилась.
        # Перед вставкой картинки/ссылки пересчитываем индексы ячеек заново.
        doc = self._docs_client.get_document(document_id=document_id)
        cell_start_indices = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )
        LOGGER.debug(
            "Google table refreshed paragraph start indices before image step: %s",
            cell_start_indices,
        )

        def _build_image_request(use_plus_one: bool, image_uri: str) -> Dict[str, Any]:
            image_index: int = _cell_start_index(
                row_index=3, col_index=0, use_plus_one=use_plus_one
            )
            return {
                "insertInlineImage": {
                    "location": {"index": image_index},
                    "uri": image_uri,
                    "objectSize": {
                        "height": {"magnitude": 180, "unit": "PT"},
                        "width": {"magnitude": 320, "unit": "PT"},
                    },
                }
            }

        def _build_image_link_text_request(use_plus_one: bool) -> Dict[str, Any]:
            image_index: int = _cell_start_index(
                row_index=3, col_index=0, use_plus_one=use_plus_one
            )
            return {
                "insertText": {
                    "location": {"index": image_index},
                    "text": f"Thumbnail: {fallback_external_thumbnail_url}\n",
                }
            }

        def _is_bounds_error(error: Exception) -> bool:
            return "inside the bounds of an existing paragraph" in str(error).lower()

        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[_build_image_request(False, docs_thumbnail_url)],
            )
            LOGGER.info("Google Docs image insert succeeded with plus_one=False.")
            return
        except Exception as image_error:
            LOGGER.warning(
                "Google Docs image insert failed with plus_one=False. Error: %s",
                image_error,
            )
            if _is_bounds_error(image_error):
                try:
                    self._docs_client.batch_update(
                        document_id=document_id,
                        requests_payload=[
                            _build_image_request(True, docs_thumbnail_url)
                        ],
                    )
                    LOGGER.info(
                        "Google Docs image insert succeeded with plus_one=True (fallback)."
                    )
                    return
                except Exception as plus_one_error:
                    LOGGER.warning(
                        "Google Docs image insert failed with plus_one=True. Error: %s",
                        plus_one_error,
                    )

        LOGGER.warning(
            "Google Docs image insert failed; inserting external thumbnail link instead."
        )
        doc = self._docs_client.get_document(document_id=document_id)
        cell_start_indices = _find_last_table_cell_paragraph_start_indices(
            doc=doc,
            rows=rows,
            columns=columns,
        )
        LOGGER.debug(
            "Google table refreshed paragraph start indices before image-link fallback: %s",
            cell_start_indices,
        )
        try:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[_build_image_link_text_request(False)],
            )
        except Exception:
            self._docs_client.batch_update(
                document_id=document_id,
                requests_payload=[_build_image_link_text_request(True)],
            )
        LOGGER.info("Google Docs image row filled with external thumbnail link.")


def _cell_index(row: int, col: int, columns: int) -> int:
    return row * columns + col


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


def _find_last_table_cell_paragraph_start_indices(
    doc: Dict[str, Any], rows: int, columns: int
) -> List[int]:
    """
    Возвращает startIndex абзаца для каждой ячейки (по строкам) последней таблицы.
    Предполагается, что в каждой ячейке есть хотя бы один абзац.
    """
    body: Dict[str, Any] = doc.get("body", {})
    content: List[Dict[str, Any]] = body.get("content", [])
    tables: List[Dict[str, Any]] = [
        item["table"] for item in content if "table" in item
    ]
    if not tables:
        raise RuntimeError("Не нашел таблицу в документе после insertTable.")
    table: Dict[str, Any] = tables[-1]

    table_rows: List[Dict[str, Any]] = table.get("tableRows", [])
    if len(table_rows) < rows:
        raise RuntimeError("Таблица имеет меньше строк, чем ожидается.")
    if any(len(r.get("tableCells", [])) < columns for r in table_rows[:rows]):
        raise RuntimeError("Таблица имеет меньше колонок, чем ожидается.")

    result: List[int] = []
    for row_index in range(rows):
        cells: List[Dict[str, Any]] = table_rows[row_index]["tableCells"]
        for col_index in range(columns):
            cell_content: List[Dict[str, Any]] = cells[col_index].get("content", [])
            paragraph: Optional[Dict[str, Any]] = None
            for element in cell_content:
                if "paragraph" in element:
                    paragraph = element["paragraph"]
                    break
            if paragraph is None:
                raise RuntimeError(
                    "Не нашел paragraph в ячейке таблицы (ожидался всегда)."
                )
            # Элементы абзаца лежат в cell_content; нужен startIndex
            # структурного контейнера, а не вложенного paragraph.
            # Поэтому ищем startIndex в том же элементе, где есть "paragraph".
            paragraph_container: Optional[Dict[str, Any]] = None
            for element in cell_content:
                if "paragraph" in element:
                    paragraph_container = element
                    break
            if paragraph_container is None or "startIndex" not in paragraph_container:
                raise RuntimeError(
                    "Не удалось определить startIndex paragraph контейнера в ячейке."
                )
            start_index: int = int(paragraph_container["startIndex"])

            # start_index: int = (
            #     int(paragraph_container["startIndex"]) + 1
            # )  # +1, чтобы вставка шла внутрь абзаца.
            result.append(start_index)

    if len(result) != rows * columns:
        raise RuntimeError("Внутренняя ошибка: неверное количество индексов ячеек.")
    return result


def _find_last_table_start_index(doc: Dict[str, Any]) -> int:
    body: Dict[str, Any] = doc.get("body", {})
    content: List[Dict[str, Any]] = body.get("content", [])
    table_items: List[Dict[str, Any]] = [item for item in content if "table" in item]
    if not table_items:
        raise RuntimeError("Не найдена таблица в документе.")
    start_index_raw: Optional[Any] = table_items[-1].get("startIndex")
    if start_index_raw is None:
        raise RuntimeError("Не найден startIndex последней таблицы.")
    return int(start_index_raw)


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


def _language_index(language: str) -> int:
    order: Tuple[str, ...] = ("uk", "en", "ru", "other")
    try:
        return order.index(language)
    except ValueError:
        return len(order)


def _language_heading(language: str) -> str:
    labels: Dict[str, str] = {
        "uk": "UA",
        "en": "ENG",
        "ru": "RU",
        "other": "OTHER",
    }
    return labels.get(language, "OTHER")


def _build_titles_summary(videos: List[PlannedVideo]) -> str:
    if not videos:
        return "1) ..."
    use_numbers: bool = len(videos) > 1
    lines: List[str] = []
    for index, video in enumerate(videos, start=1):
        if use_numbers:
            lines.append(f"{index}) {video.metadata.title}")
        else:
            lines.append(video.metadata.title)
    return "\n".join(lines)


def _build_doc_header_text(
    context: Dict[str, str],
    language_groups: Dict[str, List[PlannedVideo]],
) -> str:
    return (
        "Ежедневные стримы / Everyday streams\n"
        f"{context['time_cet']} CET/CEST ({context['time_kiev']} Kiev, {context['time_gmt']} GMT)\n\n"
        f"❇️ Эфир {context['date']}  Скинуть ключи до {context['time_kiev_minus_1']} по Киеву\n"
        f"Drop the keys off before {context['time_gmt_minus_1']} GMT\n\n"
        "❇️ Форма для ключей /  Form for keys\n"
        f"{context['form_url']}\n\n"
        "При технических проблемах / In case of technical problems\n"
        f"Contact: {context['contacts']}\n\n"
        "❇️ Опис / Description / Описание\n"
        f"\nUA - {context['time_ukr']}\n"
        f"{_build_titles_summary(language_groups.get('uk', []))}\n"
        f"\nENG - {context['time_eng']}\n"
        f"{_build_titles_summary(language_groups.get('en', []))}\n"
        f"\nRU - {context['time_ru']}\n"
        f"{_build_titles_summary(language_groups.get('ru', []))}"
    )


def _build_descriptions_summary(videos: List[PlannedVideo]) -> str:
    if not videos:
        return "1) ..."
    use_numbers: bool = len(videos) > 1
    lines: List[str] = []
    for index, video in enumerate(videos, start=1):
        description_text: str = video.metadata.description.strip() or "(no description)"
        if use_numbers:
            lines.append(f"{index}) {description_text}")
        else:
            lines.append(description_text)
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_preview_placeholder_rows(
    videos: List[PlannedVideo],
) -> List[Tuple[str, bool]]:
    if not videos:
        return [(" ", False)]
    return [(" ", False) for _ in videos]


def _youtube_video_id_from_url(video_url: str) -> Optional[str]:
    match: Optional[re.Match[str]] = re.search(
        r"(?:youtu\.be/|v=|embed/|shorts/)([A-Za-z0-9_-]{11})",
        video_url,
    )
    if not match:
        return None
    return match.group(1)


def _thumbnail_candidates(video: PlannedVideo) -> List[str]:
    candidates: List[str] = []
    video_id: Optional[str] = _youtube_video_id_from_url(video.normalized_link)
    if video_id:
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
        candidates.append(f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg")
    candidates.append(video.metadata.thumbnail_url)
    # Keep order and remove duplicates/empty values.
    deduped: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        key: str = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _build_language_table_rows(
    language: str,
    videos: List[PlannedVideo],
) -> List[Tuple[str, bool]]:
    labels: Dict[str, Tuple[str, str, str]] = {
        "uk": ("НАЗВА", "ОПИС", "ПРЕВ'Ю"),
        "en": ("TITLE", "DESCRIPTION", "PREVIEW"),
        "ru": ("НАЗВАНИЕ", "ОПИСАНИЕ", "ПРЕВЬЮ"),
        "other": ("TITLE", "DESCRIPTION", "PREVIEW"),
    }
    title_label, desc_label, preview_label = labels.get(
        language,
        labels["other"],
    )
    rows: List[Tuple[str, bool]] = [
        (_language_heading(language), True),
        (title_label, True),
        (_build_titles_summary(videos), False),
        (desc_label, True),
        (_build_descriptions_summary(videos), False),
        (preview_label, True),
    ]
    rows.extend(_build_preview_placeholder_rows(videos))
    return rows


def _build_telegram_header_text(
    context: Dict[str, str],
    generated_doc_url: str,
    config: "AppConfig",
) -> str:
    time_cet: str = _telegram_safe_time(context["time_cet"])
    time_kiev: str = _telegram_safe_time(context["time_kiev"])
    time_gmt: str = _telegram_safe_time(context["time_gmt"])
    return (
        "Ежедневные стримы / Everyday streams\n"
        f"{time_cet} CET/CEST ({time_kiev} Kiev, {time_gmt} GMT)\n\n"
        f"{config.telegram_symbol_broadcast} Эфир {config.telegram_symbol_alert} {context['date']} / Скинуть ключи за час до эфира\n"
        "Broadcast / Drop the keys off 1 hour before the stream\n\n"
        f"{config.telegram_symbol_form} Форма для ключей / Form for keys\n"
        f"{context['form_url']}\n\n"
        f"{config.telegram_symbol_description} Описание / Description\n"
        f"{generated_doc_url}"
    )


def _telegram_language_flag(language: str, config: "AppConfig") -> str:
    flag_by_language: Dict[str, str] = {
        "uk": config.telegram_flag_uk,
        "en": config.telegram_flag_en,
        "ru": config.telegram_flag_ru,
        "other": config.telegram_flag_other,
    }
    return flag_by_language.get(language, config.telegram_flag_other)


def _telegram_language_flags(language: str, config: "AppConfig") -> str:
    return _telegram_language_flag(language, config) * max(
        1, int(config.telegram_flag_repeat_count)
    )


def _telegram_language_name(language: str, config: "AppConfig") -> str:
    name_by_language: Dict[str, str] = {
        "uk": config.telegram_language_name_uk,
        "en": config.telegram_language_name_en,
        "ru": config.telegram_language_name_ru,
        "other": config.telegram_language_name_other,
    }
    return name_by_language.get(language, config.telegram_language_name_other)


def _build_telegram_language_block(video: PlannedVideo, config: "AppConfig") -> str:
    description_text: str = video.metadata.description.strip() or "(no description)"
    time_kiev_safe: str = _telegram_safe_time(
        video.scheduled_at_kiev.strftime("%H:%M")
    )
    return (
        f"{video.date_display} на {time_kiev_safe} по Киеву\n\n"
        f"{config.telegram_symbol_pin}Название и описание эфира {_telegram_language_flags(video.language, config)}\n"
        "Name and description of stream\n\n"
        f"{video.metadata.title}\n\n"
        f"{description_text}"
    )


def _build_telegram_language_digest_block(
    language: str,
    videos: List[PlannedVideo],
    context: Dict[str, str],
    config: "AppConfig",
) -> str:
    lines: List[str] = [
        f"{_telegram_language_flags(language, config)}{_telegram_language_name(language, config)} на {context['date']}",
        "",
    ]
    for video in videos:
        lines.append(f"{config.telegram_symbol_done} {video.metadata.title}")
        lines.append(video.normalized_link)
        lines.append("")
    return "\n".join(lines).rstrip()


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

    times_by_language: Dict[str, str] = {"uk": "--:--", "en": "--:--", "ru": "--:--"}
    for language in ("uk", "en", "ru"):
        same_language: List[PlannedVideo] = [
            item for item in videos if item.language == language
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
    def __init__(self, local_image_dir_template: str) -> None:
        self._local_image_dir_template: str = local_image_dir_template

    def build_doc_title(self, date_key: str, created_at: datetime) -> str:
        creation_stamp: str = created_at.strftime("%H%M_%d%m%y")
        return f"{date_key} Ежедневные стримы / Everyday streams {creation_stamp}"

    def build_image_path(
        self,
        language: str,
        date_key: str,
        language_index: int,
        title: str,
        extension: str,
    ) -> Path:
        display_language: str = _display_language_code(language)
        base_dir: str = self._local_image_dir_template.format(
            language=display_language,
            date=date_key,
        )
        safe_title: str = _build_safe_entity_name(title)
        return Path(base_dir) / f"{language_index}_{safe_title}{extension}"


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_enabled: bool
    google_enabled: bool
    google_credentials_path: Optional[Path]
    google_token_path: Optional[Path]
    google_drive_folder_id: Optional[str]
    google_doc_share_mode: str
    google_sheets_id: str
    google_sheets_range: str
    google_form_url: str
    google_contacts: str
    local_image_dir_template: str
    timezone_kiev: str
    timezone_cet: str
    telegram_symbol_separator: str
    telegram_separator_repeat_count: int
    telegram_symbol_broadcast: str
    telegram_symbol_alert: str
    telegram_symbol_form: str
    telegram_symbol_description: str
    telegram_symbol_pin: str
    telegram_symbol_done: str
    telegram_flag_uk: str
    telegram_flag_en: str
    telegram_flag_ru: str
    telegram_flag_other: str
    telegram_flag_repeat_count: int
    telegram_language_name_uk: str
    telegram_language_name_en: str
    telegram_language_name_ru: str
    telegram_language_name_other: str


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
        self._kiev_tz = _load_zoneinfo(config.timezone_kiev)
        self._cet_tz = _load_zoneinfo(config.timezone_cet)
        self._name_builder = NamePathBuilder(config.local_image_dir_template)

    def run_batch(self, dry_run: bool) -> None:
        if not self._config.google_enabled:
            raise RuntimeError("Для batch режима GOOGLE_ENABLED должен быть включен.")
        if (
            not self._config.google_credentials_path
            or not self._config.google_token_path
        ):
            raise RuntimeError(
                "Google включен, но не задан GOOGLE_CREDENTIALS_PATH / GOOGLE_TOKEN_PATH."
            )

        services_factory: GoogleServicesFactory = GoogleServicesFactory(
            credentials_path=self._config.google_credentials_path,
            token_path=self._config.google_token_path,
        )
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
            docs_client=docs_client
        )

        LOGGER.info(
            "Reading Google Sheets: spreadsheet=%s range=%s",
            self._config.google_sheets_id,
            self._config.google_sheets_range,
        )
        rows: List[SheetRow] = sheets_client.read_rows(
            spreadsheet_id=self._config.google_sheets_id,
            range_name=self._config.google_sheets_range,
        )
        LOGGER.info("Rows loaded from sheet: %d", len(rows))

        now_kiev: datetime = datetime.now(self._kiev_tz)
        processed: List[PlannedVideo] = []
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
                if scheduled_at < now_kiev:
                    LOGGER.info(
                        "Row %d: skipped, already in the past (%s)",
                        row.row_number,
                        scheduled_at.isoformat(),
                    )
                    continue

                normalized_link: str = _normalize_youtube_video_url(row.link)
                LOGGER.info("Row %d: normalized URL", row.row_number)
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

                processed.append(
                    PlannedVideo(
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
                )
            except Exception as error:
                LOGGER.exception(
                    "Row %d: processing failed. reason=%s",
                    row.row_number,
                    error,
                )

        if not processed:
            LOGGER.warning("No videos to process after filtering.")
            return

        videos_by_date: Dict[str, List[PlannedVideo]] = {}
        for video in processed:
            videos_by_date.setdefault(video.date_key, []).append(video)

        for date_key in sorted(videos_by_date.keys()):
            day_videos: List[PlannedVideo] = sorted(
                videos_by_date[date_key],
                key=lambda item: (
                    _language_index(item.language),
                    item.scheduled_at_kiev.time(),
                    item.row_number,
                ),
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
                    item for item in day_videos if item.language == language
                ]
                for language_index, video in enumerate(language_items, start=1):
                    local_image_path: Path = self._name_builder.build_image_path(
                        language=language,
                        date_key=date_key,
                        language_index=language_index,
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
                    updated_video: PlannedVideo = dataclasses.replace(
                        video,
                        local_thumbnail_path=local_image_path,
                    )
                    language_groups[language].append(updated_video)
                    updated_day_videos.append(updated_video)

            day_videos = updated_day_videos

            header_context: Dict[str, str] = _build_header_context(
                videos=day_videos,
                form_url=self._config.google_form_url,
                contacts=self._config.google_contacts,
                cet_tz=self._cet_tz,
            )
            doc_title: str = self._name_builder.build_doc_title(
                date_key=date_key,
                created_at=datetime.now(self._kiev_tz),
            )
            doc_url: str = "DRY_RUN_DOC_URL"
            if not dry_run:
                document_id: str = docs_client.create_document(title=doc_title)
                if self._config.google_doc_share_mode != "private":
                    role_by_mode: Dict[str, str] = {
                        "anyone_reader": "reader",
                        "anyone_commenter": "commenter",
                        "anyone_writer": "writer",
                    }
                    role: str = role_by_mode[self._config.google_doc_share_mode]
                    drive_client.set_anyone_permission(file_id=document_id, role=role)
                if self._config.google_drive_folder_id:
                    drive_client.move_file_to_folder(
                        file_id=document_id,
                        folder_id=self._config.google_drive_folder_id,
                    )
                report_writer.write_daily_document(
                    document_id=document_id,
                    header_text=_build_doc_header_text(
                        header_context,
                        language_groups=language_groups,
                    ),
                    language_groups=language_groups,
                )
                doc_url = f"https://docs.google.com/document/d/{document_id}/edit"
            LOGGER.info("Date %s: Google Doc created: %s", date_key, doc_url)
            self._send_telegram_for_date(
                day_videos=day_videos,
                header_context=header_context,
                doc_url=doc_url,
                dry_run=dry_run,
            )

    def _send_telegram_for_date(
        self,
        day_videos: List[PlannedVideo],
        header_context: Dict[str, str],
        doc_url: str,
        dry_run: bool,
    ) -> None:
        if not self._config.telegram_enabled:
            LOGGER.info("Telegram disabled by TELEGRAM_ENABLED=0.")
            return

        date_separator: str = (
            self._config.telegram_symbol_separator
            * max(1, int(self._config.telegram_separator_repeat_count))
        )
        header_message: str = _build_telegram_header_text(
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
        for video in day_videos:
            grouped[video.language].append(video)

        for language in ("uk", "en", "ru", "other"):
            items: List[PlannedVideo] = sorted(
                grouped[language],
                key=lambda item: (item.scheduled_at_kiev.time(), item.row_number),
            )
            for item in items:
                block_text = _build_telegram_language_block(item, config=self._config)
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

        post_header_message: str = (
            f"{self._config.telegram_symbol_broadcast} Эфир {header_context['date']}"
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
            digest_text: str = _build_telegram_language_digest_block(
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
        message_text: str = (
            f"Title:\n{metadata.title}\n\n"
            f"Language:\n{language}\n\n"
            f"Description:\n{(metadata.description or '(no description)')}\n\n"
            f"URL:\n{metadata.url}\n"
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
    return AppConfig(
        telegram_bot_token=_must_get_env("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_must_get_env("TELEGRAM_CHAT_ID"),
        telegram_enabled=_load_bool_env("TELEGRAM_ENABLED", default=True),
        google_enabled=_load_bool_env("GOOGLE_ENABLED", default=False),
        google_credentials_path=Path(_must_get_env("GOOGLE_CREDENTIALS_PATH")),
        google_token_path=Path(_must_get_env("GOOGLE_TOKEN_PATH")),
        google_drive_folder_id=os.getenv("GOOGLE_DRIVE_FOLDER_ID") or None,
        google_doc_share_mode=_load_google_doc_share_mode_from_env(),
        google_sheets_id=_must_get_env("GOOGLE_SHEETS_ID"),
        google_sheets_range=_must_get_env("GOOGLE_SHEETS_RANGE"),
        google_form_url=_must_get_env("GOOGLE_FORM_URL"),
        google_contacts=_must_get_env("GOOGLE_CONTACTS"),
        local_image_dir_template=_must_get_env("LOCAL_IMAGE_DIR_TEMPLATE"),
        timezone_kiev=os.getenv("TIMEZONE_KIEV", "Europe/Kyiv").strip(),
        timezone_cet=os.getenv("TIMEZONE_CET", "Europe/Berlin").strip(),
        telegram_symbol_separator=os.getenv("TELEGRAM_SYMBOL_SEPARATOR", "🎬").strip(),
        telegram_separator_repeat_count=_load_int_env(
            "TELEGRAM_SEPARATOR_REPEAT_COUNT", default=8
        ),
        telegram_symbol_broadcast=os.getenv("TELEGRAM_SYMBOL_BROADCAST", "❇️").strip(),
        telegram_symbol_alert=os.getenv("TELEGRAM_SYMBOL_ALERT", "🚨").strip(),
        telegram_symbol_form=os.getenv("TELEGRAM_SYMBOL_FORM", "❇️").strip(),
        telegram_symbol_description=os.getenv(
            "TELEGRAM_SYMBOL_DESCRIPTION", "❇️"
        ).strip(),
        telegram_symbol_pin=os.getenv("TELEGRAM_SYMBOL_PIN", "📌").strip(),
        telegram_symbol_done=os.getenv("TELEGRAM_SYMBOL_DONE", "✅").strip(),
        telegram_flag_uk=os.getenv("TELEGRAM_FLAG_UK", "🇺🇦").strip(),
        telegram_flag_en=os.getenv("TELEGRAM_FLAG_EN", "🇬🇧").strip(),
        telegram_flag_ru=os.getenv("TELEGRAM_FLAG_RU", "🇷🇺").strip(),
        telegram_flag_other=os.getenv("TELEGRAM_FLAG_OTHER", "🌐").strip(),
        telegram_flag_repeat_count=_load_int_env(
            "TELEGRAM_FLAG_REPEAT_COUNT", default=3
        ),
        telegram_language_name_uk=os.getenv("TELEGRAM_LANGUAGE_NAME_UK", "Укр").strip(),
        telegram_language_name_en=os.getenv("TELEGRAM_LANGUAGE_NAME_EN", "Eng").strip(),
        telegram_language_name_ru=os.getenv("TELEGRAM_LANGUAGE_NAME_RU", "Ru").strip(),
        telegram_language_name_other=os.getenv(
            "TELEGRAM_LANGUAGE_NAME_OTHER", "Other"
        ).strip(),
    )


def _load_bool_env(name: str, default: bool) -> bool:
    raw: Optional[str] = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_int_env(name: str, default: int) -> int:
    raw: Optional[str] = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as error:
        raise RuntimeError(f"Env var {name} must be int, got: {raw!r}") from error


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


def _load_google_doc_share_mode_from_env() -> str:
    mode_raw: str = os.getenv("GOOGLE_DOC_SHARE_MODE", "").strip().lower()
    alias_to_mode: Dict[str, str] = {
        "private": "private",
        "off": "private",
        "none": "private",
        "anyone_reader": "anyone_reader",
        "reader": "anyone_reader",
        "read": "anyone_reader",
        "view": "anyone_reader",
        "anyone_commenter": "anyone_commenter",
        "commenter": "anyone_commenter",
        "comment": "anyone_commenter",
        "anyone_writer": "anyone_writer",
        "writer": "anyone_writer",
        "edit": "anyone_writer",
        "editable": "anyone_writer",
    }
    if mode_raw:
        normalized_mode: Optional[str] = alias_to_mode.get(mode_raw)
        if normalized_mode is None:
            supported_modes: str = ", ".join(
                [
                    "private",
                    "anyone_reader",
                    "anyone_commenter",
                    "anyone_writer",
                ]
            )
            raise RuntimeError(
                f"Unsupported GOOGLE_DOC_SHARE_MODE={mode_raw!r}. Supported: {supported_modes}"
            )
        return normalized_mode

    # Backward compatibility with legacy boolean flag.
    legacy_raw: str = os.getenv("GOOGLE_DOC_SHARE_ANYONE_WRITER", "").strip()
    if legacy_raw:
        legacy_enabled: bool = legacy_raw in {"1", "true", "True", "yes", "YES"}
        if legacy_enabled:
            LOGGER.warning(
                "GOOGLE_DOC_SHARE_ANYONE_WRITER is deprecated; use GOOGLE_DOC_SHARE_MODE=anyone_writer."
            )
            return "anyone_writer"
        return "private"

    return "private"


def _describe_google_doc_share_mode(share_mode: str) -> str:
    descriptions: Dict[str, str] = {
        "private": "private: only explicitly granted users can access the document",
        "anyone_reader": "anyone_reader: anyone with the link can view",
        "anyone_commenter": "anyone_commenter: anyone with the link can comment",
        "anyone_writer": "anyone_writer: anyone with the link can edit",
    }
    if share_mode not in descriptions:
        raise RuntimeError(f"Unsupported google_doc_share_mode={share_mode!r}.")
    return descriptions[share_mode]


def _must_get_env(name: str) -> str:
    value: Optional[str] = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"Env var {name} is required.")
    return value.strip()


# def _setup_logging(debug: bool) -> None:
#     logging.basicConfig(
#         level=logging.DEBUG if debug else logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     )


def _setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,  # базовый уровень для всего
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # DEBUG включаем только для нашего логгера
    logging.getLogger("restreamer").setLevel(logging.DEBUG if debug else logging.INFO)

    # Шумные/опасные логгеры — прижимаем
    logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
    logging.getLogger("oauthlib").setLevel(logging.WARNING)


def main(argv: Sequence[str]) -> int:
    load_dotenv()

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Batch pipeline from Google Sheets to Google Docs and Telegram "
            "with local thumbnail saving."
        )
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL for --single mode")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single URL mode (debug only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send to Telegram and do not create Google Docs.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(list(argv))

    _setup_logging(debug=bool(args.debug))

    config: AppConfig = _load_config_from_env()
    safe_config: Dict[str, Any] = dataclasses.asdict(config)
    token_value: str = str(safe_config.get("telegram_bot_token", ""))
    safe_config["telegram_bot_token"] = (
        f"{token_value[:6]}..." if token_value else "<empty>"
    )
    LOGGER.debug(
        "Loaded config: %s",
        json.dumps(safe_config, ensure_ascii=False, default=str),
    )
    LOGGER.debug(
        "Google Doc share mode resolved: %s (%s)",
        config.google_doc_share_mode,
        _describe_google_doc_share_mode(config.google_doc_share_mode),
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

    if bool(args.single):
        if not args.url:
            raise RuntimeError("For --single mode URL is required.")
        app.run_single(video_url=str(args.url), dry_run=bool(args.dry_run))
    else:
        app.run_batch(dry_run=bool(args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
