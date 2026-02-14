from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

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
except ImportError:
    GoogleAuthRequest = None  # type: ignore
    Credentials = None  # type: ignore
    InstalledAppFlow = None  # type: ignore
    build = None  # type: ignore
    MediaFileUpload = None  # type: ignore

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


@dataclass(frozen=True)
class NormalizedImage:
    bytes_data: bytes
    extension: str
    mime_type: str


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
        )


def _normalize_youtube_video_url(video_url: str) -> str:
    video_id_match: Optional[re.Match[str]] = re.search(
        r"(?:youtu\.be/|v=|shorts/)([A-Za-z0-9_-]{11})",
        video_url,
    )
    if not video_id_match:
        return video_url
    return f"https://youtu.be/{video_id_match.group(1)}"


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

    def send_photo_bytes(
        self,
        photo_bytes: bytes,
        filename: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> None:
        url: str = self._make_api_url("sendPhoto")
        files: Dict[str, Tuple[str, bytes, str]] = {
            "photo": (filename, photo_bytes, mime_type),
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

        if creds is not None and getattr(creds, "valid", False):
            return creds

        if (
            creds is not None
            and getattr(creds, "expired", False)
            and getattr(creds, "refresh_token", None)
        ):
            creds.refresh(GoogleAuthRequest())
            creds_any: Any = creds
            self._token_path.write_text(str(creds_any.to_json()), encoding="utf-8")
            return creds

        flow: Any = InstalledAppFlow.from_client_secrets_file(
            str(self._credentials_path), scopes=list(self._SCOPES)
        )
        creds = flow.run_local_server(port=0)
        creds_any: Any = creds
        self._token_path.write_text(str(creds_any.to_json()), encoding="utf-8")
        return creds


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
    timestamp_part: str = datetime.now().strftime("%d%m%y_%H%M")
    normalized_title: str = re.sub(r"\s+", "_", video_title.strip())
    clean_title: str = "".join(
        ch for ch in normalized_title if ch.isalnum() or ch == "_"
    )
    clean_title = re.sub(r"_+", "_", clean_title).strip("_")
    if not clean_title:
        clean_title = "video"
    return f"{timestamp_part}_{clean_title}"


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


# ----------------------------
# Оркестратор (приложение)
# ----------------------------


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    telegram_chat_id: str
    google_enabled: bool
    google_credentials_path: Optional[Path]
    google_token_path: Optional[Path]
    google_drive_folder_id: Optional[str]
    google_doc_share_mode: str


class YouTubeToTelegramAndDocsApp:
    def __init__(
        self,
        config: AppConfig,
        metadata_fetcher: YouTubeMetadataFetcher,
        http_client: HttpClient,
        telegram_client: TelegramBotClient,
    ) -> None:
        self._config: AppConfig = config
        self._metadata_fetcher: YouTubeMetadataFetcher = metadata_fetcher
        self._http_client: HttpClient = http_client
        self._telegram_client: TelegramBotClient = telegram_client

    def run(self, video_url: str, create_google_doc: bool) -> None:
        if not re.match(r"^https?://", video_url):
            raise ValueError("Ожидался URL с http:// или https://")

        normalized_video_url: str = _normalize_youtube_video_url(video_url)
        if normalized_video_url != video_url:
            LOGGER.debug(
                "Normalized YouTube URL from %s to %s",
                video_url,
                normalized_video_url,
            )

        LOGGER.info("Processing video URL: %s", normalized_video_url)
        video: VideoMetadata = self._metadata_fetcher.fetch(
            video_url=normalized_video_url
        )
        safe_entity_name: str = _build_safe_entity_name(video.title)
        LOGGER.debug("Generated safe entity name: %s", safe_entity_name)

        thumbnail_bytes: bytes = self._http_client.get_bytes(video.thumbnail_url)
        normalized_thumbnail: NormalizedImage = _normalize_thumbnail(thumbnail_bytes)

        message_text: str = self._compose_message(video=video)
        self._telegram_client.send_text(text=message_text)
        self._telegram_client.send_photo_bytes(
            photo_bytes=normalized_thumbnail.bytes_data,
            filename=f"{safe_entity_name}{normalized_thumbnail.extension}",
            mime_type=normalized_thumbnail.mime_type,
            caption=video.title[:900] if video.title else None,
        )

        if create_google_doc and self._config.google_enabled:
            self._create_google_doc_with_table(
                video=video,
                normalized_thumbnail=normalized_thumbnail,
                safe_entity_name=safe_entity_name,
            )

    def _compose_message(self, video: VideoMetadata) -> str:
        description_trimmed: str = video.description.strip()
        if not description_trimmed:
            description_trimmed = "(no description)"

        return (
            f"Title:\n{video.title}\n\n"
            f"Description:\n{description_trimmed}\n\n"
            f"URL:\n{video.url}\n"
        )

    def _create_google_doc_with_table(
        self,
        video: VideoMetadata,
        normalized_thumbnail: NormalizedImage,
        safe_entity_name: str,
    ) -> None:
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
        docs_service: Any = services_factory.create_docs_service()
        drive_service: Any = services_factory.create_drive_service()

        docs_client: GoogleDocsClient = GoogleDocsClient(docs_service=docs_service)
        drive_client: GoogleDriveClient = GoogleDriveClient(drive_service=drive_service)
        report_writer: GoogleDocsReportWriter = GoogleDocsReportWriter(
            docs_client=docs_client
        )

        doc_title: str = safe_entity_name
        document_id: str = docs_client.create_document(title=doc_title)

        if self._config.google_doc_share_mode != "private":
            role_by_mode: Dict[str, str] = {
                "anyone_reader": "reader",
                "anyone_commenter": "commenter",
                "anyone_writer": "writer",
            }
            role: str = role_by_mode[self._config.google_doc_share_mode]
            drive_client.set_anyone_permission(file_id=document_id, role=role)
            LOGGER.info(
                "Google Doc sharing enabled: mode=%s (anyone role=%s).",
                self._config.google_doc_share_mode,
                role,
            )
        else:
            LOGGER.info("Google Doc sharing mode=private (no anyone-link access).")

        # Перемещаем документ в папку (опционально).
        if self._config.google_drive_folder_id:
            drive_client.move_file_to_folder(
                file_id=document_id, folder_id=self._config.google_drive_folder_id
            )

        # Сохраняем thumbnail в Drive только как хранилище.
        drive_public_thumbnail_url: Optional[str] = None
        with tempfile.TemporaryDirectory() as temp_dir:
            thumbnail_path: Path = (
                Path(temp_dir) / f"{safe_entity_name}{normalized_thumbnail.extension}"
            )
            thumbnail_path.write_bytes(normalized_thumbnail.bytes_data)

            try:
                _, drive_public_thumbnail_url = (
                    drive_client.upload_image_and_make_public(
                        image_path=thumbnail_path,
                        folder_id=self._config.google_drive_folder_id,
                        mime_type=normalized_thumbnail.mime_type,
                    )
                )
            except Exception as drive_error:
                LOGGER.warning(
                    "Drive thumbnail upload failed (storage only). Continuing without Drive link. Error: %s",
                    drive_error,
                )

        report_writer.write_video_table(
            document_id=document_id,
            video=video,
            docs_thumbnail_url=(drive_public_thumbnail_url or video.thumbnail_url),
            fallback_external_thumbnail_url=(
                drive_public_thumbnail_url or video.thumbnail_url
            ),
        )

        self._telegram_client.send_text(
            text=f"Google Doc created:\nhttps://docs.google.com/document/d/{document_id}/edit"
        )


# ----------------------------
# CLI / конфиг
# ----------------------------


def _load_config_from_env() -> AppConfig:
    telegram_bot_token: str = _must_get_env("TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = _must_get_env("TELEGRAM_CHAT_ID")

    google_enabled_raw: str = os.getenv("GOOGLE_ENABLED", "0").strip()
    google_enabled: bool = google_enabled_raw in {"1", "true", "True", "yes", "YES"}
    google_doc_share_mode: str = _load_google_doc_share_mode_from_env()

    google_credentials_path_str: Optional[str] = os.getenv("GOOGLE_CREDENTIALS_PATH")
    google_token_path_str: Optional[str] = os.getenv("GOOGLE_TOKEN_PATH")
    google_drive_folder_id: Optional[str] = os.getenv("GOOGLE_DRIVE_FOLDER_ID") or None

    google_credentials_path: Optional[Path] = (
        Path(google_credentials_path_str) if google_credentials_path_str else None
    )
    google_token_path: Optional[Path] = (
        Path(google_token_path_str) if google_token_path_str else None
    )

    return AppConfig(
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        google_enabled=google_enabled,
        google_credentials_path=google_credentials_path,
        google_token_path=google_token_path,
        google_drive_folder_id=google_drive_folder_id,
        google_doc_share_mode=google_doc_share_mode,
    )


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
            "Fetch YouTube metadata and send to Telegram. "
            "Optionally create Google Doc with table."
        )
    )
    parser.add_argument("url", type=str, help="YouTube video URL")
    parser.add_argument(
        "--google-doc",
        action="store_true",
        help="Create Google Doc with table (requires Google setup)",
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

    app: YouTubeToTelegramAndDocsApp = YouTubeToTelegramAndDocsApp(
        config=config,
        metadata_fetcher=metadata_fetcher,
        http_client=http_client,
        telegram_client=telegram_client,
    )

    app.run(video_url=str(args.url), create_google_doc=bool(args.google_doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
