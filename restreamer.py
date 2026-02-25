from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import os
import re
import secrets
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
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
from google import genai
from google.genai import types

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from yt_dlp import YoutubeDL
import yaml

# Google API (опционально)
try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials as OAuthUserCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    GoogleAuthRequest = None  # type: ignore
    OAuthUserCredentials = None  # type: ignore
    ServiceAccountCredentials = None  # type: ignore
    InstalledAppFlow = None  # type: ignore
    build = None  # type: ignore
    MediaFileUpload = None  # type: ignore
    HttpError = Exception  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


LOGGER_NAME_ENV_VAR: str = "STREAMERTG_LOGGER_NAME"
LOGGER_NAME_DEFAULT: str = "DEFAULT_NAME"
_SOURCE_URL_LINE_RE: re.Pattern[str] = re.compile(r"^https?://\S+$", re.IGNORECASE)
URL_PATTERN: re.Pattern[str] = re.compile(r"https?://\S+", re.IGNORECASE)
MERGED_DESCRIPTION_HARD_CEILING: int = 4500


def _resolve_logger_name() -> str:
    candidate: str = os.getenv(LOGGER_NAME_ENV_VAR, "").strip()
    return candidate or LOGGER_NAME_DEFAULT


def _resolve_logger_name_meta() -> Tuple[str, str, bool]:
    raw_value: Optional[str] = os.getenv(LOGGER_NAME_ENV_VAR)
    if raw_value is None:
        return (LOGGER_NAME_DEFAULT, "default", False)
    cleaned: str = raw_value.strip()
    if cleaned:
        return (cleaned, "env", True)
    return (LOGGER_NAME_DEFAULT, "default", True)


LOGGER = logging.getLogger(_resolve_logger_name())
_GEMINI_CLIENT: Optional[genai.Client] = None
_OPENAI_CLIENT: Optional[Any] = None
_ACTIVE_TEMPLATES: Optional["AppTemplates"] = None
_GEMINI_RATE_GUARD: Optional["GeminiRateGuard"] = None


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
class PlannedVideoItem:
    row_number: int
    normalized_url: str
    date_key: str
    time_key: str
    slot_key: str
    base_lang: str
    merge_langs: List[str]
    title: str
    description: str
    preview: str
    base_video: PlannedVideo


@dataclass(frozen=True)
class MergedLanguageContent:
    title: str
    description: str
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
    plain_repair_used: Optional[bool] = None
    plain_repair_fallback_model: Optional[str] = None
    validation_reasons: Optional[List[str]] = None


@dataclass(frozen=True)
class MergedPublicationPayload:
    title_text: str
    description_text: str


@dataclass(frozen=True)
class AppTemplates:
    google_doc_header: str
    google_doc_table_labels_json: str
    google_doc_bold_line_prefixes_json: str
    google_doc_language_headings_json: str
    telegram_header: str
    telegram_language_block: str
    telegram_language_merged_block: str
    telegram_key_form_reminder: str
    telegram_sparkle_separator: str
    telegram_post_header: str
    telegram_language_digest_header: str
    llm_merge_title_description_prompt: str
    llm_startup_ping_prompt: str
    llm_language_names_json: str
    files_preview_name_template: str
    files_doc_title_template: str
    files_language_codes_json: str
    common_no_description_text: str
    common_single_mode_message: str


@dataclass(frozen=True)
class OpenAIOrgUsageConfig:
    admin_api_key: str
    project_id: str
    models: Tuple[str, ...]
    monthly_budget_usd: Optional[float]
    timeout_sec: float
    verbose_log: bool


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

    def post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response: requests.Response = requests.post(
            url,
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())


# ----------------------------
# Google Docs/Drive (опционально)
# ----------------------------


class GoogleServicesFactory:
    """
    Создает аутентифицированные клиенты Google Docs + Drive.
    Поддерживает oauth (default) и service_account (через явный env override).
    """

    _SCOPES: Tuple[str, ...] = (
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/cloud-platform.read-only",
    )

    def __init__(self) -> None:
        self._cached_credentials: Optional[Any] = None
        self._auth_mode: str = _load_google_auth_mode_from_env()
        self._service_account_path: Optional[Path] = _load_google_service_account_path()
        self._oauth_credentials_path: Path = _load_google_oauth_credentials_path()
        self._oauth_token_path: Path = _load_google_oauth_token_path()

    def create_docs_service(self) -> Any:
        if build is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )
        creds: Any = self._get_credentials()
        return build("docs", "v1", credentials=creds, cache_discovery=False)

    def create_drive_service(self) -> Any:
        if build is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )
        creds: Any = self._get_credentials()
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def create_sheets_service(self) -> Any:
        if build is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )
        creds: Any = self._get_credentials()
        return build("sheets", "v4", credentials=creds, cache_discovery=False)

    def get_auth_mode(self) -> str:
        return self._auth_mode

    def get_runtime_principal_email(self) -> str:
        creds: Any = self._get_credentials()
        if self._auth_mode == "service_account":
            email_value: str = str(
                getattr(creds, "service_account_email", "")
                or getattr(creds, "_service_account_email", "")
            ).strip()
            return email_value or "unknown"
        if build is None:
            return "unknown"
        try:
            drive_service: Any = build(
                "drive",
                "v3",
                credentials=creds,
                cache_discovery=False,
            )
            response: Dict[str, Any] = cast(
                Dict[str, Any],
                drive_service.about()
                .get(fields="user(emailAddress,displayName)")
                .execute(),
            )
            user_payload: Dict[str, Any] = cast(
                Dict[str, Any], response.get("user", {})
            )
            email: str = str(user_payload.get("emailAddress") or "").strip()
            display_name: str = str(user_payload.get("displayName") or "").strip()
            return email or display_name or "unknown"
        except Exception:
            return "unknown"

    def get_google_project_info(self, *, strict: bool = False) -> Tuple[str, str]:
        creds: Any = self._get_credentials()
        project_id: str = str(getattr(creds, "project_id", "")).strip()
        if not project_id:
            if strict:
                raise RuntimeError(
                    "Google project_id is not available in active credentials."
                )
            return ("unknown", "unknown")
        if build is None:
            if strict:
                raise RuntimeError("Google API client is unavailable.")
            return (project_id, project_id)
        try:
            cloud_service: Any = build(
                "cloudresourcemanager",
                "v3",
                credentials=creds,
                cache_discovery=False,
            )
            response: Dict[str, Any] = cast(
                Dict[str, Any],
                cloud_service.projects().get(name=f"projects/{project_id}").execute(),
            )
            project_name: str = str(
                response.get("displayName") or response.get("projectId") or project_id
            ).strip()
            return (project_id, project_name or project_id)
        except Exception as error:
            parsed_reason: str = _summarize_error(cast(Exception, error))
            if strict:
                raise RuntimeError(f"server_reason={parsed_reason}") from error
            LOGGER.warning(
                "Failed to resolve Google project display name via API (%s).",
                parsed_reason,
            )
            return (project_id, project_id)

    def _get_credentials(self) -> Any:
        if self._cached_credentials is not None:
            return self._cached_credentials
        self._cached_credentials = _create_google_credentials(
            auth_mode=self._auth_mode,
            service_account_path=self._service_account_path,
            oauth_credentials_path=self._oauth_credentials_path,
            oauth_token_path=self._oauth_token_path,
            scopes=self._SCOPES,
        )
        return self._cached_credentials

    def get_oauth_paths(self) -> Tuple[Path, Path]:
        return (self._oauth_credentials_path, self._oauth_token_path)


def _create_google_credentials(
    *,
    auth_mode: str,
    service_account_path: Optional[Path],
    oauth_credentials_path: Path,
    oauth_token_path: Path,
    scopes: Sequence[str],
) -> Any:
    if auth_mode == "service_account":
        return _create_service_account_credentials(
            service_account_path=service_account_path,
            scopes=scopes,
        )
    if auth_mode == "oauth":
        return _create_oauth_credentials(
            oauth_credentials_path=oauth_credentials_path,
            oauth_token_path=oauth_token_path,
            scopes=scopes,
        )
    raise RuntimeError(f"Unsupported GOOGLE_AUTH_MODE={auth_mode!r}.")


def _create_service_account_credentials(
    *,
    service_account_path: Optional[Path],
    scopes: Sequence[str],
) -> Any:
    if ServiceAccountCredentials is None:
        raise RuntimeError(
            "Google service account auth недоступен: не установлен google-auth."
        )
    if service_account_path is None:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            "Resolved path: ''."
        )
    if not service_account_path.exists():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {str(service_account_path)!r}. File not found."
        )
    if not service_account_path.is_file():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {str(service_account_path)!r}. Path is not a file."
        )
    try:
        payload: Any = json.loads(service_account_path.read_text("utf-8-sig"))
    except Exception as error:
        message: str = (
            "Expected type=service_account in Google credentials file. "
            f"Path={service_account_path!s}. Failed to parse JSON."
        )
        LOGGER.error("%s Reason=%s", message, _summarize_error(error))
        raise RuntimeError(message) from error
    if not isinstance(payload, dict):
        message = (
            "Expected type=service_account in Google credentials file. "
            f"Path={service_account_path!s}. JSON root must be an object."
        )
        LOGGER.error(message)
        raise RuntimeError(message)
    type_value: str = str(payload.get("type") or "").strip()
    if type_value != "service_account":
        message = (
            "Expected type=service_account in Google credentials file. "
            f"Path={service_account_path!s}. Found type={type_value!r}."
        )
        LOGGER.error(message)
        raise RuntimeError(message)
    try:
        return ServiceAccountCredentials.from_service_account_file(
            str(service_account_path), scopes=list(scopes)
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load Google service account credentials from {service_account_path}: {error}"
        ) from error


def _create_oauth_credentials(
    *,
    oauth_credentials_path: Path,
    oauth_token_path: Path,
    scopes: Sequence[str],
) -> Any:
    if (
        OAuthUserCredentials is None
        or GoogleAuthRequest is None
        or InstalledAppFlow is None
    ):
        raise RuntimeError(
            "OAuth auth mode requires google-auth and google-auth-oauthlib packages."
        )
    if not oauth_credentials_path.exists() or not oauth_credentials_path.is_file():
        raise RuntimeError(
            "Google OAuth credentials file is required. "
            f"Set GOOGLE_OAUTH_CREDENTIALS_PATH (current={str(oauth_credentials_path)!r}) "
            "to your OAuth client credentials.json (Desktop app)."
        )

    creds: Optional[Any] = None
    if oauth_token_path.exists() and oauth_token_path.is_file():
        try:
            creds = OAuthUserCredentials.from_authorized_user_file(
                str(oauth_token_path),
                scopes=list(scopes),
            )
        except Exception:
            creds = None

    if creds and getattr(creds, "valid", False):
        return creds

    if (
        creds
        and getattr(creds, "expired", False)
        and getattr(creds, "refresh_token", None)
    ):
        try:
            creds.refresh(GoogleAuthRequest())
            oauth_token_path.write_text(str(creds.to_json()), encoding="utf-8")
            return creds
        except Exception:
            creds = None

    try:
        flow: Any = InstalledAppFlow.from_client_secrets_file(
            str(oauth_credentials_path),
            scopes=list(scopes),
        )
        try:
            creds = flow.run_local_server(port=0)
        except Exception:
            run_console_fn: Optional[Any] = getattr(flow, "run_console", None)
            if run_console_fn is None:
                raise RuntimeError(
                    "OAuth local server flow failed and console fallback is unavailable "
                    "in installed google-auth-oauthlib."
                )
            creds = run_console_fn()
        if creds is None:
            raise RuntimeError("OAuth flow returned empty credentials object.")
        creds_to_save: Any = creds
        oauth_token_path.write_text(str(creds_to_save.to_json()), encoding="utf-8")
        return creds_to_save
    except Exception as error:
        raise RuntimeError(
            "Failed to create OAuth token.json. "
            f"credentials_path={str(oauth_credentials_path)!r} token_path={str(oauth_token_path)!r}. "
            "Check OAuth Desktop credentials and browser authorization flow."
        ) from error


class GoogleSheetsClient:
    def __init__(self, sheets_service: Any) -> None:
        self._sheets_service: Any = sheets_service

    def ping_access(self, spreadsheet_id: str) -> Tuple[str, str]:
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._sheets_service.spreadsheets()
            .get(
                spreadsheetId=spreadsheet_id,
                fields="spreadsheetId,properties(title)",
            )
            .execute(),
        )
        resolved_id: str = str(response.get("spreadsheetId") or spreadsheet_id).strip()
        properties: Dict[str, Any] = cast(
            Dict[str, Any], response.get("properties", {})
        )
        title: str = str(properties.get("title") or "unknown").strip() or "unknown"
        return (resolved_id, title)

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
        merge_index: Optional[int] = None
        merge_aliases_specific: Tuple[str, ...] = (
            "Merge (ua/en/ru)",
            "Translate/Overwrite (ua/en/ru)",
        )
        merge_aliases_generic: Tuple[str, ...] = (
            "Merge",
            "Translate/Overwrite",
            "Translate",
            "Overwrite",
            "Перевод",
            "Переклад",
            "Замена",
            "Перезапись",
        )
        merge_aliases_specific_normalized: Tuple[str, ...] = tuple(
            _normalize_header_name(alias) for alias in merge_aliases_specific
        )
        merge_aliases_generic_normalized: Tuple[str, ...] = tuple(
            _normalize_header_name(alias) for alias in merge_aliases_generic
        )

        for idx, name in enumerate(normalized_header):
            if name in merge_aliases_specific_normalized:
                merge_index = idx
                break
        if merge_index is None:
            for idx, name in enumerate(normalized_header):
                if any(
                    alias and alias in name
                    for alias in merge_aliases_specific_normalized
                ):
                    merge_index = idx
                    break
        if merge_index is None:
            for idx, name in enumerate(normalized_header):
                if name in merge_aliases_generic_normalized:
                    merge_index = idx
                    break
        if merge_index is None:
            for idx, name in enumerate(normalized_header):
                if any(
                    alias and alias in name
                    for alias in merge_aliases_generic_normalized
                ):
                    merge_index = idx
                    break

        merge_header_text: Optional[str] = (
            header[merge_index]
            if merge_index is not None and merge_index < len(header)
            else None
        )
        LOGGER.info(
            "Sheets header detected: links_index=%s date_index=%s time_index=%s merge_index=%s merge_header=%r",
            links_index,
            date_index,
            time_index,
            merge_index,
            merge_header_text,
        )
        if merge_index is None:
            possible_translate_headers: List[str] = [
                original_name
                for original_name, normalized_name in zip(header, normalized_header)
                if "translate" in normalized_name or "overwrite" in normalized_name
            ]
            if possible_translate_headers:
                LOGGER.warning(
                    "Merge column not detected. Found possible translate/overwrite column in headers: %r",
                    possible_translate_headers,
                )
        if links_index is None or date_index is None or time_index is None:
            raise RuntimeError(
                "Google Sheets header не распознан. "
                f"Found columns: {header!r}. "
                "Нужны колонки для Links/Date/Time."
            )

        rows: List[SheetRow] = []
        for row_number, row_values in enumerate(values[1:], start=2):
            merge_raw: str = (
                _value_from_row(row_values=row_values, index=merge_index)
                if merge_index is not None
                else ""
            )
            merge_languages: List[str] = parse_merge_languages(merge_raw)
            if merge_raw.strip():
                LOGGER.info(
                    "Row %d: merge column parsed raw=%r tokens=%s",
                    row_number,
                    merge_raw,
                    merge_languages,
                )
                valid_tokens: set[str] = {"ua", "uk", "en", "ru"}
                raw_tokens: List[str] = [
                    item.strip()
                    for item in re.split(r"\s*[;,|]\s*", merge_raw)
                    if item.strip()
                ]
                for token in raw_tokens:
                    if token.lower() not in valid_tokens:
                        LOGGER.warning(
                            "Row %d: invalid Merge token %r ignored",
                            row_number,
                            token,
                        )
            rows.append(
                SheetRow(
                    row_number=row_number,
                    link=_value_from_row(row_values=row_values, index=links_index),
                    date_raw=_value_from_row(row_values=row_values, index=date_index),
                    time_raw=_value_from_row(row_values=row_values, index=time_index),
                    merge_raw=merge_raw,
                    merge_languages=merge_languages,
                    links_column_index=links_index,
                )
            )
        return rows

    def update_cell_string(
        self,
        *,
        spreadsheet_id: str,
        cell_a1: str,
        value: str,
    ) -> None:
        self._sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=cell_a1,
            valueInputOption="RAW",
            body={"values": [[str(value or "")]]},
        ).execute()


class GoogleDriveClient:
    def __init__(self, drive_service: Any) -> None:
        self._drive_service: Any = drive_service

    def ping_access(self) -> Tuple[str, str]:
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._drive_service.about()
            .get(fields="user(displayName,emailAddress)")
            .execute(),
        )
        user_payload: Dict[str, Any] = cast(Dict[str, Any], response.get("user", {}))
        display_name: str = (
            str(user_payload.get("displayName", "")).strip() or "unknown"
        )
        email: str = str(user_payload.get("emailAddress", "")).strip() or "unknown"
        return (display_name, email)

    def get_current_user_info(self) -> Tuple[str, str]:
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._drive_service.about()
            .get(fields="user(displayName,emailAddress)")
            .execute(),
        )
        user_payload: Dict[str, Any] = cast(Dict[str, Any], response.get("user", {}))
        display_name: str = (
            str(user_payload.get("displayName", "")).strip() or "unknown"
        )
        email: str = str(user_payload.get("emailAddress", "")).strip() or "unknown"
        return (display_name, email)

    def get_file_owner_info(self, file_id: str) -> str:
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._drive_service.files()
            .get(
                fileId=file_id,
                fields="owners(displayName,emailAddress)",
                supportsAllDrives=True,
            )
            .execute(),
        )
        owners: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], response.get("owners", [])
        )
        if not owners:
            return "unknown"
        owner: Dict[str, Any] = owners[0]
        owner_name: str = str(owner.get("displayName", "")).strip() or "unknown"
        owner_email: str = str(owner.get("emailAddress", "")).strip() or "unknown"
        return f"{owner_name} <{owner_email}>"

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

        # Логируем метаданные перед загрузкой, чтобы было видно, что именно мы пытаемся загрузить.
        LOGGER.info(
            "Drive upload: folder_id=%r parents=%s file_metadata=%s",
            folder_id,
            file_metadata.get("parents", []),
            file_metadata,
        )

        if MediaFileUpload is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )

        media: Any = MediaFileUpload(
            str(image_path), mimetype=mime_type, resumable=False
        )
        created: Dict[str, Any] = (
            self._drive_service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        file_id: str = str(created["id"])

        # Делаем файл публичным: доступен любому, у кого есть ссылка.
        self._drive_service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
            fields="id",
            supportsAllDrives=True,
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

    def ensure_folder(self, *, parent_folder_id: str, folder_name: str) -> str:
        safe_name: str = str(folder_name or "").strip()
        if not safe_name:
            raise RuntimeError("Drive folder name must not be empty.")
        escaped_name: str = safe_name.replace("'", "\\'")
        query: str = (
            f"name='{escaped_name}' and "
            "mimeType='application/vnd.google-apps.folder' and "
            "trashed=false and "
            f"'{parent_folder_id}' in parents"
        )
        response: Dict[str, Any] = cast(
            Dict[str, Any],
            self._drive_service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id,name)",
                pageSize=1,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute(),
        )
        files_found: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], response.get("files", [])
        )
        if files_found:
            return str(files_found[0].get("id") or "").strip()

        payload: Dict[str, Any] = {
            "name": safe_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }
        created: Dict[str, Any] = cast(
            Dict[str, Any],
            self._drive_service.files()
            .create(
                body=payload,
                fields="id,name",
                supportsAllDrives=True,
            )
            .execute(),
        )
        created_id: str = str(created.get("id") or "").strip()
        if not created_id:
            raise RuntimeError(
                f"Drive folder creation returned empty id: name={safe_name!r} parent={parent_folder_id!r}"
            )
        LOGGER.info(
            "Drive preview date folder ensured: parent_id=%s name=%s folder_id=%s",
            parent_folder_id,
            safe_name,
            created_id,
        )
        return created_id

    def ensure_folder_path(
        self,
        *,
        parent_folder_id: str,
        path_parts: Sequence[str],
    ) -> str:
        current_folder_id: str = parent_folder_id
        for part in path_parts:
            cleaned_part: str = str(part or "").strip()
            if not cleaned_part:
                continue
            current_folder_id = self.ensure_folder(
                parent_folder_id=current_folder_id,
                folder_name=cleaned_part,
            )
        return current_folder_id

    def move_file_to_folder(self, file_id: str, folder_id: str) -> None:
        file_info: Dict[str, Any] = (
            self._drive_service.files()
            .get(fileId=file_id, fields="parents", supportsAllDrives=True)
            .execute()
        )
        previous_parents: str = ",".join(file_info.get("parents", []))
        self._drive_service.files().update(
            fileId=file_id,
            addParents=folder_id,
            removeParents=previous_parents,
            fields="id, parents",
            supportsAllDrives=True,
        ).execute()

    def set_anyone_permission(self, file_id: str, role: str) -> None:
        if role not in {"reader", "commenter", "writer"}:
            raise ValueError(f"Unsupported Google Drive anyone role: {role}")
        self._drive_service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": role},
            fields="id",
            supportsAllDrives=True,
        ).execute()


class GoogleDocsClient:
    def __init__(self, docs_service: Any) -> None:
        self._docs_service: Any = docs_service

    def ping_access(self) -> str:
        # Safe ping without mutating user documents.
        try:
            self._docs_service.documents().get(documentId="streamertg-ping").execute()
            return "probe_document_found(unexpected)"
        except HttpError as error:
            status_code: Optional[int] = getattr(
                getattr(error, "resp", None), "status", None
            )
            if status_code in {400, 404}:
                return f"probe_rejected_as_expected(status={status_code})"
            raise

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

    def get_document_end_index(self, document_id: str) -> int:
        doc: Dict[str, Any] = self.get_document(document_id=document_id)
        content: List[Dict[str, Any]] = cast(
            List[Dict[str, Any]], doc.get("body", {}).get("content", [])
        )
        if not content:
            raise RuntimeError("Google Docs document body.content is empty.")
        end_index_raw: Optional[Any] = content[-1].get("endIndex")
        if end_index_raw is None:
            raise RuntimeError("Google Docs document endIndex is missing.")
        return int(end_index_raw)

    def insert_text_at_index(self, document_id: str, index: int, text: str) -> None:
        self.batch_update(
            document_id=document_id,
            requests_payload=[
                {"insertText": {"location": {"index": int(index)}, "text": text}}
            ],
        )

    def insert_table_at_index(
        self,
        document_id: str,
        rows: int,
        columns: int,
        index: int,
    ) -> None:
        self.batch_update(
            document_id=document_id,
            requests_payload=[
                {
                    "insertTable": {
                        "rows": rows,
                        "columns": columns,
                        "location": {"index": int(index)},
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

    def __init__(self, docs_client: GoogleDocsClient, templates: AppTemplates) -> None:
        self._docs_client: GoogleDocsClient = docs_client
        self._templates: AppTemplates = templates

    def write_daily_document(
        self,
        document_id: str,
        header_text: str,
        language_groups: Dict[str, List[PlannedVideo]],
        merged_content_by_language: Optional[Dict[str, MergedLanguageContent]] = None,
        merge_audit_by_language: Optional[Dict[str, LanguageMergeAttempt]] = None,
    ) -> None:
        self._insert_header_text(
            document_id=document_id,
            text=f"{header_text}\n\n",
        )
        guard_index: int = (
            self._docs_client.get_document_end_index(document_id=document_id) - 1
        )
        self._docs_client.insert_text_at_index(
            document_id=document_id,
            index=guard_index,
            text="\n",
        )
        LOGGER.info(
            "Inserted blank paragraph before first table (doc formatting guard) index=%d.",
            guard_index,
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
                merged_content=(
                    (merged_content_by_language or {}).get(language)
                    if merged_content_by_language
                    else None
                ),
                merge_attempt=(
                    (merge_audit_by_language or {}).get(language)
                    if merge_audit_by_language
                    else None
                ),
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

        bold_line_prefixes: Tuple[str, ...] = ()
        try:
            raw_prefixes: Any = json.loads(
                self._templates.google_doc_bold_line_prefixes_json
            )
            if isinstance(raw_prefixes, list):
                bold_line_prefixes = tuple(str(item) for item in raw_prefixes)
        except Exception:
            bold_line_prefixes = ()
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

    def write_header_only(
        self,
        document_id: str,
        header_text: str,
    ) -> None:
        self._insert_header_text(document_id=document_id, text=header_text)

    def insert_page_break(self, document_id: str) -> None:
        self._docs_client.batch_update(
            document_id=document_id,
            requests_payload=[{"insertPageBreak": {"endOfSegmentLocation": {}}}],
        )

    def write_language_table(
        self,
        document_id: str,
        language: str,
        videos: List[PlannedVideo],
        merged_content: Optional[MergedLanguageContent] = None,
        merge_attempt: Optional[LanguageMergeAttempt] = None,
        time_display: Optional[str] = None,
        table_insert_index: Optional[int] = None,
    ) -> None:
        row_values: List[Tuple[str, bool]] = _build_language_table_rows(
            language=language,
            videos=videos,
            merged_content=merged_content,
            merge_attempt=merge_attempt,
            time_display=time_display,
            templates=self._templates,
        )
        rows: int = len(row_values)
        columns: int = 1
        if table_insert_index is None:
            self._docs_client.insert_table_at_end(
                document_id=document_id,
                rows=rows,
                columns=columns,
            )
        else:
            self._docs_client.insert_table_at_index(
                document_id=document_id,
                rows=rows,
                columns=columns,
                index=table_insert_index,
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
            description_text: str = video.description.strip() or _no_description_text()

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


def parse_merge_languages(value: str) -> List[str]:
    token_map: Dict[str, str] = {
        "ua": "uk",
        "uk": "uk",
        "en": "en",
        "ru": "ru",
    }
    stable_order: Tuple[str, ...] = ("uk", "en", "ru")
    selected: set[str] = set()
    for raw_token in re.split(r"\s*[;,|]\s*", str(value or "")):
        token: str = raw_token.strip().lower()
        if not token:
            continue
        mapped: Optional[str] = token_map.get(token)
        if mapped:
            selected.add(mapped)
    return [language for language in stable_order if language in selected]


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
) -> None:
    old_link_value: str = str(old_link or "").strip()
    new_link_value: str = str(normalized_link or "").strip()
    if not new_link_value:
        return
    if new_link_value == old_link_value:
        return
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
            'Row %d link normalized (writeback disabled): old="%s" new="%s"',
            row_number,
            old_link_value,
            new_link_value,
        )
        return
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
            "Row %d link normalized and written back.",
            row_number,
        )
    except Exception as writeback_error:
        LOGGER.warning(
            "Row %d: link normalized but write-back failed. reason=%s",
            row_number,
            _summarize_error(writeback_error),
        )


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


def _merge_semantics_from_env() -> str:
    raw_value: str = str(os.getenv("STG_MERGE_SEMANTICS", "override") or "").strip()
    normalized_value: str = raw_value.lower()
    if normalized_value in {"override", "add"}:
        return normalized_value
    LOGGER.warning(
        "Unknown STG_MERGE_SEMANTICS=%r; falling back to override.",
        raw_value,
    )
    return "override"


def _deduplicate_planned_videos_within_date_language(
    videos: List[PlannedVideo],
) -> List[PlannedVideo]:
    sorted_by_row: List[PlannedVideo] = sorted(
        videos,
        key=lambda item: (
            item.date_key,
            _planned_video_block_language(item),
            item.row_number,
        ),
    )
    kept_by_key: Dict[Tuple[str, str, str], PlannedVideo] = {}
    deduped: List[PlannedVideo] = []
    for video in sorted_by_row:
        block_language: str = _planned_video_block_language(video)
        normalized_url: str = str(video.normalized_link or "").strip()
        dedup_key: Tuple[str, str, str] = (
            video.date_key,
            block_language,
            normalized_url,
        )
        existing: Optional[PlannedVideo] = kept_by_key.get(dedup_key)
        if existing is None:
            kept_by_key[dedup_key] = video
            deduped.append(video)
            continue
        LOGGER.warning(
            "Duplicate link skipped: date=%s lang=%s url=%s kept_row=%d skipped_row=%d",
            video.date_key,
            block_language,
            normalized_url,
            existing.row_number,
            video.row_number,
        )
    return deduped


def _planned_video_time_key(video: PlannedVideo) -> str:
    return video.scheduled_at_kiev.strftime("%H%M")


def _planned_video_slot_key(video: PlannedVideo) -> str:
    return f"{video.date_key}_{_planned_video_time_key(video)}"


def _format_time_key_for_display(time_key: str) -> str:
    stripped: str = time_key.strip()
    if re.fullmatch(r"\d{4}", stripped):
        return f"{stripped[:2]}:{stripped[2:]}"
    return stripped


def _deduplicate_planned_videos_within_slot_global(
    videos: List[PlannedVideo],
) -> List[PlannedVideo]:
    # De-duplicate by normalized URL within the same slot_key (ignore language).
    # Keep the row that has a non-empty Merge column; otherwise keep the earliest row_number.
    kept_rows_by_key: Dict[Tuple[str, str], int] = {}
    has_merge_by_row: Dict[Tuple[str, str, int], bool] = {}

    for item in videos:
        slot_key: str = _planned_video_slot_key(item)
        url_key: str = item.normalized_link
        row_key: Tuple[str, str, int] = (slot_key, url_key, item.row_number)
        if row_key not in has_merge_by_row:
            row_meta: Optional[RowVideoCharacteristics] = item.row_characteristics
            has_merge_by_row[row_key] = bool(
                row_meta is not None and row_meta.merge_languages
            )

    for item in videos:
        slot_key = _planned_video_slot_key(item)
        url_key = item.normalized_link
        group_key: Tuple[str, str] = (slot_key, url_key)
        row_key = (slot_key, url_key, item.row_number)

        current_best_row: Optional[int] = kept_rows_by_key.get(group_key)
        if current_best_row is None:
            kept_rows_by_key[group_key] = item.row_number
            continue

        best_row_key: Tuple[str, str, int] = (slot_key, url_key, current_best_row)
        best_has_merge: bool = has_merge_by_row.get(best_row_key, False)
        candidate_has_merge: bool = has_merge_by_row.get(row_key, False)

        if candidate_has_merge and not best_has_merge:
            kept_rows_by_key[group_key] = item.row_number
        elif (
            candidate_has_merge == best_has_merge and item.row_number < current_best_row
        ):
            kept_rows_by_key[group_key] = item.row_number

    skipped_rows_logged: Set[Tuple[str, str, int, int]] = set()
    filtered: List[PlannedVideo] = []
    for item in videos:
        slot_key = _planned_video_slot_key(item)
        url_key = item.normalized_link
        group_key = (slot_key, url_key)
        kept_row: int = kept_rows_by_key[group_key]
        if item.row_number == kept_row:
            filtered.append(item)
            continue

        log_key: Tuple[str, str, int, int] = (
            slot_key,
            url_key,
            kept_row,
            item.row_number,
        )
        if log_key not in skipped_rows_logged:
            skipped_rows_logged.add(log_key)
            best_has_merge = has_merge_by_row.get((slot_key, url_key, kept_row), False)
            candidate_has_merge = has_merge_by_row.get(
                (slot_key, url_key, item.row_number), False
            )
            reason: str
            if best_has_merge and not candidate_has_merge:
                reason = "kept_row_has_merge"
            else:
                reason = "kept_earliest_row"
            LOGGER.info(
                "Duplicate URL skipped in slot: slot=%s url=%s kept_row=%d skipped_row=%d reason=%s",
                slot_key,
                url_key,
                kept_row,
                item.row_number,
                reason,
            )
    return filtered


def _deduplicate_planned_videos_within_slot_language(
    videos: List[PlannedVideo],
) -> List[PlannedVideo]:
    sorted_by_row: List[PlannedVideo] = sorted(
        videos,
        key=lambda item: (
            _planned_video_slot_key(item),
            _planned_video_block_language(item),
            item.row_number,
        ),
    )
    kept_by_key: Dict[Tuple[str, str, str], PlannedVideo] = {}
    deduped: List[PlannedVideo] = []
    for video in sorted_by_row:
        block_language: str = _planned_video_block_language(video)
        normalized_url: str = str(video.normalized_link or "").strip()
        slot_key: str = _planned_video_slot_key(video)
        dedup_key: Tuple[str, str, str] = (
            slot_key,
            block_language,
            normalized_url,
        )
        existing: Optional[PlannedVideo] = kept_by_key.get(dedup_key)
        if existing is None:
            kept_by_key[dedup_key] = video
            deduped.append(video)
            continue
        LOGGER.warning(
            "Duplicate URL skipped in group: slot=%s lang=%s url=%s kept_row=%d skipped_row=%d",
            slot_key,
            block_language,
            normalized_url,
            existing.row_number,
            video.row_number,
        )
    return deduped


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


def _planned_video_block_language(video: PlannedVideo) -> str:
    if video.forced_block_language in {"uk", "en", "ru", "other"}:
        return str(video.forced_block_language)
    return video.language


def _language_index(language: str) -> int:
    order: Tuple[str, ...] = ("uk", "en", "ru", "other")
    try:
        return order.index(language)
    except ValueError:
        return len(order)


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


def _numbered_lines(values: Sequence[str]) -> str:
    cleaned_values: List[str] = [
        str(item or "").strip() for item in values if str(item or "").strip()
    ]
    if not cleaned_values:
        return "1) ..."
    if len(cleaned_values) == 1:
        return cleaned_values[0]
    return "\n".join(
        [f"{index}) {item}" for index, item in enumerate(cleaned_values, start=1)]
    )


def _no_description_text() -> str:
    templates: Optional[AppTemplates] = globals().get("_ACTIVE_TEMPLATES")
    if templates is None:
        return "no description"
    value: str = str(templates.common_no_description_text or "").strip()
    return value or "no description"


def _merged_title_for_docs(merged_content: Optional[MergedLanguageContent]) -> str:
    if not merged_content:
        return ""
    return str(merged_content.title_audit or merged_content.title or "").strip()


def _merged_title_for_selected(merged_content: Optional[MergedLanguageContent]) -> str:
    if not merged_content:
        return ""
    return str(merged_content.title_selected or merged_content.title or "").strip()


def _merged_description_for_docs(
    merged_content: Optional[MergedLanguageContent],
) -> str:
    if not merged_content:
        return ""
    return str(
        merged_content.description_audit or merged_content.description or ""
    ).strip()


def _merged_description_for_selected(
    merged_content: Optional[MergedLanguageContent],
) -> str:
    if not merged_content:
        return ""
    return str(
        merged_content.description_selected or merged_content.description or ""
    ).strip()


def _numbered_original_titles(videos: List[PlannedVideo]) -> str:
    source_titles: List[str] = [video.metadata.title for video in videos]
    return _numbered_lines(source_titles)


def _build_merged_publication_payload(
    *,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    use_audit_text: bool,
) -> MergedPublicationPayload:
    merged_title_text: str = (
        _merged_title_for_docs(merged_content)
        if use_audit_text
        else _merged_title_for_selected(merged_content)
    )
    merged_description_text: str = (
        _merged_description_for_docs(merged_content)
        if use_audit_text
        else _merged_description_for_selected(merged_content)
    )
    title_text: str = merged_title_text.strip()
    return MergedPublicationPayload(
        title_text=title_text,
        description_text=merged_description_text.strip(),
    )


def _build_titles_summary(
    videos: List[PlannedVideo],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    use_audit_text: bool = True,
) -> str:
    if merged_content:
        payload: MergedPublicationPayload = _build_merged_publication_payload(
            videos=videos,
            merged_content=merged_content,
            use_audit_text=use_audit_text,
        )
        return payload.title_text
    if merge_attempt is not None:
        salvaged_title: str = str(merge_attempt.salvaged_title or "").strip()
        if salvaged_title:
            return salvaged_title
        return _numbered_original_titles(videos) if videos else "1) ..."
    if not videos:
        return "1) ..."
    return _numbered_original_titles(videos)


def _build_doc_header_text(
    header_context: Dict[str, str],
    templates: AppTemplates,
) -> str:
    context: Dict[str, str] = dict(header_context)
    context.setdefault("language_time_titles", "")
    return _render_template(templates.google_doc_header, context)


def _fallback_source_description_text(video: PlannedVideo) -> str:
    description_text: str = video.metadata.description.strip() or _no_description_text()
    if _strip_chapter_timestamps_enabled_from_env():
        description_text = strip_chapter_timestamps(description_text)
    return description_text


def _build_descriptions_summary(
    videos: List[PlannedVideo],
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    use_audit_text: bool = True,
) -> str:
    source_descriptions: List[str] = []
    for video in videos:
        source_descriptions.append(_fallback_source_description_text(video))
    source_lines: str = _numbered_lines(source_descriptions)
    if merged_content:
        payload: MergedPublicationPayload = _build_merged_publication_payload(
            videos=videos,
            merged_content=merged_content,
            use_audit_text=use_audit_text,
        )
        return payload.description_text
    if merge_attempt is not None:
        raw_text: str = str(merge_attempt.raw_response_text or "").strip()
        if raw_text:
            return f"{raw_text}\n\n{source_lines}".strip()
        return source_lines
    if not videos:
        return "1) ..."
    if len(source_descriptions) == 1:
        return source_descriptions[0]
    lines: List[str] = []
    for index, item in enumerate(source_descriptions, start=1):
        lines.append(f"{index}) {item}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_preview_placeholder_rows(
    videos: List[PlannedVideo],
) -> List[Tuple[str, bool]]:
    if not videos:
        return [(" ", False)]
    return [(" ", False) for _ in videos]


def _youtube_video_id_from_url(video_url: str) -> Optional[str]:
    return _extract_youtube_video_id(video_url)


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
    merged_content: Optional[MergedLanguageContent] = None,
    merge_attempt: Optional[LanguageMergeAttempt] = None,
    time_display: Optional[str] = None,
    templates: Optional[AppTemplates] = None,
) -> List[Tuple[str, bool]]:
    if templates is None:
        raise RuntimeError("Templates are required for language table labels.")
    try:
        labels_payload: Any = json.loads(templates.google_doc_table_labels_json)
    except Exception as error:
        raise RuntimeError(
            f"Invalid template google_doc.table_labels: {error}"
        ) from error
    labels: Dict[str, Tuple[str, str, str]] = {}
    if isinstance(labels_payload, dict):
        for key, value in labels_payload.items():
            if isinstance(value, list) and len(value) == 3:
                labels[str(key)] = (str(value[0]), str(value[1]), str(value[2]))
    title_label, desc_label, preview_label = labels.get(
        language,
        labels.get("other", ("TITLE", "DESCRIPTION", "PREVIEW")),
    )

    heading: str = _language_heading(language)
    if time_display:
        heading = f"{heading} - {time_display}"
    description_text: str = _build_descriptions_summary(
        videos=videos,
        merged_content=merged_content,
        merge_attempt=merge_attempt,
    )

    rows: List[Tuple[str, bool]] = [
        (heading, True),
        (title_label, True),
        (
            _build_titles_summary(
                videos=videos,
                merged_content=merged_content,
                merge_attempt=merge_attempt,
            ),
            False,
        ),
        (desc_label, True),
        (description_text, False),
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
    return _render_template(
        config.templates.telegram_header,
        {
            "time_cet": time_cet,
            "time_kiev": time_kiev,
            "time_gmt": time_gmt,
            "symbol_broadcast": config.telegram_symbol_broadcast,
            "symbol_alert": config.telegram_symbol_alert,
            "date": context["date"],
            "symbol_form": config.telegram_symbol_form,
            "form_url": context["form_url"],
            "contacts": context["contacts"],
            "symbol_description": config.telegram_symbol_description,
            "generated_doc_url": generated_doc_url,
        },
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
    language: str = _planned_video_block_language(video)
    description_text: str = _fallback_source_description_text(video)
    time_kiev_safe: str = _telegram_safe_time(video.scheduled_at_kiev.strftime("%H:%M"))
    return _render_template(
        config.templates.telegram_language_block,
        {
            "date_display": video.date_display,
            "time_kiev": time_kiev_safe,
            "symbol_pin": config.telegram_symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": video.metadata.title,
            "description": description_text,
        },
    )


def _build_telegram_language_merged_block(
    language: str,
    videos: List[PlannedVideo],
    merged_content: MergedLanguageContent,
    merge_attempt: Optional[LanguageMergeAttempt],
    config: "AppConfig",
) -> str:
    if not videos:
        raise ValueError("videos must not be empty for merged telegram block")
    times_text: str = ", ".join(
        sorted({video.scheduled_at_kiev.strftime("%H:%M") for video in videos})
    )
    date_display: str = videos[0].date_display
    time_kiev_safe: str = _telegram_safe_time(times_text)
    merged_title_text: str = _build_titles_summary(
        videos,
        merged_content,
        merge_attempt,
        use_audit_text=config.telegram_use_audit,
    )
    merged_description_text: str = _build_descriptions_summary(
        videos,
        merged_content,
        merge_attempt,
        use_audit_text=config.telegram_use_audit,
    )
    return _render_template(
        config.templates.telegram_language_merged_block,
        {
            "date_display": date_display,
            "time_kiev": time_kiev_safe,
            "symbol_pin": config.telegram_symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": merged_title_text.strip(),
            "description": (merged_description_text.strip() or _no_description_text()),
        },
    )


def _build_telegram_language_nomerge_block(
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
) -> str:
    if not videos:
        raise ValueError("videos must not be empty for nomerge telegram block")
    times_text: str = ", ".join(
        sorted({video.scheduled_at_kiev.strftime("%H:%M") for video in videos})
    )
    date_display: str = videos[0].date_display
    time_kiev_safe: str = _telegram_safe_time(times_text)
    titles_text: str = _numbered_original_titles(videos).strip()
    descriptions_text: str = _build_descriptions_summary(videos=videos).strip()
    return _render_template(
        config.templates.telegram_language_merged_block,
        {
            "date_display": date_display,
            "time_kiev": time_kiev_safe,
            "symbol_pin": config.telegram_symbol_pin,
            "language_flags": _telegram_language_flags(language, config),
            "title": titles_text or "1) ...",
            "description": descriptions_text or _no_description_text(),
        },
    )


def _build_telegram_language_digest_block(
    language: str,
    videos: List[PlannedVideo],
    context: Dict[str, str],
    config: "AppConfig",
) -> str:
    times_text: str = ", ".join(
        sorted({video.scheduled_at_kiev.strftime("%H:%M") for video in videos})
    )
    digest_header: str = _render_template(
        config.templates.telegram_language_digest_header,
        {
            "language_flags": _telegram_language_flags(language, config),
            "language_name": _telegram_language_name(language, config),
            "date": context["date"],
            "time_kiev": _telegram_safe_time(times_text),
        },
    )
    lines: List[str] = [
        digest_header,
        "",
    ]
    for video in videos:
        lines.append(f"{config.telegram_symbol_done} {video.metadata.title}")
        lines.append(video.normalized_link)
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_telegram_key_form_reminder(
    context: Dict[str, str], config: "AppConfig"
) -> str:
    return _render_template(
        config.templates.telegram_key_form_reminder,
        {"form_url": context["form_url"]},
    )


def _call_stg_model(
    *,
    prompt_text: str,
    model_name: str,
    timeout_seconds: float,
    config: "AppConfig",
    attempt_label: str,
    max_tokens_override: Optional[int] = None,
    force_json: bool = False,
    response_schema: Optional[Any] = None,
    pre_delay_sec: float = 0.0,
) -> str:
    resolved_max_tokens: int = config.gemini_max_output_tokens
    if max_tokens_override is not None and int(max_tokens_override) > 0:
        resolved_max_tokens = int(max_tokens_override)
    return _call_gemini_generate_once(
        prompt_text=prompt_text,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
        attempt_label=attempt_label,
        debug=LOGGER.isEnabledFor(logging.DEBUG),
        temperature_override=config.gemini_temperature,
        max_tokens_override=resolved_max_tokens,
        force_json=force_json,
        response_schema=response_schema,
        pre_delay_sec=pre_delay_sec,
    )


def _language_name_for_merge_prompt(language: str) -> str:
    default_names: Dict[str, str] = {
        "uk": "Ukrainian",
        "en": "English",
        "ru": "Russian",
        "other": "the original language of sources",
    }
    templates: Optional[AppTemplates] = globals().get("_ACTIVE_TEMPLATES")
    if templates is None:
        return default_names.get(language, default_names["other"])
    try:
        payload: Any = json.loads(templates.llm_language_names_json)
        if isinstance(payload, dict):
            return str(
                payload.get(language, payload.get("other", default_names["other"]))
            )
    except Exception:
        pass
    return default_names.get(language, default_names["other"])


def _description_block_length_guidance_for_sources(source_count: int) -> str:
    if source_count in {2, 3}:
        return (
            "- Target total length for the DESCRIPTION block: 3200-3600 characters.\n"
            "- Keep clearly below hard limits."
        )
    if source_count in {4, 5}:
        return (
            "- Hard max for DESCRIPTION block: 4000 characters.\n"
            "- Target total length: 3600-3900 characters."
        )
    return (
        f"- Hard max for DESCRIPTION block: {MERGED_DESCRIPTION_HARD_CEILING} characters.\n"
        "- Target total length: 4000-4400 characters."
    )


def _strip_urls(text: str) -> str:
    without_urls: str = URL_PATTERN.sub("", str(text or ""))
    normalized: str = without_urls.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"\n\s+", "\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _build_llm_merge_prompt_text(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
) -> str:
    if len(videos) < 2:
        raise ValueError("Expected at least 2 videos for merged generation.")

    unique_dates: List[str] = sorted(
        {video.date_display for video in videos if video.date_display}
    )
    date_or_period: str = ", ".join(unique_dates) if unique_dates else "n/a"
    sources_blocks: List[str] = []
    for index, video in enumerate(videos, start=1):
        title_text: str = video.metadata.title.strip()
        source_description: str = (
            video.metadata.description.strip() or _no_description_text()
        )
        if _strip_chapter_timestamps_enabled_from_env():
            source_description = strip_chapter_timestamps(source_description)
        description_text_raw: str = _strip_urls(source_description)
        description_text: str = _truncate_head_tail(
            description_text_raw,
            limit=config.llm_source_desc_max_chars,
        )
        source_url: str = video.normalized_link.strip()
        sources_blocks.append(
            f"VIDEO {index}:\nURL: {source_url}\nTITLE: {title_text}\nDESCRIPTION: {description_text}"
        )

    base_prompt: str = _render_template(
        config.templates.llm_merge_title_description_prompt,
        {
            "language_name": _language_name_for_merge_prompt(language),
            "sources_block": "\n\n".join(sources_blocks),
        },
    )
    strict_prompt: str = (
        "You are a careful editor. You MUST use ONLY facts explicitly present in source content.\n"
        "Do NOT invent names, dates, places, numbers, quotes, or events.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "Line 1: TITLE: <title>\n"
        "Line 2+: DESCRIPTION: <description text; may contain newlines>\n"
        "No other headers. No JSON. No markdown. No code fences.\n\n"
        f"LANGUAGE:\nWrite in: {_language_name_for_merge_prompt(language)}.\n\n"
        "Final title and description MUST be written in the target language block.\n"
        "If some source videos are in other languages, translate/adapt them before composing the final text.\n"
        "Do not mention translation.\n\n"
        "TITLE RULES:\n"
        "- 1-98 characters\n"
        "- no emoji\n"
        "- one line\n"
        "- must reflect combined sources accurately\n"
        "- do NOT include model name\n\n"
        "DESCRIPTION RULES:\n"
        f"{_description_block_length_guidance_for_sources(len(videos))}\n"
        f"- There are exactly {len(videos)} input sources, so DESCRIPTION paragraph block MUST contain exactly {len(videos)} paragraphs.\n"
        "- Paragraph i must summarize only source i, in the same order.\n"
        "- Paragraphs MUST be separated by exactly one blank line (\\n\\n).\n"
        '- No numbering or source labels in paragraphs: no "1)", "2)", "Video 1:", "Описание 2:", "Source:".\n'
        "- No headers, no preamble, no conclusion, no CTA, no hashtags, no meta labels inside the DESCRIPTION paragraph block.\n"
        "- Each paragraph must be 1-3 sentences, informative, neutral newsroom style.\n"
        "- Keep source-specific factual anchors in each paragraph (names, dates, places, organizations, numbers, awards, quotes).\n"
        "- Never move facts from one source paragraph to another.\n"
        "- Avoid long bios; keep facts only and concise.\n"
        "- If close to the limit, compress each paragraph proportionally.\n"
        "FORMAT CONTRACT:\n"
        f"- DESCRIPTION paragraph block must have exactly {len(videos)} paragraphs separated by one blank line.\n"
        "- Example for N=3:\n"
        "  Paragraph 1 text...\n\n"
        "  Paragraph 2 text...\n\n"
        "  Paragraph 3 text...\n"
        "- If you cannot comply, output an empty string.\n"
        "- After these paragraphs output exactly in this order, without labels:\n"
        "  1) CTA line (single line, 1-2 sentences)\n"
        "  2) HASHTAGS line (single line, space-separated hashtags)\n"
        f"  3) {len(videos)} source URL lines (one URL per line, same source order)\n"
        '- Forbidden labels anywhere in output: "CTA:", "HASHTAGS:", "SOURCES:".\n'
        "- Output only TITLE and DESCRIPTION blocks, with no extra commentary.\n\n"
        "FACT SAFETY:\n"
        "- Never merge different people into one.\n"
        "- Never assume relationships unless explicitly stated.\n"
        "- Keep neutral newsroom tone.\n\n"
        f"Date/period (optional): {date_or_period}\n"
        "Target location (optional): n/a\n"
    )
    return f"{base_prompt}\n\n{strict_prompt}"


def _gemini_merge_call_raw(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    attempt_label: str,
    prompt_text_override: Optional[str] = None,
) -> str:
    prompt_text: str = str(
        prompt_text_override or ""
    ).strip() or _build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
    )
    pre_delay_sec: float = max(0.0, float(config.gemini_pre_delay_sec))
    rate_guard_wait_sec: float = _get_gemini_rate_guard().planned_wait_seconds()
    resolved_delay_sec: float = max(pre_delay_sec, rate_guard_wait_sec)
    LOGGER.info(
        "LLM merge: sleeping %.2fs before request ... provider=gemini language=%s sources=%d (pre_delay=%.2fs rate_guard=%.2fs)",
        resolved_delay_sec,
        language,
        len(videos),
        pre_delay_sec,
        rate_guard_wait_sec,
    )
    LOGGER.debug(
        "LLM merge prompt: expected_paragraphs=%d prompt_chars=%d model=%s",
        len(videos),
        len(prompt_text),
        config.gemini_model,
    )
    raw_text: str = _call_stg_model(
        prompt_text=prompt_text,
        model_name=config.gemini_model,
        timeout_seconds=config.gemini_timeout_sec,
        config=config,
        attempt_label=attempt_label,
        max_tokens_override=config.gemini_max_output_tokens,
        force_json=False,
        response_schema=None,
        pre_delay_sec=resolved_delay_sec,
    )
    return raw_text


def _normalize_source_url_for_compare(url: str) -> str:
    cleaned: str = str(url or "").strip()
    if not cleaned:
        return ""
    try:
        return _normalize_youtube_video_url(cleaned)
    except Exception:
        return cleaned.rstrip("/")


def _normalize_source_url_for_validation(url: str) -> str:
    original: str = str(url or "")
    cleaned: str = original.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("<") and cleaned.endswith(">") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    if cleaned.startswith("(") and cleaned.endswith(")") and len(cleaned) >= 2:
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.rstrip(".,;)")
    if cleaned != original:
        LOGGER.debug("Normalized source URL: old=%r new=%r", original, cleaned)
    return cleaned


def _normalize_single_line_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _sanitize_forbidden_section_labels(text: str) -> Tuple[str, bool]:
    raw_text: str = str(text or "")
    sanitized_text: str = re.sub(
        r"(?im)^\s*(cta|hashtags|sources)\s*:\s*",
        "",
        raw_text,
    )
    return (sanitized_text, sanitized_text != raw_text)


def _meta_guard_max_nonempty_lines_from_env() -> int:
    return _load_int_env("STG_META_GUARD_MAX_NONEMPTY_LINES", 8, min_value=1)


def _sheets_link_writeback_enabled_from_env() -> bool:
    return _load_bool_env("STG_SHEETS_LINK_WRITEBACK", True)


def _strip_chapter_timestamps_enabled_from_env() -> bool:
    return _load_bool_env("STG_STRIP_CHAPTER_TIMESTAMPS", True)


def _sheets_link_normalize_report_limit_from_env() -> int:
    return _load_int_env("STG_SHEETS_LINK_NORMALIZE_REPORT_LIMIT", 20, min_value=1)


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


def strip_paragraph_labels(text: str) -> str:
    lines: List[str] = (
        str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    cleaned_lines: List[str] = []
    pattern: re.Pattern[str] = re.compile(
        r"(?i)^\s*(?:video|paragraph|описание|абзац|видео)\s*\d+\s*[:\-]\s*"
    )
    for line in lines:
        cleaned_lines.append(pattern.sub("", line))
    return "\n".join(cleaned_lines).strip()


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


def strip_meta_lines(text: str) -> str:
    lines: List[str] = (
        str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    cleaned_lines: List[str] = []
    meta_pattern: re.Pattern[str] = re.compile(
        r"(?i)^\s*(?:cta(?:\s*line)?|hashtags(?:\s*line)?|preview|title|description|sources?|source|название|описание|прев[ью'’]+)\b"
    )
    max_nonempty_guard_lines: int = _meta_guard_max_nonempty_lines_from_env()
    nonempty_seen: int = 0
    meta_lines_removed: int = 0
    for line in lines:
        normalized_line: str = line.strip()
        if not normalized_line:
            cleaned_lines.append("")
            continue
        nonempty_seen += 1
        in_guard_zone: bool = nonempty_seen <= max_nonempty_guard_lines
        if in_guard_zone and meta_pattern.match(normalized_line):
            meta_lines_removed += 1
            continue
        if in_guard_zone and re.match(
            r"(?i)^https?://(?:www\.)?(?:youtu\.be|youtube\.com)/\S*$", normalized_line
        ):
            meta_lines_removed += 1
            continue
        cleaned_lines.append(line)
    LOGGER.debug(
        "meta_guard_lines=%d meta_lines_removed=%d",
        max_nonempty_guard_lines,
        meta_lines_removed,
    )
    return "\n".join(cleaned_lines).strip()


def validate_description_plain(text: str) -> Tuple[bool, List[str]]:
    normalized: str = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    reasons: List[str] = []
    paragraph_label_pattern: re.Pattern[str] = re.compile(
        r"(?i)^\s*(?:video|paragraph|описание|абзац|видео)\s*\d+\s*[:\-]"
    )
    ordinal_source_artifact_patterns: Tuple[Tuple[str, re.Pattern[str]], ...] = (
        (
            "en_source_paragraph_order",
            re.compile(
                r"(?i)\bthe\s+(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+(?:source|paragraph)\b"
            ),
        ),
        (
            "ru_source_order",
            re.compile(
                r"(?i)\b(?:перв(?:ый|ого)|втор(?:ой|ого)|трет(?:ий|ьего))\s+источник(?:а)?\b"
            ),
        ),
        (
            "ua_source_order",
            re.compile(
                r"(?i)\b(?:перш(?:ий|ого)|друг(?:ий|ого)|трет(?:ій|ього))\s+джерел(?:о|а)\b"
            ),
        ),
        (
            "ru_video_source_order",
            re.compile(
                r"(?i)\bвидеоматериал\w*\s+(?:перв(?:ого|ый)|втор(?:ого|ой)|трет(?:ьего|ий))\s+источник(?:а)?\b"
            ),
        ),
    )
    max_nonempty_guard_lines: int = _meta_guard_max_nonempty_lines_from_env()
    header_token_patterns: Tuple[Tuple[str, re.Pattern[str]], ...] = (
        ("title:", re.compile(r"(?i)\btitle\s*:")),
        ("description:", re.compile(r"(?i)\bdescription\s*:")),
        ("preview:", re.compile(r"(?i)\bpreview\s*:")),
        ("cta:", re.compile(r"(?i)\bcta\s*:")),
        ("hashtags:", re.compile(r"(?i)\bhashtags\s*:")),
        ("source:", re.compile(r"(?i)\bsources?\s*:")),
        ("название:", re.compile(r"(?i)\bназвание\s*:")),
        ("описание:", re.compile(r"(?i)\bописание\s*:")),
        ("превью:", re.compile(r"(?i)\bпрев[ью'’]+\s*:")),
    )
    nonempty_seen: int = 0
    for line_number, line in enumerate(normalized.split("\n"), start=1):
        stripped_line: str = line.strip()
        if not stripped_line:
            continue
        nonempty_seen += 1
        if paragraph_label_pattern.search(stripped_line):
            reasons.append(
                f"line {line_number}: banned label remains ({stripped_line[:80]!r})"
            )
            continue
        artifact_found: bool = False
        for artifact_label, artifact_pattern in ordinal_source_artifact_patterns:
            matched_artifact: Optional[re.Match[str]] = artifact_pattern.search(
                stripped_line
            )
            if matched_artifact is None:
                continue
            reasons.append(
                f'banned_source_order_artifact: "{artifact_label}" token={matched_artifact.group(0)!r} line={line_number}'
            )
            artifact_found = True
            break
        if artifact_found:
            continue
        if nonempty_seen > max_nonempty_guard_lines:
            continue
        for token_label, token_pattern in header_token_patterns:
            if token_pattern.search(stripped_line):
                reasons.append(f'meta_token_in_header: "{token_label}"')
                break
    return (len(reasons) == 0, reasons)


def _clean_and_validate_llm_description(
    *,
    text: str,
) -> Tuple[str, bool, List[str]]:
    cleaned: str = strip_meta_lines(strip_paragraph_labels(text))
    if _strip_chapter_timestamps_enabled_from_env():
        cleaned = strip_chapter_timestamps(cleaned)
    ok, reasons = validate_description_plain(cleaned)
    return (cleaned, ok, reasons)


def _recover_merge_description_paragraphs(
    *,
    provider_name: str,
    merged_text: str,
    expected_paragraphs: int,
) -> List[str]:
    paragraphs, _, _ = _safe_split_paragraph_region_to_expected(
        paragraph_region_text=str(merged_text or ""),
        expected_source_count=expected_paragraphs,
        provider_name=provider_name,
    )
    if len(paragraphs) != expected_paragraphs:
        raise RuntimeError(
            f"{provider_name} paragraph recovery failed: {len(paragraphs)} != {expected_paragraphs}"
        )
    if any(not str(item or "").strip() for item in paragraphs):
        raise RuntimeError(
            f"{provider_name} paragraph recovery produced empty paragraph"
        )
    return [_normalize_single_line_text(item) for item in paragraphs]


def _build_deterministic_group_paragraphs(
    *,
    videos: List[PlannedVideo],
    paragraph_limit: int,
) -> List[str]:
    paragraphs: List[str] = []
    for video in videos:
        source_text: str = _fallback_source_description_text(video)
        source_text = _normalize_single_line_text(_strip_urls(source_text))
        if not source_text:
            source_text = "n/a"
        source_text = _truncate_to_limit(source_text, limit=max(200, paragraph_limit))
        paragraphs.append(source_text)
    return paragraphs


def _enforce_merged_paragraphs_for_group(
    *,
    provider_name: str,
    merged_content: MergedLanguageContent,
    videos: List[PlannedVideo],
    paragraph_limit: int,
) -> MergedLanguageContent:
    expected_paragraphs: int = len(videos)
    if expected_paragraphs <= 1:
        return merged_content
    merged_text: str = str(merged_content.description or "").strip()
    recovered_paragraphs: List[str]
    try:
        recovered_paragraphs = _recover_merge_description_paragraphs(
            provider_name=provider_name,
            merged_text=merged_text,
            expected_paragraphs=expected_paragraphs,
        )
        LOGGER.info(
            "%s paragraph enforcement: expected=%d final=%d source=recovered",
            provider_name,
            expected_paragraphs,
            len(recovered_paragraphs),
        )
    except Exception as error:
        LOGGER.warning(
            "%s paragraph enforcement fallback to deterministic source paragraphs. expected=%d reason=%s",
            provider_name,
            expected_paragraphs,
            _summarize_error(error),
        )
        recovered_paragraphs = _build_deterministic_group_paragraphs(
            videos=videos,
            paragraph_limit=paragraph_limit,
        )

    enforced_description: str = "\n\n".join(recovered_paragraphs).strip()
    cleaned_description, ok, reasons = _clean_and_validate_llm_description(
        text=enforced_description
    )
    if not ok or not cleaned_description:
        fallback_paragraphs: List[str] = _build_deterministic_group_paragraphs(
            videos=videos,
            paragraph_limit=paragraph_limit,
        )
        cleaned_description = "\n\n".join(fallback_paragraphs).strip()
        LOGGER.warning(
            "%s paragraph enforcement validation fallback: reasons=%s",
            provider_name,
            reasons,
        )
    return dataclasses.replace(
        merged_content,
        description=cleaned_description,
        description_selected=cleaned_description,
        description_audit=cleaned_description,
    )


def _truncate_validation_reasons(reasons: List[str], limit: int = 3) -> List[str]:
    cleaned_reasons: List[str] = [
        str(item or "").strip() for item in reasons if str(item or "").strip()
    ]
    return cleaned_reasons[: max(1, int(limit))]


def _is_valid_hashtags_line(line: str) -> bool:
    cleaned_line: str = str(line or "").strip()
    if not cleaned_line:
        return False
    tokens: List[str] = [token for token in re.split(r"\s+", cleaned_line) if token]
    if not tokens:
        return False
    return all(re.match(r"^#[^\s#]+$", token) is not None for token in tokens)


def _normalize_hashtags_line(line: str) -> str:
    tokens: List[str] = [
        token for token in re.split(r"\s+", str(line or "").strip()) if token
    ]
    return " ".join(tokens)


def _hashtags_stopwords_for_language(language: str) -> set[str]:
    lang_code: str = str(language or "").strip().lower()
    stopwords_common: set[str] = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "about",
        "into",
        "after",
        "before",
        "under",
        "over",
        "between",
        "without",
        "your",
        "you",
        "our",
        "are",
        "was",
        "were",
        "will",
        "just",
        "more",
        "most",
        "less",
        "news",
        "live",
        "update",
        "и",
        "в",
        "во",
        "на",
        "по",
        "за",
        "из",
        "к",
        "ко",
        "от",
        "до",
        "для",
        "при",
        "про",
        "что",
        "это",
        "как",
        "или",
        "а",
        "но",
        "мы",
        "вы",
        "они",
        "он",
        "она",
        "оно",
        "у",
        "та",
        "то",
        "це",
        "цей",
        "ця",
        "ці",
        "про",
        "щодо",
        "если",
        "еслибы",
    }
    stopwords_by_language: Dict[str, set[str]] = {
        "en": {
            "in",
            "on",
            "at",
            "of",
            "to",
            "is",
            "be",
            "an",
            "or",
            "as",
            "by",
            "it",
            "its",
            "new",
        },
        "ru": {
            "не",
            "об",
            "мы",
            "вы",
            "их",
            "его",
            "ее",
            "также",
            "только",
            "уже",
            "еще",
            "стрим",
            "обзор",
            "новости",
        },
        "uk": {
            "не",
            "та",
            "як",
            "аби",
            "щоб",
            "вже",
            "ще",
            "також",
            "стрім",
            "огляд",
            "новини",
        },
    }
    return stopwords_common | stopwords_by_language.get(lang_code, set())


def _sanitize_hashtag_token(raw_token: str) -> str:
    token: str = str(raw_token or "").strip().lower()
    token = token.strip("_")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"[^a-zа-яёіїєґ0-9_]+", "", token, flags=re.IGNORECASE)
    return token


def _generate_hashtags_line_deterministic(
    language: str,
    merged_title: str,
    merged_description_paragraphs: List[str],
    source_titles: Optional[List[str]] = None,
) -> str:
    text_parts: List[str] = [str(merged_title or "")]
    text_parts.extend(str(item or "") for item in (merged_description_paragraphs or []))
    text_parts.extend(str(item or "") for item in (source_titles or []))
    combined_text: str = "\n".join(text_parts)
    raw_tokens: List[str] = [
        token
        for token in re.split(
            r"[^A-Za-zА-Яа-яЁёІіЇїЄєҐґ0-9_]+",
            combined_text,
        )
        if token
    ]

    stopwords: set[str] = _hashtags_stopwords_for_language(language)
    selected_tokens: List[str] = []
    seen_tokens: set[str] = set()
    for raw_token in raw_tokens:
        normalized_token: str = _sanitize_hashtag_token(raw_token)
        if not normalized_token:
            continue
        if normalized_token.startswith(("http", "www")):
            continue
        if re.fullmatch(r"_+", normalized_token):
            continue
        if len(normalized_token) < 3:
            continue
        if normalized_token in stopwords:
            continue
        if normalized_token in seen_tokens:
            continue
        seen_tokens.add(normalized_token)
        selected_tokens.append(normalized_token)

    max_hashtags: int = 10
    hashtags_tokens: List[str] = selected_tokens[:max_hashtags]
    lang_code: str = str(language or "").strip().lower()
    default_tokens_by_language: Dict[str, List[str]] = {
        "en": ["news", "live", "update"],
        "ru": ["новости", "стрим", "обзор"],
        "uk": ["новини", "стрім", "огляд"],
    }
    default_tokens: List[str] = default_tokens_by_language.get(
        lang_code, default_tokens_by_language["en"]
    )
    if len(hashtags_tokens) < 3:
        for default_token in default_tokens:
            normalized_default: str = _sanitize_hashtag_token(default_token)
            if not normalized_default:
                continue
            if normalized_default in seen_tokens:
                continue
            seen_tokens.add(normalized_default)
            hashtags_tokens.append(normalized_default)
            if len(hashtags_tokens) >= 3:
                break

    hashtags_line: str = " ".join(
        f"#{token}" for token in hashtags_tokens[:max_hashtags]
    )
    hashtags_line = _normalize_hashtags_line(hashtags_line)
    if not _is_valid_hashtags_line(hashtags_line):
        fallback_line: str = "#news #live #update"
        if lang_code == "ru":
            fallback_line = "#новости #стрим #обзор"
        elif lang_code == "uk":
            fallback_line = "#новини #стрім #огляд"
        hashtags_line = fallback_line
    if not _is_valid_hashtags_line(hashtags_line):
        raise RuntimeError("Local hashtags generation failed validation.")
    return hashtags_line


def _remove_urls_from_paragraph(text: str) -> str:
    without_urls: str = URL_PATTERN.sub("", str(text or ""))
    normalized: str = re.sub(r"\s+", " ", without_urls).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized.strip()


def _remove_cta_style_sentences_from_paragraphs(
    paragraphs: List[str],
) -> Tuple[List[str], bool]:
    cta_patterns: Tuple[str, ...] = (
        "watch",
        "learn more",
        "join",
        "subscribe",
        "links below",
        "for more information",
        "подписывай",
        "підпис",
        "долуч",
    )
    filtered_paragraphs: List[str] = []
    changed: bool = False
    for paragraph_text in paragraphs:
        normalized_paragraph: str = _normalize_single_line_text(paragraph_text)
        sentence_parts: List[str] = re.split(r"(?<=[.!?…])\s+", normalized_paragraph)
        kept_sentences: List[str] = []
        removed_any_sentence: bool = False
        for sentence in sentence_parts:
            sentence_clean: str = sentence.strip()
            if not sentence_clean:
                continue
            sentence_lower: str = sentence_clean.lower()
            if any(pattern in sentence_lower for pattern in cta_patterns):
                removed_any_sentence = True
                continue
            kept_sentences.append(sentence_clean)
        if removed_any_sentence and kept_sentences:
            cleaned_paragraph: str = _normalize_single_line_text(
                " ".join(kept_sentences)
            )
            if cleaned_paragraph != normalized_paragraph:
                changed = True
            filtered_paragraphs.append(cleaned_paragraph)
            continue
        filtered_paragraphs.append(normalized_paragraph)
    return (filtered_paragraphs, changed)


def _split_text_by_positions(text: str, split_positions: List[int]) -> List[str]:
    if not split_positions:
        return [text]
    positions: List[int] = sorted(set(split_positions))
    chunks: List[str] = []
    previous_index: int = 0
    for position in positions:
        if position <= previous_index or position >= len(text):
            continue
        chunks.append(text[previous_index:position])
        previous_index = position
    chunks.append(text[previous_index:])
    return chunks


def _choose_split_positions_near_targets(
    *,
    text_len: int,
    candidate_positions: List[int],
    parts_count: int,
) -> Optional[List[int]]:
    required_splits: int = max(0, parts_count - 1)
    if required_splits == 0:
        return []
    unique_candidates: List[int] = sorted(
        {position for position in candidate_positions if 0 < position < text_len}
    )
    if len(unique_candidates) < required_splits:
        return None
    chosen_positions: List[int] = []
    previous_position: int = 0
    for split_index in range(1, parts_count):
        target_position: float = (text_len * split_index) / parts_count
        remaining_splits: int = required_splits - len(chosen_positions) - 1
        available: List[int] = [
            position
            for position in unique_candidates
            if position > previous_position
            and position <= text_len - max(1, remaining_splits)
        ]
        if not available:
            return None
        best_position: int = min(
            available, key=lambda position: (abs(position - target_position), position)
        )
        chosen_positions.append(best_position)
        previous_position = best_position
    if len(chosen_positions) != required_splits:
        return None
    return chosen_positions


def _merge_extra_paragraphs_to_expected(
    *,
    paragraphs: List[str],
    expected_count: int,
) -> List[str]:
    if expected_count <= 0:
        return []
    merged: List[str] = [
        _normalize_single_line_text(item) for item in paragraphs if item.strip()
    ]
    while len(merged) > expected_count:
        last_index: int = len(merged) - 1
        merged[last_index - 1] = _normalize_single_line_text(
            f"{merged[last_index - 1]} {merged[last_index]}"
        )
        merged.pop(last_index)
    return merged


def _safe_split_paragraph_region_to_expected(
    *,
    paragraph_region_text: str,
    expected_source_count: int,
    provider_name: str,
) -> Tuple[List[str], int, str]:
    normalized_region: str = (
        str(paragraph_region_text or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )
    if not normalized_region:
        raise RuntimeError(f"{provider_name} merge description is empty.")

    blank_split_paragraphs: List[str] = [
        _normalize_single_line_text(item)
        for item in re.split(r"\n\s*\n", normalized_region)
        if str(item).strip()
    ]
    paragraphs_detected: int = len(blank_split_paragraphs)
    if len(blank_split_paragraphs) == expected_source_count:
        return (blank_split_paragraphs, paragraphs_detected, "")
    if len(blank_split_paragraphs) > expected_source_count:
        LOGGER.warning(
            "%s merge description has %d paragraphs; expected %d. Merging extras into previous paragraphs.",
            provider_name,
            len(blank_split_paragraphs),
            expected_source_count,
        )
        merged_tail_paragraphs: List[str] = _merge_extra_paragraphs_to_expected(
            paragraphs=blank_split_paragraphs,
            expected_count=expected_source_count,
        )
        LOGGER.debug(
            "%s paragraph recovery: paragraphs_detected=%d expected=%d recovery_path=merge_tail final_paragraphs=%d",
            provider_name,
            paragraphs_detected,
            expected_source_count,
            len(merged_tail_paragraphs),
        )
        return (merged_tail_paragraphs, paragraphs_detected, "merge_tail")

    single_newline_paragraphs: List[str] = [
        _normalize_single_line_text(line)
        for line in normalized_region.split("\n")
        if line.strip()
    ]
    if len(single_newline_paragraphs) >= expected_source_count:
        LOGGER.warning(
            "%s merge description has %d paragraphs via blank split; rebuilding from single-newline segments to %d.",
            provider_name,
            len(blank_split_paragraphs),
            expected_source_count,
        )
        rebuilt_newline_paragraphs: List[str] = _merge_extra_paragraphs_to_expected(
            paragraphs=single_newline_paragraphs,
            expected_count=expected_source_count,
        )
        LOGGER.debug(
            "%s paragraph recovery: paragraphs_detected=%d expected=%d recovery_path=split_newline final_paragraphs=%d",
            provider_name,
            paragraphs_detected,
            expected_source_count,
            len(rebuilt_newline_paragraphs),
        )
        return (rebuilt_newline_paragraphs, paragraphs_detected, "split_newline")

    text_length: int = len(normalized_region)
    newline_positions: List[int] = [
        match.start() for match in re.finditer(r"\n+", normalized_region)
    ]
    split_positions: Optional[List[int]] = _choose_split_positions_near_targets(
        text_len=text_length,
        candidate_positions=newline_positions,
        parts_count=expected_source_count,
    )
    if split_positions:
        newline_chunks: List[str] = [
            _normalize_single_line_text(chunk)
            for chunk in _split_text_by_positions(normalized_region, split_positions)
            if chunk.strip()
        ]
        if len(newline_chunks) == expected_source_count and all(newline_chunks):
            LOGGER.warning(
                "%s merge description paragraphs recovered via newline-boundary splitting (%d -> %d).",
                provider_name,
                len(blank_split_paragraphs),
                expected_source_count,
            )
            LOGGER.debug(
                "%s paragraph recovery: paragraphs_detected=%d expected=%d recovery_path=split_newline final_paragraphs=%d",
                provider_name,
                paragraphs_detected,
                expected_source_count,
                len(newline_chunks),
            )
            return (newline_chunks, paragraphs_detected, "split_newline")

    sentence_positions: List[int] = [
        match.end()
        for match in re.finditer(r"[.!?…]+(?:[\"'»”)\]]+)?\s+", normalized_region)
    ]
    split_positions = _choose_split_positions_near_targets(
        text_len=text_length,
        candidate_positions=sentence_positions,
        parts_count=expected_source_count,
    )
    if split_positions:
        sentence_chunks: List[str] = [
            _normalize_single_line_text(chunk)
            for chunk in _split_text_by_positions(normalized_region, split_positions)
            if chunk.strip()
        ]
        if len(sentence_chunks) == expected_source_count and all(sentence_chunks):
            LOGGER.warning(
                "%s merge description paragraphs recovered via sentence-boundary splitting (%d -> %d).",
                provider_name,
                len(blank_split_paragraphs),
                expected_source_count,
            )
            LOGGER.debug(
                "%s paragraph recovery: paragraphs_detected=%d expected=%d recovery_path=split_sentence final_paragraphs=%d",
                provider_name,
                paragraphs_detected,
                expected_source_count,
                len(sentence_chunks),
            )
            return (sentence_chunks, paragraphs_detected, "split_sentence")

    hard_split_positions: List[int] = []
    for split_index in range(1, expected_source_count):
        position: int = int(round((text_length * split_index) / expected_source_count))
        position = max(1, min(text_length - 1, position))
        hard_split_positions.append(position)
    hard_chunks: List[str] = [
        _normalize_single_line_text(chunk)
        for chunk in _split_text_by_positions(normalized_region, hard_split_positions)
        if chunk.strip()
    ]
    if len(hard_chunks) == expected_source_count and all(hard_chunks):
        LOGGER.warning(
            "%s merge description paragraphs recovered via hard even split (%d -> %d).",
            provider_name,
            len(blank_split_paragraphs),
            expected_source_count,
        )
        LOGGER.debug(
            "%s paragraph recovery: paragraphs_detected=%d expected=%d recovery_path=split_hard final_paragraphs=%d",
            provider_name,
            paragraphs_detected,
            expected_source_count,
            len(hard_chunks),
        )
        return (hard_chunks, paragraphs_detected, "split_hard")

    raise RuntimeError(
        f"{provider_name} merge description has {paragraphs_detected} paragraphs; expected {expected_source_count}."
    )


def _parse_structured_merged_description_or_raise(
    *,
    provider_name: str,
    description_text: str,
    expected_source_count: int,
    expected_source_urls: Optional[List[str]] = None,
) -> Tuple[List[str], str, str, List[str]]:
    if expected_source_count <= 0:
        raise RuntimeError("expected_source_count must be >= 1 for merge parsing.")
    normalized: str = (
        str(description_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    if not normalized:
        raise RuntimeError(f"{provider_name} merge description is empty.")

    lines: List[str] = normalized.split("\n")
    indexed_non_empty: List[Tuple[int, str]] = [
        (line_index, line.strip())
        for line_index, line in enumerate(lines)
        if line.strip()
    ]
    required_non_empty: int = expected_source_count + 2
    if len(indexed_non_empty) < required_non_empty:
        raise RuntimeError(
            f"{provider_name} merge output has too few non-empty lines: {len(indexed_non_empty)} < {required_non_empty}."
        )

    source_pairs: List[Tuple[int, str]] = indexed_non_empty[-expected_source_count:]
    hashtags_pair: Tuple[int, str] = indexed_non_empty[-(expected_source_count + 1)]
    cta_pair: Tuple[int, str] = indexed_non_empty[-(expected_source_count + 2)]

    cta_line: str = _normalize_single_line_text(cta_pair[1])
    if not cta_line:
        raise RuntimeError(f"{provider_name} merge CTA line is empty.")
    hashtags_line: str = str(hashtags_pair[1] or "").strip()

    paragraph_region_text: str = "\n".join(lines[: cta_pair[0]]).strip()
    paragraphs, paragraphs_detected, recovery_path = (
        _safe_split_paragraph_region_to_expected(
            paragraph_region_text=paragraph_region_text,
            expected_source_count=expected_source_count,
            provider_name=provider_name,
        )
    )
    if recovery_path:
        LOGGER.debug(
            "%s paragraph recovery summary: paragraphs_detected=%d expected=%d recovery_path=%s final_paragraphs=%d",
            provider_name,
            paragraphs_detected,
            expected_source_count,
            recovery_path,
            len(paragraphs),
        )
    if any(not paragraph for paragraph in paragraphs):
        raise RuntimeError(
            f"{provider_name} merge description contains empty paragraph."
        )

    parsed_source_urls: List[str] = []
    for source_index, source_pair in enumerate(source_pairs, start=1):
        candidate_url: str = _normalize_source_url_for_validation(source_pair[1])
        if not _SOURCE_URL_LINE_RE.match(candidate_url):
            raise RuntimeError(
                f"{provider_name} merge source URL {source_index} is invalid: {candidate_url!r}."
            )
        parsed_source_urls.append(candidate_url)

    if cta_pair[0] >= hashtags_pair[0]:
        raise RuntimeError(
            f"{provider_name} merge output order is invalid: CTA/HASHTAGS."
        )
    if hashtags_pair[0] >= source_pairs[0][0]:
        raise RuntimeError(
            f"{provider_name} merge output order is invalid: HASHTAGS/SOURCES."
        )
    if any(line.strip() for line in lines[source_pairs[-1][0] + 1 :]):
        raise RuntimeError(
            f"{provider_name} merge output has extra non-empty lines after {expected_source_count} source URLs."
        )

    canonical_source_urls: List[str] = [item.strip() for item in parsed_source_urls]
    expected_urls_clean: List[str] = [
        str(item).strip() for item in (expected_source_urls or []) if str(item).strip()
    ]
    if expected_urls_clean:
        if len(expected_urls_clean) != expected_source_count:
            raise RuntimeError(
                f"{provider_name} expected source URLs count mismatch: {len(expected_urls_clean)} vs {expected_source_count}."
            )
        for source_index in range(expected_source_count):
            expected_url: str = _normalize_source_url_for_compare(
                expected_urls_clean[source_index]
            )
            actual_url: str = _normalize_source_url_for_compare(
                parsed_source_urls[source_index]
            )
            if expected_url != actual_url:
                raise RuntimeError(
                    f"{provider_name} merge source URL order mismatch at position {source_index + 1}."
                )
        canonical_source_urls = expected_urls_clean

    paragraph_block_text: str = "\n".join(paragraphs)
    urls_inside_description: List[str] = [
        match.group(0).rstrip(".,;:!?)\"'")
        for match in re.finditer(r"https?://\S+", paragraph_block_text)
    ]
    if urls_inside_description:
        LOGGER.warning(
            "%s merge description still contains URL-like tokens inside paragraphs: count=%d sample=%s",
            provider_name,
            len(urls_inside_description),
            _compact_single_line(urls_inside_description[0], max_len=120),
        )

    return (paragraphs, cta_line, hashtags_line, canonical_source_urls)


def _format_structured_merged_description(
    *,
    paragraphs: List[str],
    cta_line: str,
    hashtags_line: str,
    source_urls: List[str],
) -> str:
    paragraphs_block: str = "\n\n".join(
        item.strip() for item in paragraphs if item.strip()
    ).strip()
    sources_block: str = "\n".join(item.strip() for item in source_urls if item.strip())
    return f"{paragraphs_block}\n{cta_line.strip()}\n{hashtags_line.strip()}\n{sources_block}".strip()


def _auto_trim_structured_description_to_ceiling(
    *,
    paragraphs: List[str],
    cta_line: str,
    hashtags_line: str,
    source_urls: List[str],
    ceiling: int,
) -> Tuple[List[str], str, str, List[str], bool]:
    trimmed_paragraphs: List[str] = [
        _normalize_single_line_text(item) for item in paragraphs
    ]
    trimmed_cta: str = _normalize_single_line_text(cta_line)
    trimmed_hashtags: str = _normalize_hashtags_line(hashtags_line)
    trimmed_sources: List[str] = [
        str(item).strip() for item in source_urls if str(item).strip()
    ]

    def _render_length() -> int:
        return len(
            _format_structured_merged_description(
                paragraphs=trimmed_paragraphs,
                cta_line=trimmed_cta,
                hashtags_line=trimmed_hashtags,
                source_urls=trimmed_sources,
            )
        )

    before_length: int = _render_length()
    if before_length <= ceiling:
        return (
            trimmed_paragraphs,
            trimmed_cta,
            trimmed_hashtags,
            trimmed_sources,
            False,
        )
    hashtags_tokens_before: int = len(
        [token for token in trimmed_hashtags.split() if token]
    )

    while _render_length() > ceiling:
        overflow: int = _render_length() - ceiling
        reducible_by_paragraphs: List[int] = [
            max(0, len(item) - 1) for item in trimmed_paragraphs
        ]
        total_reducible_paragraphs: int = sum(reducible_by_paragraphs)
        if total_reducible_paragraphs > 0:
            remaining_overflow: int = overflow
            for paragraph_index, paragraph_text in enumerate(trimmed_paragraphs):
                max_reduce: int = reducible_by_paragraphs[paragraph_index]
                if max_reduce <= 0:
                    continue
                proportional_reduce: int = int(
                    round((overflow * max_reduce) / max(1, total_reducible_paragraphs))
                )
                if proportional_reduce <= 0 and remaining_overflow > 0:
                    proportional_reduce = 1
                actual_reduce: int = min(
                    max_reduce, proportional_reduce, remaining_overflow
                )
                if actual_reduce <= 0:
                    continue
                new_limit: int = max(1, len(paragraph_text) - actual_reduce)
                trimmed_paragraphs[paragraph_index] = _truncate_to_limit(
                    paragraph_text, limit=new_limit
                )
                remaining_overflow = max(0, remaining_overflow - actual_reduce)
            continue

        if len(trimmed_cta) > 1:
            overflow = _render_length() - ceiling
            cta_new_limit: int = max(1, len(trimmed_cta) - overflow)
            trimmed_cta = _truncate_to_limit(trimmed_cta, limit=cta_new_limit)
            continue
        hashtag_tokens: List[str] = [
            token for token in trimmed_hashtags.split() if token
        ]
        if len(hashtag_tokens) > 1:
            hashtag_tokens.pop()
            trimmed_hashtags = " ".join(hashtag_tokens).strip()
            continue
        break

    after_length: int = _render_length()
    was_trimmed: bool = after_length < before_length
    hashtags_tokens_after: int = len(
        [token for token in trimmed_hashtags.split() if token]
    )
    if hashtags_tokens_after < hashtags_tokens_before:
        LOGGER.debug(
            "openai merge auto-trim: hashtags trimmed tokens_before=%d tokens_after=%d",
            hashtags_tokens_before,
            hashtags_tokens_after,
        )
    if was_trimmed:
        LOGGER.info(
            "merge auto-trim applied: before=%d after=%d ceiling=%d",
            before_length,
            after_length,
            ceiling,
        )
    return (
        trimmed_paragraphs,
        trimmed_cta,
        trimmed_hashtags,
        trimmed_sources,
        was_trimmed,
    )


def _self_check_merged_output_format_or_raise(
    *,
    description_text: str,
    expected_source_count: int,
) -> None:
    normalized: str = (
        str(description_text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    )
    non_empty_lines: List[str] = [
        line.strip() for line in normalized.split("\n") if line.strip()
    ]
    if len(non_empty_lines) < expected_source_count + 2:
        raise RuntimeError(
            f"Self-check failed: expected at least {expected_source_count + 2} non-empty lines."
        )
    tail_urls: List[str] = non_empty_lines[-expected_source_count:]
    paragraphs, _, hashtags_line, parsed_sources = (
        _parse_structured_merged_description_or_raise(
            provider_name="self-check",
            description_text=normalized,
            expected_source_count=expected_source_count,
            expected_source_urls=tail_urls,
        )
    )
    if len(paragraphs) != expected_source_count:
        raise RuntimeError(
            f"Self-check failed: paragraph count {len(paragraphs)} != {expected_source_count}."
        )
    if len(parsed_sources) != expected_source_count:
        raise RuntimeError(
            f"Self-check failed: sources count {len(parsed_sources)} != {expected_source_count}."
        )
    if not _is_valid_hashtags_line(hashtags_line):
        raise RuntimeError("Self-check failed: invalid hashtags line.")


def _debug_run_merge_format_self_test() -> None:
    case1_description: str = (
        "Paragraph one source A facts.\n"
        "Paragraph two source B facts.\n"
        "Paragraph three source C facts.\n"
        "Follow for updates\n"
        "#Tag1 #Tag2 #Tag3\n"
        "https://youtu.be/AAAAAAAAAAA\n"
        "https://youtu.be/BBBBBBBBBBB\n"
        "https://youtu.be/CCCCCCCCCCC"
    )
    case1_paragraphs, case1_cta, case1_hashtags, case1_sources = (
        _parse_structured_merged_description_or_raise(
            provider_name="debug-self-test-case1",
            description_text=case1_description,
            expected_source_count=3,
            expected_source_urls=[
                "https://youtu.be/AAAAAAAAAAA",
                "https://youtu.be/BBBBBBBBBBB",
                "https://youtu.be/CCCCCCCCCCC",
            ],
        )
    )
    assert len(case1_paragraphs) == 3
    assert _is_valid_hashtags_line(case1_hashtags)
    case1_rendered: str = _format_structured_merged_description(
        paragraphs=case1_paragraphs,
        cta_line=case1_cta,
        hashtags_line=case1_hashtags,
        source_urls=case1_sources,
    )
    case1_rendered_sanitized, _ = _sanitize_forbidden_section_labels(case1_rendered)
    _self_check_merged_output_format_or_raise(
        description_text=case1_rendered_sanitized,
        expected_source_count=3,
    )

    case2_description: str = (
        "Alpha source one has long factual text without line breaks and without punctuation "
        "beta source two has another long factual text without line breaks and without punctuation "
        "gamma source three has additional long factual text without line breaks and without punctuation\n"
        "Watch now\n"
        "#TagA #TagB #TagC\n"
        "https://youtu.be/DDDDDDDDDDD\n"
        "https://youtu.be/EEEEEEEEEEE\n"
        "https://youtu.be/FFFFFFFFFFF"
    )
    case2_paragraphs, case2_cta, case2_hashtags, case2_sources = (
        _parse_structured_merged_description_or_raise(
            provider_name="debug-self-test-case2",
            description_text=case2_description,
            expected_source_count=3,
            expected_source_urls=[
                "https://youtu.be/DDDDDDDDDDD",
                "https://youtu.be/EEEEEEEEEEE",
                "https://youtu.be/FFFFFFFFFFF",
            ],
        )
    )
    assert len(case2_paragraphs) == 3
    assert _is_valid_hashtags_line(case2_hashtags)
    case2_rendered: str = _format_structured_merged_description(
        paragraphs=case2_paragraphs,
        cta_line=case2_cta,
        hashtags_line=case2_hashtags,
        source_urls=case2_sources,
    )
    case2_rendered_sanitized, _ = _sanitize_forbidden_section_labels(case2_rendered)
    _self_check_merged_output_format_or_raise(
        description_text=case2_rendered_sanitized,
        expected_source_count=3,
    )

    case3_hashtags_line: str = "   #One   #Two    #Three   "
    assert _is_valid_hashtags_line(case3_hashtags_line)
    assert _normalize_hashtags_line(case3_hashtags_line) == "#One #Two #Three"

    case4_raw_output: str = (
        "TITLE: Valid merged title\n"
        "DESCRIPTION: Fact paragraph one for source one.\n"
        "Fact paragraph two for source two.\n"
        "Call to action text.\n"
        "Підпишіться #One #Two\n"
        "https://youtu.be/GGGGGGGGGGG\n"
        "https://youtu.be/HHHHHHHHHHH"
    )
    case4_merged: MergedLanguageContent = _parse_llm_merge_raw_or_raise(
        provider_name="debug-self-test-case4",
        model_name="debug-model",
        language="uk",
        raw_text=case4_raw_output,
        source_urls=[
            "https://youtu.be/GGGGGGGGGGG",
            "https://youtu.be/HHHHHHHHHHH",
        ],
        source_titles=[
            "Source title one",
            "Source title two",
        ],
    )
    case4_cleaned, case4_ok, case4_reasons = _clean_and_validate_llm_description(
        text=case4_merged.description
    )
    assert case4_ok, case4_reasons
    assert case4_cleaned == case4_merged.description


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
    assert [_planned_video_block_language(item) for item in unchanged] == ["en"]

    cloned: List[PlannedVideo] = [base_video] + [
        dataclasses.replace(base_video, forced_block_language=language_code)
        for language_code in parse_merge_languages("ua,ru")
    ]
    assert len(cloned) == 3
    assert [_planned_video_block_language(item) for item in cloned] == [
        "en",
        "uk",
        "ru",
    ]

    base_block_lang: str = _planned_video_block_language(base_video)
    clones_with_skip: List[PlannedVideo] = []
    for merge_language in parse_merge_languages("uk"):
        if merge_language == base_block_lang:
            continue
        clones_with_skip.append(
            dataclasses.replace(base_video, forced_block_language=merge_language)
        )
    assert len(clones_with_skip) == 1
    assert [_planned_video_block_language(item) for item in clones_with_skip] == ["uk"]

    uk_base_video: PlannedVideo = dataclasses.replace(base_video, language="uk")
    uk_base_block_lang: str = _planned_video_block_language(uk_base_video)
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
    base_block_lang: str = _planned_video_block_language(base_video)
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
    assert [_planned_video_block_language(item) for item in override_items] == [
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
    assert [_planned_video_block_language(item) for item in add_items] == [
        "en",
        "uk",
        "ru",
    ]

    duplicate_row_video: PlannedVideo = dataclasses.replace(base_video, row_number=8)
    deduped_items: List[PlannedVideo] = (
        _deduplicate_planned_videos_within_date_language(
            [base_video, duplicate_row_video]
        )
    )
    assert len(deduped_items) == 1
    assert deduped_items[0].row_number == 2


def _debug_run_plain_description_self_test() -> None:
    labels_sample: str = "Video 1: Hello\nОписание 2: Привет"
    labels_cleaned: str = strip_paragraph_labels(labels_sample)
    assert "Video 1:" not in labels_cleaned
    assert "Описание 2:" not in labels_cleaned

    previous_meta_guard: Optional[str] = os.getenv("STG_META_GUARD_MAX_NONEMPTY_LINES")
    try:
        os.environ["STG_META_GUARD_MAX_NONEMPTY_LINES"] = "2"
        meta_sample: str = (
            "Title: Header line\n"
            "Description: Header line two\n"
            "Quote block starts\n"
            'He said: "Title: keep this in the middle."\n'
            "Tail paragraph"
        )
        meta_cleaned: str = strip_meta_lines(meta_sample)
        assert "Title: Header line" not in meta_cleaned
        assert "Description: Header line two" not in meta_cleaned
        assert 'He said: "Title: keep this in the middle."' in meta_cleaned

        valid_with_middle_title, valid_reasons = validate_description_plain(
            "First paragraph\nSecond paragraph\nQuote: Title: should stay"
        )
        assert valid_with_middle_title, valid_reasons

        invalid_header, invalid_header_reasons = validate_description_plain(
            "Title: must be removed from header\nBody paragraph"
        )
        assert not invalid_header
        assert any(
            'meta_token_in_header: "title:"' == reason
            for reason in invalid_header_reasons
        )
    finally:
        if previous_meta_guard is None:
            os.environ.pop("STG_META_GUARD_MAX_NONEMPTY_LINES", None)
        else:
            os.environ["STG_META_GUARD_MAX_NONEMPTY_LINES"] = previous_meta_guard

    merge_languages: List[str] = ["uk", "ru"]
    base_language: str = "en"
    routed_languages: List[str] = []
    if merge_languages:
        routed_languages = list(merge_languages)
    else:
        routed_languages = [base_language]
    assert "en" not in routed_languages

    class _SheetsStub:
        def __init__(self) -> None:
            self.write_calls: int = 0

        def update_cell_string(
            self,
            *,
            spreadsheet_id: str,
            cell_a1: str,
            value: str,
        ) -> None:
            self.write_calls += 1

    sheets_stub = _SheetsStub()
    _handle_normalized_link_writeback(
        sheets_client=cast(GoogleSheetsClient, sheets_stub),
        spreadsheet_id="sheet-id",
        sheet_name_for_writeback="Sheet1",
        row_number=10,
        links_column_index=0,
        old_link="https://youtube.com/watch?v=ABCDEFGHIJK",
        normalized_link="https://youtu.be/ABCDEFGHIJK",
        writeback_enabled=False,
    )
    assert sheets_stub.write_calls == 0
    _handle_normalized_link_writeback(
        sheets_client=cast(GoogleSheetsClient, sheets_stub),
        spreadsheet_id="sheet-id",
        sheet_name_for_writeback="Sheet1",
        row_number=10,
        links_column_index=0,
        old_link="https://youtube.com/watch?v=ABCDEFGHIJK",
        normalized_link="https://youtu.be/ABCDEFGHIJK",
        writeback_enabled=True,
    )
    assert sheets_stub.write_calls == 1

    assert _extract_youtube_video_id("prefixABCDEFGHIJKsuffix") is None
    assert _extract_youtube_video_id(" ABCDEFGHIJK ") == "ABCDEFGHIJK"
    assert (
        _extract_youtube_video_id("https://youtube.com/watch?v=ZYXWVUTSRQP")
        == "ZYXWVUTSRQP"
    )


def _build_llm_merge_repair_prompt_text(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    previous_output: str,
    parse_error: str,
) -> str:
    base_prompt: str = _build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
    )
    return (
        f"{base_prompt}\n\n"
        "Your previous output was invalid. Rewrite the full answer from scratch.\n"
        "Do not explain errors. Return only TITLE and DESCRIPTION blocks.\n"
        "Keep one DESCRIPTION paragraph per source, in source order.\n\n"
        f"Validation error: {parse_error}\n\n"
        "Previous invalid output:\n"
        f"{previous_output.strip()}"
    ).strip()


def _build_plain_description_repair_prompt_text(
    *,
    target_language: str,
    invalid_text: str,
) -> str:
    return (
        "Remove any labels/headings/meta lines. "
        f"Return ONLY plain paragraph text in {_language_name_for_merge_prompt(target_language)}.\n"
        "No TITLE/DESCRIPTION/CTA/HASHTAGS/PREVIEW labels.\n"
        "No markdown. No JSON. No commentary.\n\n"
        "Text:\n"
        f"{str(invalid_text or '').strip()}"
    ).strip()


def _build_single_source_translate_prompt_text(
    *,
    source_language: str,
    target_language: str,
    source_description: str,
) -> str:
    return (
        "You are a precise editor and translator.\n"
        f"Source language: {_language_name_for_merge_prompt(source_language)}.\n"
        f"Target language: {_language_name_for_merge_prompt(target_language)}.\n"
        "Task: translate and lightly rewrite for readability while preserving facts.\n"
        "Return ONLY plain paragraph text in target language.\n"
        "No labels/headings/meta lines.\n"
        "No TITLE/DESCRIPTION/CTA/HASHTAGS/PREVIEW.\n"
        "No markdown. No JSON.\n\n"
        "Source description:\n"
        f"{str(source_description or '').strip()}"
    ).strip()


def _build_hashtags_repair_prompt_text(
    *,
    invalid_hashtags_line: str,
) -> str:
    return (
        "You are a strict hashtag formatter.\n"
        "Return exactly one line with only space-separated hashtags.\n"
        "Rules:\n"
        "1) Keep existing hashtag words if possible.\n"
        "2) Remove all non-hashtag tokens.\n"
        '3) Do not output labels like "HASHTAGS:".\n'
        "4) No extra commentary.\n\n"
        "Invalid hashtags line:\n"
        f"{invalid_hashtags_line.strip()}"
    ).strip()


def _extract_hashtags_line_from_repair_output(raw_text: str) -> str:
    cleaned: str = _strip_json_code_fences(str(raw_text or ""))
    for line in cleaned.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        candidate: str = line.strip()
        if not candidate:
            continue
        if re.match(r"(?i)^hashtags\s*:", candidate):
            candidate = candidate.split(":", 1)[1].strip()
        candidate = candidate.strip("\"'` ")
        return candidate
    return ""


def _extract_valid_title_from_raw_or_none(raw_text: str) -> Optional[str]:
    text: str = _strip_json_code_fences(str(raw_text or ""))
    if not text.strip():
        return None
    candidate_titles: List[str] = []
    payload: Optional[Dict[str, str]] = _parse_merge_payload_plaintext_tolerant(text)
    if payload is not None:
        payload_title: str = str(payload.get("merged_title") or "").strip()
        if payload_title:
            candidate_titles.append(payload_title)
    lines: List[str] = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for line_index, line in enumerate(lines):
        title_match: Optional[re.Match[str]] = re.match(
            r"(?i)^\s*title\s*:\s*(.*)$", line
        )
        if not title_match:
            continue
        title_value: str = title_match.group(1).strip()
        if not title_value:
            for tail_line in lines[line_index + 1 :]:
                candidate: str = tail_line.strip()
                if candidate:
                    title_value = candidate
                    break
        if title_value:
            candidate_titles.append(title_value)
        break
    for candidate_title in candidate_titles:
        try:
            sanitized_title: str = _sanitize_title(
                re.sub(r"[.!?…]{2,}$", "", candidate_title).strip(),
                min_chars=1,
                max_chars=98,
                allow_emoji=False,
            )
            if sanitized_title.strip("\"'«»` ").strip():
                return sanitized_title
        except Exception:
            continue
    return None


def _parse_llm_merge_raw_or_raise(
    *,
    provider_name: str,
    model_name: str,
    language: str,
    raw_text: str,
    source_urls: List[str],
    source_titles: Optional[List[str]] = None,
    hashtags_repair_once: Optional[Callable[[str], str]] = None,
) -> MergedLanguageContent:
    payload: Optional[Dict[str, str]] = _parse_merge_payload_plaintext_tolerant(
        raw_text
    )
    if payload is None:
        raise RuntimeError(
            f"{provider_name} merge output is not parseable as TITLE/DESCRIPTION text."
        )

    merged_title: str = str(payload.get("merged_title") or "").strip()
    merged_description: str = str(payload.get("merged_description") or "").strip()
    merged_description, labels_sanitized = _sanitize_forbidden_section_labels(
        merged_description
    )
    if labels_sanitized:
        LOGGER.warning(
            "%s merge output contained forbidden labels CTA/HASHTAGS/SOURCES; sanitized before parsing.",
            provider_name,
        )
    merged_title = _sanitize_title(
        merged_title,
        min_chars=1,
        max_chars=98,
        allow_emoji=False,
    )
    merged_title = re.sub(r"[.!?…]{2,}$", "", merged_title).strip()
    merged_title = _sanitize_title(
        merged_title,
        min_chars=1,
        max_chars=98,
        allow_emoji=False,
    )
    merged_title_unquoted: str = merged_title.strip("\"'«»` ").strip()
    if not merged_title_unquoted:
        raise RuntimeError(
            f"{provider_name} merge title is invalid (quotes-only or empty)."
        )
    source_urls_clean: List[str] = [
        str(item).strip() for item in (source_urls or []) if str(item).strip()
    ]
    if not source_urls_clean:
        raise RuntimeError(f"{provider_name} merge source URLs are missing.")
    (
        description_paragraphs,
        cta_line,
        hashtags_line,
        description_source_urls,
    ) = _parse_structured_merged_description_or_raise(
        provider_name=provider_name,
        description_text=merged_description,
        expected_source_count=len(source_urls_clean),
        expected_source_urls=source_urls_clean,
    )
    removed_url_tokens_count: int = 0
    cleaned_paragraphs: List[str] = []
    for paragraph_text in description_paragraphs:
        paragraph_before: str = _normalize_single_line_text(paragraph_text)
        url_tokens_before: int = len(URL_PATTERN.findall(paragraph_before))
        paragraph_after: str = _remove_urls_from_paragraph(paragraph_before)
        if not paragraph_after:
            paragraph_after = "n/a"
        url_tokens_after: int = len(URL_PATTERN.findall(paragraph_after))
        if url_tokens_before > url_tokens_after:
            removed_url_tokens_count += url_tokens_before - url_tokens_after
        cleaned_paragraphs.append(paragraph_after)
    if removed_url_tokens_count > 0:
        LOGGER.warning(
            "%s merge: removed URL-like tokens inside paragraphs locally: removed_count=%d",
            provider_name,
            removed_url_tokens_count,
        )
    description_paragraphs = cleaned_paragraphs
    description_paragraphs, cta_style_cleaned = (
        _remove_cta_style_sentences_from_paragraphs(description_paragraphs)
    )
    if cta_style_cleaned:
        LOGGER.warning(
            "%s merge description paragraphs contained CTA-style language; cleaned automatically.",
            provider_name,
        )
    if not _is_valid_hashtags_line(hashtags_line):
        LOGGER.warning(
            "%s merge hashtags line invalid or missing; generating hashtags locally.",
            provider_name,
        )
        hashtags_line = _generate_hashtags_line_deterministic(
            language=language,
            merged_title=merged_title,
            merged_description_paragraphs=description_paragraphs,
            source_titles=source_titles,
        )
    hashtags_line = _normalize_hashtags_line(hashtags_line)
    (
        description_paragraphs,
        cta_line,
        hashtags_line,
        description_source_urls,
        _,
    ) = _auto_trim_structured_description_to_ceiling(
        paragraphs=description_paragraphs,
        cta_line=cta_line,
        hashtags_line=hashtags_line,
        source_urls=description_source_urls,
        ceiling=MERGED_DESCRIPTION_HARD_CEILING,
    )
    merged_description_structured: str = _format_structured_merged_description(
        paragraphs=description_paragraphs,
        cta_line=cta_line,
        hashtags_line=hashtags_line,
        source_urls=description_source_urls,
    )
    merged_description_for_check, _ = _sanitize_forbidden_section_labels(
        merged_description_structured
    )
    _self_check_merged_output_format_or_raise(
        description_text=merged_description_for_check,
        expected_source_count=len(source_urls_clean),
    )
    plain_description_text: str = "\n\n".join(description_paragraphs).strip()
    cleaned_plain_description, plain_ok, plain_reasons = (
        _clean_and_validate_llm_description(text=plain_description_text)
    )
    if not plain_ok:
        LOGGER.debug(
            "%s merge plain description validation failed reasons=%s",
            provider_name,
            plain_reasons,
        )
        reason_text: str = "; ".join(plain_reasons) or "unknown validation failure"
        raise RuntimeError(
            f"{provider_name} merge plain description validation failed: {reason_text}"
        )
    if not cleaned_plain_description:
        raise RuntimeError(f"{provider_name} merge description is empty.")
    if len(cleaned_plain_description) > MERGED_DESCRIPTION_HARD_CEILING:
        raise RuntimeError(
            f"{provider_name} merge description exceeds {MERGED_DESCRIPTION_HARD_CEILING} chars."
        )

    return MergedLanguageContent(
        title=merged_title,
        description=cleaned_plain_description,
        title_selected=merged_title,
        description_selected=cleaned_plain_description,
        title_audit=merged_title,
        description_audit=cleaned_plain_description,
        llm_model=model_name,
    )


def _extract_openai_response_text(response: Any) -> str:
    output_text: Any = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(response, dict):
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

    output_items: Any = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    if not isinstance(output_items, list):
        return ""
    chunks: List[str] = []
    for output_item in output_items:
        item_text: Any = None
        item_type: str = ""
        if isinstance(output_item, dict):
            item_type = str(output_item.get("type", "")).strip().lower()
            if item_type in {"output_text", "text"}:
                item_text = output_item.get("text")
        else:
            item_type = str(getattr(output_item, "type", "")).strip().lower()
            if item_type in {"output_text", "text"}:
                item_text = getattr(output_item, "text", None)
        if isinstance(item_text, str) and item_text.strip():
            chunks.append(item_text.strip())

        content_items: Any = None
        if isinstance(output_item, dict):
            content_items = output_item.get("content")
        else:
            content_items = getattr(output_item, "content", None)
        if not isinstance(content_items, list):
            continue
        for content_item in content_items:
            text_value: Any = None
            if isinstance(content_item, dict):
                if content_item.get("type") in {"output_text", "text"}:
                    text_value = content_item.get("text")
            else:
                item_type: str = str(getattr(content_item, "type", "")).strip().lower()
                if item_type in {"output_text", "text"}:
                    text_value = getattr(content_item, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                chunks.append(text_value.strip())
    return "\n".join(chunks).strip()


def _debug_openai_output_shape(response: Any) -> str:
    response_type: str = type(response).__name__
    output_text_value: Any = getattr(response, "output_text", None)
    output_text_len: int = (
        len(output_text_value.strip()) if isinstance(output_text_value, str) else 0
    )
    output_text_repr: str = (
        repr(output_text_value[:200]) if isinstance(output_text_value, str) else "None"
    )
    output_items: Any = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    output_len: int = len(output_items) if isinstance(output_items, list) else 0

    output_types: List[str] = []
    if isinstance(output_items, list):
        for item in output_items[:6]:
            if isinstance(item, dict):
                item_type: str = str(item.get("type", "")).strip() or "?"
                role_value: str = str(item.get("role", "")).strip()
            else:
                item_type = str(getattr(item, "type", "")).strip() or "?"
                role_value = str(getattr(item, "role", "")).strip()
            if role_value:
                output_types.append(f"{item_type}/{role_value}")
            else:
                output_types.append(item_type)

    available_attrs: str = ""
    if not isinstance(response, dict):
        attrs: List[str] = []
        for name in (
            "id",
            "model",
            "status",
            "output_text",
            "output",
            "usage",
            "error",
        ):
            if hasattr(response, name):
                attrs.append(name)
        available_attrs = ",".join(attrs)
    response_preview: str = _compact_single_line(str(response), max_len=500)

    return (
        f"type={response_type} output_text_len={output_text_len} "
        f"output_text_repr={output_text_repr} output_items_len={output_len} "
        f"output_item_types={output_types} attrs={available_attrs} "
        f"response_preview={response_preview}"
    )


def _openai_response_output_item_types(response: Any) -> List[str]:
    output_items: Any = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    if not isinstance(output_items, list):
        return []
    item_types: List[str] = []
    for item in output_items:
        if isinstance(item, dict):
            item_type: str = str(item.get("type", "")).strip().lower()
        else:
            item_type = str(getattr(item, "type", "")).strip().lower()
        if item_type:
            item_types.append(item_type)
    return item_types


def _openai_incomplete_reason(response: Any) -> str:
    incomplete: Any = getattr(response, "incomplete_details", None)
    if incomplete is None and isinstance(response, dict):
        incomplete = response.get("incomplete_details")
    if incomplete is None:
        return ""
    if isinstance(incomplete, dict):
        return str(incomplete.get("reason", "")).strip().lower()
    return str(getattr(incomplete, "reason", "")).strip().lower()


def _is_openai_temperature_unsupported_error(error: Exception) -> bool:
    message: str = str(error or "").lower()
    return (
        "unsupported parameter" in message
        and "temperature" in message
        and "invalid_request_error" in message
    )


def _openai_should_send_temperature(model_name: str) -> bool:
    normalized_model_name: str = str(model_name or "").strip().lower()
    if normalized_model_name.startswith("gpt"):
        return False
    return True


def _load_openai_org_usage_config_or_none() -> Optional[OpenAIOrgUsageConfig]:
    admin_api_key: str = str(os.getenv("OPENAI_ADMIN_KEY", "") or "").strip()
    project_id: str = str(os.getenv("OPENAI_PROJECT_ID", "") or "").strip()
    if not admin_api_key or not project_id:
        return None
    models_raw: str = str(
        os.getenv("OPENAI_USAGE_MODELS", "gpt-5-mini,gpt-5-nano") or ""
    ).strip()
    models: Tuple[str, ...] = tuple(
        model.strip() for model in models_raw.split(",") if model.strip()
    )
    if not models:
        models = ("gpt-5-mini", "gpt-5-nano")
    monthly_budget_raw: str = str(
        os.getenv("OPENAI_MONTHLY_BUDGET_USD", "") or ""
    ).strip()
    monthly_budget_usd: Optional[float] = None
    if monthly_budget_raw:
        try:
            monthly_budget_usd = float(monthly_budget_raw)
        except Exception:
            monthly_budget_usd = None
    timeout_raw: str = str(os.getenv("OPENAI_USAGE_TIMEOUT_SEC", "30.0") or "30.0")
    try:
        timeout_sec: float = max(5.0, float(timeout_raw))
    except Exception:
        timeout_sec = 30.0
    verbose_raw: str = str(os.getenv("OPENAI_USAGE_VERBOSE_LOG", "1") or "1").strip()
    verbose_log: bool = verbose_raw.lower() in {"1", "true", "yes", "on"}
    return OpenAIOrgUsageConfig(
        admin_api_key=admin_api_key,
        project_id=project_id,
        models=models,
        monthly_budget_usd=monthly_budget_usd,
        timeout_sec=timeout_sec,
        verbose_log=verbose_log,
    )


def _unix_seconds_utc(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return int(value.timestamp())


class OpenAIAdminClient:
    def __init__(self, *, admin_api_key: str, timeout_sec: float) -> None:
        self._admin_api_key: str = admin_api_key
        self._timeout_sec: float = timeout_sec
        self._session: requests.Session = requests.Session()

    def _get(self, *, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        response: requests.Response = self._session.get(
            url=url,
            params=params,
            headers={
                "Authorization": f"Bearer {self._admin_api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._timeout_sec,
        )
        response.raise_for_status()
        payload: Any = response.json()
        if isinstance(payload, dict):
            return cast(Dict[str, Any], payload)
        return {}

    def list_project_rate_limits(self, *, project_id: str) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = self._get(
            url=f"https://api.openai.com/v1/organization/projects/{project_id}/rate_limits",
            params={"limit": 200},
        )
        data: Any = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def get_completions_usage_daily_by_model(
        self,
        *,
        start_time_utc: datetime,
        end_time_utc: datetime,
        models: Tuple[str, ...],
        project_id: str,
        limit_days: int,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "start_time": _unix_seconds_utc(start_time_utc),
            "end_time": _unix_seconds_utc(end_time_utc),
            "bucket_width": "1d",
            "group_by": ["model"],
            "project_ids": [project_id],
            "limit": min(max(limit_days, 1), 31),
        }
        if models:
            params["models"] = list(models)
        return self._get(
            url="https://api.openai.com/v1/organization/usage/completions",
            params=params,
        )

    def get_costs_daily(
        self,
        *,
        start_time_utc: datetime,
        end_time_utc: datetime,
        project_id: str,
        limit_days: int,
    ) -> Dict[str, Any]:
        return self._get(
            url="https://api.openai.com/v1/organization/costs",
            params={
                "start_time": _unix_seconds_utc(start_time_utc),
                "end_time": _unix_seconds_utc(end_time_utc),
                "bucket_width": "1d",
                "group_by": ["project_id", "line_item"],
                "project_ids": [project_id],
                "limit": min(max(limit_days, 1), 180),
            },
        )


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


def _rate_limit_metric_value(
    rate_limit_obj: Optional[Dict[str, Any]],
    candidate_keys: Tuple[str, ...],
) -> Optional[float]:
    if not isinstance(rate_limit_obj, dict):
        return None
    for key in candidate_keys:
        raw_value: Any = rate_limit_obj.get(key)
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
    return None


def _json_dumps_for_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _json_dumps_pretty_for_log(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _iso_from_unix_seconds(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return "n/a"


def _safe_int_or_none(raw_value: Any) -> Optional[int]:
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    text_value: str = str(raw_value or "").strip()
    if not text_value:
        return None
    text_value = text_value.replace(",", "")
    if not re.fullmatch(r"-?\d+", text_value):
        return None
    try:
        return int(text_value)
    except Exception:
        return None


def _parse_reset_value_to_seconds(raw_value: str) -> Optional[float]:
    text_value: str = str(raw_value or "").strip().lower()
    if not text_value:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", text_value):
        numeric_value: float = float(text_value)
        if numeric_value > 1_000_000_000:
            return max(0.0, numeric_value - time.time())
        return max(0.0, numeric_value)
    total_seconds: float = 0.0
    matched_any: bool = False
    for match in re.finditer(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text_value):
        matched_any = True
        numeric_part: float = float(match.group(1))
        unit_part: str = match.group(2)
        if unit_part == "ms":
            total_seconds += numeric_part / 1000.0
        elif unit_part == "s":
            total_seconds += numeric_part
        elif unit_part == "m":
            total_seconds += numeric_part * 60.0
        elif unit_part == "h":
            total_seconds += numeric_part * 3600.0
    if matched_any:
        return max(0.0, total_seconds)
    return None


def extract_rate_limit_snapshot(headers: Mapping[str, str]) -> Dict[str, Any]:
    rate_limit_headers: Dict[str, str] = {}
    for header_name, header_value in headers.items():
        name_normalized: str = str(header_name or "").strip().lower()
        if not name_normalized.startswith("x-ratelimit-"):
            continue
        rate_limit_headers[name_normalized] = str(header_value or "").strip()

    remaining_requests_candidates: List[int] = []
    remaining_tokens_candidates: List[int] = []
    reset_requests_candidates: List[Tuple[float, str]] = []
    reset_tokens_candidates: List[Tuple[float, str]] = []
    raw_reset_requests: List[str] = []
    raw_reset_tokens: List[str] = []

    for header_name, header_value in rate_limit_headers.items():
        lower_name: str = header_name.lower()
        if "remaining" in lower_name and "request" in lower_name:
            maybe_remaining_requests: Optional[int] = _safe_int_or_none(header_value)
            if maybe_remaining_requests is not None:
                remaining_requests_candidates.append(maybe_remaining_requests)
        if "remaining" in lower_name and "token" in lower_name:
            maybe_remaining_tokens: Optional[int] = _safe_int_or_none(header_value)
            if maybe_remaining_tokens is not None:
                remaining_tokens_candidates.append(maybe_remaining_tokens)
        if "reset" in lower_name and "request" in lower_name:
            raw_reset_requests.append(header_value)
            maybe_reset_seconds: Optional[float] = _parse_reset_value_to_seconds(
                header_value
            )
            if maybe_reset_seconds is not None:
                reset_requests_candidates.append((maybe_reset_seconds, header_value))
        if "reset" in lower_name and "token" in lower_name:
            raw_reset_tokens.append(header_value)
            maybe_reset_seconds = _parse_reset_value_to_seconds(header_value)
            if maybe_reset_seconds is not None:
                reset_tokens_candidates.append((maybe_reset_seconds, header_value))

    snapshot: Dict[str, Any] = {}
    if remaining_requests_candidates:
        snapshot["remaining_requests"] = min(remaining_requests_candidates)
    if remaining_tokens_candidates:
        snapshot["remaining_tokens"] = min(remaining_tokens_candidates)
    if reset_requests_candidates:
        reset_requests_candidates.sort(key=lambda item: item[0])
        reset_seconds, _ = reset_requests_candidates[0]
        snapshot["reset_requests"] = f"{int(round(reset_seconds))}s"
    elif raw_reset_requests:
        snapshot["reset_requests"] = sorted(raw_reset_requests, key=len)[0]
    if reset_tokens_candidates:
        reset_tokens_candidates.sort(key=lambda item: item[0])
        reset_seconds, _ = reset_tokens_candidates[0]
        snapshot["reset_tokens"] = f"{int(round(reset_seconds))}s"
    elif raw_reset_tokens:
        snapshot["reset_tokens"] = sorted(raw_reset_tokens, key=len)[0]
    return snapshot


def format_rate_limit_one_liner(
    snapshot: Dict[str, Any],
    model_name: Optional[str],
    request_kind: str,
) -> str:
    line_parts: List[str] = [
        "OPENAI RL",
        f"model={(str(model_name or '').strip() or 'unknown')}",
        f"kind={(str(request_kind or '').strip() or 'request')}",
    ]
    remaining_requests: Any = snapshot.get("remaining_requests")
    remaining_tokens: Any = snapshot.get("remaining_tokens")
    reset_requests: Any = snapshot.get("reset_requests")
    reset_tokens: Any = snapshot.get("reset_tokens")
    if isinstance(remaining_requests, int):
        line_parts.append(f"rem_req={remaining_requests}")
    if isinstance(remaining_tokens, int):
        line_parts.append(f"rem_tok={remaining_tokens}")
    if isinstance(reset_requests, str) and reset_requests.strip():
        line_parts.append(f"reset_req={reset_requests}")
    if isinstance(reset_tokens, str) and reset_tokens.strip():
        line_parts.append(f"reset_tok={reset_tokens}")
    return " ".join(line_parts)


def _extract_openai_headers_from_raw_response(raw_response: Any) -> Dict[str, str]:
    headers_any: Any = getattr(raw_response, "headers", None)
    if headers_any is None:
        response_any: Any = getattr(raw_response, "response", None)
        headers_any = getattr(response_any, "headers", None)
    if headers_any is None:
        http_response_any: Any = getattr(raw_response, "http_response", None)
        headers_any = getattr(http_response_any, "headers", None)
    if headers_any is None:
        return {}
    if isinstance(headers_any, Mapping):
        return {
            str(key).strip(): str(value).strip()
            for key, value in headers_any.items()
            if str(key).strip()
        }
    items_attr: Any = getattr(headers_any, "items", None)
    if callable(items_attr):
        try:
            items_result: Any = items_attr()
            if not isinstance(items_result, Iterable):
                return {}
            items_pairs: Iterable[Tuple[Any, Any]] = cast(
                Iterable[Tuple[Any, Any]],
                items_result,
            )
            return {
                str(key).strip(): str(value).strip()
                for key, value in items_pairs
                if str(key).strip()
            }
        except Exception:
            return {}
    return {}


def _log_openai_rate_limit_snapshot(
    *,
    response_or_raw: Any,
    model_name: str,
    request_kind: str,
) -> None:
    headers: Dict[str, str] = _extract_openai_headers_from_raw_response(response_or_raw)
    if not headers:
        LOGGER.info(
            "OPENAI RL model=%s kind=%s headers=unavailable",
            str(model_name or "").strip() or "unknown",
            str(request_kind or "").strip() or "request",
        )
        return
    snapshot: Dict[str, Any] = extract_rate_limit_snapshot(headers)
    if not snapshot:
        LOGGER.info(
            "OPENAI RL model=%s kind=%s",
            str(model_name or "").strip() or "unknown",
            str(request_kind or "").strip() or "request",
        )
        return
    LOGGER.info(
        "%s",
        format_rate_limit_one_liner(
            snapshot=snapshot,
            model_name=model_name,
            request_kind=request_kind,
        ),
    )


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
    timeout_sec: float = 30.0
    timeout_raw: str = str(os.getenv("OPENAI_USAGE_TIMEOUT_SEC", "30.0") or "30.0")
    try:
        timeout_sec = max(5.0, float(timeout_raw))
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
    usage_endpoint_path: str = "/v1/organization/usage/completions"
    usage_url: str = f"https://api.openai.com{usage_endpoint_path}"
    usage_max_retries: int = 2
    usage_backoff_sec: Tuple[float, ...] = (0.5, 1.0)
    usage_payload: Dict[str, Any] = {}
    usage_error_status: Optional[int] = None
    usage_error_message: str = ""
    usage_error_kind: str = ""
    usage_success: bool = False
    for usage_attempt_index in range(usage_max_retries + 1):
        try:
            usage_response: requests.Response = session.get(
                url=usage_url,
                params=usage_params,
                headers=common_headers,
                timeout=timeout_sec,
            )
            usage_response.raise_for_status()
            usage_payload_any: Any = usage_response.json()
            usage_payload = (
                cast(Dict[str, Any], usage_payload_any)
                if isinstance(usage_payload_any, dict)
                else {}
            )
            usage_success = True
            break
        except requests.HTTPError as error:
            status_code: Optional[int] = None
            if error.response is not None:
                status_code = int(error.response.status_code)
            usage_error_status = status_code
            usage_error_message = str(error).strip() or "HTTP error"
            usage_error_kind = "http_500" if status_code == 500 else "http_error"
            if status_code == 500 and usage_attempt_index < usage_max_retries:
                if usage_attempt_index == 0:
                    LOGGER.debug(
                        "openai_usage_summary=request_failed_retrying http=500 attempt=%d endpoint=%s backoff_sec=%.1f",
                        usage_attempt_index + 1,
                        usage_endpoint_path,
                        usage_backoff_sec[usage_attempt_index],
                    )
                time.sleep(usage_backoff_sec[usage_attempt_index])
                continue
            break
        except Exception as error:
            usage_error_message = _summarize_error(cast(Exception, error))
            usage_error_kind = "request_error"
            break
    if not usage_success:
        error_payload: Dict[str, Any] = {
            "error": "usage request failed",
            "usage_endpoint": usage_endpoint_path,
            "usage_attempts": usage_max_retries,
        }
        if usage_error_kind:
            error_payload["usage_error_kind"] = usage_error_kind
        if usage_error_status is not None:
            error_payload["usage_http_status"] = usage_error_status
        elif usage_error_message:
            error_payload["usage_error_message"] = usage_error_message
        return error_payload
    try:
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
        return {
            "error": f"costs request failed: {_summarize_error(cast(Exception, error))}"
        }

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
        usage_error_kind: str = str(summary_payload.get("usage_error_kind", "")).strip()
        usage_http_status: Optional[int] = _safe_int_or_none(
            summary_payload.get("usage_http_status")
        )
        usage_attempts: int = (
            _safe_int_or_none(summary_payload.get("usage_attempts")) or 0
        )
        usage_endpoint: str = str(summary_payload.get("usage_endpoint", "")).strip()
        if (
            usage_error_kind == "http_500"
            and usage_http_status == 500
            and usage_endpoint
        ):
            logger.warning(
                "openai_usage_summary=skipped http=500 attempts=%d endpoint=%s",
                usage_attempts,
                usage_endpoint,
            )
        else:
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
        logger.info(
            "OPENAI COST month spent_usd=%.6f",
            spent_usd_month,
        )


def _get_openai_client(config: "AppConfig") -> Any:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if OpenAI is None:
        raise RuntimeError("Package 'openai' is not installed.")
    api_key: str = os.getenv("GPT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Env var GPT_API_KEY is required for OpenAI calls.")
    _OPENAI_CLIENT = OpenAI(
        api_key=api_key,
        timeout=config.openai_timeout_sec,
        max_retries=0,
    )
    return _OPENAI_CLIENT


def _openai_merge_call_raw(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    attempt_label: str,
    model_name: str,
    prompt_text_override: Optional[str] = None,
) -> str:
    prompt_text: str = str(
        prompt_text_override or ""
    ).strip() or _build_llm_merge_prompt_text(
        language=language,
        videos=videos,
        config=config,
    )
    pre_delay_sec: float = max(0.0, float(config.openai_pre_delay_sec))
    LOGGER.info(
        "LLM merge: sleeping %.2fs before request ... provider=openai language=%s sources=%d",
        pre_delay_sec,
        language,
        len(videos),
    )
    if pre_delay_sec > 0.0:
        time.sleep(pre_delay_sec)
    LOGGER.info(
        "LLM provider=openai attempt=%s model=%s timeout_sec=%.1f",
        attempt_label,
        model_name,
        config.openai_timeout_sec,
    )
    LOGGER.debug(
        "LLM merge prompt: expected_paragraphs=%d prompt_chars=%d model=%s",
        len(videos),
        len(prompt_text),
        model_name,
    )
    client: Any = _get_openai_client(config).with_options(
        timeout=config.openai_timeout_sec
    )
    temperature: float = float(getattr(config, "openai_temperature", 0.0))
    reasoning_effort: str = "low"
    initial_max_output_tokens: int = int(config.openai_max_output_tokens)
    temperature_enabled: bool = _openai_should_send_temperature(model_name)
    use_structured_output: bool = prompt_text_override is None
    if not temperature_enabled:
        LOGGER.debug(
            "OpenAI model=%s: sending request without temperature (known unsupported family).",
            model_name,
        )

    def _structured_json_schema(expected_source_count: int) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["title", "paragraphs", "cta", "hashtags", "sources"],
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 98,
                    "pattern": r"^[^\ud800-\udfff]+$",
                },
                "paragraphs": {
                    "type": "array",
                    "minItems": expected_source_count,
                    "maxItems": expected_source_count,
                    "items": {"type": "string", "minLength": 1},
                },
                "cta": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": r"^[^\r\n]+$",
                },
                "hashtags": {
                    "type": "string",
                    "minLength": 1,
                    "pattern": r"^[^\r\n]+$",
                },
                "sources": {
                    "type": "array",
                    "minItems": expected_source_count,
                    "maxItems": expected_source_count,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"^https?://\S+$",
                    },
                },
            },
        }

    def _extract_structured_payload_or_none(response: Any) -> Optional[Dict[str, Any]]:
        raw_json_text: str = _strip_json_code_fences(
            _extract_openai_response_text(response)
        )
        if not raw_json_text:
            return None
        candidates: List[str] = [raw_json_text]
        candidates.extend(_extract_json_object_candidates(raw_json_text))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                parsed: Any = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return cast(Dict[str, Any], parsed)
        return None

    def _build_structured_output_text_or_raise(payload: Dict[str, Any]) -> str:
        expected_source_count: int = len(videos)
        title_raw: str = str(payload.get("title") or "").strip()
        title_value: str = _sanitize_title(
            title_raw,
            min_chars=1,
            max_chars=98,
            allow_emoji=False,
        )
        if title_value != title_raw:
            raise RuntimeError("structured title is invalid")
        paragraphs_raw: Any = payload.get("paragraphs")
        if not isinstance(paragraphs_raw, list):
            raise RuntimeError("structured paragraphs must be an array")
        if len(paragraphs_raw) != expected_source_count:
            raise RuntimeError(
                f"structured paragraphs count {len(paragraphs_raw)} != {expected_source_count}"
            )
        paragraphs: List[str] = []
        for paragraph_value in paragraphs_raw:
            paragraph_text: str = _normalize_single_line_text(paragraph_value)
            if not paragraph_text:
                raise RuntimeError("structured paragraph is empty")
            paragraphs.append(paragraph_text)
        cta_line: str = _normalize_single_line_text(str(payload.get("cta") or ""))
        if not cta_line:
            raise RuntimeError("structured cta is empty")
        hashtags_line: str = _normalize_single_line_text(
            str(payload.get("hashtags") or "")
        )
        if not hashtags_line:
            raise RuntimeError("structured hashtags are empty")
        sources_raw: Any = payload.get("sources")
        if not isinstance(sources_raw, list):
            raise RuntimeError("structured sources must be an array")
        if len(sources_raw) != expected_source_count:
            raise RuntimeError(
                f"structured sources count {len(sources_raw)} != {expected_source_count}"
            )
        source_urls: List[str] = []
        for source_value in sources_raw:
            source_url: str = _normalize_source_url_for_validation(source_value)
            if not _SOURCE_URL_LINE_RE.match(source_url):
                raise RuntimeError(f"structured source URL is invalid: {source_url!r}")
            source_urls.append(source_url)
        description_text: str = _format_structured_merged_description(
            paragraphs=paragraphs,
            cta_line=cta_line,
            hashtags_line=hashtags_line,
            source_urls=source_urls,
        )
        return f"TITLE: {title_value}\nDESCRIPTION:\n{description_text}".strip()

    def _create_response(max_output_tokens: int, *, structured: bool) -> Any:
        request_kwargs: Dict[str, Any] = {
            "model": model_name,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt_text}],
                }
            ],
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
        }
        if structured:
            schema_name: str = f"streamertg_merge_{language}_v1"
            request_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": _structured_json_schema(len(videos)),
                }
            }
            LOGGER.info(
                "OpenAI structured_output=enabled schema_name=%s expected_sources=%d",
                schema_name,
                len(videos),
            )
        if temperature_enabled:
            request_kwargs["temperature"] = temperature
        try:
            raw_response: Any = client.responses.with_raw_response.create(
                **request_kwargs
            )
            _log_openai_rate_limit_snapshot(
                response_or_raw=raw_response,
                model_name=model_name,
                request_kind="responses.create",
            )
            parse_method: Any = getattr(raw_response, "parse", None)
            if callable(parse_method):
                return parse_method()
            return raw_response
        except AttributeError:
            response: Any = client.responses.create(**request_kwargs)
            _log_openai_rate_limit_snapshot(
                response_or_raw=response,
                model_name=model_name,
                request_kind="responses.create",
            )
            return response

    def _create_response_with_temperature_fallback(
        max_output_tokens: int, *, structured: bool
    ) -> Any:
        nonlocal temperature_enabled
        try:
            return _create_response(max_output_tokens, structured=structured)
        except Exception as error:
            if temperature_enabled and _is_openai_temperature_unsupported_error(
                cast(Exception, error)
            ):
                temperature_enabled = False
                LOGGER.warning(
                    "OpenAI model=%s does not support temperature; retrying_without_temperature.",
                    model_name,
                )
                return _create_response(max_output_tokens, structured=structured)
            raise

    if use_structured_output:
        try:
            used_max_output_tokens_structured: int = initial_max_output_tokens
            structured_response: Any = _create_response_with_temperature_fallback(
                used_max_output_tokens_structured,
                structured=True,
            )
            structured_incomplete_reason: str = _openai_incomplete_reason(
                structured_response
            )
            if structured_incomplete_reason == "max_output_tokens":
                used_max_output_tokens_structured = min(
                    3000, max(1, initial_max_output_tokens * 2)
                )
                LOGGER.warning(
                    "OpenAI response hit max_output_tokens attempt=%s model=%s retrying_once_with_max_output_tokens=%d",
                    attempt_label,
                    model_name,
                    used_max_output_tokens_structured,
                )
                structured_response = _create_response_with_temperature_fallback(
                    used_max_output_tokens_structured,
                    structured=True,
                )
            structured_payload: Optional[Dict[str, Any]] = (
                _extract_structured_payload_or_none(structured_response)
            )
            if structured_payload is None:
                raw_text_structured: str = _extract_openai_response_text(
                    structured_response
                )
                tolerant_payload, tolerant_method = parse_json_tolerant(
                    raw_text_structured
                )
                if tolerant_payload is not None:
                    structured_payload = cast(Dict[str, Any], tolerant_payload)
                    LOGGER.debug(
                        "structured_output=tolerant_json_success method=%s raw_len=%d",
                        tolerant_method,
                        len(raw_text_structured),
                    )
                else:
                    LOGGER.debug(
                        "structured_output=tolerant_json_fail raw_len=%d",
                        len(raw_text_structured),
                    )
                    raise RuntimeError("structured response is not valid JSON object")
            structured_output_text: str = _build_structured_output_text_or_raise(
                structured_payload
            )
            LOGGER.info("structured_output=enabled")
            return structured_output_text
        except Exception as structured_error:
            LOGGER.warning(
                'structured_output=fallback_plain_text reason="%s"',
                _compact_single_line(_summarize_error(structured_error), max_len=220),
            )

    used_max_output_tokens: int = initial_max_output_tokens
    response: Any = _create_response_with_temperature_fallback(
        used_max_output_tokens,
        structured=False,
    )
    incomplete_reason: str = _openai_incomplete_reason(response)
    if incomplete_reason == "max_output_tokens":
        used_max_output_tokens = min(3000, max(1, initial_max_output_tokens * 2))
        LOGGER.warning(
            "OpenAI response hit max_output_tokens attempt=%s model=%s retrying_once_with_max_output_tokens=%d",
            attempt_label,
            model_name,
            used_max_output_tokens,
        )
        response = _create_response_with_temperature_fallback(
            used_max_output_tokens,
            structured=False,
        )
        incomplete_reason = _openai_incomplete_reason(response)

    raw_text: str = _extract_openai_response_text(response)
    output_item_types: List[str] = _openai_response_output_item_types(response)
    only_reasoning: bool = bool(output_item_types) and all(
        item_type == "reasoning" for item_type in output_item_types
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        preview: str = _compact_single_line(raw_text, max_len=300)
        temperature_debug: str = (
            f"{temperature:.2f}" if temperature_enabled else "disabled"
        )
        LOGGER.debug(
            "OpenAI merge response preview attempt=%s model=%s max_output_tokens=%d temperature=%s reasoning_effort=%s incomplete_reason=%s output_item_types=%s only_reasoning=%s text_len=%d text_repr=%s preview=%s",
            attempt_label,
            model_name,
            used_max_output_tokens,
            temperature_debug,
            reasoning_effort,
            incomplete_reason or "n/a",
            output_item_types,
            "yes" if only_reasoning else "no",
            len(raw_text),
            repr(raw_text[:200]),
            preview,
        )
        if not raw_text.strip():
            LOGGER.debug(
                "OpenAI merge response shape attempt=%s model=%s: %s",
                attempt_label,
                model_name,
                _debug_openai_output_shape(response),
            )
    if not raw_text.strip():
        raise RuntimeError("openai merge returned empty output text")
    return raw_text


def _gemini_merge_title_and_description_once(
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
) -> MergedLanguageContent:
    raw_text: str = _gemini_merge_call_raw(
        language=language,
        videos=videos,
        config=config,
        attempt_label=f"GEMINI_MERGE_{language.upper()}",
    )
    return _parse_llm_merge_raw_or_raise(
        provider_name="gemini",
        model_name=config.gemini_model,
        language=language,
        raw_text=raw_text,
        source_urls=[item.normalized_link for item in videos],
        source_titles=[item.metadata.title for item in videos],
    )


def _build_plain_merged_content_or_raise(
    *,
    model_name: str,
    title_text: str,
    description_text: str,
) -> MergedLanguageContent:
    merged_title: str = _sanitize_title(
        str(title_text or "").strip(),
        min_chars=1,
        max_chars=98,
        allow_emoji=False,
    )
    merged_title = re.sub(r"[.!?…]{2,}$", "", merged_title).strip()
    merged_title = _sanitize_title(
        merged_title,
        min_chars=1,
        max_chars=98,
        allow_emoji=False,
    )
    cleaned_description, is_valid, reasons = _clean_and_validate_llm_description(
        text=description_text
    )
    if not is_valid:
        raise RuntimeError(
            "plain description validation failed: " + ("; ".join(reasons) or "unknown")
        )
    if not cleaned_description:
        raise RuntimeError("plain description is empty")
    return MergedLanguageContent(
        title=merged_title,
        description=cleaned_description,
        title_selected=merged_title,
        description_selected=cleaned_description,
        title_audit=merged_title,
        description_audit=cleaned_description,
        llm_model=model_name,
    )


def _extract_title_and_description_payload_or_none(
    raw_text: str,
) -> Optional[Tuple[str, str]]:
    payload: Optional[Dict[str, str]] = _parse_merge_payload_plaintext_tolerant(
        raw_text
    )
    if payload is None:
        payload = _parse_merge_payload_tolerant(raw_text)
    if payload is None:
        return None
    title_value: str = str(payload.get("merged_title") or "").strip()
    description_value: str = str(payload.get("merged_description") or "").strip()
    if not title_value or not description_value:
        return None
    return (title_value, description_value)


def _attempt_openai_plain_description_repair_once(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    attempt_label: str,
    model_name: str,
    invalid_text: str,
) -> str:
    prompt_text: str = _build_plain_description_repair_prompt_text(
        target_language=language,
        invalid_text=invalid_text,
    )
    repaired_text: str = _openai_merge_call_raw(
        language=language,
        videos=videos,
        config=config,
        attempt_label=attempt_label,
        model_name=model_name,
        prompt_text_override=prompt_text,
    )
    cleaned_text, is_valid, reasons = _clean_and_validate_llm_description(
        text=repaired_text
    )
    if not is_valid:
        raise RuntimeError(
            "repair output validation failed: " + ("; ".join(reasons) or "unknown")
        )
    if not cleaned_text:
        raise RuntimeError("repair output is empty")
    return cleaned_text


def _attempt_openai_single_source_translate_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    attempt_label: str,
) -> LanguageMergeAttempt:
    if len(videos) != 1:
        raise RuntimeError("single-source translate expects exactly one video.")
    source_video: PlannedVideo = videos[0]
    source_language: str = (
        source_video.row_characteristics.detected_source_language
        if source_video.row_characteristics is not None
        else source_video.language
    )
    model_sequence: List[str] = [
        str(config.openai_model_primary or "").strip() or "gpt-5-nano",
        "gpt-5-mini",
    ]
    if model_sequence[1] == model_sequence[0]:
        model_sequence = [model_sequence[0]]
    source_description: str = (
        source_video.metadata.description.strip() or _no_description_text()
    )
    base_title: str = source_video.metadata.title.strip() or "Untitled"
    last_raw_response: str = ""
    last_error_summary: Optional[str] = None
    plain_repair_used: bool = False
    plain_repair_fallback_model: str = ""
    validation_reasons: List[str] = []
    for attempt_index, model_name in enumerate(model_sequence, start=1):
        call_label: str = f"{attempt_label}_TRY{attempt_index}"
        prompt_text: str = _build_single_source_translate_prompt_text(
            source_language=source_language,
            target_language=language,
            source_description=source_description,
        )
        try:
            last_raw_response = _openai_merge_call_raw(
                language=language,
                videos=videos,
                config=config,
                attempt_label=call_label,
                model_name=model_name,
                prompt_text_override=prompt_text,
            )
            cleaned_description, is_valid, reasons = (
                _clean_and_validate_llm_description(text=last_raw_response)
            )
            if not is_valid:
                validation_reasons = _truncate_validation_reasons(reasons)
                LOGGER.debug(
                    "Single-source validation failed language=%s model=%s reasons=%s",
                    language,
                    model_name,
                    reasons,
                )
                repair_label: str = f"{call_label}_REPAIR"
                plain_repair_used = True
                cleaned_description = _attempt_openai_plain_description_repair_once(
                    language=language,
                    videos=videos,
                    config=config,
                    attempt_label=repair_label,
                    model_name=model_name,
                    invalid_text=last_raw_response,
                )
            merged_content: MergedLanguageContent = (
                _build_plain_merged_content_or_raise(
                    model_name=model_name,
                    title_text=base_title,
                    description_text=cleaned_description,
                )
            )
            return LanguageMergeAttempt(
                language=language,
                model_name=model_name,
                raw_response_text=last_raw_response,
                merged=merged_content,
                error_summary=None,
                salvaged_title=merged_content.title,
                plain_repair_used=plain_repair_used,
                plain_repair_fallback_model=plain_repair_fallback_model or None,
                validation_reasons=validation_reasons or None,
            )
        except Exception as error:
            last_error_summary = _summarize_error(error)
            if attempt_index < len(model_sequence):
                if model_sequence[attempt_index] == "gpt-5-mini":
                    plain_repair_fallback_model = "gpt-5-mini"
                LOGGER.debug(
                    "Single-source translate fallback model will be used language=%s from_model=%s reason=%s",
                    language,
                    model_name,
                    last_error_summary,
                )
            continue
    return LanguageMergeAttempt(
        language=language,
        model_name=model_sequence[-1],
        raw_response_text=last_raw_response,
        merged=None,
        error_summary=last_error_summary or "unknown error",
        salvaged_title=base_title,
        plain_repair_used=plain_repair_used or None,
        plain_repair_fallback_model=plain_repair_fallback_model or None,
        validation_reasons=validation_reasons or None,
    )


def _attempt_gemini_single_source_translate_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    attempt_label: str,
) -> LanguageMergeAttempt:
    if len(videos) != 1:
        raise RuntimeError("single-source translate expects exactly one video.")
    source_video: PlannedVideo = videos[0]
    source_language: str = (
        source_video.row_characteristics.detected_source_language
        if source_video.row_characteristics is not None
        else source_video.language
    )
    source_description: str = (
        source_video.metadata.description.strip() or _no_description_text()
    )
    prompt_text: str = _build_single_source_translate_prompt_text(
        source_language=source_language,
        target_language=language,
        source_description=source_description,
    )
    raw_response_text: str = ""
    model_name: str = str(config.gemini_model or "").strip() or "gemini"
    base_title: str = source_video.metadata.title.strip() or "Untitled"
    plain_repair_used: bool = False
    validation_reasons: List[str] = []
    try:
        raw_response_text = _gemini_merge_call_raw(
            language=language,
            videos=videos,
            config=config,
            attempt_label=attempt_label,
            prompt_text_override=prompt_text,
        )
        cleaned_description, is_valid, reasons = _clean_and_validate_llm_description(
            text=raw_response_text
        )
        if not is_valid:
            validation_reasons = _truncate_validation_reasons(reasons)
            LOGGER.debug(
                "Gemini single-source validation failed language=%s reasons=%s",
                language,
                reasons,
            )
            plain_repair_used = True
            repair_prompt: str = _build_plain_description_repair_prompt_text(
                target_language=language,
                invalid_text=raw_response_text,
            )
            repaired_text: str = _gemini_merge_call_raw(
                language=language,
                videos=videos,
                config=config,
                attempt_label=f"{attempt_label}_REPAIR",
                prompt_text_override=repair_prompt,
            )
            cleaned_description, is_valid, reasons = (
                _clean_and_validate_llm_description(text=repaired_text)
            )
            if not is_valid:
                raise RuntimeError(
                    "single-source description validation failed after repair: "
                    + ("; ".join(reasons) or "unknown")
                )
        merged_content: MergedLanguageContent = _build_plain_merged_content_or_raise(
            model_name=model_name,
            title_text=base_title,
            description_text=cleaned_description,
        )
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text=raw_response_text,
            merged=merged_content,
            error_summary=None,
            salvaged_title=merged_content.title,
            plain_repair_used=plain_repair_used,
            plain_repair_fallback_model=None,
            validation_reasons=validation_reasons or None,
        )
    except Exception as error:
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text=raw_response_text,
            merged=None,
            error_summary=_summarize_error(error),
            salvaged_title=base_title,
            plain_repair_used=plain_repair_used or None,
            plain_repair_fallback_model=None,
            validation_reasons=validation_reasons or None,
        )


def _attempt_gemini_merge_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    attempt_label: str,
) -> LanguageMergeAttempt:
    raw_response_text: str = ""
    model_name: str = str(config.gemini_model or "").strip() or "gemini"
    source_urls: List[str] = [item.normalized_link for item in videos]
    source_titles: List[str] = [item.metadata.title for item in videos]
    salvaged_title: Optional[str] = None

    def _repair_hashtags_once(invalid_hashtags_line: str) -> str:
        repair_prompt: str = _build_hashtags_repair_prompt_text(
            invalid_hashtags_line=invalid_hashtags_line
        )
        repaired_raw_text: str = _gemini_merge_call_raw(
            language=language,
            videos=videos,
            config=config,
            attempt_label=f"{attempt_label}_HASHTAGS_REPAIR",
            prompt_text_override=repair_prompt,
        )
        return _extract_hashtags_line_from_repair_output(repaired_raw_text)

    try:
        raw_response_text = _gemini_merge_call_raw(
            language=language,
            videos=videos,
            config=config,
            attempt_label=attempt_label,
        )
        salvaged_title = _extract_valid_title_from_raw_or_none(raw_response_text)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "LLM preview provider=gemini model=%s text=%s",
                model_name,
                _compact_single_line(raw_response_text, max_len=300),
            )
    except Exception as error:
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text=raw_response_text,
            merged=None,
            error_summary=_summarize_error(error),
            salvaged_title=_extract_valid_title_from_raw_or_none(raw_response_text),
        )
    try:
        merged_content: MergedLanguageContent = _parse_llm_merge_raw_or_raise(
            provider_name="gemini",
            model_name=model_name,
            language=language,
            raw_text=raw_response_text,
            source_urls=source_urls,
            source_titles=source_titles,
            hashtags_repair_once=_repair_hashtags_once,
        )
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text=raw_response_text,
            merged=merged_content,
            error_summary=None,
            salvaged_title=merged_content.title,
        )
    except Exception as error:
        parse_error_summary: str = _summarize_error(error)
        LOGGER.warning(
            "Gemini merge parse failed language=%s model=%s reason=%s; running one repair pass.",
            language,
            model_name,
            parse_error_summary,
        )
        repair_prompt: str = _build_llm_merge_repair_prompt_text(
            language=language,
            videos=videos,
            config=config,
            previous_output=raw_response_text,
            parse_error=parse_error_summary,
        )
        try:
            repaired_raw_text: str = _gemini_merge_call_raw(
                language=language,
                videos=videos,
                config=config,
                attempt_label=f"{attempt_label}_REPAIR",
                prompt_text_override=repair_prompt,
            )
            repaired_title: Optional[str] = _extract_valid_title_from_raw_or_none(
                repaired_raw_text
            )
            if repaired_title:
                salvaged_title = repaired_title
            repaired_merged: MergedLanguageContent = _parse_llm_merge_raw_or_raise(
                provider_name="gemini",
                model_name=model_name,
                language=language,
                raw_text=repaired_raw_text,
                source_urls=source_urls,
                source_titles=source_titles,
                hashtags_repair_once=_repair_hashtags_once,
            )
            LOGGER.warning(
                "Gemini merge repair succeeded language=%s model=%s",
                language,
                model_name,
            )
            return LanguageMergeAttempt(
                language=language,
                model_name=model_name,
                raw_response_text=repaired_raw_text,
                merged=repaired_merged,
                error_summary=None,
                salvaged_title=repaired_merged.title,
            )
        except Exception as repair_error:
            combined_error: str = (
                f"parse_error={parse_error_summary}; repair_error={_summarize_error(repair_error)}"
            )
            LOGGER.warning(
                "Gemini merge repair failed language=%s model=%s reason=%s",
                language,
                model_name,
                combined_error,
            )
        return LanguageMergeAttempt(
            language=language,
            model_name=model_name,
            raw_response_text=raw_response_text,
            merged=None,
            error_summary=combined_error,
            salvaged_title=salvaged_title,
        )


def _attempt_openai_merge_with_audit(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
    attempt_label: str,
) -> LanguageMergeAttempt:
    model_sequence: List[str] = [
        str(config.openai_model_primary or "").strip() or "gpt-5-nano",
        str(config.openai_model_fallback or "").strip() or "gpt-5-mini",
    ]
    if model_sequence[1] == model_sequence[0]:
        model_sequence = [model_sequence[0]]
    last_raw_response: str = ""
    last_error_summary: Optional[str] = None
    source_urls: List[str] = [item.normalized_link for item in videos]
    source_titles: List[str] = [item.metadata.title for item in videos]
    best_salvaged_title: Optional[str] = None
    plain_repair_used: bool = False
    plain_repair_fallback_model: str = ""
    validation_reasons: List[str] = []

    def _repair_hashtags_once(
        invalid_hashtags_line: str,
        *,
        model_name_for_repair: str,
        attempt_suffix: str,
    ) -> str:
        repair_prompt: str = _build_hashtags_repair_prompt_text(
            invalid_hashtags_line=invalid_hashtags_line
        )
        repaired_raw_text: str = _openai_merge_call_raw(
            language=language,
            videos=videos,
            config=config,
            attempt_label=f"{attempt_label}_{attempt_suffix}_HASHTAGS_REPAIR",
            model_name=model_name_for_repair,
            prompt_text_override=repair_prompt,
        )
        return _extract_hashtags_line_from_repair_output(repaired_raw_text)

    for attempt_index, model_name in enumerate(model_sequence, start=1):
        current_label: str = f"{attempt_label}_TRY{attempt_index}"
        current_try_suffix: str = f"TRY{attempt_index}"
        try:
            last_raw_response = _openai_merge_call_raw(
                language=language,
                videos=videos,
                config=config,
                attempt_label=current_label,
                model_name=model_name,
            )
            maybe_title: Optional[str] = _extract_valid_title_from_raw_or_none(
                last_raw_response
            )
            if maybe_title:
                best_salvaged_title = maybe_title
        except Exception as error:
            last_error_summary = _summarize_error(error)
            LOGGER.warning(
                "OpenAI merge call failed language=%s model=%s reason=%s",
                language,
                model_name,
                last_error_summary,
            )
            continue
        try:
            merged_content: MergedLanguageContent = _parse_llm_merge_raw_or_raise(
                provider_name="openai",
                model_name=model_name,
                language=language,
                raw_text=last_raw_response,
                source_urls=source_urls,
                source_titles=source_titles,
                hashtags_repair_once=lambda invalid_hashtags_line, current_model=model_name, current_suffix=current_try_suffix: _repair_hashtags_once(
                    invalid_hashtags_line,
                    model_name_for_repair=current_model,
                    attempt_suffix=current_suffix,
                ),
            )
            return LanguageMergeAttempt(
                language=language,
                model_name=model_name,
                raw_response_text=last_raw_response,
                merged=merged_content,
                error_summary=None,
                salvaged_title=merged_content.title,
                plain_repair_used=plain_repair_used or None,
                plain_repair_fallback_model=plain_repair_fallback_model or None,
                validation_reasons=validation_reasons or None,
            )
        except Exception as parse_error:
            parse_error_summary: str = _summarize_error(parse_error)
            LOGGER.warning(
                "OpenAI merge parse failed language=%s model=%s reason=%s; running one repair pass.",
                language,
                model_name,
                parse_error_summary,
            )
            payload_pair: Optional[Tuple[str, str]] = (
                _extract_title_and_description_payload_or_none(last_raw_response)
            )
            if payload_pair is not None:
                payload_title, payload_description = payload_pair
                _, payload_is_valid, payload_reasons = (
                    _clean_and_validate_llm_description(text=payload_description)
                )
                if not payload_is_valid:
                    validation_reasons = _truncate_validation_reasons(payload_reasons)
                try:
                    plain_repair_used = True
                    repaired_description: str = (
                        _attempt_openai_plain_description_repair_once(
                            language=language,
                            videos=videos,
                            config=config,
                            attempt_label=f"{current_label}_PLAIN_REPAIR",
                            model_name=model_name,
                            invalid_text=payload_description,
                        )
                    )
                    repaired_content: MergedLanguageContent = (
                        _build_plain_merged_content_or_raise(
                            model_name=model_name,
                            title_text=payload_title,
                            description_text=repaired_description,
                        )
                    )
                    LOGGER.warning(
                        "OpenAI merge plain-description repair succeeded language=%s model=%s",
                        language,
                        model_name,
                    )
                    return LanguageMergeAttempt(
                        language=language,
                        model_name=model_name,
                        raw_response_text=last_raw_response,
                        merged=repaired_content,
                        error_summary=None,
                        salvaged_title=repaired_content.title,
                        plain_repair_used=True,
                        plain_repair_fallback_model=None,
                        validation_reasons=validation_reasons or None,
                    )
                except Exception as plain_repair_error:
                    LOGGER.debug(
                        "OpenAI merge plain-description repair failed language=%s model=%s reason=%s",
                        language,
                        model_name,
                        _summarize_error(plain_repair_error),
                    )
                    if model_name != "gpt-5-mini":
                        plain_repair_fallback_model = "gpt-5-mini"
                        LOGGER.debug(
                            "OpenAI merge plain-description fallback model used language=%s from_model=%s to_model=gpt-5-mini",
                            language,
                            model_name,
                        )
                        try:
                            fallback_repaired_description: str = (
                                _attempt_openai_plain_description_repair_once(
                                    language=language,
                                    videos=videos,
                                    config=config,
                                    attempt_label=f"{current_label}_PLAIN_REPAIR_FALLBACK",
                                    model_name="gpt-5-mini",
                                    invalid_text=payload_description,
                                )
                            )
                            fallback_content: MergedLanguageContent = (
                                _build_plain_merged_content_or_raise(
                                    model_name="gpt-5-mini",
                                    title_text=payload_title,
                                    description_text=fallback_repaired_description,
                                )
                            )
                            return LanguageMergeAttempt(
                                language=language,
                                model_name="gpt-5-mini",
                                raw_response_text=last_raw_response,
                                merged=fallback_content,
                                error_summary=None,
                                salvaged_title=fallback_content.title,
                                plain_repair_used=True,
                                plain_repair_fallback_model="gpt-5-mini",
                                validation_reasons=validation_reasons or None,
                            )
                        except Exception as fallback_plain_error:
                            LOGGER.debug(
                                "OpenAI merge fallback plain-description repair failed language=%s reason=%s",
                                language,
                                _summarize_error(fallback_plain_error),
                            )
            repair_prompt: str = _build_llm_merge_repair_prompt_text(
                language=language,
                videos=videos,
                config=config,
                previous_output=last_raw_response,
                parse_error=parse_error_summary,
            )
            try:
                repaired_raw_text: str = _openai_merge_call_raw(
                    language=language,
                    videos=videos,
                    config=config,
                    attempt_label=f"{current_label}_REPAIR",
                    model_name=model_name,
                    prompt_text_override=repair_prompt,
                )
                repaired_title: Optional[str] = _extract_valid_title_from_raw_or_none(
                    repaired_raw_text
                )
                if repaired_title:
                    best_salvaged_title = repaired_title
                repaired_merged: MergedLanguageContent = _parse_llm_merge_raw_or_raise(
                    provider_name="openai",
                    model_name=model_name,
                    language=language,
                    raw_text=repaired_raw_text,
                    source_urls=source_urls,
                    source_titles=source_titles,
                    hashtags_repair_once=lambda invalid_hashtags_line, current_model=model_name, current_suffix=current_try_suffix: _repair_hashtags_once(
                        invalid_hashtags_line,
                        model_name_for_repair=current_model,
                        attempt_suffix=f"{current_suffix}_REPAIR",
                    ),
                )
                LOGGER.warning(
                    "OpenAI merge repair succeeded language=%s model=%s",
                    language,
                    model_name,
                )
                return LanguageMergeAttempt(
                    language=language,
                    model_name=model_name,
                    raw_response_text=repaired_raw_text,
                    merged=repaired_merged,
                    error_summary=None,
                    salvaged_title=repaired_merged.title,
                    plain_repair_used=plain_repair_used or None,
                    plain_repair_fallback_model=plain_repair_fallback_model or None,
                    validation_reasons=validation_reasons or None,
                )
            except Exception as repair_error:
                last_error_summary = f"parse_error={parse_error_summary}; repair_error={_summarize_error(repair_error)}"
            LOGGER.warning(
                "OpenAI merge attempt failed language=%s model=%s reason=%s",
                language,
                model_name,
                last_error_summary,
            )
            continue
    return LanguageMergeAttempt(
        language=language,
        model_name=model_sequence[-1],
        raw_response_text=last_raw_response,
        merged=None,
        error_summary=last_error_summary or "unknown error",
        salvaged_title=best_salvaged_title,
        plain_repair_used=plain_repair_used or None,
        plain_repair_fallback_model=plain_repair_fallback_model or None,
        validation_reasons=validation_reasons or None,
    )


def _parse_merge_payload_tolerant(raw_text: str) -> Optional[Dict[str, str]]:
    text: str = _strip_json_code_fences(str(raw_text or "").strip())
    if not text:
        return None

    candidates: List[str] = [text]
    candidates.extend(_extract_json_object_candidates(text))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed: Any = json.loads(candidate)
            if isinstance(parsed, dict):
                title_raw: str = str(parsed.get("merged_title") or "").strip()
                description_raw: str = str(
                    parsed.get("merged_description") or ""
                ).strip()
                if title_raw and description_raw:
                    return {
                        "merged_title": title_raw,
                        "merged_description": description_raw,
                    }
        except Exception:
            continue

    title_match: Optional[re.Match[str]] = re.search(
        r'"merged_title"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    description_match: Optional[re.Match[str]] = re.search(
        r'"merged_description"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        flags=re.DOTALL,
    )
    if not title_match or not description_match:
        return None
    try:
        title_value: str = json.loads(f'"{title_match.group(1)}"')
        description_value: str = json.loads(f'"{description_match.group(1)}"')
    except Exception:
        return None
    title_value = title_value.strip()
    description_value = description_value.strip()
    if not title_value or not description_value:
        return None
    return {
        "merged_title": title_value,
        "merged_description": description_value,
    }


def _parse_merge_payload_plaintext_tolerant(raw_text: str) -> Optional[Dict[str, str]]:
    text: str = _strip_json_code_fences(str(raw_text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip()
    if not text:
        return None

    lines: List[str] = text.split("\n")
    title_index: Optional[int] = None
    title_value: str = ""
    for index, line in enumerate(lines):
        title_match: Optional[re.Match[str]] = re.match(
            r"(?i)^\s*title\s*:\s*(.*)$", line
        )
        if not title_match:
            continue
        title_index = index
        title_value = title_match.group(1).strip()
        if not title_value:
            for tail_line in lines[index + 1 :]:
                candidate: str = tail_line.strip()
                if candidate:
                    title_value = candidate
                    break
        break
    if title_index is None or not title_value:
        return None

    description_index: Optional[int] = None
    description_first_line: str = ""
    for index in range(title_index + 1, len(lines)):
        desc_match: Optional[re.Match[str]] = re.match(
            r"(?i)^\s*description\s*:\s*(.*)$", lines[index]
        )
        if not desc_match:
            continue
        description_index = index
        description_first_line = desc_match.group(1).strip()
        break
    if description_index is None:
        return None

    description_tail: str = "\n".join(lines[description_index + 1 :]).strip()
    description_value: str = (
        f"{description_first_line}\n{description_tail}".strip()
        if description_first_line and description_tail
        else (description_first_line or description_tail)
    )
    if title_value and description_value:
        return {
            "merged_title": title_value,
            "merged_description": description_value,
        }

    return None


def _strip_json_code_fences(text: str) -> str:
    cleaned: str = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
    return cleaned.strip()


def parse_json_tolerant(raw_text: str) -> tuple[dict[str, object] | None, str]:
    """
    Try to parse a JSON object from raw_text.
    Returns (json_obj_or_none, method_tag).
    method_tag examples: "direct", "code_fence", "first_brace", "fail".
    """

    def _load_object(candidate_text: str) -> dict[str, object] | None:
        candidate: str = str(candidate_text or "").strip()
        if not candidate:
            return None
        try:
            parsed: Any = json.loads(candidate)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return cast(dict[str, object], parsed)
        return None

    text: str = str(raw_text or "").strip()
    if not text:
        return (None, "fail")

    direct_obj: dict[str, object] | None = _load_object(text)
    if direct_obj is not None:
        return (direct_obj, "direct")

    fence_matches: List[Tuple[int, str]] = []
    for fence_match in re.finditer(r"```([^\n`]*)\s*\n?(.*?)```", text, re.DOTALL):
        info_string: str = str(fence_match.group(1) or "").strip().lower()
        block_text: str = str(fence_match.group(2) or "").strip()
        if not block_text:
            continue
        priority: int = 0 if "json" in info_string else 1
        fence_matches.append((priority, block_text))
    if fence_matches:
        fence_matches.sort(key=lambda item: item[0])
        fence_obj: dict[str, object] | None = _load_object(fence_matches[0][1])
        if fence_obj is not None:
            return (fence_obj, "code_fence")

    first_brace_index: int = text.find("{")
    last_brace_index: int = text.rfind("}")
    if first_brace_index >= 0 and last_brace_index > first_brace_index:
        first_last_obj: dict[str, object] | None = _load_object(
            text[first_brace_index : last_brace_index + 1]
        )
        if first_last_obj is not None:
            return (first_last_obj, "first_brace")

        depth: int = 0
        in_string: bool = False
        escaped: bool = False
        for index in range(first_brace_index, len(text)):
            char: str = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                if depth > 0:
                    depth -= 1
                if depth == 0:
                    balanced_obj: dict[str, object] | None = _load_object(
                        text[first_brace_index : index + 1]
                    )
                    if balanced_obj is not None:
                        return (balanced_obj, "first_brace")
                    break
    return (None, "fail")


def _extract_json_object_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    in_string: bool = False
    escaped: bool = False
    depth: int = 0
    start_index: int = -1
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start_index >= 0:
                fragment: str = text[start_index : index + 1].strip()
                if fragment:
                    candidates.append(fragment)
                start_index = -1
    return candidates


def _merge_language_content_with_multi_mode(
    *,
    language: str,
    videos: List[PlannedVideo],
    config: "AppConfig",
) -> Optional[MergedLanguageContent]:
    if len(videos) <= 1:
        return None
    if config.llm_provider == "gemini":
        return _gemini_merge_title_and_description_once(
            language=language,
            videos=videos,
            config=config,
        )
    if config.llm_provider == "openai":
        attempt: LanguageMergeAttempt = _attempt_openai_merge_with_audit(
            language=language,
            videos=videos,
            config=config,
            attempt_label=f"OPENAI_MERGE_{language.upper()}",
        )
        return attempt.merged
    raise RuntimeError(f"Unsupported llm provider: {config.llm_provider!r}")


def _is_emoji_like_codepoint(codepoint: int) -> bool:
    emoji_ranges: Tuple[Tuple[int, int], ...] = (
        (0x1F1E6, 0x1F1FF),
        (0x1F300, 0x1F5FF),
        (0x1F600, 0x1F64F),
        (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F),
        (0x1F780, 0x1F7FF),
        (0x1F800, 0x1F8FF),
        (0x1F900, 0x1F9FF),
        (0x1FA00, 0x1FA6F),
        (0x1FA70, 0x1FAFF),
        (0x2600, 0x26FF),
        (0x2700, 0x27BF),
        (0x24C2, 0x1F251),
    )
    for range_start, range_end in emoji_ranges:
        if range_start <= codepoint <= range_end:
            return True
    return False


def _strip_emoji(text: str) -> str:
    result_characters: List[str] = []
    for symbol in text:
        codepoint: int = ord(symbol)
        if codepoint in {0x200D, 0xFE0F, 0x20E3}:
            continue
        if _is_emoji_like_codepoint(codepoint):
            continue
        result_characters.append(symbol)
    cleaned_text: str = "".join(result_characters)
    return re.sub(r"\s+", " ", cleaned_text).strip()


def _truncate_to_limit(text: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"

    cut_limit: int = limit - 1
    raw_cut: str = text[:cut_limit]
    last_space_index: int = raw_cut.rfind(" ")
    if last_space_index > 0:
        raw_cut = raw_cut[:last_space_index]
    truncated: str = raw_cut.rstrip()
    if not truncated:
        truncated = text[:cut_limit]
    return f"{truncated}…"


def _truncate_head_tail(text: str, *, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 2:
        return "…"

    body_limit: int = limit - 1
    head_len: int = max(1, int(body_limit * 0.7))
    tail_len: int = max(1, body_limit - head_len)
    if head_len + tail_len > len(text):
        head_len = max(1, min(head_len, len(text) - 1))
        tail_len = max(1, len(text) - head_len)
    head_part: str = text[:head_len].rstrip()
    tail_part: str = text[-tail_len:].lstrip()
    if not head_part or not tail_part:
        return _truncate_to_limit(text, limit=limit)
    return f"{head_part}…{tail_part}"


def _sanitize_title(
    title: str,
    *,
    min_chars: int,
    max_chars: int,
    allow_emoji: bool,
) -> str:
    normalized_title: str = title.strip()
    if not allow_emoji:
        normalized_title = _strip_emoji(normalized_title)
    normalized_title = re.sub(r"\s+", " ", normalized_title).strip()
    normalized_title = _truncate_to_limit(normalized_title, limit=max_chars)
    if not normalized_title:
        return ""
    if len(normalized_title) < min_chars:
        return ""
    return normalized_title


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


def _extract_gemini_reason(text: str) -> Optional[str]:
    compact: str = re.sub(r"\s+", " ", text).strip()
    if (
        "RESOURCE_EXHAUSTED" not in compact
        and "Quota exceeded for metric:" not in compact
    ):
        return None
    status_match: Optional[re.Match[str]] = re.search(
        r"RESOURCE_EXHAUSTED|PERMISSION_DENIED|UNAUTHENTICATED|INVALID_ARGUMENT",
        compact,
    )
    code_match: Optional[re.Match[str]] = re.search(
        r"(?:'code'|\"code\"):\s*(\d+)", compact
    )
    metric_match: Optional[re.Match[str]] = re.search(
        r"Quota exceeded for metric:\s*([A-Za-z0-9._/-]+)",
        compact,
    )
    retry_match: Optional[re.Match[str]] = re.search(
        r"retry in\s*([0-9.]+s?)",
        compact,
        flags=re.IGNORECASE,
    )
    parts: List[str] = []
    if code_match:
        parts.append(f"http={code_match.group(1)}")
    if status_match:
        parts.append(f"status={status_match.group(0)}")
    if metric_match:
        parts.append(f"quota_metric={metric_match.group(1)}")
    if retry_match:
        parts.append(f"retry_in={retry_match.group(1)}")
    if parts:
        return " ".join(parts)
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
    gemini_reason: Optional[str] = _extract_gemini_reason(raw)
    if gemini_reason:
        return gemini_reason
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
        "run_id=%s Config summary: config_processing_mode=%s resolved_processing_mode=%s now_tz_mode=%s llm_provider=%s openai_primary=%s openai_fallback=%s openai_timeout_sec=%.1f openai_max_output_tokens=%d llm_source_desc_max_chars=%d llm_run_if_single_source=%s openai_pre_delay_sec=%.1f gemini_pre_delay_sec=%.1f",
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
        config.gemini_pre_delay_sec,
    )
    LOGGER.info(
        "run_id=%s Config summary: gemini_candidates=%s",
        run_id,
        ", ".join(config.gemini_models),
    )
    LOGGER.info(
        "run_id=%s Config summary: gemini_limits rpm_only GEMINI_LIMIT_RPM=%d min_interval_sec=%.2f dry_run_allowed=%s",
        run_id,
        _load_int_env("GEMINI_LIMIT_RPM", 4, min_value=1),
        60.0 / float(_load_int_env("GEMINI_LIMIT_RPM", 4, min_value=1)),
        _load_bool_env("GEMINI_ALLOW_IN_DRY_RUN", False),
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


class GeminiRateGuard:
    def __init__(self, rpm_limit: int) -> None:
        self._rpm_limit: int = max(1, int(rpm_limit))
        self._min_interval_sec: float = 60.0 / float(self._rpm_limit)
        self._lock = threading.Lock()
        self._next_allowed_monotonic: float = 0.0

    @classmethod
    def from_env(cls) -> "GeminiRateGuard":
        rpm_limit: int = _load_int_env("GEMINI_LIMIT_RPM", 4, min_value=1)
        return cls(rpm_limit=rpm_limit)

    def current_limits(self) -> Tuple[int, float]:
        return (self._rpm_limit, self._min_interval_sec)

    def planned_wait_seconds(self) -> float:
        with self._lock:
            now_monotonic: float = time.monotonic()
            return max(0.0, self._next_allowed_monotonic - now_monotonic)

    def run_with_min_delay(self, call: Callable[[], Any], min_delay_sec: float) -> Any:
        with self._lock:
            now_monotonic: float = time.monotonic()
            guard_wait_seconds: float = max(
                0.0, self._next_allowed_monotonic - now_monotonic
            )
            resolved_wait_seconds: float = max(guard_wait_seconds, min_delay_sec)
            if resolved_wait_seconds > 0.0:
                LOGGER.info(
                    "GeminiRateGuard: sleeping %.2fs (guard_wait=%.2fs pre_delay=%.2fs GEMINI_LIMIT_RPM=%d min_interval_sec=%.2f).",
                    resolved_wait_seconds,
                    guard_wait_seconds,
                    min_delay_sec,
                    self._rpm_limit,
                    self._min_interval_sec,
                )
                time.sleep(resolved_wait_seconds)
            self._next_allowed_monotonic = time.monotonic() + self._min_interval_sec
            return call()

    def run(self, call: Callable[[], Any]) -> Any:
        return self.run_with_min_delay(call, 0.0)


def _get_gemini_rate_guard() -> GeminiRateGuard:
    global _GEMINI_RATE_GUARD
    if _GEMINI_RATE_GUARD is None:
        _GEMINI_RATE_GUARD = GeminiRateGuard.from_env()
    return _GEMINI_RATE_GUARD


def _dump_prompt_to_file(
    *,
    prompt_text: str,
    attempt_label: str,
    model_name: str,
) -> Path:
    output_dir: Path = Path(_load_stg_debug_prompt_dir_from_env())
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_attempt: str = _safe_debug_filename_part(attempt_label)
    safe_model: str = _safe_debug_filename_part(model_name)
    output_path: Path = output_dir / f"{timestamp}__{safe_attempt}__{safe_model}.txt"
    with output_path.open("w", encoding="utf-8") as file_obj:
        file_obj.write(prompt_text)
    return output_path


class _TransientGeminiError(RuntimeError):
    pass


def _get_gemini_client() -> genai.Client:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is not None:
        return _GEMINI_CLIENT
    api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Env var GEMINI_API_KEY is required for Gemini calls.")
    _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def _is_transient_gemini_error(error: Exception) -> bool:
    if _is_gemini_quota_error(error):
        return False
    message: str = str(error).lower()
    class_name: str = error.__class__.__name__.lower()
    transient_markers: Tuple[str, ...] = (
        "429",
        "503",
        "rate limit",
        "resource exhausted",
        "temporarily unavailable",
        "service unavailable",
        "connection reset",
        "connection aborted",
        "ssl",
        "deadline exceeded",
        "unavailable",
    )
    for marker in transient_markers:
        if marker in message:
            return True
    return class_name in {
        "resourceexhausted",
        "serviceunavailable",
        "toomanyrequests",
    }


def _extract_error_status_code(error: Exception) -> Optional[int]:
    candidates: List[Any] = [
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(error, "code", None),
    ]
    response_obj: Any = getattr(error, "response", None)
    if response_obj is not None:
        candidates.append(getattr(response_obj, "status_code", None))
        candidates.append(getattr(response_obj, "status", None))
    resp_obj: Any = getattr(error, "resp", None)
    if resp_obj is not None:
        candidates.append(getattr(resp_obj, "status", None))
        candidates.append(getattr(resp_obj, "status_code", None))
    for candidate in candidates:
        try:
            if candidate is None:
                continue
            code: int = int(candidate)
            if 100 <= code <= 599:
                return code
        except Exception:
            continue
    return None


def _is_gemini_quota_error(error: Exception) -> bool:
    message: str = str(error).lower()
    status_code: Optional[int] = _extract_error_status_code(error)
    quota_markers: Tuple[str, ...] = (
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "ratelimitexceeded",
        "quota exceeded",
    )
    if status_code == 429:
        return True
    if status_code in {400, 403} and any(marker in message for marker in quota_markers):
        return True
    if "429" in message:
        return True
    return any(marker in message for marker in quota_markers)


def _log_gemini_quota_hit(*, model_name: str, error: Exception) -> None:
    rpm_limit, min_interval_sec = _get_gemini_rate_guard().current_limits()
    status_code: Optional[int] = _extract_error_status_code(error)
    status_marker: str = (
        f"http={status_code}" if status_code is not None else "http=unknown"
    )
    LOGGER.error(
        "!!! GEMINI RATE LIMIT / QUOTA HIT (%s RESOURCE_EXHAUSTED/QUOTA) !!! model=%s GEMINI_LIMIT_RPM=%d min_interval_sec=%.2f reason=%s",
        status_marker,
        model_name,
        rpm_limit,
        min_interval_sec,
        _summarize_error(error),
    )


def _call_gemini_generate_once(
    *,
    prompt_text: str,
    model_name: str,
    timeout_seconds: float,
    attempt_label: str,
    debug: bool,
    temperature_override: Optional[float] = None,
    max_tokens_override: Optional[int] = None,
    force_json: bool = False,
    response_schema: Optional[Any] = None,
    pre_delay_sec: float = 0.0,
) -> str:
    if debug:
        _safe_console_print(f"LLM attempt={attempt_label} model={model_name}")
        prompt_to_file: bool = _load_stg_debug_prompt_to_file_from_env()
        if prompt_to_file:
            prompt_file_path: Path = _dump_prompt_to_file(
                prompt_text=prompt_text,
                attempt_label=attempt_label,
                model_name=model_name,
            )
            _safe_console_print(f"LLM prompt saved: {prompt_file_path}")
            preview_lines: List[str] = prompt_text.splitlines()[:20]
            if preview_lines:
                _safe_console_print(
                    f"----- LLM PROMPT PREVIEW ({attempt_label} {model_name}) -----"
                )
                _safe_console_print("\n".join(preview_lines))
                _safe_console_print(
                    f"----- LLM PROMPT PREVIEW END ({attempt_label} {model_name}) -----"
                )
        else:
            _safe_console_print(
                f"----- LLM PROMPT BEGIN ({attempt_label} {model_name}) -----"
            )
            _safe_console_print(prompt_text)
            _safe_console_print(
                f"----- LLM PROMPT END ({attempt_label} {model_name}) -----"
            )

    config_kwargs: Dict[str, Any] = {}
    if temperature_override is not None:
        config_kwargs["temperature"] = float(temperature_override)
    if max_tokens_override is not None and int(max_tokens_override) > 0:
        config_kwargs["max_output_tokens"] = int(max_tokens_override)
    if force_json:
        config_kwargs["response_mime_type"] = "application/json"
        if response_schema is not None:
            config_kwargs["response_schema"] = response_schema
    generation_config = types.GenerateContentConfig(**config_kwargs)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(_TransientGeminiError),
        reraise=True,
    )
    def _invoke_with_retry() -> Any:
        def _call_api() -> Any:
            return _get_gemini_rate_guard().run_with_min_delay(
                lambda: _get_gemini_client().models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=generation_config,
                ),
                max(0.0, float(pre_delay_sec)),
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_api)
            try:
                return future.result(timeout=float(timeout_seconds))
            except FuturesTimeoutError as error:
                future.cancel()
                raise RuntimeError(
                    f"Gemini request timed out after {timeout_seconds} seconds."
                ) from error
            except Exception as error:
                if _is_gemini_quota_error(error):
                    _log_gemini_quota_hit(model_name=model_name, error=error)
                if _is_transient_gemini_error(error):
                    raise _TransientGeminiError(str(error)) from error
                raise

    response: Any = _invoke_with_retry()
    raw_output_text: str = str(getattr(response, "text", "") or "").strip()
    if not raw_output_text:
        raise RuntimeError("Gemini response is empty.")
    _safe_console_print(
        f"----- LLM RESPONSE BEGIN ({attempt_label} {model_name}) -----"
    )
    _safe_console_print(raw_output_text)
    _safe_console_print(f"----- LLM RESPONSE END ({attempt_label} {model_name}) -----")
    return raw_output_text


def _gemini_startup_ping(config: "AppConfig", model_name: str) -> Dict[str, str]:
    prompt_text: str = _render_template(
        config.templates.llm_startup_ping_prompt,
        {
            "model_name": model_name,
            "utc_now": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
    )
    response_text: str = _call_gemini_generate_once(
        prompt_text=prompt_text,
        model_name=model_name,
        timeout_seconds=config.gemini_timeout_sec,
        attempt_label="GEMINI_STARTUP_PING",
        debug=False,
        force_json=True,
    )
    LOGGER.info("Gemini startup ping raw response: %s", response_text)
    try:
        payload: Any = json.loads(response_text)
    except Exception as error:
        raise RuntimeError(f"Gemini startup ping returned non-JSON response: {error}")
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini startup ping JSON root must be an object.")

    status: str = str(payload.get("status", "")).strip().lower()
    provider: str = str(payload.get("provider", "")).strip().lower()
    model: str = str(payload.get("model", "")).strip()
    version: str = str(payload.get("version", "")).strip()
    if status != "ok":
        raise RuntimeError(f"Gemini startup ping status is not ok: status={status!r}")
    if provider and provider != "gemini":
        raise RuntimeError(
            f"Gemini startup ping provider mismatch: provider={provider!r}"
        )
    if not model:
        raise RuntimeError("Gemini startup ping returned empty model field.")
    if not version:
        raise RuntimeError("Gemini startup ping returned empty version field.")
    return {
        "status": status,
        "provider": provider or "gemini",
        "model": model,
        "version": version,
    }


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
            item for item in videos if _planned_video_block_language(item) == language
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
        preview_name_template: str,
        doc_title_template: str,
        language_codes_json: str,
        max_filename_stem: int,
    ) -> None:
        self._local_image_dir_template: str = local_image_dir_template
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

    def build_doc_title(self, date_key: str, created_at: datetime) -> str:
        creation_stamp: str = created_at.strftime("%H%M_%d%m%y")
        return _render_template(
            self._doc_title_template,
            {"date": date_key, "creation_stamp": creation_stamp},
        )

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
        file_language_code: str = self._language_codes.get(language, "OT")
        safe_title: str = _build_safe_entity_name(title)
        raw_stem: str = _render_template(
            self._preview_name_template,
            {
                "index": language_index,
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


@dataclass(frozen=True)
class AppConfig:
    telegram_bot_token: str
    telegram_chat_id: str
    telegram_enabled: bool
    google_enabled: bool
    google_service_account_path: Optional[Path]
    google_drive_folder_id: Optional[str]
    google_drive_preview_folder_id: Optional[str]
    google_drive_preview_path_template: str
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
    processing_mode: str
    now_tz_mode: str
    llm_provider: str
    openai_model_primary: str
    openai_model_fallback: str
    openai_timeout_sec: float
    openai_max_output_tokens: int
    openai_pre_delay_sec: float
    llm_source_desc_max_chars: int
    llm_run_if_single_source: bool
    gemini_pre_delay_sec: float
    gemini_model: str
    gemini_models: Tuple[str, ...]
    gemini_timeout_sec: float
    gemini_temperature: float
    gemini_max_output_tokens: int
    preview_filename_max_stem: int
    stg_templates_path: Path
    telegram_use_audit: bool
    templates: AppTemplates


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
        self._name_builder = NamePathBuilder(
            local_image_dir_template=config.local_image_dir_template,
            preview_name_template=config.templates.files_preview_name_template,
            doc_title_template=config.templates.files_doc_title_template,
            language_codes_json=config.templates.files_language_codes_json,
            max_filename_stem=config.preview_filename_max_stem,
        )

    def run_batch(self, dry_run: bool, *, processing_mode: str, run_id: str) -> None:
        if not self._config.google_enabled:
            raise RuntimeError("Для batch режима GOOGLE_ENABLED должен быть включен.")
        resolved_processing_mode: str = _normalize_processing_mode(
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
            _resolve_logger_name_meta()
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
        LOGGER.info("Gemini request: startup ping skipped to preserve quota.")
        LOGGER.info(
            "Gemini request models.candidates (config only): %s",
            ", ".join(self._config.gemini_models),
        )
        LOGGER.info(
            "Gemini request model.timeout_sec: %s", self._config.gemini_timeout_sec
        )
        LOGGER.info(
            "Gemini request model.temperature: %s", self._config.gemini_temperature
        )
        LOGGER.info(
            "Gemini request model.max_output_tokens: %d",
            self._config.gemini_max_output_tokens,
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
        LOGGER.info(
            "Gemini request pre_delay_sec=%.1f",
            self._config.gemini_pre_delay_sec,
        )
        merge_config: AppConfig = self._config
        gemini_api_key_present: bool = bool(os.getenv("GEMINI_API_KEY", "").strip())
        gpt_api_key_present: bool = bool(os.getenv("GPT_API_KEY", "").strip())
        llm_allow_in_dry_run: bool = _load_bool_env(
            "STG_LLM_ALLOW_IN_DRY_RUN",
            _load_bool_env("GEMINI_ALLOW_IN_DRY_RUN", False),
        )
        llm_merge_enabled: bool = False
        merge_mode_enabled: bool = resolved_processing_mode == "merge"
        if merge_mode_enabled:
            if self._config.llm_provider == "gemini":
                llm_merge_enabled = bool(self._config.gemini_model.strip()) and (
                    gemini_api_key_present
                )
            elif self._config.llm_provider == "openai":
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

        now_filter_tz: timezone | ZoneInfo = _now_filter_timezone(
            self._config.now_tz_mode,
            self._kiev_tz,
        )
        now_for_filter: datetime = datetime.now(now_filter_tz)
        LOGGER.info(
            "now_tz=%s now=%s",
            self._config.now_tz_mode,
            now_for_filter.isoformat(),
        )
        merge_semantics: str = _merge_semantics_from_env()
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
                base_block_lang: str = _planned_video_block_language(base_video)
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
        processed = _deduplicate_planned_videos_within_slot_language(processed)
        if not processed:
            _log_link_normalization_report(
                normalization_candidates=link_normalization_candidates,
                writeback_enabled=sheets_link_writeback_enabled,
            )
            LOGGER.warning("No videos to process after filtering.")
            return

        processed_by_slot: Dict[str, List[PlannedVideo]] = {}
        for item in processed:
            processed_by_slot.setdefault(_planned_video_slot_key(item), []).append(item)
        for slot_key in sorted(processed_by_slot.keys()):
            slot_items: List[PlannedVideo] = processed_by_slot[slot_key]
            date_key_for_slot: str = slot_items[0].date_key
            slot_lang_counts: Dict[str, int] = {
                language: sum(
                    1
                    for slot_item in slot_items
                    if _planned_video_block_language(slot_item) == language
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
                slots_by_time.setdefault(_planned_video_time_key(item), []).append(item)

            slot_results: List[Dict[str, Any]] = []
            for slot_time_key in sorted(slots_by_time.keys()):
                slot_key: str = f"{date_key}_{slot_time_key}"
                day_videos: List[PlannedVideo] = sorted(
                    slots_by_time[slot_time_key],
                    key=lambda item: (
                        _language_index(_planned_video_block_language(item)),
                        item.scheduled_at_kiev.time(),
                        item.row_number,
                    ),
                )
                slot_lang_counts: Dict[str, int] = {
                    language: sum(
                        1
                        for item in day_videos
                        if _planned_video_block_language(item) == language
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
                        if _planned_video_block_language(item) == language
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
                            if self._config.llm_provider == "gemini":
                                merge_attempt = _attempt_gemini_single_source_translate_with_audit(
                                    language=language,
                                    videos=language_items_for_merge,
                                    config=merge_config,
                                    attempt_label=f"GEMINI_TRANSLATE_{language.upper()}",
                                )
                            elif self._config.llm_provider == "openai":
                                merge_attempt = _attempt_openai_single_source_translate_with_audit(
                                    language=language,
                                    videos=language_items_for_merge,
                                    config=merge_config,
                                    attempt_label=f"OPENAI_TRANSLATE_{language.upper()}",
                                )
                            else:
                                raise RuntimeError(
                                    f"Unsupported llm provider: {self._config.llm_provider!r}"
                                )
                        else:
                            if self._config.llm_provider == "gemini":
                                merge_attempt = _attempt_gemini_merge_with_audit(
                                    language=language,
                                    videos=language_items_for_merge,
                                    config=merge_config,
                                    attempt_label=f"GEMINI_MERGE_{language.upper()}",
                                )
                            elif self._config.llm_provider == "openai":
                                merge_attempt = _attempt_openai_merge_with_audit(
                                    language=language,
                                    videos=language_items_for_merge,
                                    config=merge_config,
                                    attempt_label=f"OPENAI_MERGE_{language.upper()}",
                                )
                            else:
                                raise RuntimeError(
                                    f"Unsupported llm provider: {self._config.llm_provider!r}"
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
                                merged_content_value = _enforce_merged_paragraphs_for_group(
                                    provider_name=self._config.llm_provider,
                                    merged_content=merged_content_value,
                                    videos=language_items_for_merge,
                                    paragraph_limit=self._config.llm_source_desc_max_chars,
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
                time_display: str = _format_time_key_for_display(
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
                        merged_title = _numbered_original_titles(
                            slot_language_items
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
                        time_display: str = _format_time_key_for_display(
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
        merged_map: Dict[str, MergedLanguageContent] = merged_content_by_language or {}
        merge_attempt_map: Dict[str, LanguageMergeAttempt] = (
            merge_audit_by_language or {}
        )
        for video in day_videos:
            grouped[_planned_video_block_language(video)].append(video)
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
                merged_block_text: str = _build_telegram_language_merged_block(
                    language=language,
                    videos=items,
                    merged_content=merged_for_language,
                    merge_attempt=merge_attempt_map.get(language),
                    config=self._config,
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
                nomerge_block_text: str = _build_telegram_language_nomerge_block(
                    language=language,
                    videos=items,
                    config=self._config,
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

        key_form_reminder: str = _build_telegram_key_form_reminder(
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

        post_header_message: str = _render_template(
            self._config.templates.telegram_post_header,
            {
                "symbol_broadcast": self._config.telegram_symbol_broadcast,
                "date": header_context["date"],
                "time_kiev": _telegram_safe_time(header_context["time_kiev"]),
            },
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
        message_text: str = _render_template(
            self._config.templates.common_single_mode_message,
            {
                "title": metadata.title,
                "language": language,
                "description": (metadata.description or _no_description_text()),
                "url": metadata.url,
            },
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


def _must_get_template_value(payload: Dict[str, Any], dotted_key: str) -> str:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"Template key is missing: {dotted_key}")
        current = current[key]
    value: str = str(current or "").strip()
    if not value:
        raise RuntimeError(f"Template key is empty: {dotted_key}")
    return value


def _must_get_template_object(payload: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"Template key is missing: {dotted_key}")
        current = current[key]
    return current


def _load_templates_from_path(path: Path) -> AppTemplates:
    if not path.exists():
        raise RuntimeError(f"Templates file does not exist: {path}")
    raw_text: str = path.read_text(encoding="utf-8")
    try:
        payload: Any = yaml.safe_load(raw_text)
    except Exception as error:
        raise RuntimeError(f"Failed to parse templates file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"Templates root must be a mapping in {path}")
    google_doc_table_labels_raw: Any = _must_get_template_object(
        payload, "google_doc.table_labels"
    )
    google_doc_bold_line_prefixes_raw: Any = _must_get_template_object(
        payload, "google_doc.bold_line_prefixes"
    )
    google_doc_language_headings_raw: Any = _must_get_template_object(
        payload, "google_doc.language_headings"
    )
    llm_language_names_raw: Any = _must_get_template_object(
        payload, "llm.language_names"
    )
    files_language_codes_raw: Any = _must_get_template_object(
        payload, "files.language_codes"
    )
    try:
        google_doc_table_labels_json: str = json.dumps(
            google_doc_table_labels_raw, ensure_ascii=False
        )
        google_doc_bold_line_prefixes_json: str = json.dumps(
            google_doc_bold_line_prefixes_raw, ensure_ascii=False
        )
        google_doc_language_headings_json: str = json.dumps(
            google_doc_language_headings_raw, ensure_ascii=False
        )
        llm_language_names_json: str = json.dumps(
            llm_language_names_raw, ensure_ascii=False
        )
        files_language_codes_json: str = json.dumps(
            files_language_codes_raw, ensure_ascii=False
        )
    except Exception as error:
        raise RuntimeError(f"Templates JSON conversion failed: {error}") from error
    return AppTemplates(
        google_doc_header=_must_get_template_value(payload, "google_doc.header"),
        google_doc_table_labels_json=google_doc_table_labels_json,
        google_doc_bold_line_prefixes_json=google_doc_bold_line_prefixes_json,
        google_doc_language_headings_json=google_doc_language_headings_json,
        telegram_header=_must_get_template_value(payload, "telegram.header"),
        telegram_language_block=_must_get_template_value(
            payload, "telegram.language_block"
        ),
        telegram_language_merged_block=_must_get_template_value(
            payload, "telegram.language_merged_block"
        ),
        telegram_key_form_reminder=_must_get_template_value(
            payload, "telegram.key_form_reminder"
        ),
        telegram_sparkle_separator=_must_get_template_value(
            payload, "telegram.sparkle_separator"
        ),
        telegram_post_header=_must_get_template_value(payload, "telegram.post_header"),
        telegram_language_digest_header=_must_get_template_value(
            payload, "telegram.language_digest_header"
        ),
        llm_merge_title_description_prompt=_must_get_template_value(
            payload, "llm.merge_title_description_prompt"
        ),
        llm_startup_ping_prompt=_must_get_template_value(
            payload, "llm.startup_ping_prompt"
        ),
        llm_language_names_json=llm_language_names_json,
        files_preview_name_template=_must_get_template_value(
            payload, "files.preview_name_template"
        ),
        files_doc_title_template=_must_get_template_value(
            payload, "files.doc_title_template"
        ),
        files_language_codes_json=files_language_codes_json,
        common_no_description_text=_must_get_template_value(
            payload, "common.no_description_text"
        ),
        common_single_mode_message=_must_get_template_value(
            payload, "common.single_mode_message"
        ),
    )


def _must_get_setting_value(payload: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(f"Config key is missing: {dotted_key}")
        current = current[key]
    return current


def _setting_as_bool(payload: Dict[str, Any], dotted_key: str) -> bool:
    value: Any = _must_get_setting_value(payload, dotted_key)
    if isinstance(value, bool):
        return value
    raw: str = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Config key must be bool: {dotted_key}={value!r}")


def _setting_as_int(payload: Dict[str, Any], dotted_key: str) -> int:
    value: Any = _must_get_setting_value(payload, dotted_key)
    try:
        return int(value)
    except Exception as error:
        raise RuntimeError(f"Config key must be int: {dotted_key}={value!r}") from error


def _setting_as_float(payload: Dict[str, Any], dotted_key: str) -> float:
    value: Any = _must_get_setting_value(payload, dotted_key)
    try:
        return float(value)
    except Exception as error:
        raise RuntimeError(
            f"Config key must be float: {dotted_key}={value!r}"
        ) from error


def _setting_as_str(payload: Dict[str, Any], dotted_key: str) -> str:
    value: Any = _must_get_setting_value(payload, dotted_key)
    return str(value).strip()


def _setting_as_optional_str(payload: Dict[str, Any], dotted_key: str) -> Optional[str]:
    current: Any = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    value: str = str(current).strip()
    return value or None


def _resolve_gemini_models(app_settings: Dict[str, Any]) -> Tuple[str, ...]:
    # Quota protection: runtime is pinned to a single model.
    return ("gemini-2.5-flash",)


def _resolve_llm_provider_from_env() -> str:
    raw_value: str = os.getenv("STG_LLM_PROVIDER", "gpt").strip().lower()
    if raw_value in {"gpt", "openai"}:
        return "openai"
    if raw_value == "gemini":
        return "gemini"
    LOGGER.warning(
        "Unknown STG_LLM_PROVIDER=%r. Falling back to OpenAI provider.",
        raw_value,
    )
    return "openai"


def _normalize_processing_mode(value: str, *, source: str) -> str:
    raw_value: str = str(value or "").strip().lower()
    normalized_input: str = raw_value[2:] if raw_value.startswith("--") else raw_value
    alias_to_mode: Dict[str, str] = {
        "merge": "merge",
        "nomerge": "nomerge",
        "no-merge": "nomerge",
        "no_merge": "nomerge",
    }
    normalized_mode: Optional[str] = alias_to_mode.get(normalized_input)
    if normalized_mode is None:
        raise RuntimeError(
            f"Unsupported {source}={value!r}. Supported modes: merge, nomerge."
        )
    return normalized_mode


def _normalize_now_tz_mode(value: str, *, source: str) -> str:
    raw_value: str = str(value or "").strip().lower()
    alias_to_mode: Dict[str, str] = {
        "kyiv": "kyiv",
        "kiev": "kyiv",
        "fixed_gmt_plus_2": "fixed_gmt_plus_2",
        "fixedgmtplus2": "fixed_gmt_plus_2",
        "gmt+2": "fixed_gmt_plus_2",
        "utc+2": "fixed_gmt_plus_2",
    }
    normalized_mode: Optional[str] = alias_to_mode.get(raw_value)
    if normalized_mode is None:
        raise RuntimeError(
            f"Unsupported {source}={value!r}. Supported: kyiv, fixed_gmt_plus_2."
        )
    return normalized_mode


def _now_filter_timezone(now_tz_mode: str, kiev_tz: ZoneInfo) -> timezone | ZoneInfo:
    if now_tz_mode == "fixed_gmt_plus_2":
        return timezone(timedelta(hours=2))
    return kiev_tz


def _validate_app_settings(payload: Dict[str, Any]) -> None:
    checks: List[Tuple[str, str]] = [
        ("templates_path", "str"),
        ("processing.mode", "str"),
        ("google.enabled", "bool"),
        ("google.drive_folder_id", "str"),
        ("google.drive_preview_folder_id", "str"),
        ("google.drive_preview_path_template", "str"),
        ("google.doc_share_mode", "str"),
        ("google.sheets_id", "str"),
        ("google.sheets_range", "str"),
        ("google.form_url", "str"),
        ("google.contacts", "str"),
        ("paths.local_image_dir_template", "str"),
        ("timezones.kiev", "str"),
        ("timezones.cet", "str"),
        ("telegram.enabled", "bool"),
        ("telegram.use_audit", "bool"),
        ("telegram.symbol_separator", "str"),
        ("telegram.separator_repeat_count", "int"),
        ("telegram.symbol_broadcast", "str"),
        ("telegram.symbol_alert", "str"),
        ("telegram.symbol_form", "str"),
        ("telegram.symbol_description", "str"),
        ("telegram.symbol_pin", "str"),
        ("telegram.symbol_done", "str"),
        ("telegram.flag_uk", "str"),
        ("telegram.flag_en", "str"),
        ("telegram.flag_ru", "str"),
        ("telegram.flag_other", "str"),
        ("telegram.flag_repeat_count", "int"),
        ("telegram.language_name_uk", "str"),
        ("telegram.language_name_en", "str"),
        ("telegram.language_name_ru", "str"),
        ("telegram.language_name_other", "str"),
        ("gemini.model", "str"),
        ("gemini.timeout_sec", "float"),
        ("gemini.temperature", "float"),
        ("gemini.max_output_tokens", "int"),
        ("files.preview_filename_max_stem", "int"),
    ]
    errors: List[str] = []
    for key, key_type in checks:
        try:
            if key_type == "bool":
                _setting_as_bool(payload, key)
            elif key_type == "int":
                _setting_as_int(payload, key)
            elif key_type == "float":
                _setting_as_float(payload, key)
            else:
                value: str = _setting_as_str(payload, key)
                if not value:
                    errors.append(f"{key}: must not be empty")
        except Exception as error:
            errors.append(f"{key}: {error}")
    if errors:
        joined: str = "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(
            "Invalid app config. Please fix these keys:\n"
            f"{joined}\n"
            "You can start from app_config.example.yaml."
        )


def _load_app_settings_from_path(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"App config file does not exist: {path}")
    raw_text: str = path.read_text(encoding="utf-8")
    try:
        payload: Any = yaml.safe_load(raw_text)
    except Exception as error:
        raise RuntimeError(
            f"Failed to parse app config file {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"App config root must be a mapping in {path}")
    app_payload: Any = payload.get("app")
    if not isinstance(app_payload, dict):
        raise RuntimeError(f"App config must contain mapping key 'app' in {path}")
    app_settings: Dict[str, Any] = cast(Dict[str, Any], app_payload)
    _validate_app_settings(app_settings)
    return app_settings


def _load_config_from_env() -> AppConfig:
    app_config_path: Path = Path(
        os.getenv("APP_CONFIG_PATH", "app_config.yaml").strip() or "app_config.yaml"
    )
    app_settings: Dict[str, Any] = _load_app_settings_from_path(app_config_path)
    _warn_ignored_google_auth_mode_in_config(app_settings)

    stg_templates_path: Path = Path(_setting_as_str(app_settings, "templates_path"))
    default_templates_path: Path = Path("templates.yaml")
    templates: AppTemplates
    try:
        templates = _load_templates_from_path(stg_templates_path)
    except Exception as error:
        if stg_templates_path == default_templates_path:
            raise
        LOGGER.warning(
            "Templates loading failed for %s (%s). Falling back to %s.",
            stg_templates_path,
            error,
            default_templates_path,
        )
        templates = _load_templates_from_path(default_templates_path)
    globals()["_ACTIVE_TEMPLATES"] = templates

    configured_gemini_model: str = _setting_as_str(app_settings, "gemini.model")
    gemini_model: str = "gemini-2.5-flash"
    if configured_gemini_model and configured_gemini_model != gemini_model:
        LOGGER.warning(
            "Config gemini.model=%s is ignored. Using fixed model=%s.",
            configured_gemini_model,
            gemini_model,
        )
    gemini_models: Tuple[str, ...] = _resolve_gemini_models(app_settings)
    processing_mode: str = _normalize_processing_mode(
        _setting_as_str(app_settings, "processing.mode"),
        source="app.processing.mode",
    )
    config_now_tz_raw: str = (
        _setting_as_optional_str(app_settings, "processing.now_tz") or ""
    )
    env_now_tz_raw: str = os.getenv("STG_NOW_TZ", "").strip()
    if env_now_tz_raw and config_now_tz_raw and env_now_tz_raw != config_now_tz_raw:
        LOGGER.warning(
            "STG_NOW_TZ overrides processing.now_tz: config=%r env=%r",
            config_now_tz_raw,
            env_now_tz_raw,
        )
    now_tz_mode_input: str = env_now_tz_raw or config_now_tz_raw or "kyiv"
    now_tz_mode: str = _normalize_now_tz_mode(
        now_tz_mode_input,
        source="now timezone mode",
    )
    llm_provider: str = _resolve_llm_provider_from_env()
    openai_model_primary: str = (
        os.getenv("STG_OPENAI_MODEL_PRIMARY", "").strip() or "gpt-5-nano"
    )
    openai_model_fallback: str = (
        os.getenv("STG_OPENAI_MODEL_FALLBACK", "").strip() or "gpt-5-mini"
    )
    openai_timeout_sec: float = _load_float_env(
        "STG_OPENAI_TIMEOUT_SEC", 120.0, min_value=1.0
    )
    openai_max_output_tokens: int = _load_int_env(
        "STG_OPENAI_MAX_OUTPUT_TOKENS", 1000, min_value=1
    )
    openai_pre_delay_sec: float = _load_float_env(
        "STG_OPENAI_PRE_DELAY_SEC", 5.0, min_value=0.0
    )
    llm_source_desc_max_chars: int = _load_int_env(
        "STG_LLM_SOURCE_DESC_MAX_CHARS", 2000, min_value=200
    )
    llm_run_if_single_source: bool = _load_bool_env(
        "STG_LLM_RUN_IF_SINGLE_SOURCE", False
    )
    gemini_pre_delay_sec: float = _load_float_env(
        "STG_GEMINI_PRE_DELAY_SEC", 20.0, min_value=0.0
    )
    google_auth_mode: str = _load_google_auth_mode_from_env()
    google_service_account_path_raw: str = os.getenv(
        "GOOGLE_SERVICE_ACCOUNT_PATH", ""
    ).strip()
    if (
        os.getenv("GOOGLE_CREDENTIALS_PATH", "").strip()
        or os.getenv("GOOGLE_TOKEN_PATH", "").strip()
    ):
        LOGGER.warning(
            "GOOGLE_CREDENTIALS_PATH/GOOGLE_TOKEN_PATH are deprecated and ignored. "
            "Use GOOGLE_OAUTH_CREDENTIALS_PATH/GOOGLE_OAUTH_TOKEN_PATH."
        )
    google_enabled: bool = _setting_as_bool(app_settings, "google.enabled")
    config_kwargs: Dict[str, Any] = {
        "telegram_bot_token": _must_get_env("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": _must_get_env("TELEGRAM_CHAT_ID"),
        "telegram_enabled": _setting_as_bool(app_settings, "telegram.enabled"),
        "google_enabled": google_enabled,
        "google_service_account_path": (
            Path(google_service_account_path_raw)
            if google_service_account_path_raw
            else None
        ),
        "google_drive_folder_id": (
            _setting_as_str(app_settings, "google.drive_folder_id") or None
        ),
        "google_drive_preview_folder_id": (
            _setting_as_str(app_settings, "google.drive_preview_folder_id")
            or _setting_as_str(app_settings, "google.drive_folder_id")
            or None
        ),
        "google_drive_preview_path_template": _setting_as_str(
            app_settings, "google.drive_preview_path_template"
        ),
        "google_doc_share_mode": _normalize_google_doc_share_mode(
            _setting_as_str(app_settings, "google.doc_share_mode")
        ),
        "google_sheets_id": _setting_as_str(app_settings, "google.sheets_id"),
        "google_sheets_range": _setting_as_str(app_settings, "google.sheets_range"),
        "google_form_url": _setting_as_str(app_settings, "google.form_url"),
        "google_contacts": _setting_as_str(app_settings, "google.contacts"),
        "local_image_dir_template": _setting_as_str(
            app_settings, "paths.local_image_dir_template"
        ),
        "timezone_kiev": _setting_as_str(app_settings, "timezones.kiev"),
        "timezone_cet": _setting_as_str(app_settings, "timezones.cet"),
        "telegram_symbol_separator": _setting_as_str(
            app_settings, "telegram.symbol_separator"
        ),
        "telegram_separator_repeat_count": _setting_as_int(
            app_settings, "telegram.separator_repeat_count"
        ),
        "telegram_symbol_broadcast": _setting_as_str(
            app_settings, "telegram.symbol_broadcast"
        ),
        "telegram_symbol_alert": _setting_as_str(app_settings, "telegram.symbol_alert"),
        "telegram_symbol_form": _setting_as_str(app_settings, "telegram.symbol_form"),
        "telegram_symbol_description": _setting_as_str(
            app_settings, "telegram.symbol_description"
        ),
        "telegram_symbol_pin": _setting_as_str(app_settings, "telegram.symbol_pin"),
        "telegram_symbol_done": _setting_as_str(app_settings, "telegram.symbol_done"),
        "telegram_flag_uk": _setting_as_str(app_settings, "telegram.flag_uk"),
        "telegram_flag_en": _setting_as_str(app_settings, "telegram.flag_en"),
        "telegram_flag_ru": _setting_as_str(app_settings, "telegram.flag_ru"),
        "telegram_flag_other": _setting_as_str(app_settings, "telegram.flag_other"),
        "telegram_flag_repeat_count": _setting_as_int(
            app_settings, "telegram.flag_repeat_count"
        ),
        "telegram_language_name_uk": _setting_as_str(
            app_settings, "telegram.language_name_uk"
        ),
        "telegram_language_name_en": _setting_as_str(
            app_settings, "telegram.language_name_en"
        ),
        "telegram_language_name_ru": _setting_as_str(
            app_settings, "telegram.language_name_ru"
        ),
        "telegram_language_name_other": _setting_as_str(
            app_settings, "telegram.language_name_other"
        ),
        "processing_mode": processing_mode,
        "now_tz_mode": now_tz_mode,
        "llm_provider": llm_provider,
        "openai_model_primary": openai_model_primary,
        "openai_model_fallback": openai_model_fallback,
        "openai_timeout_sec": openai_timeout_sec,
        "openai_max_output_tokens": openai_max_output_tokens,
        "openai_pre_delay_sec": openai_pre_delay_sec,
        "llm_source_desc_max_chars": llm_source_desc_max_chars,
        "llm_run_if_single_source": llm_run_if_single_source,
        "gemini_pre_delay_sec": gemini_pre_delay_sec,
        "gemini_model": gemini_model,
        "gemini_models": gemini_models,
        "gemini_timeout_sec": _setting_as_float(app_settings, "gemini.timeout_sec"),
        "gemini_temperature": _setting_as_float(app_settings, "gemini.temperature"),
        "gemini_max_output_tokens": _setting_as_int(
            app_settings, "gemini.max_output_tokens"
        ),
        "preview_filename_max_stem": _setting_as_int(
            app_settings, "files.preview_filename_max_stem"
        ),
        "stg_templates_path": stg_templates_path,
        "telegram_use_audit": _setting_as_bool(app_settings, "telegram.use_audit"),
        "templates": templates,
    }
    _validate_google_service_account_path_requirement(
        google_enabled=google_enabled,
        google_auth_mode=google_auth_mode,
        service_account_path_raw=google_service_account_path_raw,
        service_account_path=cast(
            Optional[Path], config_kwargs["google_service_account_path"]
        ),
    )
    return AppConfig(**config_kwargs)


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


def _normalize_google_doc_share_mode(mode_raw_input: str) -> str:
    mode_raw: str = str(mode_raw_input or "").strip().lower()
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
            f"Unsupported google.doc_share_mode={mode_raw!r}. Supported: {supported_modes}"
        )
    return normalized_mode


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


def _load_google_auth_mode_from_env() -> str:
    raw_value: str = os.getenv("GOOGLE_AUTH_MODE", "").strip().lower()
    if not raw_value:
        return "oauth"
    if raw_value in {"oauth", "service_account"}:
        return raw_value
    raise RuntimeError(
        "Invalid GOOGLE_AUTH_MODE. Allowed values: 'oauth', 'service_account'. "
        f"Current value: {raw_value!r}"
    )


def _load_google_service_account_path() -> Optional[Path]:
    raw_value: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH", "").strip()
    if not raw_value:
        return None
    return Path(raw_value)


def _load_google_oauth_credentials_path() -> Path:
    raw_value: str = os.getenv("GOOGLE_OAUTH_CREDENTIALS_PATH", "").strip()
    return Path(raw_value or "credentials.json")


def _load_google_oauth_token_path() -> Path:
    raw_value: str = os.getenv("GOOGLE_OAUTH_TOKEN_PATH", "").strip()
    return Path(raw_value or "token.json")


def _warn_ignored_google_auth_mode_in_config(app_settings: Dict[str, Any]) -> None:
    google_payload: Any = app_settings.get("google")
    if not isinstance(google_payload, dict):
        return
    for key, value in google_payload.items():
        normalized_key: str = str(key or "").strip().lower()
        if "auth" in normalized_key and "mode" in normalized_key:
            LOGGER.warning(
                "auth_mode in config is ignored; use GOOGLE_AUTH_MODE env; ignored_value=%r",
                value,
            )
            return


def _validate_google_service_account_path_requirement(
    *,
    google_enabled: bool,
    google_auth_mode: str,
    service_account_path_raw: str,
    service_account_path: Optional[Path],
) -> None:
    if not google_enabled:
        return
    if google_auth_mode != "service_account":
        return
    resolved_path: str = str(service_account_path_raw or "").strip()
    if not resolved_path or service_account_path is None:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}"
        )
    if not service_account_path.exists():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}. File not found."
        )
    if not service_account_path.is_file():
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}. Path is not a file."
        )
    try:
        with service_account_path.open("r", encoding="utf-8-sig") as file_obj:
            file_obj.read(1)
    except Exception as error:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_PATH is required when google integration is enabled. "
            f"Resolved path: {resolved_path!r}. File is not readable: {_summarize_error(error)}"
        ) from error


# def _setup_logging(debug: bool) -> None:
#     logging.basicConfig(
#         level=logging.DEBUG if debug else logging.INFO,
#         format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     )


def _resolve_log_dir() -> Path:
    env_log_dir_raw: str = str(os.getenv("STG_LOG_DIR", "") or "").strip()
    script_dir: Path = Path(__file__).resolve().parent
    if env_log_dir_raw:
        candidate_path: Path = Path(env_log_dir_raw)
        if not candidate_path.is_absolute():
            candidate_path = (script_dir / candidate_path).resolve()
        return candidate_path
    windows_default: Path = Path(r"D:\_projects\restreamer\logs")
    if windows_default.exists():
        return windows_default
    return script_dir / "logs"


def _setup_logging(debug: bool) -> None:
    logger_name: str = _resolve_logger_name()
    configured_level_name: str = (
        str(os.getenv("STG_LOG_LEVEL", "INFO") or "INFO").strip().upper()
    )
    configured_level: int = getattr(logging, configured_level_name, logging.INFO)
    app_level: int = logging.DEBUG if debug else configured_level

    log_dir_path: Path = _resolve_log_dir()
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_filename: str = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_restreamer.log"
    log_file_path: Path = log_dir_path / log_filename

    formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler: logging.StreamHandler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(log_file_path),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    app_logger: logging.Logger = logging.getLogger(logger_name)
    app_logger.setLevel(app_level)
    root_logger.info(
        "Logging initialized level=%s file=%s",
        logging.getLevelName(app_level),
        str(log_file_path),
    )
    # Env examples:
    # STG_LOG_DIR=D:\_projects\streamertg\logs
    # OPENAI_ADMIN_KEY=...
    # OPENAI_MONTHLY_BUDGET_USD=100

    # Шумные/опасные логгеры — прижимаем
    logging.getLogger("requests_oauthlib").setLevel(logging.WARNING)
    logging.getLogger("oauthlib").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient.http").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: Sequence[str]) -> int:
    load_dotenv()
    globals()["LOGGER"] = logging.getLogger(_resolve_logger_name())
    argv_list: List[str] = list(argv)
    run_id: str = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Batch pipeline from Google Sheets to Google Docs and Telegram "
            "with local thumbnail saving."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--merge",
        action="store_true",
        help="Run with LLM merge enabled for grouped descriptions.",
    )
    mode_group.add_argument(
        "--nomerge",
        action="store_true",
        help="Run without LLM merge; publish original titles/descriptions with numbering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not send to Telegram and do not create Google Docs.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv_list)

    _setup_logging(debug=bool(args.debug))
    if bool(args.debug):
        _debug_run_merge_format_self_test()
        _debug_run_merge_column_self_test()
        _debug_run_merge_range_semantics_and_dedup_self_test()
        _debug_run_plain_description_self_test()

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
    processing_mode = _normalize_processing_mode(
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
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
