from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, cast, Dict

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.app_config_loader import (
    load_google_auth_mode_from_env as _load_google_auth_mode_from_env_impl,
    load_google_oauth_credentials_path as _load_google_oauth_credentials_path_impl,
    load_google_oauth_token_path as _load_google_oauth_token_path_impl,
    load_google_service_account_path as _load_google_service_account_path_impl,
)

try:
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials as OAuthUserCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
except ImportError:
    GoogleAuthRequest = None  # type: ignore
    OAuthUserCredentials = None  # type: ignore
    ServiceAccountCredentials = None  # type: ignore
    InstalledAppFlow = None  # type: ignore
    build = None  # type: ignore


LOGGER = _get_logger_impl(__name__)


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
        self._auth_mode: str = _load_google_auth_mode_from_env_impl()
        self._service_account_path: Optional[Path] = (
            _load_google_service_account_path_impl()
        )
        self._oauth_credentials_path: Path = _load_google_oauth_credentials_path_impl()
        self._oauth_token_path: Path = _load_google_oauth_token_path_impl()

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
        LOGGER.error("%s Reason=%s", message, _summarize_error(cast(Exception, error)))
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
            "to your OAuth client credentials file (Desktop app)."
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
            "Failed to create OAuth token file. "
            f"credentials_path={str(oauth_credentials_path)!r} token_path={str(oauth_token_path)!r}. "
            "Check OAuth Desktop credentials and browser authorization flow."
        ) from error


def _summarize_error(error: Exception, max_len: int = 220) -> str:
    compact: str = re.sub(r"\s+", " ", str(error).replace("\r", " ").replace("\n", " ")).strip()
    if len(compact) <= max_len:
        return compact
    return f"{compact[:max_len].rstrip()}..."
