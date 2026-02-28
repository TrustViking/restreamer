from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import requests

from app.bootstrap.logging_config import get_logger as _get_logger_impl

try:
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
except ImportError:
    MediaFileUpload = None  # type: ignore
    MediaIoBaseDownload = None  # type: ignore


LOGGER = _get_logger_impl(__name__)


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

    def export_google_doc_as_docx(self, file_id: str) -> bytes:
        if MediaIoBaseDownload is None:
            raise RuntimeError(
                "Google API client недоступен (googleapiclient не установлен)."
            )
        request: Any = self._drive_service.files().export_media(
            fileId=file_id,
            mimeType=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )
        output_buffer: io.BytesIO = io.BytesIO()
        downloader: Any = MediaIoBaseDownload(output_buffer, request)
        done: bool = False
        while not done:
            _, done = downloader.next_chunk()
        return output_buffer.getvalue()
