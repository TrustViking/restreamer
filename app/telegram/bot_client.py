from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import requests

LOGGER: logging.Logger = logging.getLogger(__name__)


class TelegramChatMigratedError(Exception):
    """Raised when Telegram reports a group -> supergroup migration."""

    def __init__(self, old_chat_id: str, new_chat_id: str, raw_response: str) -> None:
        self.old_chat_id: str = old_chat_id
        self.new_chat_id: str = new_chat_id
        self.raw_response: str = raw_response
        super().__init__(f"Chat migrated from {old_chat_id} to {new_chat_id}")


def _raise_for_telegram_response(response: requests.Response, *, chat_id: str = "") -> None:
    if response.status_code != 200:
        try:
            payload: Dict[str, Any] = response.json()
            migrate_to: Any = payload.get("parameters", {}).get("migrate_to_chat_id")
            if migrate_to is not None:
                new_chat_id: str = str(migrate_to).strip()
                if new_chat_id:
                    raise TelegramChatMigratedError(
                        old_chat_id=chat_id,
                        new_chat_id=new_chat_id,
                        raw_response=response.text,
                    )
        except TelegramChatMigratedError:
            raise
        except Exception:
            pass
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
        newline_index: int = normalized.rfind("\n", start_index, end_index)
        if newline_index > start_index + 200:
            end_index = newline_index + 1
        chunks.append(normalized[start_index:end_index])
        start_index = end_index
    return chunks


class TelegramBotClient:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        timeout_seconds: float = 20.0,
        send_delay_seconds: float = 1.0,
        max_retries: int = 3,
    ) -> None:
        self._bot_token: str = bot_token
        self._chat_id: str = chat_id
        self._timeout_seconds: float = timeout_seconds
        self._send_delay: float = send_delay_seconds
        self._max_retries: int = max_retries
        self._on_migration_callback: Optional[Callable[[str, str], None]] = None

    @property
    def chat_id(self) -> str:
        return self._chat_id

    def set_migration_callback(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback(old_chat_id, new_chat_id) invoked on chat migration."""
        self._on_migration_callback = callback

    def send_text(self, text: str) -> None:
        chunks: List[str] = _split_text_for_telegram(text=text, max_chunk_size=3500)
        self._send_text_chunks(chunks=chunks)

    def get_me(self) -> Dict[str, Any]:
        url: str = self._make_api_url("getMe")
        response: requests.Response = requests.get(url, timeout=self._timeout_seconds)
        _raise_for_telegram_response(response=response)
        payload: Dict[str, Any] = response.json()
        return cast(Dict[str, Any], payload.get("result", {}))

    def send_photo_as_file_bytes(
        self,
        photo_bytes: bytes,
        filename: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> None:
        self.send_document_as_file_bytes(
            file_bytes=photo_bytes,
            filename=filename,
            mime_type=mime_type,
            caption=caption,
        )

    def send_document_as_file_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        caption: Optional[str] = None,
    ) -> None:
        url: str = self._make_api_url("sendDocument")
        files: Dict[str, Tuple[str, bytes, str]] = {
            "document": (filename, file_bytes, mime_type),
        }
        data: Dict[str, str] = {"chat_id": self._chat_id}
        if caption:
            data["caption"] = caption

        response: requests.Response = self._execute_with_throttle(
            lambda: requests.post(
                url,
                data=data,
                files=files,
                timeout=self._timeout_seconds,
            )
        )
        try:
            _raise_for_telegram_response(response=response, chat_id=self._chat_id)
        except TelegramChatMigratedError as migration_error:
            LOGGER.warning(
                "telegram_chat_migrated old_chat_id=%s new_chat_id=%s method=sendDocument retrying=yes",
                migration_error.old_chat_id,
                migration_error.new_chat_id,
            )
            self._chat_id = migration_error.new_chat_id
            if self._on_migration_callback is not None:
                try:
                    self._on_migration_callback(
                        migration_error.old_chat_id,
                        migration_error.new_chat_id,
                    )
                except Exception as callback_error:
                    LOGGER.warning(
                        "telegram_migration_callback_failed error=%s",
                        callback_error,
                    )
            data["chat_id"] = self._chat_id
            retry_response: requests.Response = self._execute_with_throttle(
                lambda: requests.post(
                    url,
                    data=data,
                    files=files,
                    timeout=self._timeout_seconds,
                )
            )
            _raise_for_telegram_response(
                response=retry_response,
                chat_id=self._chat_id,
            )

    def _send_text_chunks(self, chunks: Iterable[str]) -> None:
        for chunk in chunks:
            self._post_json(
                method="sendMessage",
                payload={
                    "chat_id": self._chat_id,
                    "text": chunk,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": False,
                },
            )

    def _post_json(self, method: str, payload: Dict[str, Any]) -> None:
        url: str = self._make_api_url(method)
        response: requests.Response = self._execute_with_throttle(
            lambda: requests.post(
                url,
                json=payload,
                timeout=self._timeout_seconds,
            )
        )
        try:
            _raise_for_telegram_response(response=response, chat_id=self._chat_id)
        except TelegramChatMigratedError as migration_error:
            LOGGER.warning(
                "telegram_chat_migrated old_chat_id=%s new_chat_id=%s method=%s retrying=yes",
                migration_error.old_chat_id,
                migration_error.new_chat_id,
                method,
            )
            self._chat_id = migration_error.new_chat_id
            if self._on_migration_callback is not None:
                try:
                    self._on_migration_callback(
                        migration_error.old_chat_id,
                        migration_error.new_chat_id,
                    )
                except Exception as callback_error:
                    LOGGER.warning(
                        "telegram_migration_callback_failed error=%s",
                        callback_error,
                    )
            payload["chat_id"] = self._chat_id
            retry_response: requests.Response = self._execute_with_throttle(
                lambda: requests.post(
                    url,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            )
            _raise_for_telegram_response(
                response=retry_response,
                chat_id=self._chat_id,
            )

    def _execute_with_throttle(
        self,
        func: Callable[[], requests.Response],
    ) -> requests.Response:
        last_response: requests.Response | None = None
        attempt: int
        for attempt in range(1, self._max_retries + 1):
            time.sleep(self._send_delay)
            response: requests.Response = func()
            if response.status_code != 429:
                return response
            last_response = response
            retry_after: int = 5
            try:
                payload: Dict[str, Any] = response.json()
                raw_retry_after: Any = (
                    payload.get("parameters", {}).get("retry_after", 5)
                )
                retry_after = int(raw_retry_after)
            except (ValueError, TypeError, requests.RequestException):
                retry_after = 5
            LOGGER.warning(
                "Telegram 429: retry_after=%ss (attempt %d/%d)",
                retry_after,
                attempt,
                self._max_retries,
            )
            time.sleep(retry_after)
        if last_response is not None:
            return last_response
        return func()

    def _make_api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self._bot_token}/{method}"
