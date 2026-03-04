from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, cast

import requests


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
        newline_index: int = normalized.rfind("\n", start_index, end_index)
        if newline_index > start_index + 200:
            end_index = newline_index + 1
        chunks.append(normalized[start_index:end_index])
        start_index = end_index
    return chunks


class TelegramBotClient:
    def __init__(
        self, bot_token: str, chat_id: str, timeout_seconds: float = 20.0
    ) -> None:
        self._bot_token: str = bot_token
        self._chat_id: str = chat_id
        self._timeout_seconds: float = timeout_seconds

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

    def _post_json(self, method: str, payload: Dict[str, Any]) -> None:
        url: str = self._make_api_url(method)
        response: requests.Response = requests.post(
            url, json=payload, timeout=self._timeout_seconds
        )
        _raise_for_telegram_response(response=response)

    def _make_api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self._bot_token}/{method}"
