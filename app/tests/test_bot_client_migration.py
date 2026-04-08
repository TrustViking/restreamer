from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.telegram.bot_client import TelegramBotClient


def _make_migration_response(new_chat_id: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 400
    response.text = json.dumps(
        {
            "ok": False,
            "error_code": 400,
            "description": "Bad Request: group chat was upgraded to a supergroup chat",
            "parameters": {"migrate_to_chat_id": int(new_chat_id)},
        }
    )
    response.json.return_value = json.loads(response.text)
    return response


def _make_ok_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.text = '{"ok": true, "result": {}}'
    response.json.return_value = {"ok": True, "result": {}}
    return response


@patch("app.telegram.bot_client.requests.post")
def test_post_json_retries_on_migration(mock_post: MagicMock) -> None:
    migration_response = _make_migration_response("-1003867270959")
    ok_response = _make_ok_response()
    mock_post.side_effect = [migration_response, ok_response]

    client = TelegramBotClient(
        bot_token="test_token",
        chat_id="-5268509425",
        send_delay_seconds=0.0,
    )
    client._post_json("sendMessage", {"chat_id": "-5268509425", "text": "hello"})
    assert client.chat_id == "-1003867270959"
    assert mock_post.call_count == 2


@patch("app.telegram.bot_client.requests.post")
def test_send_document_retries_on_migration(mock_post: MagicMock) -> None:
    migration_response = _make_migration_response("-1003867270959")
    ok_response = _make_ok_response()
    mock_post.side_effect = [migration_response, ok_response]

    client = TelegramBotClient(
        bot_token="test_token",
        chat_id="-5268509425",
        send_delay_seconds=0.0,
    )
    client.send_document_as_file_bytes(
        file_bytes=b"test",
        filename="test.json",
        mime_type="application/json",
    )
    assert client.chat_id == "-1003867270959"


@patch("app.telegram.bot_client.requests.post")
def test_migration_callback_is_invoked(mock_post: MagicMock) -> None:
    migration_response = _make_migration_response("-1003867270959")
    ok_response = _make_ok_response()
    mock_post.side_effect = [migration_response, ok_response]

    callback = MagicMock()
    client = TelegramBotClient(
        bot_token="test_token",
        chat_id="-5268509425",
        send_delay_seconds=0.0,
    )
    client.set_migration_callback(callback)
    client._post_json("sendMessage", {"chat_id": "-5268509425", "text": "hello"})
    callback.assert_called_once_with("-5268509425", "-1003867270959")


def test_send_text_uses_html_parse_mode() -> None:
    client = TelegramBotClient(
        bot_token="test_token",
        chat_id="-5268509425",
        send_delay_seconds=0.0,
    )
    client._post_json = MagicMock()
    client.send_text("hello")
    client._post_json.assert_called_once()
    kwargs = client._post_json.call_args.kwargs
    assert kwargs["method"] == "sendMessage"
    assert kwargs["payload"]["parse_mode"] == "HTML"
