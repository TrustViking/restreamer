from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.telegram_bot.group_registry import _load_registry_payload, handle_group_migration


def test_migration_removes_old_and_keeps_new() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as temp_file:
        json.dump(
            {
                "admin_ids": [123],
                "user_ids": [],
                "groups": {
                    "-123456789": {
                        "chat_id": "-123456789",
                        "chat_type": "group",
                        "chat_title": "Test Group",
                        "chat_username": "",
                    },
                    "-1001234567890": {
                        "chat_id": "-1001234567890",
                        "chat_type": "supergroup",
                        "chat_title": "Test Group",
                        "chat_username": "",
                    },
                },
            },
            temp_file,
        )
        temp_path = Path(temp_file.name)

    try:
        with patch("app.telegram_bot.group_registry._registry_path", return_value=temp_path):
            logger = logging.getLogger("test_group_registry_migration")
            handle_group_migration(
                logger=logger,
                old_chat_id="-123456789",
                new_chat_id="-1001234567890",
            )
            payload = _load_registry_payload(temp_path)
            assert "-123456789" not in payload["groups"]
            assert "-1001234567890" in payload["groups"]
    finally:
        temp_path.unlink(missing_ok=True)
