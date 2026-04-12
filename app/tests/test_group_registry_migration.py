from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.telegram_bot.group_registry import (
    _load_registry_payload,
    handle_group_migration,
    load_known_groups,
)


def test_migration_removes_old_and_keeps_new() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as temp_file:
        json.dump(
            {
                "admin_ids": [123],
                "user_ids": [],
                "groups": {
                    "-5268509425": {
                        "chat_id": "-5268509425",
                        "chat_type": "group",
                        "chat_title": "Streamertg",
                        "chat_username": "",
                    },
                    "-1003867270959": {
                        "chat_id": "-1003867270959",
                        "chat_type": "supergroup",
                        "chat_title": "Streamertg",
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
                old_chat_id="-5268509425",
                new_chat_id="-1003867270959",
            )
            payload = _load_registry_payload(temp_path)
            assert "-5268509425" not in payload["groups"]
            assert "-1003867270959" in payload["groups"]
    finally:
        temp_path.unlink(missing_ok=True)


def test_legacy_registry_file_is_migrated_from_logs_to_state() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root: Path = Path(temp_dir)
        logs_dir: Path = project_root / "logs"
        state_dir: Path = project_root / "state"
        logs_dir.mkdir(parents=True, exist_ok=True)
        legacy_path: Path = logs_dir / "bot_known_groups.json"
        legacy_payload: dict[str, object] = {
            "admin_ids": [123],
            "user_ids": [456],
            "groups": {
                "-1003867270959": {
                    "chat_id": "-1003867270959",
                    "chat_type": "supergroup",
                    "chat_title": "Streamertg",
                    "chat_username": "@stream",
                }
            },
        }
        legacy_path.write_text(
            json.dumps(legacy_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        expected_path: Path = state_dir / "bot_known_groups.json"
        fake_paths = SimpleNamespace(
            project_root=project_root,
            logs_dir=logs_dir,
            state_dir=state_dir,
        )

        with patch("app.telegram_bot.group_registry.get_project_paths", return_value=fake_paths):
            groups = load_known_groups(logger=logging.getLogger("test_group_registry_load"))

        assert not legacy_path.exists()
        assert expected_path.exists()
        payload = _load_registry_payload(expected_path)
        assert payload["groups"]["-1003867270959"]["chat_title"] == "Streamertg"
        assert len(groups) == 1
        assert groups[0].chat_id == "-1003867270959"
