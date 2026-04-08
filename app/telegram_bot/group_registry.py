"""Registry of Telegram group chats observed by the bot."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogram.types import TelegramObject

from app.paths import get_project_paths


@dataclass(frozen=True)
class KnownGroup:
    chat_id: str
    chat_type: str
    chat_title: str
    chat_username: str


def _registry_path() -> Path:
    project_root: Path = get_project_paths().project_root
    return project_root / "logs" / "bot_known_groups.json"


def _normalize_username(raw_username: str) -> str:
    normalized_username: str = str(raw_username or "").strip()
    if not normalized_username:
        return ""
    if normalized_username.startswith("@"):
        return normalized_username
    return f"@{normalized_username}"


def _extract_group_from_event(event: TelegramObject) -> KnownGroup | None:
    chat_obj: Any = getattr(event, "chat", None)
    if chat_obj is None:
        message_obj: Any = getattr(event, "message", None)
        chat_obj = getattr(message_obj, "chat", None)
    if chat_obj is None:
        return None

    chat_type: str = str(getattr(chat_obj, "type", "") or "").strip().lower()
    if chat_type not in {"group", "supergroup"}:
        return None

    raw_chat_id: Any = getattr(chat_obj, "id", None)
    chat_id: str = str(raw_chat_id or "").strip()
    if not chat_id:
        return None

    chat_title: str = str(getattr(chat_obj, "title", "") or "").strip() or "unknown"
    chat_username: str = _normalize_username(str(getattr(chat_obj, "username", "") or ""))

    return KnownGroup(
        chat_id=chat_id,
        chat_type=chat_type,
        chat_title=chat_title,
        chat_username=chat_username,
    )


def _default_registry_payload() -> dict[str, Any]:
    return {
        "admin_ids": [],
        "user_ids": [],
        "groups": {},
    }


def _load_registry_payload(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = _default_registry_payload()
    if not path.exists() or not path.is_file():
        return payload
    try:
        raw_text: str = path.read_text(encoding="utf-8")
        parsed_payload: Any = json.loads(raw_text)
    except Exception:
        return payload
    if not isinstance(parsed_payload, dict):
        return payload

    raw_admin_ids: Any = parsed_payload.get("admin_ids")
    if isinstance(raw_admin_ids, list):
        normalized_admin_ids: list[int] = []
        item: Any
        for item in raw_admin_ids:
            try:
                normalized_admin_ids.append(int(item))
            except (TypeError, ValueError):
                continue
        payload["admin_ids"] = sorted(set(normalized_admin_ids))

    raw_user_ids: Any = parsed_payload.get("user_ids")
    if isinstance(raw_user_ids, list):
        normalized_user_ids: list[int] = []
        user_item: Any
        for user_item in raw_user_ids:
            try:
                normalized_user_ids.append(int(user_item))
            except (TypeError, ValueError):
                continue
        payload["user_ids"] = sorted(set(normalized_user_ids))

    raw_groups: Any = parsed_payload.get("groups")
    groups_payload: dict[str, dict[str, str]] = {}
    if isinstance(raw_groups, dict):
        chat_id: str
        chat_payload: Any
        for chat_id, chat_payload in raw_groups.items():
            if not isinstance(chat_payload, dict):
                continue
            normalized_chat_id: str = str(chat_id or "").strip()
            if not normalized_chat_id:
                continue
            groups_payload[normalized_chat_id] = {
                "chat_id": normalized_chat_id,
                "chat_type": str(chat_payload.get("chat_type", "") or "").strip(),
                "chat_title": str(chat_payload.get("chat_title", "") or "").strip(),
                "chat_username": str(chat_payload.get("chat_username", "") or "").strip(),
            }
    payload["groups"] = groups_payload
    return payload


def _save_registry_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_payload: str = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(encoded_payload + "\n", encoding="utf-8")


def _normalize_access_ids(ids: frozenset[int]) -> list[int]:
    return sorted({int(item) for item in ids})


def sync_access_lists(
    *,
    logger: logging.Logger,
    admin_ids: frozenset[int],
    user_ids: frozenset[int],
) -> None:
    path: Path = _registry_path()
    payload: dict[str, Any] = _load_registry_payload(path)
    normalized_admin_ids: list[int] = _normalize_access_ids(admin_ids)
    normalized_user_ids: list[int] = _normalize_access_ids(user_ids)

    has_changes: bool = False
    if payload.get("admin_ids") != normalized_admin_ids:
        payload["admin_ids"] = normalized_admin_ids
        has_changes = True
    if payload.get("user_ids") != normalized_user_ids:
        payload["user_ids"] = normalized_user_ids
        has_changes = True

    if not has_changes:
        return

    try:
        _save_registry_payload(path, payload)
    except Exception as error:
        logger.warning("group_registry_save_failed path=%s error=%s", path, error)
        return

    logger.info(
        "group_registry_access_synced admins=%d users=%d",
        len(normalized_admin_ids),
        len(normalized_user_ids),
    )


def register_group_from_event(
    *,
    event: TelegramObject,
    logger: logging.Logger,
    admin_ids: frozenset[int],
    user_ids: frozenset[int],
) -> None:
    group: KnownGroup | None = _extract_group_from_event(event)
    path: Path = _registry_path()
    payload: dict[str, Any] = _load_registry_payload(path)

    normalized_admin_ids: list[int] = _normalize_access_ids(admin_ids)
    normalized_user_ids: list[int] = _normalize_access_ids(user_ids)
    has_changes: bool = False
    if payload.get("admin_ids") != normalized_admin_ids:
        payload["admin_ids"] = normalized_admin_ids
        has_changes = True
    if payload.get("user_ids") != normalized_user_ids:
        payload["user_ids"] = normalized_user_ids
        has_changes = True

    if group is None:
        if not has_changes:
            return
        try:
            _save_registry_payload(path, payload)
        except Exception as error:
            logger.warning("group_registry_save_failed path=%s error=%s", path, error)
        return

    groups_map: dict[str, dict[str, str]] = payload.get("groups", {})
    previous_payload: dict[str, str] | None = groups_map.get(group.chat_id)
    current_payload: dict[str, str] = {
        "chat_id": group.chat_id,
        "chat_type": group.chat_type,
        "chat_title": group.chat_title,
        "chat_username": group.chat_username,
    }

    if previous_payload == current_payload and not has_changes:
        return

    groups_map[group.chat_id] = current_payload
    payload["groups"] = groups_map
    try:
        _save_registry_payload(path, payload)
    except Exception as error:
        logger.warning("group_registry_save_failed path=%s error=%s", path, error)
        return

    logger.info(
        "group_registry_updated chat_id=%s chat_type=%s title=%s username=%s",
        group.chat_id,
        group.chat_type,
        group.chat_title,
        group.chat_username or "-",
    )


def handle_group_migration(
    *,
    logger: logging.Logger,
    old_chat_id: str,
    new_chat_id: str,
    new_chat_type: str = "supergroup",
    new_chat_title: str = "",
) -> None:
    """Remove old group chat_id and ensure new supergroup chat_id is registered."""
    path: Path = _registry_path()
    payload: dict[str, Any] = _load_registry_payload(path)
    groups_map: dict[str, dict[str, str]] = payload.get("groups", {})

    old_normalized: str = str(old_chat_id or "").strip()
    new_normalized: str = str(new_chat_id or "").strip()
    has_changes: bool = False

    if old_normalized and old_normalized in groups_map:
        old_title: str = groups_map[old_normalized].get("chat_title", "unknown")
        del groups_map[old_normalized]
        has_changes = True
        logger.info(
            "group_registry_migration_removed old_chat_id=%s old_title=%s",
            old_normalized,
            old_title,
        )

    if new_normalized:
        existing: dict[str, str] | None = groups_map.get(new_normalized)
        title: str = new_chat_title or (existing or {}).get("chat_title", "unknown")
        new_entry: dict[str, str] = {
            "chat_id": new_normalized,
            "chat_type": new_chat_type,
            "chat_title": title,
            "chat_username": (existing or {}).get("chat_username", ""),
        }
        if existing != new_entry:
            groups_map[new_normalized] = new_entry
            has_changes = True

    if not has_changes:
        return

    payload["groups"] = groups_map
    try:
        _save_registry_payload(path, payload)
    except Exception as error:
        logger.warning("group_registry_migration_save_failed path=%s error=%s", path, error)
        return

    logger.info(
        "group_registry_migration_completed old_chat_id=%s new_chat_id=%s",
        old_normalized,
        new_normalized,
    )


def load_known_groups(*, logger: logging.Logger) -> list[KnownGroup]:
    path: Path = _registry_path()
    payload: dict[str, Any] = _load_registry_payload(path)
    groups_map: dict[str, dict[str, str]] = payload.get("groups", {})
    groups: list[KnownGroup] = []

    chat_id: str
    chat_payload: dict[str, str]
    for chat_id, chat_payload in groups_map.items():
        chat_type: str = str(chat_payload.get("chat_type", "") or "").strip() or "unknown"
        chat_title: str = str(chat_payload.get("chat_title", "") or "").strip() or "unknown"
        chat_username: str = str(chat_payload.get("chat_username", "") or "").strip()
        groups.append(
            KnownGroup(
                chat_id=chat_id,
                chat_type=chat_type,
                chat_title=chat_title,
                chat_username=chat_username,
            )
        )

    groups.sort(key=lambda item: (item.chat_title.lower(), item.chat_id))
    logger.debug("group_registry_loaded path=%s groups=%d", path, len(groups))
    return groups
