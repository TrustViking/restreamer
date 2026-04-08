"""Telegram bot authorization: role-based access control."""
from __future__ import annotations

import enum
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from aiogram import BaseMiddleware, types
from aiogram.types import TelegramObject

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.telegram_bot.group_registry import register_group_from_event

LOGGER: logging.Logger = _get_logger_impl("bot")


class UserRole(enum.Enum):
    """Bot user roles. Order matters: higher = more privileges."""

    ADMIN = "admin"
    USER = "user"


def _format_username(username: str | None) -> str:
    normalized_username: str = str(username or "").strip()
    if not normalized_username:
        return "@unknown"
    if normalized_username.startswith("@"):
        return normalized_username
    return f"@{normalized_username}"


def resolve_user_role(
    user_id: int,
    *,
    admin_ids: frozenset[int],
    user_ids: frozenset[int],
) -> Optional[UserRole]:
    """Return the role for a Telegram user, or None if not authorized.

    If a user_id appears in both admin_ids and user_ids, admin wins.
    """
    if user_id in admin_ids:
        return UserRole.ADMIN
    if user_id in user_ids:
        return UserRole.USER
    return None


def is_authorized(
    user_id: int,
    *,
    admin_ids: frozenset[int],
    user_ids: frozenset[int],
) -> bool:
    """Return True if user has any role (admin or user)."""
    return resolve_user_role(user_id, admin_ids=admin_ids, user_ids=user_ids) is not None


def is_admin(
    user_id: int,
    *,
    admin_ids: frozenset[int],
) -> bool:
    """Return True if user is an admin."""
    return user_id in admin_ids


class RoleMiddleware(BaseMiddleware):
    """Middleware that checks authorization and injects user_role into handler data.

    Replaces AdminOnlyMiddleware. Now allows both admins and users,
    and passes UserRole to handlers via data["user_role"].
    """

    def __init__(
        self,
        *,
        admin_ids: frozenset[int],
        user_ids: frozenset[int],
    ) -> None:
        self._admin_ids: frozenset[int] = admin_ids
        self._user_ids: frozenset[int] = user_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        register_group_from_event(
            event=event,
            logger=LOGGER,
            admin_ids=self._admin_ids,
            user_ids=self._user_ids,
        )
        user: types.User | None = getattr(event, "from_user", None)
        if user is None:
            return None

        role: Optional[UserRole] = resolve_user_role(
            user.id,
            admin_ids=self._admin_ids,
            user_ids=self._user_ids,
        )
        username_label: str = _format_username(user.username)
        event_type: str = type(event).__name__
        if role is None:
            LOGGER.info(
                "access_check decision=deny user_id=%d username=%s event=%s",
                user.id,
                username_label,
                event_type,
            )
            answer_method: Any = getattr(event, "answer", None)
            if callable(answer_method):
                await answer_method("⛔ Access denied.")
            return None

        LOGGER.info(
            "access_check decision=allow role=%s user_id=%d username=%s event=%s",
            role.value,
            user.id,
            username_label,
            event_type,
        )
        data["user_role"] = role
        return await handler(event, data)
