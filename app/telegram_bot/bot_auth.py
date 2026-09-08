"""Telegram bot authorization: role-based access control."""
from __future__ import annotations

import enum
import logging
from typing import Any, Awaitable, Callable, Dict, Optional

from aiogram import BaseMiddleware, types
from aiogram.types import TelegramObject

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.telegram_bot.group_registry import load_known_group_ids, register_group_from_event

# Telegram system account ID used as `from_user.id` when an anonymous
# group administrator (or "send as group/channel") sends a message.
# Real sender's user_id is NOT delivered to bots in this mode.
GROUP_ANONYMOUS_BOT_ID: int = 1087968824

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


def _extract_chat_id(event: TelegramObject) -> Optional[int]:
    """Return chat.id from a Message or CallbackQuery event, or None."""
    chat_obj: Any = getattr(event, "chat", None)
    if chat_obj is None:
        # CallbackQuery has no direct .chat; chat is on .message
        message_obj: Any = getattr(event, "message", None)
        chat_obj = getattr(message_obj, "chat", None)
    if chat_obj is None:
        return None
    raw_chat_id: Any = getattr(chat_obj, "id", None)
    try:
        return int(raw_chat_id) if raw_chat_id is not None else None
    except (TypeError, ValueError):
        return None


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
        user: types.User | None = getattr(event, "from_user", None)
        if user is None:
            # Без отправителя нет смысла авторизовать; но можем зарегистрировать
            # группу (это безопасно — кейс анонима к этой ветке не относится).
            register_group_from_event(
                event=event,
                logger=LOGGER,
                admin_ids=self._admin_ids,
                user_ids=self._user_ids,
            )
            return None

        username_label: str = _format_username(user.username)
        event_type: str = type(event).__name__

        # Path 2 (Anonymous Group Admin) ПРОВЕРЯЕМ ПЕРВЫМ для GROUP_ANONYMOUS_BOT_ID,
        # потому что иначе register_group_from_event ниже автоматически
        # зарегистрирует неизвестную группу и сломает фильтр.
        if user.id == GROUP_ANONYMOUS_BOT_ID:
            chat_id: Optional[int] = _extract_chat_id(event)
            # Читаем список известных групп ДО любой регистрации.
            known_group_ids: frozenset[int] = load_known_group_ids()

            if chat_id is not None and chat_id in known_group_ids:
                # Группа уже была зарегистрирована раньше — доверяем.
                # Теперь безопасно вызвать register_group_from_event (обновит
                # title/username, если поменялись, без изменения семантики).
                register_group_from_event(
                    event=event,
                    logger=LOGGER,
                    admin_ids=self._admin_ids,
                    user_ids=self._user_ids,
                )
                LOGGER.info(
                    "access_check decision=allow role=admin source=anonymous_group_admin "
                    "chat_id=%d user_id=%d username=%s event=%s",
                    chat_id,
                    user.id,
                    username_label,
                    event_type,
                )
                data["user_role"] = UserRole.ADMIN
                return await handler(event, data)

            # Анонимный отправитель из неизвестной группы:
            # НЕ регистрируем эту группу, иначе следующая попытка пройдёт.
            LOGGER.info(
                "access_check decision=deny reason=anonymous_unknown_chat "
                "chat_id=%s user_id=%d username=%s event=%s",
                chat_id if chat_id is not None else "-",
                user.id,
                username_label,
                event_type,
            )
            answer_method: Any = getattr(event, "answer", None)
            if callable(answer_method):
                await answer_method(
                    "⛔ Вы отправляете сообщение анонимно от имени группы.\n"
                    "Бот не видит ваш реальный Telegram ID, "
                    "а эта группа не зарегистрирована.\n"
                    "Отключите режим «Анонимный администратор» "
                    "или отправьте команду от личного аккаунта."
                )
            return None

        # Обычный отправитель (не аноним). Регистрируем группу как раньше:
        # на регулярных пользователей фильтр «известная группа» не действует,
        # их авторизация определяется списками admin_ids/user_ids.
        register_group_from_event(
            event=event,
            logger=LOGGER,
            admin_ids=self._admin_ids,
            user_ids=self._user_ids,
        )

        # Path 1: стандартная авторизация по user.id
        role: Optional[UserRole] = resolve_user_role(
            user.id,
            admin_ids=self._admin_ids,
            user_ids=self._user_ids,
        )
        if role is not None:
            LOGGER.info(
                "access_check decision=allow role=%s user_id=%d username=%s event=%s",
                role.value,
                user.id,
                username_label,
                event_type,
            )
            data["user_role"] = role
            return await handler(event, data)

        # Path 3: обычный deny
        LOGGER.info(
            "access_check decision=deny user_id=%d username=%s event=%s",
            user.id,
            username_label,
            event_type,
        )
        answer_method = getattr(event, "answer", None)
        if callable(answer_method):
            await answer_method("⛔ Access denied.")
        return None
