from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware, types
from aiogram.types import TelegramObject


def is_admin(user_id: int, admin_ids: frozenset[int]) -> bool:
    return user_id in admin_ids


class AdminOnlyMiddleware(BaseMiddleware):
    def __init__(self, admin_ids: frozenset[int]) -> None:
        self._admin_ids: frozenset[int] = admin_ids

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        user: types.User | None = getattr(event, "from_user", None)
        if user is None or not is_admin(user.id, self._admin_ids):
            answer_method: Any = getattr(event, "answer", None)
            if callable(answer_method):
                await answer_method("⛔ Access denied.")
            return None
        return await handler(event, data)
