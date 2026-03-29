from __future__ import annotations

import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, User

from app.telegram_bot.bot_auth import AdminOnlyMiddleware
from app.telegram_bot.bot_handlers_info import router

LOGGER: logging.Logger = logging.getLogger("restreamer.bot")


def _read_required_env(var_name: str) -> str:
    raw_value: str = str(os.environ.get(var_name, "")).strip()
    if not raw_value:
        raise RuntimeError(f"Missing required environment variable: {var_name}")
    return raw_value


def _parse_admin_ids(raw_admin_ids: str) -> frozenset[int]:
    parsed_admin_ids: set[int] = set()
    token: str
    for token in raw_admin_ids.split(","):
        normalized_token: str = token.strip()
        if not normalized_token:
            continue
        try:
            admin_id: int = int(normalized_token)
        except ValueError as error:
            raise RuntimeError(
                "Invalid TELEGRAM_ADMIN_USER_IDS value: expected comma-separated integers"
            ) from error
        parsed_admin_ids.add(admin_id)
    admin_ids_set: frozenset[int] = frozenset(parsed_admin_ids)
    if not admin_ids_set:
        raise RuntimeError("TELEGRAM_ADMIN_USER_IDS must contain at least one admin user ID")
    return admin_ids_set


def _build_bot_and_dispatcher() -> tuple[Bot, Dispatcher, frozenset[int]]:
    bot_token: str = _read_required_env("TELEGRAM_BOT_TOKEN")
    raw_admin_ids: str = _read_required_env("TELEGRAM_ADMIN_USER_IDS")
    admin_ids: frozenset[int] = _parse_admin_ids(raw_admin_ids)
    bot: Bot = Bot(token=bot_token)
    dp: Dispatcher = Dispatcher()
    dp.include_router(router)
    dp.message.middleware(AdminOnlyMiddleware(admin_ids=admin_ids))
    dp.callback_query.middleware(AdminOnlyMiddleware(admin_ids=admin_ids))
    return bot, dp, admin_ids


async def run_bot() -> None:
    bot: Bot
    dp: Dispatcher
    admin_ids: frozenset[int]
    bot, dp, admin_ids = _build_bot_and_dispatcher()
    LOGGER.info("Bot starting...")
    bot_info: User = await bot.get_me()
    username: str = str(bot_info.username or "")
    LOGGER.info("Bot username: %s", username)
    LOGGER.info("Admin IDs configured: %d", len(admin_ids))
    commands: list[BotCommand] = [
        BotCommand(command="start", description="Начать работу"),
        BotCommand(command="status", description="Последний запуск"),
        BotCommand(command="run", description="Запустить обработку"),
    ]
    await bot.set_my_commands(commands)
    LOGGER.info("Bot menu commands registered: %d", len(commands))
    LOGGER.info("Bot polling started")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
