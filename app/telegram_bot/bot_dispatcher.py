from __future__ import annotations

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, User

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.telegram_bot.bot_auth import RoleMiddleware
from app.telegram_bot.bot_handlers_info import router
from app.telegram_bot.group_registry import KnownGroup, load_known_groups, sync_access_lists

LOGGER: logging.Logger = _get_logger_impl("bot")
_STARTUP_API_TIMEOUT_SEC: float = 20.0


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


def _parse_optional_ids(raw_ids: str) -> frozenset[int]:
    """Parse comma-separated IDs. Returns empty frozenset if input is empty."""
    parsed_ids: set[int] = set()
    token: str
    for token in raw_ids.split(","):
        normalized: str = token.strip()
        if not normalized:
            continue
        try:
            parsed_ids.add(int(normalized))
        except ValueError as error:
            raise RuntimeError(
                "Invalid TELEGRAM_USER_IDS value: expected comma-separated integers"
            ) from error
    return frozenset(parsed_ids)


def _build_bot_and_dispatcher() -> tuple[Bot, Dispatcher, frozenset[int], frozenset[int]]:
    bot_token: str = _read_required_env("TELEGRAM_BOT_TOKEN")
    raw_admin_ids: str = _read_required_env("TELEGRAM_ADMIN_USER_IDS")
    admin_ids: frozenset[int] = _parse_admin_ids(raw_admin_ids)
    raw_user_ids: str = str(os.environ.get("TELEGRAM_USER_IDS", ""))
    user_ids: frozenset[int] = _parse_optional_ids(raw_user_ids)
    bot: Bot = Bot(token=bot_token)
    dp: Dispatcher = Dispatcher()
    dp.include_router(router)
    role_middleware: RoleMiddleware = RoleMiddleware(admin_ids=admin_ids, user_ids=user_ids)
    dp.message.middleware(role_middleware)
    dp.callback_query.middleware(role_middleware)
    return bot, dp, admin_ids, user_ids


async def run_bot() -> None:
    bot: Bot
    dp: Dispatcher
    admin_ids: frozenset[int]
    user_ids: frozenset[int]
    bot, dp, admin_ids, user_ids = _build_bot_and_dispatcher()
    LOGGER.info("Bot starting...")
    LOGGER.info("Bot startup step=sync_access_lists")
    sync_access_lists(
        logger=LOGGER,
        admin_ids=admin_ids,
        user_ids=user_ids,
    )
    LOGGER.info("Bot startup step=get_me timeout_sec=%.1f", _STARTUP_API_TIMEOUT_SEC)
    try:
        bot_info: User = await asyncio.wait_for(
            bot.get_me(),
            timeout=_STARTUP_API_TIMEOUT_SEC,
        )
    except TimeoutError as error:
        raise RuntimeError(
            "Telegram API timeout on getMe during bot startup. "
            "Check internet access, proxy/firewall and TELEGRAM_BOT_TOKEN."
        ) from error
    username: str = str(bot_info.username or "")
    LOGGER.info("Bot username: %s", username)
    LOGGER.info("Admin IDs configured: %d", len(admin_ids))
    LOGGER.info("User IDs configured: %d", len(user_ids))
    known_groups: list[KnownGroup] = load_known_groups(logger=LOGGER)
    if not known_groups:
        LOGGER.info("Known group chats: 0")
    else:
        LOGGER.info("Known group chats: %d", len(known_groups))
        group: KnownGroup
        for group in known_groups:
            LOGGER.info(
                "Known group chat_id=%s type=%s title=%s username=%s",
                group.chat_id,
                group.chat_type,
                group.chat_title,
                group.chat_username or "-",
            )
    commands: list[BotCommand] = [
        BotCommand(command="start", description="Начать работу"),
        BotCommand(command="stop", description="Остановить бота"),
    ]
    LOGGER.info(
        "Bot startup step=set_my_commands timeout_sec=%.1f count=%d",
        _STARTUP_API_TIMEOUT_SEC,
        len(commands),
    )
    try:
        await asyncio.wait_for(
            bot.set_my_commands(commands),
            timeout=_STARTUP_API_TIMEOUT_SEC,
        )
        LOGGER.info("Bot menu commands registered: %d", len(commands))
    except TimeoutError:
        LOGGER.warning("Bot menu command registration timed out; continuing without menu refresh")
    LOGGER.info("Bot polling started")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()
