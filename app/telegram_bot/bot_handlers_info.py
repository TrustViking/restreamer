from __future__ import annotations

import asyncio
import logging
import threading
import time
import re
from typing import Any

from aiogram import Dispatcher, F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)

from app.application.application import RestreamerApplication
from app.bootstrap.logging_config import get_logger as _get_logger_impl

router: Router = Router()
LOGGER: logging.Logger = _get_logger_impl("bot")
_pipeline_lock: threading.Lock = threading.Lock()
_VALID_AUDIT_MODES: tuple[str, str, str] = ("nomerge", "merge", "audit")
_MODE_DISPLAY_NAMES: dict[str, str] = {
    "nomerge": "nomerge",
    "merge": "merge",
    "audit": "nomerge + merge",
}
_MODE_KEYBOARD: InlineKeyboardMarkup = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="nomerge", callback_data="run:nomerge"),
            InlineKeyboardButton(text="merge", callback_data="run:merge"),
        ],
        [
            InlineKeyboardButton(
                text="nomerge + merge", callback_data="run:audit"
            ),
            InlineKeyboardButton(text="cancel", callback_data="run:cancel"),
        ],
    ]
)

_STALE_CALLBACK_PATTERN: re.Pattern[str] = re.compile(
    r"query is too old|query ID is invalid",
    re.IGNORECASE,
)


def _format_username(username: str | None) -> str:
    normalized_username: str = str(username or "").strip()
    if not normalized_username:
        return "@unknown"
    if normalized_username.startswith("@"):
        return normalized_username
    return f"@{normalized_username}"


def _resolve_callback_chat_id(callback: CallbackQuery) -> str | None:
    message_obj: Any = callback.message
    chat_obj: Any = getattr(message_obj, "chat", None)
    chat_id_raw: Any = getattr(chat_obj, "id", None)
    normalized_chat_id: str = str(chat_id_raw or "").strip()
    if not normalized_chat_id:
        return None
    return normalized_chat_id


async def _safe_callback_answer(
    callback: CallbackQuery,
    text: str = "",
    *,
    show_alert: bool = False,
) -> bool:
    """Answer a callback query, gracefully handling stale/expired queries.

    Returns True if answered successfully, False if the query was stale.
    """
    try:
        if text:
            await callback.answer(text, show_alert=show_alert)
        else:
            await callback.answer()
        return True
    except TelegramBadRequest as exc:
        if _STALE_CALLBACK_PATTERN.search(str(exc)):
            LOGGER.debug("stale_callback_ignored callback_id=%s error=%s", callback.id, exc)
            return False
        raise


async def _safe_send_message(
    callback: CallbackQuery,
    text: str,
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
) -> bool:
    """Send a new message in the same chat. Returns True on success."""
    message_obj: Any = callback.message
    answer_method: Any = getattr(message_obj, "answer", None)
    if not callable(answer_method):
        try:
            await callback.answer(text[:200], show_alert=True)
        except TelegramBadRequest:
            return False
        return False
    try:
        await answer_method(text, reply_markup=reply_markup)
        return True
    except TelegramBadRequest:
        return False


def _format_elapsed(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f} сек"
    minutes: int = int(seconds) // 60
    remaining_sec: int = int(seconds) % 60
    return f"{minutes} мин {remaining_sec:02d} сек"


@router.message(Command("start"))
async def start_handler(message: types.Message) -> None:
    response_text: str = (
        "Привет! Я запускаю обработку видео в разных режимах.\n\n"
        "Выберите режим обработки:"
    )
    await message.answer(
        response_text,
        reply_markup=_MODE_KEYBOARD,
    )


@router.message(Command(commands=("stop", "stopbot")))
async def stop_bot_handler(message: types.Message, dispatcher: Dispatcher) -> None:
    actor: types.User | None = message.from_user
    actor_id: int = int(actor.id) if actor is not None else 0
    actor_username: str = _format_username(actor.username if actor is not None else None)
    LOGGER.warning(
        "bot_stop_requested user_id=%d username=%s",
        actor_id,
        actor_username,
    )
    await message.answer("⏹ Останавливаю бота...")
    try:
        await dispatcher.stop_polling()
    except RuntimeError:
        LOGGER.warning("bot_stop_ignored reason=polling_not_running")


@router.callback_query(F.data == "action:start")
async def start_callback_handler(callback: CallbackQuery) -> None:
    if not await _safe_callback_answer(callback):
        return
    await _safe_send_message(
        callback,
        "Выберите режим обработки:",
        reply_markup=_MODE_KEYBOARD,
    )


@router.callback_query(F.data.startswith("run:"))
async def run_callback_handler(callback: CallbackQuery) -> None:
    callback_data: str = str(callback.data or "")
    callback_parts: list[str] = callback_data.split(":", maxsplit=1)
    audit_mode: str = callback_parts[1] if len(callback_parts) == 2 else ""
    if audit_mode == "cancel":
        if not await _safe_callback_answer(callback):
            return
        await _safe_send_message(
            callback,
            "Запуск отменён.",
            reply_markup=None,
        )
        return
    if audit_mode not in _VALID_AUDIT_MODES:
        if not await _safe_callback_answer(callback, "Неизвестный режим"):
            return
        return
    if not await _safe_callback_answer(callback):
        return
    display_mode: str = _MODE_DISPLAY_NAMES.get(audit_mode, audit_mode)
    actor: types.User | None = callback.from_user
    actor_id: int = int(actor.id) if actor is not None else 0
    actor_username: str = _format_username(actor.username if actor is not None else None)
    source_chat_id: str | None = _resolve_callback_chat_id(callback)
    LOGGER.info(
        "pipeline_run_requested user_id=%d username=%s mode=%s source_chat_id=%s",
        actor_id,
        actor_username,
        audit_mode,
        source_chat_id or "fallback_config",
    )
    await _safe_send_message(
        callback,
        f"⏳ Запуск обработки... (режим: {display_mode})",
        reply_markup=None,
    )

    lock_acquired: bool = _pipeline_lock.acquire(blocking=False)
    if not lock_acquired:
        LOGGER.info(
            "pipeline_run_rejected_busy user_id=%d username=%s mode=%s source_chat_id=%s",
            actor_id,
            actor_username,
            audit_mode,
            source_chat_id or "fallback_config",
        )
        await _safe_send_message(
            callback,
            "⚠️ Обработка уже выполняется. Подождите.",
            reply_markup=None,
        )
        return

    _progress_loop: asyncio.AbstractEventLoop | None = None
    try:
        _progress_loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    def _on_progress(text: str) -> None:
        if _progress_loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            _safe_send_message(
                callback,
                text,
                reply_markup=None,
            ),
            _progress_loop,
        )

    def _run_pipeline(mode: str, target_chat_id: str | None) -> int:
        application: RestreamerApplication = RestreamerApplication(
            telegram_chat_id_override=target_chat_id,
            progress_callback=_on_progress,
        )
        argv: list[str] = ["--audit-mode", mode]
        exit_code: int = application.run(argv)
        return exit_code

    started_at: float = time.monotonic()
    try:
        LOGGER.info(
            "pipeline_run_started user_id=%d username=%s mode=%s target_chat_id=%s",
            actor_id,
            actor_username,
            audit_mode,
            source_chat_id or "fallback_config",
        )
        exit_code: int = await asyncio.to_thread(_run_pipeline, audit_mode, source_chat_id)
        elapsed_sec: float = round(time.monotonic() - started_at, 1)
        if exit_code == 0:
            LOGGER.info(
                "pipeline_run_finished user_id=%d username=%s mode=%s target_chat_id=%s exit_code=%d elapsed_sec=%.1f",
                actor_id,
                actor_username,
                audit_mode,
                source_chat_id or "fallback_config",
                exit_code,
                elapsed_sec,
            )
            await _safe_send_message(
                callback,
                (
                    "✅ Обработка завершена\n\n"
                    f"Режим: {display_mode}\n"
                    f"Время: {_format_elapsed(elapsed_sec)}"
                ),
                reply_markup=_MODE_KEYBOARD,
            )
            return
        LOGGER.info(
            "pipeline_run_finished user_id=%d username=%s mode=%s target_chat_id=%s exit_code=%d elapsed_sec=%.1f",
            actor_id,
            actor_username,
            audit_mode,
            source_chat_id or "fallback_config",
            exit_code,
            elapsed_sec,
        )
        await _safe_send_message(
            callback,
            (
                "❌ Обработка завершена с ошибками\n\n"
                f"Режим: {display_mode}\n"
                f"Код ошибки: {exit_code}\n"
                f"Время: {_format_elapsed(elapsed_sec)}"
            ),
            reply_markup=_MODE_KEYBOARD,
        )
    except Exception as exc:
        elapsed_sec: float = round(time.monotonic() - started_at, 1)
        LOGGER.exception(
            "pipeline_run_failed user_id=%d username=%s mode=%s target_chat_id=%s elapsed_sec=%.1f",
            actor_id,
            actor_username,
            audit_mode,
            source_chat_id or "fallback_config",
            elapsed_sec,
        )
        error_text: str = str(exc)[:500]
        await _safe_send_message(
            callback,
            (
                "❌ Обработка не удалась\n\n"
                f"Режим: {display_mode}\n"
                f"Ошибка: {error_text}\n"
                f"Время: {_format_elapsed(elapsed_sec)}"
            ),
            reply_markup=_MODE_KEYBOARD,
        )
    finally:
        _pipeline_lock.release()
