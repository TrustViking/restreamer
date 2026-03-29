from __future__ import annotations

import asyncio
import logging
import threading
import time
import re
from typing import Any

from aiogram import F, Router, types
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
)

from app.application.application import RestreamerApplication

router: Router = Router()
LOGGER: logging.Logger = logging.getLogger("restreamer.bot")
_pipeline_lock: threading.Lock = threading.Lock()
_VALID_AUDIT_MODES: tuple[str, str, str] = ("nomerge", "merge", "audit")
_MODE_DISPLAY_NAMES: dict[str, str] = {
    "nomerge": "без объединения",
    "merge": "с объединением",
    "audit": "без объединения + объединение",
}
_MAIN_KEYBOARD: ReplyKeyboardMarkup = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="▶ Запустить"),
            KeyboardButton(text="📊 Статус"),
        ],
        [
            KeyboardButton(text="ℹ Помощь"),
        ],
    ],
    resize_keyboard=True,
)
_POST_RUN_KEYBOARD: InlineKeyboardMarkup = InlineKeyboardMarkup(
    inline_keyboard=[
        [
            InlineKeyboardButton(text="🔁 Запустить ещё", callback_data="action:run_again"),
            InlineKeyboardButton(text="📊 Статус", callback_data="action:status"),
        ],
    ]
)

_STALE_CALLBACK_PATTERN: re.Pattern[str] = re.compile(
    r"query is too old|query ID is invalid",
    re.IGNORECASE,
)


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
        "Что можно сделать:\n"
        "• запустить обработку\n"
        "• посмотреть статус последнего запуска\n"
        "• получить справку по режимам\n\n"
        "Выберите действие ниже."
    )
    await message.answer(response_text, reply_markup=_MAIN_KEYBOARD)


@router.message(F.text == "ℹ Помощь")
async def help_handler(message: types.Message) -> None:
    response_text: str = (
        "Доступные режимы обработки:\n\n"
        "• Без объединения — каждый источник обрабатывается отдельно\n"
        "• С объединением — материалы объединяются в один результат через LLM\n"
        "• Без объединения + объединение — оба режима последовательно\n\n"
        "Нажмите «▶ Запустить», чтобы выбрать режим."
    )
    await message.answer(response_text, reply_markup=_MAIN_KEYBOARD)


@router.message(Command("status"))
@router.message(F.text == "📊 Статус")
async def status_handler(message: types.Message) -> None:
    response_text: str = (
        "📊 Статус\n\n"
        "Состояние: ожидание\n"
        "Последний запуск: нет данных"
    )
    await message.answer(response_text, reply_markup=_MAIN_KEYBOARD)


@router.message(Command("run"))
@router.message(F.text == "▶ Запустить")
async def run_handler(message: types.Message) -> None:
    keyboard: InlineKeyboardMarkup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Без объединения", callback_data="run:nomerge"),
                InlineKeyboardButton(text="С объединением", callback_data="run:merge"),
            ],
            [
                InlineKeyboardButton(
                    text="Без объединения + объединение", callback_data="run:audit"
                ),
                InlineKeyboardButton(text="Отмена", callback_data="run:cancel"),
            ],
        ]
    )
    await message.answer("Выберите режим обработки:", reply_markup=keyboard)


async def _send_callback_message(callback: CallbackQuery, text: str) -> None:
    message_obj: Any = callback.message
    answer_method: Any = getattr(message_obj, "answer", None)
    if callable(answer_method):
        await answer_method(text)
        return
    await callback.answer(text, show_alert=True)


async def _send_final_message(callback: CallbackQuery, text: str) -> None:
    message_obj: Any = callback.message
    answer_method: Any = getattr(message_obj, "answer", None)
    if callable(answer_method):
        await answer_method(text, reply_markup=_POST_RUN_KEYBOARD)
        return
    await callback.answer(text, show_alert=True)


@router.callback_query(F.data == "action:run_again")
async def post_run_again_handler(callback: CallbackQuery) -> None:
    if not await _safe_callback_answer(callback):
        return
    keyboard: InlineKeyboardMarkup = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Без объединения", callback_data="run:nomerge"),
                InlineKeyboardButton(text="С объединением", callback_data="run:merge"),
            ],
            [
                InlineKeyboardButton(
                    text="Без объединения + объединение", callback_data="run:audit"
                ),
                InlineKeyboardButton(text="Отмена", callback_data="run:cancel"),
            ],
        ]
    )
    message_obj: Any = callback.message
    answer_method: Any = getattr(message_obj, "answer", None)
    if callable(answer_method):
        await answer_method("Выберите режим обработки:", reply_markup=keyboard)
        return
    await callback.answer("Выберите режим обработки:", show_alert=True)


@router.callback_query(F.data == "action:status")
async def post_run_status_handler(callback: CallbackQuery) -> None:
    if not await _safe_callback_answer(callback):
        return
    response_text: str = (
        "📊 Статус\n\n"
        "Состояние: ожидание\n"
        "Последний запуск: нет данных"
    )
    await _send_callback_message(callback, response_text)


@router.callback_query(F.data.startswith("run:"))
async def run_callback_handler(callback: CallbackQuery) -> None:
    callback_data: str = str(callback.data or "")
    callback_parts: list[str] = callback_data.split(":", maxsplit=1)
    audit_mode: str = callback_parts[1] if len(callback_parts) == 2 else ""
    if audit_mode == "cancel":
        if not await _safe_callback_answer(callback):
            return
        await _send_callback_message(callback, "Запуск отменён.")
        return
    if audit_mode not in _VALID_AUDIT_MODES:
        if not await _safe_callback_answer(callback, "Неизвестный режим"):
            return
        return
    if not await _safe_callback_answer(callback):
        return
    display_mode: str = _MODE_DISPLAY_NAMES.get(audit_mode, audit_mode)

    lock_acquired: bool = _pipeline_lock.acquire(blocking=False)
    if not lock_acquired:
        await _send_callback_message(
            callback,
            "⚠️ Обработка уже выполняется. Подождите.",
        )
        return

    await _send_callback_message(
        callback,
        f"⏳ Запуск обработки... (режим: {display_mode})",
    )

    def _run_pipeline(mode: str) -> int:
        application: RestreamerApplication = RestreamerApplication()
        argv: list[str] = ["--audit-mode", mode]
        exit_code: int = application.run(argv)
        return exit_code

    started_at: float = time.monotonic()
    try:
        exit_code: int = await asyncio.to_thread(_run_pipeline, audit_mode)
        elapsed_sec: float = round(time.monotonic() - started_at, 1)
        if exit_code == 0:
            await _send_final_message(
                callback,
                (
                    "✅ Обработка завершена\n\n"
                    f"Режим: {display_mode}\n"
                    f"Время: {_format_elapsed(elapsed_sec)}"
                ),
            )
            return
        await _send_final_message(
            callback,
            (
                "⚠️ Обработка завершена с ошибками\n\n"
                f"Режим: {display_mode}\n"
                f"Код ошибки: {exit_code}\n"
                f"Время: {_format_elapsed(elapsed_sec)}"
            ),
        )
    except Exception as exc:
        elapsed_sec: float = round(time.monotonic() - started_at, 1)
        error_text: str = str(exc)[:500]
        await _send_final_message(
            callback,
            (
                "❌ Обработка не удалась\n\n"
                f"Режим: {display_mode}\n"
                f"Ошибка: {error_text}\n"
                f"Время: {_format_elapsed(elapsed_sec)}"
            ),
        )
        LOGGER.exception("Pipeline failed via bot command")
    finally:
        _pipeline_lock.release()
