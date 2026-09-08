"""Tests for RoleMiddleware handling of anonymous group admin messages."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.telegram_bot.bot_auth import (
    GROUP_ANONYMOUS_BOT_ID,
    RoleMiddleware,
    UserRole,
)


def _make_message_event(
    *,
    user_id: int,
    username: str | None,
    chat_id: int | None,
    chat_type: str = "supergroup",
) -> SimpleNamespace:
    """Build a Message-like event with from_user and chat."""
    answer_mock: AsyncMock = AsyncMock()
    user_ns: SimpleNamespace = SimpleNamespace(id=user_id, username=username)
    chat_ns: SimpleNamespace | None
    if chat_id is None:
        chat_ns = None
    else:
        chat_ns = SimpleNamespace(id=chat_id, type=chat_type, title="Test", username="")
    event: SimpleNamespace = SimpleNamespace(
        from_user=user_ns,
        chat=chat_ns,
        answer=answer_mock,
    )
    # type(event).__name__ becomes "SimpleNamespace" -- override for log clarity
    return event


def _write_registry(state_dir: Path, *, group_chat_ids: list[int]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    groups_payload: dict[str, dict[str, str]] = {
        str(cid): {
            "chat_id": str(cid),
            "chat_type": "supergroup",
            "chat_title": "Test",
            "chat_username": "",
        }
        for cid in group_chat_ids
    }
    payload: dict[str, Any] = {
        "admin_ids": [],
        "user_ids": [],
        "groups": groups_payload,
    }
    (state_dir / "bot_known_groups.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def isolated_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect get_project_paths().project_root to tmp_path so the
    registry file lives in tmp_path/state/bot_known_groups.json."""
    from app.paths import project_paths as paths_module

    fake_paths: SimpleNamespace = SimpleNamespace(project_root=tmp_path)
    monkeypatch.setattr(paths_module, "get_project_paths", lambda: fake_paths)
    # Also patch the import site inside group_registry
    from app.telegram_bot import group_registry as gr_module
    monkeypatch.setattr(gr_module, "get_project_paths", lambda: fake_paths)
    return tmp_path


@pytest.mark.asyncio
async def test_anonymous_admin_from_known_group_is_allowed_as_admin(
    isolated_project_root: Path,
) -> None:
    """GroupAnonymousBot from a registered group must get ADMIN role."""
    known_chat_id: int = -1001234567890
    _write_registry(isolated_project_root / "state", group_chat_ids=[known_chat_id])

    middleware: RoleMiddleware = RoleMiddleware(
        admin_ids=frozenset({999}),  # GroupAnonymousBot ID is NOT in admin list
        user_ids=frozenset(),
    )
    event: SimpleNamespace = _make_message_event(
        user_id=GROUP_ANONYMOUS_BOT_ID,
        username="GroupAnonymousBot",
        chat_id=known_chat_id,
    )
    handler: AsyncMock = AsyncMock(return_value="HANDLER_CALLED")
    data: dict[str, Any] = {}

    result: Any = await middleware(handler, event, data)

    handler.assert_awaited_once()
    assert result == "HANDLER_CALLED"
    assert data.get("user_role") == UserRole.ADMIN
    event.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_anonymous_admin_from_unknown_group_is_denied_with_explanatory_message(
    isolated_project_root: Path,
) -> None:
    """GroupAnonymousBot from an unregistered group must be denied with
    an explanatory message (not generic 'Access denied').

    REGRESSION GUARD: the unknown chat MUST NOT be auto-registered as a
    side effect of the denial — otherwise the next anonymous attempt
    from the same chat would pass, breaking the 'known = trusted' filter.
    """
    _write_registry(isolated_project_root / "state", group_chat_ids=[-100111])

    middleware: RoleMiddleware = RoleMiddleware(
        admin_ids=frozenset({999}),
        user_ids=frozenset(),
    )
    unknown_chat_id: int = -100222
    event: SimpleNamespace = _make_message_event(
        user_id=GROUP_ANONYMOUS_BOT_ID,
        username="GroupAnonymousBot",
        chat_id=unknown_chat_id,
    )
    handler: AsyncMock = AsyncMock()
    data: dict[str, Any] = {}

    result: Any = await middleware(handler, event, data)

    handler.assert_not_awaited()
    assert result is None
    assert "user_role" not in data
    event.answer.assert_awaited_once()
    answer_text: str = event.answer.await_args.args[0]
    assert "анонимно" in answer_text.lower()
    assert "не зарегистрирована" in answer_text

    # REGRESSION GUARD: the registry must NOT contain the unknown chat after denial.
    from app.telegram_bot.group_registry import load_known_group_ids
    assert unknown_chat_id not in load_known_group_ids(), (
        "Unknown chat was auto-registered during anonymous-admin denial; "
        "this breaks the 'known group = trusted' semantics. "
        "register_group_from_event must not run before the known-group check."
    )


@pytest.mark.asyncio
async def test_anonymous_admin_from_unknown_group_twice_stays_denied(
    isolated_project_root: Path,
) -> None:
    """REGRESSION GUARD: a second attempt from the same unknown chat
    must still be denied. If the first attempt had auto-registered the
    chat, the second one would slip through."""
    _write_registry(isolated_project_root / "state", group_chat_ids=[])

    middleware: RoleMiddleware = RoleMiddleware(
        admin_ids=frozenset({999}),
        user_ids=frozenset(),
    )
    unknown_chat_id: int = -100999

    for attempt in range(2):
        event: SimpleNamespace = _make_message_event(
            user_id=GROUP_ANONYMOUS_BOT_ID,
            username="GroupAnonymousBot",
            chat_id=unknown_chat_id,
        )
        handler: AsyncMock = AsyncMock()
        data: dict[str, Any] = {}

        await middleware(handler, event, data)

        handler.assert_not_awaited()
        assert "user_role" not in data, f"attempt {attempt + 1} unexpectedly allowed"
        event.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_regular_user_in_admin_list_still_works(
    isolated_project_root: Path,
) -> None:
    """Path 1 (standard authorization) must keep working unchanged."""
    _write_registry(isolated_project_root / "state", group_chat_ids=[])

    middleware: RoleMiddleware = RoleMiddleware(
        admin_ids=frozenset({987654321}),
        user_ids=frozenset(),
    )
    event: SimpleNamespace = _make_message_event(
        user_id=987654321,
        username="agniyoga65",
        chat_id=-100333,
    )
    handler: AsyncMock = AsyncMock(return_value="OK")
    data: dict[str, Any] = {}

    await middleware(handler, event, data)

    handler.assert_awaited_once()
    assert data.get("user_role") == UserRole.ADMIN
    event.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_user_still_denied_generically(
    isolated_project_root: Path,
) -> None:
    """Path 3: unauthorized user with a real user_id keeps the original
    'Access denied.' message (no anonymous-specific text)."""
    _write_registry(isolated_project_root / "state", group_chat_ids=[])

    middleware: RoleMiddleware = RoleMiddleware(
        admin_ids=frozenset({999}),
        user_ids=frozenset(),
    )
    event: SimpleNamespace = _make_message_event(
        user_id=12345,
        username="randomguy",
        chat_id=-100444,
    )
    handler: AsyncMock = AsyncMock()
    data: dict[str, Any] = {}

    await middleware(handler, event, data)

    handler.assert_not_awaited()
    event.answer.assert_awaited_once()
    answer_text: str = event.answer.await_args.args[0]
    assert answer_text == "⛔ Access denied."


@pytest.mark.asyncio
async def test_load_known_group_ids_reads_fresh_on_each_call(
    isolated_project_root: Path,
) -> None:
    """No caching: new group registrations must be visible immediately."""
    from app.telegram_bot.group_registry import load_known_group_ids

    _write_registry(isolated_project_root / "state", group_chat_ids=[-100111])
    assert load_known_group_ids() == frozenset({-100111})

    _write_registry(
        isolated_project_root / "state",
        group_chat_ids=[-100111, -100222],
    )
    assert load_known_group_ids() == frozenset({-100111, -100222})
