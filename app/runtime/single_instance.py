"""Process-level single-instance lock for broadcaster.

Used to prevent multiple broadcaster.exe / bot_main.py processes from running
concurrently on the same machine. Without this, Telegram's polling protocol
round-robins updates between conflicting clients, causing duplicate pipeline
publications (verified in production logs 20260512_15*).

Lock file location: <project_root>/state/broadcaster.lock
Lock file content: one line, two whitespace-separated tokens:
    <pid> <iso_timestamp>

Acquire is atomic via os.open(..., O_CREAT|O_EXCL|O_WRONLY). On EEXIST we
inspect the existing PID and either reject (live) or overwrite (stale).

Stale-lock detection:
- POSIX: os.kill(pid, 0). If the process is not found, the lock is stale.
- Windows: ctypes OpenProcess + GetExitCodeProcess with explicit argtypes
  and restype (mandatory on Win64 so handle pointers are not truncated to
  32 bits). If the process cannot be opened or has exited (exit code !=
  STILL_ACTIVE), the lock is stale.

No external dependencies. Do not add psutil or pywin32.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Final

LOGGER: Final[logging.Logger] = logging.getLogger("pipeline.bot")

# Windows API constants
_PROCESS_QUERY_LIMITED_INFORMATION: Final[int] = 0x1000
_STILL_ACTIVE: Final[int] = 259


def _mirror_to_startup_log(message: str) -> None:
    """Append release events to logs/bot_startup.log.

    Mirrors the LOGGER.info call so release is still recorded even if
    setup_bot_logging() never ran (rare - only when startup itself
    failed before logging was configured).

    Silently swallows errors - never raise from a logging helper.
    """
    try:
        from datetime import datetime

        from app.bootstrap.logging_config import resolve_log_dir

        log_dir = resolve_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        startup_log = log_dir / "bot_startup.log"
        ts = datetime.now().isoformat(timespec="seconds")
        with startup_log.open("a", encoding="utf-8") as fp:
            fp.write(f"{ts} | pid={os.getpid()} | {message}\n")
    except Exception:
        pass


class AnotherInstanceRunning(RuntimeError):
    """Raised when another broadcaster process is detected as alive."""

    def __init__(self, pid: int, started_at: str) -> None:
        super().__init__(
            f"Another broadcaster instance is already running "
            f"(pid={pid}, started_at={started_at})"
        )
        self.pid = pid
        self.started_at = started_at


def _is_pid_alive(pid: int) -> bool:
    """Return True if a process with this PID is currently alive."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        # Explicit signatures are MANDATORY on 64-bit Windows.
        # Without them ctypes treats HANDLE returns as 32-bit ints and
        # truncates the upper 32 bits of pointer-sized handles, causing
        # silent corruption (CloseHandle on garbage, GetExitCodeProcess
        # always returning 0).
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return False
            return exit_code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # PID exists but we cannot signal it - still treat as alive.
            return True
        except OSError:
            return False
        return True


def _read_lock(lock_path: Path) -> tuple[int, str] | None:
    """Return (pid, iso_timestamp) from the lock file, or None if unreadable."""
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        LOGGER.warning("single_instance: cannot read lock file %s: %s", lock_path, exc)
        return None
    if not raw:
        return None
    parts = raw.split(maxsplit=1)
    if not parts:
        return None
    try:
        pid = int(parts[0])
    except ValueError:
        LOGGER.warning("single_instance: lock file %s has malformed pid", lock_path)
        return None
    timestamp = parts[1] if len(parts) > 1 else "unknown"
    return pid, timestamp


def _atomic_create_and_write(lock_path: Path) -> bool:
    """Try to atomically create the lock file with our PID + timestamp.

    Returns True on success, False if the file already exists (caller must
    then inspect the existing PID and decide stale vs live).
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{os.getpid()} {datetime.now().isoformat(timespec='seconds')}\n"
    try:
        fd = os.open(
            str(lock_path),
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError:
        return False
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return True


def acquire(lock_path: Path) -> None:
    """Acquire the single-instance lock at the given path.

    Atomic against concurrent starts: uses O_CREAT|O_EXCL so only one
    process can win the create race. The loser inspects the existing lock
    and either yields (live owner) or recovers (stale owner).

    Raises AnotherInstanceRunning if a live process already holds the lock.
    Overwrites stale locks (file exists but PID is dead) at most once per
    acquire call.
    """
    # First attempt: atomic create.
    if _atomic_create_and_write(lock_path):
        LOGGER.info(
            "single_instance: lock acquired pid=%d path=%s",
            os.getpid(),
            lock_path,
        )
        return

    # File already exists - inspect.
    existing = _read_lock(lock_path)
    if existing is None:
        # Race: someone created the file then we couldn't read it yet, or
        # file content is malformed/empty. Treat as stale, try once more.
        LOGGER.warning(
            "single_instance: lock at %s exists but unreadable - attempting recovery",
            lock_path,
        )
    else:
        pid, started_at = existing
        if pid == os.getpid():
            # Same process re-acquiring (shouldn't happen, but be permissive)
            return
        if _is_pid_alive(pid):
            raise AnotherInstanceRunning(pid=pid, started_at=started_at)
        LOGGER.info(
            "single_instance: stale lock detected (pid=%d started_at=%s) - overwriting",
            pid,
            started_at,
        )

    # Stale or unreadable lock - remove and retry atomic create once.
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        LOGGER.warning(
            "single_instance: cannot remove stale lock %s: %s", lock_path, exc
        )
        # Fall through; a racing process may have removed it already.
    if not _atomic_create_and_write(lock_path):
        # Someone else won the recovery race. Inspect and fail closed -
        # NEVER overwrite a lock we did not atomically create. Overwriting
        # would reintroduce the very race we are trying to prevent: two
        # processes both believing they hold the lock.
        existing_after = _read_lock(lock_path)
        if existing_after is not None:
            pid_after, ts_after = existing_after
            if pid_after != os.getpid() and _is_pid_alive(pid_after):
                raise AnotherInstanceRunning(pid=pid_after, started_at=ts_after)
            # If we reach here the file exists but the owner is either us
            # (impossible - we just lost O_EXCL to someone) or already
            # dead. Fall through to AnotherInstanceRunning anyway: this
            # branch is genuinely unreachable on a healthy machine, and
            # failing closed is safer than racing again.
        raise AnotherInstanceRunning(
            pid=existing_after[0] if existing_after else -1,
            started_at=(
                existing_after[1]
                if existing_after
                else "unknown (lost atomic-create race during stale-lock recovery)"
            ),
        )
    LOGGER.info(
        "single_instance: lock acquired pid=%d path=%s (after recovery)",
        os.getpid(),
        lock_path,
    )


def release(lock_path: Path) -> None:
    """Release the lock if owned by this process. Tolerates missing/foreign lock."""
    existing = _read_lock(lock_path)
    if existing is None:
        return
    pid, _ = existing
    if pid != os.getpid():
        LOGGER.warning(
            "single_instance: lock at %s owned by foreign pid=%d (we are pid=%d) - not releasing",
            lock_path,
            pid,
            os.getpid(),
        )
        return
    try:
        lock_path.unlink()
        LOGGER.info("single_instance: lock released pid=%d", os.getpid())
        _mirror_to_startup_log(f"lock_released path={lock_path}")
    except FileNotFoundError:
        pass
    except OSError as exc:
        LOGGER.warning("single_instance: failed to remove lock %s: %s", lock_path, exc)
