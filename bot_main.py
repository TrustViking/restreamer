from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from app.bootstrap.logging_config import setup_bot_logging
from app.paths.project_paths import get_project_paths
from app.telegram_bot.bot_dispatcher import run_bot

if TYPE_CHECKING:
    from app.paths.project_paths import ProjectPaths


def main() -> int:
    project_paths: ProjectPaths = get_project_paths()
    load_dotenv(dotenv_path=project_paths.secrets_env_path, override=False)

    from app.bootstrap.logging_config import resolve_log_dir
    from app.runtime.single_instance import (
        AnotherInstanceRunning,
        acquire as acquire_lock,
        release as release_lock,
    )

    def _write_startup_event(message: str) -> None:
        """Append a timestamped line to logs/bot_startup.log.

        Used for startup-phase events that happen BEFORE setup_bot_logging()
        configures the normal logger. Errors swallowed - the operator must
        still see stderr regardless of any file-write failure.
        """
        try:
            from datetime import datetime

            log_dir = resolve_log_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            startup_log = log_dir / "bot_startup.log"
            ts = datetime.now().isoformat(timespec="seconds")
            with startup_log.open("a", encoding="utf-8") as fp:
                fp.write(f"{ts} | pid={os.getpid()} | {message}\n")
        except Exception:
            pass

    lock_path = project_paths.state_dir / "broadcaster.lock"
    try:
        acquire_lock(lock_path)
    except AnotherInstanceRunning as exc:
        _write_startup_event(
            f"lock_rejected another_pid={exc.pid} another_started_at={exc.started_at}"
        )
        # Console message stays - operator must see this immediately
        # without opening any log file.
        sys.stderr.write(
            f"\n⚠️  Уже работает другой экземпляр Broadcaster "
            f"(pid={exc.pid}, started_at={exc.started_at}).\n"
            f"   Закройте предыдущее окно и запустите снова.\n\n"
        )
        sys.stderr.flush()
        return 1

    _write_startup_event(f"lock_acquired path={lock_path}")

    try:
        try:
            setup_bot_logging()
            asyncio.run(run_bot())
            return 0
        except BaseException as exc:
            import traceback
            from datetime import datetime

            # Write to logs/bot_startup.log for fatal startup diagnostics.
            try:
                log_dir = resolve_log_dir()
                log_dir.mkdir(parents=True, exist_ok=True)
                fatal_log_path = log_dir / "bot_startup.log"
                with fatal_log_path.open("a", encoding="utf-8") as fp:
                    fp.write(
                        f"\n=== FATAL bot startup failure at {datetime.now().isoformat()} ===\n"
                    )
                    fp.write(f"pid={os.getpid()}\n")
                    fp.write(f"exception: {type(exc).__name__}: {exc}\n")
                    fp.write(traceback.format_exc())
                    fp.write("\n")
            except Exception:
                pass
            # Also try the regular logger in case it managed to come up
            try:
                import logging

                logging.getLogger("pipeline.bot").exception(
                    "bot_startup_failed exception=%s", exc
                )
            except Exception:
                pass
            raise
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
