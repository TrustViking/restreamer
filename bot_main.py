from __future__ import annotations

import asyncio
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
    setup_bot_logging()
    asyncio.run(run_bot())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
