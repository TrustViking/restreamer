from __future__ import annotations

import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from app.bootstrap.logging_config import resolve_base_logger_name, resolve_log_file_paths
from app.paths.project_paths import get_project_paths
from app.telegram_bot.bot_dispatcher import run_bot

if TYPE_CHECKING:
    from app.paths.project_paths import ProjectPaths

 
def main() -> int:
    detailed_log_file_path, screen_log_file_path = resolve_log_file_paths(
        entrypoint_label=f"{resolve_base_logger_name()}_bot"
    )
    detailed_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    screen_log_file_path.parent.mkdir(parents=True, exist_ok=True)
    stream_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
    )
    file_formatter: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    root_logger: logging.Logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler: logging.StreamHandler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(stream_formatter)
    root_logger.addHandler(stream_handler)

    detailed_file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(detailed_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    detailed_file_handler.setLevel(logging.DEBUG)
    detailed_file_handler.setFormatter(file_formatter)
    root_logger.addHandler(detailed_file_handler)

    screen_file_handler: logging.FileHandler = logging.FileHandler(
        filename=str(screen_log_file_path),
        mode="a",
        encoding="utf-8",
    )
    screen_file_handler.setLevel(logging.INFO)
    screen_file_handler.setFormatter(stream_formatter)
    root_logger.addHandler(screen_file_handler)

    root_logger.info(
        "Bot logging initialized stream_level=INFO detailed_file_level=DEBUG screen_file_level=INFO detailed_file=%s screen_file=%s",
        str(detailed_log_file_path),
        str(screen_log_file_path),
    )
    project_paths: ProjectPaths = get_project_paths()
    load_dotenv(dotenv_path=project_paths.secrets_env_path, override=False)
    asyncio.run(run_bot())
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 
