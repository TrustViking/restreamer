from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable
from zoneinfo import ZoneInfo

from app.config.settings import AppConfig
from app.ingest.youtube_metadata import YouTubeMetadataFetcher, YtDlpYouTubeMetadataFetcher
from app.net.http_client import HttpClient
from app.paths.name_builder import NamePathBuilder
from app.pipeline.batch_runner import BatchRunner
from app.telegram.bot_client import TelegramBotClient


@dataclass(frozen=True)
class EntrypointRuntimeServices:
    metadata_fetcher: YouTubeMetadataFetcher
    http_client: HttpClient
    telegram_client: TelegramBotClient
    name_builder: NamePathBuilder
    batch_runner: BatchRunner


def build_entrypoint_runtime_services(
    *,
    logger: logging.Logger,
    config: AppConfig,
    kiev_tz: ZoneInfo,
    cet_tz: ZoneInfo,
    resolve_logger_name_meta: Callable[[], tuple[str, str, bool]],
) -> EntrypointRuntimeServices:
    metadata_fetcher: YouTubeMetadataFetcher = YtDlpYouTubeMetadataFetcher()
    http_client: HttpClient = HttpClient()
    telegram_client: TelegramBotClient = TelegramBotClient(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )
    name_builder: NamePathBuilder = NamePathBuilder(
        local_image_dir_template=config.local_image_dir_template,
        local_doc_dir_template=config.local_doc_dir_template,
        preview_name_template=config.templates.files_preview_name_template,
        doc_title_template=config.templates.files_doc_title_template,
        language_codes=config.templates.files_language_codes,
        max_filename_stem=config.preview_filename_max_stem,
    )
    batch_runner: BatchRunner = BatchRunner(
        logger=logger,
        config=config,
        metadata_fetcher=metadata_fetcher,
        http_client=http_client,
        telegram_client=telegram_client,
        name_builder=name_builder,
        kiev_tz=kiev_tz,
        cet_tz=cet_tz,
        resolve_logger_name_meta=resolve_logger_name_meta,
    )
    return EntrypointRuntimeServices(
        metadata_fetcher=metadata_fetcher,
        http_client=http_client,
        telegram_client=telegram_client,
        name_builder=name_builder,
        batch_runner=batch_runner,
    )
