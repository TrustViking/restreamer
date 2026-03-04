from __future__ import annotations

import logging
import re
from typing import Optional

from app.config.settings import AppConfig, AppTemplates
from app.core.language import detect_language
from app.core.models import VideoMetadata
from app.ingest.youtube_metadata import YouTubeMetadataFetcher, normalize_youtube_video_url
from app.media.image_normalizer import normalize_thumbnail
from app.net.http_client import HttpClient
from app.paths.name_builder import build_safe_entity_name
from app.publish.telegram_renderer import build_single_mode_message
from app.telegram.bot_client import TelegramBotClient


class SingleModeRunner:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        config: AppConfig,
        metadata_fetcher: YouTubeMetadataFetcher,
        http_client: HttpClient,
        telegram_client: TelegramBotClient,
        templates: Optional[AppTemplates],
    ) -> None:
        self._logger = logger
        self._config = config
        self._metadata_fetcher = metadata_fetcher
        self._http_client = http_client
        self._telegram_client = telegram_client
        self._templates = templates

    def run(self, video_url: str, dry_run: bool) -> None:
        if not re.match(r"^https?://", video_url):
            raise ValueError("Ожидался URL с http:// или https://")
        normalized_video_url: str = normalize_youtube_video_url(video_url)
        metadata: VideoMetadata = self._metadata_fetcher.fetch(
            video_url=normalized_video_url
        )
        language: str = detect_language(metadata)
        thumbnail_bytes: bytes = self._http_client.get_bytes(metadata.thumbnail_url)
        normalized_thumbnail = normalize_thumbnail(
            thumbnail_bytes,
            logger=self._logger,
        )
        message_text: str = build_single_mode_message(
            metadata=metadata,
            language=language,
            config=self._config,
            templates=self._templates,
        )
        if dry_run:
            self._logger.info("DRY RUN single mode message:\n%s", message_text)
            return
        self._telegram_client.send_text(text=message_text)
        self._telegram_client.send_photo_as_file_bytes(
            photo_bytes=normalized_thumbnail.bytes_data,
            filename=(
                f"{build_safe_entity_name(metadata.title)}"
                f"{normalized_thumbnail.extension}"
            ),
            mime_type=normalized_thumbnail.mime_type,
            caption=metadata.title[:900] if metadata.title else None,
        )
