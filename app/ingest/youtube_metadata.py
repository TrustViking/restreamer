from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from app.bootstrap.logging_config import get_logger
from app.core.models import VideoMetadata
from app.core.video_title_cleanup import sanitize_source_video_title


LOGGER = get_logger(__name__)


class YouTubeMetadataFetcher:
    def fetch(self, video_url: str) -> VideoMetadata:
        raise NotImplementedError


class YtDlpYouTubeMetadataFetcher(YouTubeMetadataFetcher):
    """Fetcher через локальный бинарник yt-dlp.exe (subprocess + JSON).

    Принимает ytdlp_path=None для обратной совместимости - в этом случае
    путь резолвится из ProjectPaths при первом вызове.
    """

    def __init__(
        self,
        ytdlp_path: Optional[Path] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._ytdlp_path: Optional[Path] = ytdlp_path
        self._timeout_seconds: float = timeout_seconds

    def _resolve_path(self) -> Path:
        if self._ytdlp_path is not None:
            return self._ytdlp_path
        from app.paths.project_paths import get_project_paths
        return get_project_paths().ytdlp_exe_path

    def fetch(self, video_url: str) -> VideoMetadata:
        ytdlp_path: Path = self._resolve_path()
        command: list[str] = [
            str(ytdlp_path),
            "--dump-single-json",
            "--no-warnings",
            "--skip-download",
            "--no-playlist",
        ]

        from app.paths.project_paths import get_project_paths
        project_paths = get_project_paths()

        cookies_path: Path = project_paths.cookies_file_path
        if cookies_path.exists():
            command += ["--cookies", str(cookies_path)]

        deno_path: Path = project_paths.deno_exe_path
        if deno_path.exists():
            command += ["--js-runtimes", f"deno:{deno_path}"]

        command.append(video_url)
        LOGGER.debug("yt-dlp metadata command: %s", " ".join(command))

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=self._timeout_seconds,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"yt-dlp не смог получить metadata для {video_url}. "
                f"stderr: {(result.stderr or '').strip()}"
            )

        try:
            info: Dict[str, Any] = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"yt-dlp вернул невалидный JSON для {video_url}: {exc}"
            ) from exc

        return self._build_metadata(video_url=video_url, info=info)

    def _build_metadata(self, *, video_url: str, info: Dict[str, Any]) -> VideoMetadata:
        title: str = str(info.get("title") or "").strip()
        title = sanitize_source_video_title(title)
        description: str = str(info.get("description") or "").strip()
        thumbnail_url: str = str(info.get("thumbnail") or "").strip()
        video_id: str = str(info.get("id") or "").strip()
        youtube_language: Optional[str] = str(info.get("language") or "").strip() or None
        channel_language: Optional[str] = (
            str(info.get("channel_language") or "").strip() or None
        )
        duration_value: Any = info.get("duration")
        duration_seconds: Optional[int] = None
        if isinstance(duration_value, (int, float)) and duration_value > 0:
            duration_seconds = int(duration_value)
        canonical_url: str = (
            str(info.get("webpage_url") or info.get("original_url") or video_url).strip()
            or video_url
        )

        formats_list: list = info.get("formats") or []
        audio_langs: list[str] = []
        for fmt in formats_list:
            acodec_val: str = str(fmt.get("acodec") or "").strip()
            fmt_lang: str = str(fmt.get("language") or "").strip()
            if acodec_val and acodec_val != "none" and fmt_lang and fmt_lang not in audio_langs:
                audio_langs.append(fmt_lang)

        raw_subtitles: dict = info.get("subtitles") or {}
        raw_auto_captions: dict = info.get("automatic_captions") or {}
        subtitle_lang_keys: tuple[str, ...] = tuple(raw_subtitles.keys())
        auto_caption_lang_keys: tuple[str, ...] = tuple(raw_auto_captions.keys())

        LOGGER.debug(
            "yt-dlp diagnostics: url=%s title=%r youtube_language=%s "
            "channel_language=%s audio_languages=%s subtitle_languages=%s "
            "auto_caption_languages=%s formats_count=%d",
            video_url,
            title,
            youtube_language,
            channel_language,
            audio_langs,
            subtitle_lang_keys,
            auto_caption_lang_keys,
            len(formats_list),
        )

        if not title:
            raise ValueError("Не удалось получить title (yt-dlp вернул пусто).")
        if not thumbnail_url:
            fallback_id: Optional[str] = None
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                fallback_id = video_id
            else:
                fallback_match: Optional[re.Match[str]] = re.search(
                    r"(?:youtu\.be/|v=)([A-Za-z0-9_-]{11})",
                    video_url,
                )
                if fallback_match:
                    fallback_id = fallback_match.group(1)

            if fallback_id:
                thumbnail_url = f"https://i.ytimg.com/vi/{fallback_id}/hqdefault.jpg"
                LOGGER.warning(
                    "yt-dlp returned empty thumbnail_url; using fallback video id %s: %s",
                    fallback_id,
                    thumbnail_url,
                )
            else:
                raise ValueError("Не удалось получить thumbnail_url (yt-dlp вернул пусто).")
        if not description:
            LOGGER.warning(
                "yt-dlp returned empty description for url=%s (id=%s).",
                video_url,
                video_id or "unknown",
            )

        return VideoMetadata(
            url=video_url,
            title=title,
            description=description,
            thumbnail_url=thumbnail_url,
            youtube_language=youtube_language,
            channel_language=channel_language,
            duration_seconds=duration_seconds,
            canonical_url=canonical_url,
            audio_languages=tuple(audio_langs),
            subtitle_languages=subtitle_lang_keys,
            auto_caption_languages=auto_caption_lang_keys,
        )


def extract_youtube_video_id(raw_value: str) -> Optional[str]:
    text: str = str(raw_value or "").strip()
    if not text:
        return None
    context_patterns: Tuple[Tuple[str, str], ...] = (
        (
            "context",
            r"(?:https?://)?(?:www\.)?youtu\.be/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/watch\?[^#\s]*?(?:[?&]v=|&v=)([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
        (
            "context",
            r"(?:https?://)?(?:www\.)?(?:m\.)?youtube\.com/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})(?:[^A-Za-z0-9_-]|$)",
        ),
    )
    first_context_match: Optional[Tuple[int, str, str]] = None
    for method_name, pattern in context_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate: str = str(match.group(1) or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                continue
            candidate_pos: int = match.start(1)
            if first_context_match is None or candidate_pos < first_context_match[0]:
                first_context_match = (candidate_pos, method_name, candidate)
    if first_context_match is not None:
        _, method_name, candidate = first_context_match
        LOGGER.debug("youtube_id_extracted method=%s id=%s", method_name, candidate)
        return candidate

    for match in re.finditer(r"([A-Za-z0-9_-]{11})", text):
        candidate = str(match.group(1) or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            continue
        start_index: int = match.start(1)
        end_index: int = match.end(1)
        char_before: str = text[start_index - 1] if start_index > 0 else ""
        char_after: str = text[end_index] if end_index < len(text) else ""
        if char_before and re.fullmatch(r"[A-Za-z0-9_-]", char_before):
            continue
        if char_after and re.fullmatch(r"[A-Za-z0-9_-]", char_after):
            continue
        LOGGER.debug("youtube_id_extracted method=raw id=%s", candidate)
        return candidate
    return None


def normalize_youtube_link(raw: str) -> Optional[str]:
    video_id: Optional[str] = extract_youtube_video_id(raw)
    if not video_id:
        return None
    return f"https://youtu.be/{video_id}"


def normalize_youtube_video_url(video_url: str) -> str:
    normalized: Optional[str] = normalize_youtube_link(video_url)
    if not normalized:
        raise ValueError(f"Не удалось извлечь YouTube video id из URL: {video_url}")
    return normalized
