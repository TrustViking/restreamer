from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.llm.models.model_identity import resolve_effective_llm_model
from app.paths import get_project_paths

LOGGER: logging.Logger = _get_logger_impl(__name__)

_HARDCODED_SEED: dict[str, dict[str, str]] = {
    "official_links": {
        "uk": "🌐 Офіційні ресурси:",
        "en": "🌐 Official links:",
        "ru": "🌐 Официальные ссылки:",
    },
    "recommended_materials": {
        "uk": "Рекомендовані матеріали:",
        "en": "Recommended materials:",
        "ru": "Рекомендуемые материалы:",
    },
}

_SEED_LANGUAGES: frozenset[str] = frozenset({"uk", "en", "ru"})
_INVALID_LANGUAGE_INPUTS: frozenset[str] = frozenset(
    {"", "unknown", "other", "none", "und", "xx"}
)

_LOCK: threading.Lock = threading.Lock()
_DISK_CACHE_LOADED: bool = False
_IN_MEMORY_CACHE: dict[str, dict[str, str]] = {
    "official_links": {},
    "recommended_materials": {},
}
_STORED_CONFIG: Optional[AppConfig] = None


def openai_request_merge(*args: object, **kwargs: object) -> object:
    from app.llm.llm_client import openai_request_merge as _openai_request_merge

    return _openai_request_merge(*args, **kwargs)


def init_heading_resolver(*, config: AppConfig) -> None:
    global _STORED_CONFIG
    _STORED_CONFIG = config


def resolve_official_links_heading(language: str) -> str:
    return _resolve(kind="official_links", language=language)


def resolve_recommended_materials_heading(language: str) -> str:
    return _resolve(kind="recommended_materials", language=language)


def _resolve(*, kind: str, language: str) -> str:
    normalized: str = str(language or "").strip().lower()

    if normalized in _INVALID_LANGUAGE_INPUTS:
        LOGGER.debug("heading_cache_invalid_input kind=%s language=%r", kind, language)
        return _HARDCODED_SEED[kind]["en"]

    if not (len(normalized) == 2 and normalized.isascii() and normalized.isalpha()):
        LOGGER.warning(
            "heading_cache_invalid_language_format kind=%s language=%r",
            kind,
            language,
        )
        return _HARDCODED_SEED[kind]["en"]

    if normalized in _HARDCODED_SEED[kind]:
        LOGGER.debug("heading_cache_seed_hit kind=%s language=%s", kind, normalized)
        return _HARDCODED_SEED[kind][normalized]

    with _LOCK:
        cached: Optional[str] = _IN_MEMORY_CACHE[kind].get(normalized)
        if cached:
            LOGGER.info("heading_cache_memory_hit kind=%s language=%s", kind, normalized)
            return cached

        if not _DISK_CACHE_LOADED:
            _load_cache_from_disk()

        cached = _IN_MEMORY_CACHE[kind].get(normalized)
        if cached:
            LOGGER.info("heading_cache_disk_hit kind=%s language=%s", kind, normalized)
            return cached

    if _STORED_CONFIG is None:
        LOGGER.warning(
            "heading_resolver_not_initialized kind=%s language=%s using_fallback=en",
            kind,
            normalized,
        )
        return _HARDCODED_SEED[kind]["en"]

    LOGGER.info("heading_cache_miss kind=%s language=%s", kind, normalized)
    translated: Optional[str] = _translate_heading_via_llm(kind=kind, language=normalized)
    if translated is None:
        LOGGER.info(
            "heading_cache_fallback_used kind=%s language=%s reason=translation_failed",
            kind,
            normalized,
        )
        return _HARDCODED_SEED[kind]["en"]

    if kind == "official_links":
        wrapped: str = f"🌐 {translated}"
    else:
        wrapped = translated

    with _LOCK:
        _IN_MEMORY_CACHE[kind][normalized] = wrapped
        _persist_cache_to_disk()

    LOGGER.info(
        "heading_cache_populated kind=%s language=%s value=%r",
        kind,
        normalized,
        wrapped,
    )
    return wrapped


def _translate_heading_via_llm(*, kind: str, language: str) -> Optional[str]:
    config: Optional[AppConfig] = _STORED_CONFIG
    if config is None:
        return None
    source_phrase: str = (
        "Official links:" if kind == "official_links" else "Recommended materials:"
    )

    prompt_text: str = (
        f"You are a translator. Translate the exact English phrase below into the language "
        f'identified by the ISO 639-1 code "{language}".\n'
        f"\n"
        f'Source phrase: "{source_phrase}"\n'
        f"\n"
        f"Output rules:\n"
        f"- Return ONLY the translated phrase, on a single line.\n"
        f'- Preserve the trailing colon ":".\n'
        f"- Do not add quotation marks, explanations, romanization, or any prefix or suffix.\n"
        f"- Do not include the original English phrase.\n"
        f"- Do not include the language code or language name.\n"
        f"- Do not include emoji.\n"
    )

    try:
        from app.llm.llm_client import LlmTraceContext

        model_name: str = resolve_effective_llm_model(config)
        trace_context: LlmTraceContext = LlmTraceContext(
            branch_label="heading_translation",
            date_key="-",
            slot_key="-",
            language=language,
            provider="openai",
            model_name=model_name,
            attempt_index=0,
            request_kind="heading_translation",
            source_count=0,
        )
        result = openai_request_merge(
            prompt_text=prompt_text,
            model_name=model_name,
            timeout_sec=min(15.0, float(config.llm.timeout_sec)),
            attempt_label="heading_translation",
            max_output_tokens=64,
            reasoning_effort="low",  # 64-token budget: heavier reasoning would starve the answer
            structured_schema=None,
            temperature=0.0,
            trace_context=trace_context,
        )
    except Exception as error:
        LOGGER.warning(
            "heading_translation_failed kind=%s language=%s error=%s",
            kind,
            language,
            type(error).__name__,
        )
        return None

    raw_text: str = str(result.raw_text or "").strip()
    rejection_reason: Optional[str] = _validate_translation(raw_text)
    if rejection_reason is not None:
        LOGGER.warning(
            "heading_translation_rejected kind=%s language=%s reason=%s raw=%r",
            kind,
            language,
            rejection_reason,
            raw_text,
        )
        return None

    if raw_text.startswith("🌐"):
        raw_text = raw_text.lstrip("🌐").strip()

    LOGGER.info(
        "heading_translation_succeeded kind=%s language=%s value=%r",
        kind,
        language,
        raw_text,
    )
    return raw_text


def _validate_translation(raw: str) -> Optional[str]:
    if not raw:
        return "empty"
    if "\n" in raw:
        return "multiline"
    if len(raw) > 80:
        return "too_long"
    check_text: str = raw.lstrip("🌐").strip() if raw.startswith("🌐") else raw
    if not check_text:
        return "empty_after_emoji_strip"
    if not check_text.endswith(":"):
        return "missing_trailing_colon"
    forbidden_markers: tuple[str, ...] = ("<", ">", "*", "`", "[", "]")
    for marker in forbidden_markers:
        if marker in check_text:
            return f"forbidden_marker_{marker!r}"
    return None


def _load_cache_from_disk() -> None:
    global _DISK_CACHE_LOADED
    cache_path: Path = _get_cache_path()
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for kind in ("official_links", "recommended_materials"):
                    kind_payload = raw.get(kind)
                    if isinstance(kind_payload, dict):
                        for lang_key, value in kind_payload.items():
                            if isinstance(lang_key, str) and isinstance(value, str):
                                _IN_MEMORY_CACHE[kind][lang_key.strip().lower()] = value
        except (OSError, json.JSONDecodeError, ValueError) as error:
            LOGGER.warning(
                "heading_cache_disk_load_failed error=%s",
                type(error).__name__,
            )
    _DISK_CACHE_LOADED = True


def _persist_cache_to_disk() -> None:
    cache_path: Path = _get_cache_path()
    payload: dict[str, dict[str, str]] = {
        "official_links": dict(_IN_MEMORY_CACHE["official_links"]),
        "recommended_materials": dict(_IN_MEMORY_CACHE["recommended_materials"]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp_path, cache_path)
    except OSError as error:
        LOGGER.warning(
            "heading_cache_disk_persist_failed error=%s",
            type(error).__name__,
        )
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _get_cache_path() -> Path:
    return get_project_paths().state_dir / "heading_cache.json"
