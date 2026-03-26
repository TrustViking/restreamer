from __future__ import annotations

import dataclasses
import time
from typing import Callable, List

from app.bootstrap.logging_config import get_logger as _get_logger_impl
from app.config.settings import AppConfig
from app.core.branching import BRANCH_MERGE
from app.core.models import (
    BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
    BLOCK_GENERATION_MODE_REAL_MERGE,
    LanguageMergeAttempt,
    MergedLanguageContent,
    PlannedVideo,
)
from app.llm.llm_client import LlmTraceContext, OpenAITransportResult
from app.llm.llm_factory import get_llm_provider
from app.llm.merges.merge_parser import build_plain_merged_content_or_raise, clean_and_validate_llm_description
from app.llm.merges.merge_prompt import _single_source_translate_prompt
from app.llm.models.model_compatibility import LlmModelConfigurationError
from app.llm.models.model_identity import resolve_effective_llm_model
from app.llm.providers.provider_base import LlmProvider

LOGGER = _get_logger_impl(__name__)


def _main_stage_field_source(*, provider_name: str) -> str:
    normalized_provider_name: str = str(provider_name or "").strip().lower()
    return normalized_provider_name or "main_model"


class SingleSourceTranslator:
    """Translates a single-source video description via LLM."""

    _FALLBACK_TITLE_SOURCE: str = "fallback_titles"
    _FALLBACK_HOOK_SOURCE: str = "fallback_none"
    _FALLBACK_HASHTAGS_SOURCE: str = "fallback_none"
    _FALLBACK_BODY_SOURCE: str = "fallback_source_descriptions"
    _SUCCESS_PUBLISH_LABEL: str = "single_source_plain_ok"
    _FAILED_PUBLISH_LABEL: str = "single_source_plain_failed"
    _DEFAULT_TITLE: str = "Untitled"
    _BODY_SOURCE_LABEL: str = "main_merge"

    def __init__(
        self,
        *,
        config: AppConfig,
        branch_label: str,
        date_key: str,
        slot_key: str,
    ) -> None:
        self._config: AppConfig = config
        self._branch_label: str = branch_label
        self._date_key: str = date_key
        self._slot_key: str = slot_key
        self._model_name: str = resolve_effective_llm_model(config)
        self._provider: LlmProvider = get_llm_provider(config=config)
        self._main_source_label: str = _main_stage_field_source(
            provider_name=self._provider.name
        )

    def _request_plain_text(
        self,
        *,
        prompt_text: str,
        attempt_label: str,
        trace_context: LlmTraceContext,
    ) -> str:
        if self._provider.pre_delay_sec(config=self._config) > 0:
            time.sleep(max(0.0, float(self._provider.pre_delay_sec(config=self._config))))
        response: OpenAITransportResult = self._provider.request_merge(
            prompt_text=prompt_text,
            config=self._config,
            model_name=self._model_name,
            attempt_label=attempt_label,
            max_output_tokens=self._config.llm.max_output_tokens,
            structured_schema=None,
            temperature=0.0,
            trace_context=trace_context,
        )
        return response.raw_text

    def run(
        self,
        *,
        language: str,
        videos: List[PlannedVideo],
        attempt_label: str,
        summarize_error: Callable[[Exception], str],
        no_description_text: str,
    ) -> LanguageMergeAttempt:
        if len(videos) != 1:
            raise RuntimeError("single-source translate expects exactly one video")
        source_video: PlannedVideo = videos[0]
        try:
            raw_text: str = self._request_plain_text(
                prompt_text=_single_source_translate_prompt(
                    source_language=source_video.language,
                    target_language=language,
                    source_description=source_video.metadata.description.strip() or no_description_text,
                ),
                attempt_label=attempt_label,
                trace_context=LlmTraceContext(
                    branch_label=self._branch_label,
                    date_key=self._date_key,
                    slot_key=self._slot_key,
                    language=language,
                    provider=self._provider.name,
                    model_name=self._model_name,
                    attempt_index=1,
                    request_kind="single_source_plain",
                    source_count=1,
                ),
            )
            cleaned_text, is_valid, reasons = clean_and_validate_llm_description(text=raw_text)
            if not is_valid:
                raise RuntimeError(
                    "single-source plain validation failed: " + ("; ".join(reasons) or "unknown")
                )
            merged_content: MergedLanguageContent = build_plain_merged_content_or_raise(
                model_name=self._model_name,
                title_text=source_video.metadata.title.strip() or self._DEFAULT_TITLE,
                description_text=cleaned_text,
            )
            return LanguageMergeAttempt(
                language=language,
                model_name=self._model_name,
                raw_response_text=raw_text,
                merged=dataclasses.replace(
                    merged_content,
                    branch_type=BRANCH_MERGE,
                    title_source=self._main_source_label,
                    hook_source=self._main_source_label,
                    hashtags_source=self._main_source_label,
                    body_source=self._BODY_SOURCE_LABEL,
                    block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
                ),
                error_summary=None,
                salvaged_title=merged_content.title,
                publish_source_label=self._SUCCESS_PUBLISH_LABEL,
                generator_model_name=self._model_name,
                used_model_names=(self._model_name,),
                branch_type=BRANCH_MERGE,
                title_source=self._main_source_label,
                hook_source=self._main_source_label,
                hashtags_source=self._main_source_label,
                body_source=self._BODY_SOURCE_LABEL,
                block_generation_mode=BLOCK_GENERATION_MODE_REAL_MERGE,
            )
        except LlmModelConfigurationError:
            raise
        except Exception as error:
            return LanguageMergeAttempt(
                language=language,
                model_name=self._model_name,
                raw_response_text="",
                merged=None,
                error_summary=summarize_error(error),
                salvaged_title=source_video.metadata.title.strip() or self._DEFAULT_TITLE,
                publish_source_label=self._FAILED_PUBLISH_LABEL,
                generator_model_name=self._model_name,
                used_model_names=(self._model_name,),
                branch_type=BRANCH_MERGE,
                title_source=self._FALLBACK_TITLE_SOURCE,
                hook_source=self._FALLBACK_HOOK_SOURCE,
                hashtags_source=self._FALLBACK_HASHTAGS_SOURCE,
                body_source=self._FALLBACK_BODY_SOURCE,
                block_generation_mode=BLOCK_GENERATION_MODE_FALLBACK_AFTER_MERGE_FAILURE,
            )
