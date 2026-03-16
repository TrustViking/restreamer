from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OpenAIRequestCompatibility:
    model_name: str
    normalized_model_name: str
    model_family: str
    reasoning_effort_supported: bool
    reasoning_effort_enabled: bool
    structured_output_supported: bool
    structured_output_requested: bool
    temperature_supported: bool
    temperature_enabled: bool
    capability_source: str


@dataclass(frozen=True)
class LlmRequestErrorClassification:
    reason_code: str
    retryable: bool
    fatal_model_configuration: bool
    status_code: Optional[int]
    api_error_code: str
    api_error_param: str
    detail: str


@dataclass(frozen=True)
class LlmModelConfigurationError(RuntimeError):
    provider_name: str
    model_name: str
    reason_code: str
    detail: str
    status_code: Optional[int] = None
    api_error_code: str = ""
    api_error_param: str = ""

    def __str__(self) -> str:
        return self.detail


def _normalize_model_name(model_name: str) -> str:
    return str(model_name or "").strip().lower()


def _openai_model_family(normalized_model_name: str) -> str:
    if normalized_model_name.startswith("gpt-5"):
        return "gpt-5"
    if normalized_model_name.startswith("gpt-4o"):
        return "gpt-4o"
    if normalized_model_name.startswith("gpt-4.1"):
        return "gpt-4.1"
    if normalized_model_name.startswith("gpt-4"):
        return "gpt-4"
    if normalized_model_name.startswith(("o1", "o3", "o4")):
        return "o-series"
    if normalized_model_name.startswith("gpt"):
        return "gpt-other"
    if normalized_model_name:
        return "other"
    return "unknown"


def _openai_reasoning_effort_supported(model_family: str) -> bool:
    return model_family in {"gpt-5", "o-series"}


def _openai_structured_output_supported(normalized_model_name: str) -> bool:
    return bool(normalized_model_name)


def _openai_temperature_supported(normalized_model_name: str) -> bool:
    # Temperature is intentionally disabled for all gpt-* models:
    # gpt-5 uses reasoning via the Responses API (temperature not applicable),
    # and for gpt-4x models we also send no temperature by design.
    # o-series models technically don't support it either, hence the blanket rule.
    return not normalized_model_name.startswith("gpt")


def resolve_openai_request_compatibility(
    *,
    model_name: str,
    structured_output_requested: bool,
    temperature_requested: bool,
) -> OpenAIRequestCompatibility:
    normalized_model_name: str = _normalize_model_name(model_name)
    model_family: str = _openai_model_family(normalized_model_name)
    reasoning_effort_supported: bool = _openai_reasoning_effort_supported(model_family)
    structured_output_supported: bool = _openai_structured_output_supported(
        normalized_model_name
    )
    temperature_supported: bool = _openai_temperature_supported(normalized_model_name)
    return OpenAIRequestCompatibility(
        model_name=str(model_name or "").strip(),
        normalized_model_name=normalized_model_name,
        model_family=model_family,
        reasoning_effort_supported=reasoning_effort_supported,
        reasoning_effort_enabled=reasoning_effort_supported,
        structured_output_supported=structured_output_supported,
        structured_output_requested=structured_output_requested,
        temperature_supported=temperature_supported,
        temperature_enabled=temperature_requested and temperature_supported,
        capability_source="heuristic_name_rules",
    )


def classify_openai_request_error(error: Exception) -> LlmRequestErrorClassification:
    error_type_name: str = type(error).__name__
    detail: str = str(error or "").strip() or error_type_name
    status_code: Optional[int] = getattr(error, "status_code", None)
    api_error_code: str = str(getattr(error, "code", "") or "").strip().lower()
    api_error_param: str = str(getattr(error, "param", "") or "").strip().lower()
    detail_lower: str = detail.lower()

    if error_type_name == "APITimeoutError":
        return LlmRequestErrorClassification(
            reason_code="openai_timeout",
            retryable=True,
            fatal_model_configuration=False,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    if error_type_name == "APIConnectionError":
        return LlmRequestErrorClassification(
            reason_code="openai_connection_error",
            retryable=True,
            fatal_model_configuration=False,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    if error_type_name == "RateLimitError" or status_code == 429:
        return LlmRequestErrorClassification(
            reason_code="openai_rate_limit",
            retryable=True,
            fatal_model_configuration=False,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    if error_type_name == "InternalServerError" or (
        isinstance(status_code, int) and status_code >= 500
    ):
        return LlmRequestErrorClassification(
            reason_code="openai_server_error",
            retryable=True,
            fatal_model_configuration=False,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    if error_type_name == "AuthenticationError" or status_code == 401:
        return LlmRequestErrorClassification(
            reason_code="openai_authentication_failed",
            retryable=False,
            fatal_model_configuration=True,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    if error_type_name == "PermissionDeniedError" or status_code == 403:
        return LlmRequestErrorClassification(
            reason_code="openai_model_access_denied",
            retryable=False,
            fatal_model_configuration=True,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    if error_type_name == "NotFoundError" or api_error_code == "model_not_found" or status_code == 404:
        return LlmRequestErrorClassification(
            reason_code="openai_model_not_found",
            retryable=False,
            fatal_model_configuration=True,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    if error_type_name in {"BadRequestError", "UnprocessableEntityError"} or status_code in {400, 422}:
        if api_error_code == "unsupported_parameter" or "unsupported parameter" in detail_lower:
            reason_code: str = (
                "openai_incompatible_request_shape"
                if api_error_param.startswith("text")
                else "openai_unsupported_parameter"
            )
            return LlmRequestErrorClassification(
                reason_code=reason_code,
                retryable=False,
                fatal_model_configuration=True,
                status_code=status_code,
                api_error_code=api_error_code,
                api_error_param=api_error_param,
                detail=detail,
            )
        if any(
            token in detail_lower
            for token in ("json_schema", "response format", "response_format", "text.format")
        ) or api_error_param.startswith("text"):
            return LlmRequestErrorClassification(
                reason_code="openai_incompatible_request_shape",
                retryable=False,
                fatal_model_configuration=True,
                status_code=status_code,
                api_error_code=api_error_code,
                api_error_param=api_error_param,
                detail=detail,
            )
        return LlmRequestErrorClassification(
            reason_code="openai_bad_request",
            retryable=False,
            fatal_model_configuration=True,
            status_code=status_code,
            api_error_code=api_error_code,
            api_error_param=api_error_param,
            detail=detail,
        )
    return LlmRequestErrorClassification(
        reason_code="openai_request_failed",
        retryable=False,
        fatal_model_configuration=False,
        status_code=status_code,
        api_error_code=api_error_code,
        api_error_param=api_error_param,
        detail=detail,
    )
