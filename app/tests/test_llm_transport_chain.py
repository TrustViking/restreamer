from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import httpx
import pytest

from app.llm.llm_client import (
    OpenAIResponsesRequestSpec,
    OpenAIResponsesTransport,
)
from app.llm.models.model_compatibility import resolve_openai_request_compatibility
from app.llm.models.model_identity import SERVICE_TIER_DEFAULT, SERVICE_TIER_FLEX

pytest.importorskip("openai")

from openai._exceptions import RateLimitError

_MODEL_NAME: str = "gpt-5.6-sol"
# gpt-* models are never sent a temperature, so the temperature fallback can only be
# reached with a model whose compatibility keeps temperature enabled.
_TEMPERATURE_MODEL_NAME: str = "custom-llm-v1"


def _flex_unavailable() -> RateLimitError:
    return RateLimitError(
        "Flex processing is currently unavailable. Please retry later.",
        response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
        body={"code": "resource_unavailable", "type": "requests"},
    )


def _temperature_unsupported() -> RuntimeError:
    return RuntimeError(
        "Error code: 400 - {'error': {'message': \"Unsupported parameter: 'temperature' is not "
        "supported with this model.\", 'type': 'invalid_request_error', 'param': 'temperature'}}"
    )


def _ok_response(*, incomplete_reason: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_ok",
        model=_MODEL_NAME,
        service_tier="flex",
        output_text='{"title": "T", "description": "D"}',
        incomplete_details=(
            SimpleNamespace(reason=incomplete_reason) if incomplete_reason else None
        ),
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _build_transport(
    create: Callable[..., Any],
    *,
    service_tier: str,
    model_name: str = _MODEL_NAME,
) -> OpenAIResponsesTransport:
    # No with_raw_response on the fake client -> the plain create path is exercised.
    client: SimpleNamespace = SimpleNamespace(responses=SimpleNamespace(create=create))
    spec: OpenAIResponsesRequestSpec = OpenAIResponsesRequestSpec(
        provider_name="openai",
        model_name=model_name,
        prompt_text="Prompt",
        structured_schema=None,
        temperature=0.0,
        reasoning_effort="low",
        max_output_tokens=100,
        attempt_label="test",
        request_kind="plain",
    )
    return OpenAIResponsesTransport(
        client=client,
        spec=spec,
        compatibility=resolve_openai_request_compatibility(
            model_name=model_name,
            structured_output_requested=False,
            temperature_requested=True,
        ),
        service_tier=service_tier,
    )


class TemperatureFallbackTests(unittest.TestCase):
    def test_temperature_unsupported_retries_without_temperature(self) -> None:
        calls: List[Dict[str, Any]] = []

        def _create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if len(calls) == 1:
                raise _temperature_unsupported()
            return _ok_response()

        transport: OpenAIResponsesTransport = _build_transport(
            _create,
            service_tier=SERVICE_TIER_DEFAULT,
            model_name=_TEMPERATURE_MODEL_NAME,
        )
        with patch("app.llm.llm_client.time.sleep") as sleep_mock:
            result = transport.send()

        self.assertEqual(2, len(calls))
        self.assertIn("temperature", calls[0])
        self.assertNotIn("temperature", calls[1])
        sleep_mock.assert_not_called()
        self.assertIn('"title"', result.raw_text)


class MaxOutputTokensRetryTests(unittest.TestCase):
    def test_max_output_tokens_retries_once_with_doubled_budget(self) -> None:
        calls: List[Dict[str, Any]] = []

        def _create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if len(calls) == 1:
                return _ok_response(incomplete_reason="max_output_tokens")
            return _ok_response()

        transport: OpenAIResponsesTransport = _build_transport(
            _create, service_tier=SERVICE_TIER_DEFAULT
        )
        with patch("app.llm.llm_client.time.sleep"):
            transport.send()

        self.assertEqual(2, len(calls))
        self.assertEqual(100, calls[0]["max_output_tokens"])
        self.assertEqual(200, calls[1]["max_output_tokens"])


class FlexFallbackChainTests(unittest.TestCase):
    def test_default_tier_does_not_retry_rate_limit(self) -> None:
        calls: List[Dict[str, Any]] = []

        def _create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            raise _flex_unavailable()

        transport: OpenAIResponsesTransport = _build_transport(
            _create, service_tier=SERVICE_TIER_DEFAULT
        )
        with patch("app.llm.llm_client.time.sleep") as sleep_mock:
            with self.assertRaises(RateLimitError):
                transport.send()

        self.assertEqual(1, len(calls))
        sleep_mock.assert_not_called()

    def test_flex_exhausts_retries_then_switches_to_default_tier(self) -> None:
        calls: List[Dict[str, Any]] = []

        def _create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if len(calls) <= 4:
                raise _flex_unavailable()
            return _ok_response()

        transport: OpenAIResponsesTransport = _build_transport(
            _create, service_tier=SERVICE_TIER_FLEX
        )
        with patch("app.llm.llm_client.time.sleep") as sleep_mock:
            transport.send()

        self.assertEqual(5, len(calls))
        self.assertEqual([SERVICE_TIER_FLEX] * 4, [call.get("service_tier") for call in calls[:4]])
        self.assertNotIn("service_tier", calls[4])
        self.assertEqual([20.0, 40.0, 80.0], [call.args[0] for call in sleep_mock.call_args_list])
        self.assertEqual(SERVICE_TIER_DEFAULT, transport._active_service_tier)

    def test_flex_does_not_retry_non_rate_limit_error(self) -> None:
        calls: List[Dict[str, Any]] = []

        def _create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            raise RuntimeError("boom")

        transport: OpenAIResponsesTransport = _build_transport(
            _create, service_tier=SERVICE_TIER_FLEX
        )
        with patch("app.llm.llm_client.time.sleep") as sleep_mock:
            with self.assertRaises(RuntimeError):
                transport.send()

        self.assertEqual(1, len(calls))
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
