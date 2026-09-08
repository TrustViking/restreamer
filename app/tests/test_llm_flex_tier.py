from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from app.llm.llm_client import (
    _build_openai_responses_request_kwargs,
    openai_compatible_request_merge,
)
from app.llm.models.model_compatibility import resolve_openai_request_compatibility

pytest.importorskip("openai")

from openai._exceptions import RateLimitError


def _flex_unavailable() -> RateLimitError:
    return RateLimitError(
        "Flex processing is currently unavailable. Please retry later.",
        response=httpx.Response(429, request=httpx.Request("POST", "https://api.openai.com/v1/responses")),
        body={"code": "resource_unavailable", "type": "requests"},
    )


def _ok_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_ok",
        model="gpt-5.6-sol",
        service_tier="flex",
        output_text='{"title": "T", "description": "D"}',
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


class RequestKwargsServiceTierTests(unittest.TestCase):
    def _kwargs(self, service_tier: str) -> dict:
        compatibility = resolve_openai_request_compatibility(
            model_name="gpt-5.6-sol", structured_output_requested=False, temperature_requested=False
        )
        return _build_openai_responses_request_kwargs(
            prompt_text="Prompt",
            model_name="gpt-5.6-sol",
            max_tokens=100,
            structured_schema=None,
            temperature=0.0,
            compatibility=compatibility,
            reasoning_effort="low",
            service_tier=service_tier,
        )

    def test_default_tier_is_not_sent(self) -> None:
        self.assertNotIn("service_tier", self._kwargs("default"))

    def test_flex_tier_is_sent(self) -> None:
        self.assertEqual("flex", self._kwargs("flex")["service_tier"])


class FlexFallbackTests(unittest.TestCase):
    def _client(self, create) -> SimpleNamespace:
        client = SimpleNamespace(responses=SimpleNamespace(create=create))  # no with_raw_response -> plain create path
        client.with_options = lambda **_kwargs: client
        return client

    def _merge(self, create) -> object:
        with patch("app.llm.llm_client.get_openai_client", return_value=self._client(create)), patch(
            "app.llm.llm_client.time.sleep"
        ) as sleep_mock:
            result = openai_compatible_request_merge(
                provider_name="openai",
                api_key_env="GPT_API_KEY",
                base_url=None,
                prompt_text="Prompt",
                model_name="gpt-5.6-sol",
                timeout_sec=30.0,
                max_retries=0,
                attempt_label="test",
                max_output_tokens=100,
                reasoning_effort="low",
                service_tier="flex",
            )
        return result, sleep_mock

    def test_flex_success_first_try(self) -> None:
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            return _ok_response()

        result, sleep_mock = self._merge(_create)
        self.assertEqual(1, len(calls))
        self.assertEqual("flex", calls[0]["service_tier"])
        self.assertEqual({"effort": "low"}, calls[0]["reasoning"])
        sleep_mock.assert_not_called()
        self.assertIn('"title"', result.raw_text)

    def test_flex_unavailable_retries_then_falls_back_to_default(self) -> None:
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            if len(calls) <= 4:
                raise _flex_unavailable()
            return _ok_response()

        result, sleep_mock = self._merge(_create)
        self.assertEqual(5, len(calls))
        self.assertEqual(["flex"] * 4, [c.get("service_tier") for c in calls[:4]])
        self.assertNotIn("service_tier", calls[4])  # last attempt on the default tier
        self.assertEqual([20.0, 40.0, 80.0], [c.args[0] for c in sleep_mock.call_args_list])
        self.assertIn('"title"', result.raw_text)

    def test_flex_unavailable_recovers_after_one_wait(self) -> None:
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _flex_unavailable()
            return _ok_response()

        _result, sleep_mock = self._merge(_create)
        self.assertEqual(2, len(calls))
        self.assertEqual("flex", calls[1]["service_tier"])
        self.assertEqual(1, sleep_mock.call_count)

    def test_non_rate_limit_error_is_not_retried(self) -> None:
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            self._merge(_create)
        self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
