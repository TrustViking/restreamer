from __future__ import annotations

import unittest

import httpx
import pytest

from app.llm.models.model_compatibility import (
    classify_openai_request_error,
    resolve_openai_request_compatibility,
)
from app.llm.llm_client import _build_openai_responses_request_kwargs

pytest.importorskip("openai")

from openai._exceptions import (
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
)


class OpenAIModelCompatibilityTests(unittest.TestCase):
    def _request(self) -> httpx.Request:
        return httpx.Request("POST", "https://api.openai.com/v1/responses")

    def _response(self, status_code: int) -> httpx.Response:
        return httpx.Response(status_code=status_code, request=self._request())

    def test_gpt5_request_includes_reasoning_effort(self) -> None:
        compatibility = resolve_openai_request_compatibility(
            model_name="gpt-5.1",
            structured_output_requested=True,
            temperature_requested=True,
        )
        request_kwargs = _build_openai_responses_request_kwargs(
            prompt_text="Prompt",
            model_name="gpt-5.1",
            max_tokens=1000,
            structured_schema={"name": "schema", "schema": {"type": "object"}},
            temperature=0.0,
            compatibility=compatibility,
        )
        self.assertIn("reasoning", request_kwargs)
        self.assertEqual({"effort": "low"}, request_kwargs["reasoning"])
        self.assertIn("text", request_kwargs)

    def test_gpt4o_request_omits_reasoning_effort(self) -> None:
        compatibility = resolve_openai_request_compatibility(
            model_name="gpt-4o",
            structured_output_requested=True,
            temperature_requested=True,
        )
        request_kwargs = _build_openai_responses_request_kwargs(
            prompt_text="Prompt",
            model_name="gpt-4o",
            max_tokens=1000,
            structured_schema={"name": "schema", "schema": {"type": "object"}},
            temperature=0.0,
            compatibility=compatibility,
        )
        self.assertNotIn("reasoning", request_kwargs)
        self.assertIn("text", request_kwargs)

    def test_model_not_found_is_classified_as_fatal_model_config_error(self) -> None:
        error = NotFoundError(
            "The model `gpt-5.3` does not exist",
            response=self._response(404),
            body={"code": "model_not_found", "param": "model", "type": "invalid_request_error"},
        )
        classification = classify_openai_request_error(error)
        self.assertEqual("openai_model_not_found", classification.reason_code)
        self.assertTrue(classification.fatal_model_configuration)
        self.assertFalse(classification.retryable)

    def test_access_denied_is_classified_as_fatal_model_config_error(self) -> None:
        error = PermissionDeniedError(
            "Project does not have access to model `gpt-5.4`",
            response=self._response(403),
            body={"code": "access_denied", "param": "model", "type": "invalid_request_error"},
        )
        classification = classify_openai_request_error(error)
        self.assertEqual("openai_model_access_denied", classification.reason_code)
        self.assertTrue(classification.fatal_model_configuration)
        self.assertFalse(classification.retryable)

    def test_unsupported_parameter_is_classified_as_fatal_model_config_error(self) -> None:
        error = BadRequestError(
            "Unsupported parameter: 'reasoning.effort'",
            response=self._response(400),
            body={
                "code": "unsupported_parameter",
                "param": "reasoning.effort",
                "type": "invalid_request_error",
            },
        )
        classification = classify_openai_request_error(error)
        self.assertEqual("openai_unsupported_parameter", classification.reason_code)
        self.assertTrue(classification.fatal_model_configuration)
        self.assertFalse(classification.retryable)

    def test_timeout_is_not_classified_as_fatal_model_config_error(self) -> None:
        error = APITimeoutError(request=self._request())
        classification = classify_openai_request_error(error)
        self.assertEqual("openai_timeout", classification.reason_code)
        self.assertFalse(classification.fatal_model_configuration)
        self.assertTrue(classification.retryable)

    def test_server_error_is_not_classified_as_fatal_model_config_error(self) -> None:
        error = InternalServerError(
            "Internal server error",
            response=self._response(500),
            body={"code": "server_error", "type": "server_error"},
        )
        classification = classify_openai_request_error(error)
        self.assertEqual("openai_server_error", classification.reason_code)
        self.assertFalse(classification.fatal_model_configuration)
        self.assertTrue(classification.retryable)


if __name__ == "__main__":
    unittest.main()
