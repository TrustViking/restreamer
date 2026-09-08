from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from app.config.settings import AppConfig, LlmConfig
from app.llm.llm_client import probe_openai_model_access
from app.llm.model_selection import llm_merge_requested, select_llm_model
from app.llm.models.model_compatibility import LlmModelConfigurationError

pytest.importorskip("openai")

from openai._exceptions import NotFoundError, PermissionDeniedError, RateLimitError


def _llm_config(*, model: str, fallback_model: str) -> LlmConfig:
    return LlmConfig(
        provider="openai",
        model=model,
        fallback_model=fallback_model,
        reasoning_effort="medium",
        service_tier="default",
        timeout_sec=120.0,
        max_output_tokens=2000,
        pre_delay_sec=0.0,
        source_desc_max_chars=2000,
    )


def _config(*, model: str = "gpt-5.4", fallback_model: str = "gpt-5.2") -> AppConfig:
    # select_llm_model reads only config.llm; the other sections stay empty.
    fields: dict[str, object] = {name: None for name in AppConfig.__dataclass_fields__}
    fields["llm"] = _llm_config(model=model, fallback_model=fallback_model)
    return AppConfig(**fields)  # type: ignore[arg-type]


def _access_error(model_name: str, status_code: int) -> LlmModelConfigurationError:
    reason_code: str = "openai_model_not_found" if status_code == 404 else "openai_model_access_denied"
    return LlmModelConfigurationError(
        provider_name="openai",
        model_name=model_name,
        reason_code=reason_code,
        detail=f"model {model_name} unavailable",
        status_code=status_code,
    )


class SelectLlmModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._logger: logging.Logger = logging.getLogger("test-llm-model-selection")

    def test_primary_passes_probe_and_is_kept(self) -> None:
        config: AppConfig = _config()
        with patch("app.llm.model_selection.probe_openai_model_access", return_value=None) as probe:
            selected: AppConfig = select_llm_model(config=config, logger=self._logger)
        self.assertIs(config, selected)
        self.assertEqual(1, probe.call_count)
        self.assertEqual("gpt-5.4", probe.call_args.kwargs["model_name"])
        self.assertEqual("medium", probe.call_args.kwargs["reasoning_effort"])

    def test_primary_not_found_falls_back(self) -> None:
        config: AppConfig = _config()
        with patch(
            "app.llm.model_selection.probe_openai_model_access",
            side_effect=[_access_error("gpt-5.4", 404), None],
        ) as probe, self.assertLogs(self._logger, level="WARNING") as captured:
            selected: AppConfig = select_llm_model(config=config, logger=self._logger)
        self.assertEqual("gpt-5.2", selected.llm.model)
        self.assertEqual("gpt-5.4", config.llm.model)
        self.assertEqual(2, probe.call_count)
        self.assertIn("llm_model_fallback from=gpt-5.4 to=gpt-5.2 reason_code=openai_model_not_found", "\n".join(captured.output))

    def test_primary_access_denied_falls_back(self) -> None:
        with patch(
            "app.llm.model_selection.probe_openai_model_access",
            side_effect=[_access_error("gpt-5.4", 403), None],
        ):
            selected: AppConfig = select_llm_model(config=_config(), logger=self._logger)
        self.assertEqual("gpt-5.2", selected.llm.model)

    def test_both_models_unavailable_raises(self) -> None:
        with patch(
            "app.llm.model_selection.probe_openai_model_access",
            side_effect=[_access_error("gpt-5.4", 404), _access_error("gpt-5.2", 404)],
        ):
            with self.assertRaises(LlmModelConfigurationError) as raised:
                select_llm_model(config=_config(), logger=self._logger)
        self.assertEqual("gpt-5.2", raised.exception.model_name)

    def test_non_access_config_error_does_not_fall_back(self) -> None:
        bad_request = LlmModelConfigurationError(
            provider_name="openai",
            model_name="gpt-5.4",
            reason_code="openai_unsupported_parameter",
            detail="Unsupported parameter: 'reasoning.effort'",
            status_code=400,
        )
        with patch("app.llm.model_selection.probe_openai_model_access", side_effect=bad_request) as probe:
            with self.assertRaises(LlmModelConfigurationError):
                select_llm_model(config=_config(), logger=self._logger)
        self.assertEqual(1, probe.call_count)

    def test_inconclusive_probe_keeps_primary(self) -> None:
        with patch(
            "app.llm.model_selection.probe_openai_model_access",
            side_effect=TimeoutError("probe timeout"),
        ) as probe, self.assertLogs(self._logger, level="WARNING") as captured:
            selected: AppConfig = select_llm_model(config=_config(), logger=self._logger)
        self.assertEqual("gpt-5.4", selected.llm.model)
        self.assertEqual(1, probe.call_count)
        self.assertIn("llm_model_probe_inconclusive model=gpt-5.4", "\n".join(captured.output))

    def test_same_primary_and_fallback_probe_once_and_raise(self) -> None:
        with patch(
            "app.llm.model_selection.probe_openai_model_access",
            side_effect=_access_error("gpt-5.2", 404),
        ) as probe:
            with self.assertRaises(LlmModelConfigurationError):
                select_llm_model(config=_config(model="gpt-5.2", fallback_model="gpt-5.2"), logger=self._logger)
        self.assertEqual(1, probe.call_count)


class LlmMergeRequestedTests(unittest.TestCase):
    def test_nomerge_never_probes(self) -> None:
        with patch.dict("os.environ", {"GPT_API_KEY": "key"}, clear=True):
            self.assertFalse(llm_merge_requested(audit_mode="nomerge", dry_run=False))

    def test_merge_requires_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(llm_merge_requested(audit_mode="merge", dry_run=False))
        with patch.dict("os.environ", {"GPT_API_KEY": "key"}, clear=True):
            self.assertTrue(llm_merge_requested(audit_mode="audit", dry_run=False))

    def test_dry_run_skips_probe_unless_allowed(self) -> None:
        with patch.dict("os.environ", {"GPT_API_KEY": "key"}, clear=True):
            self.assertFalse(llm_merge_requested(audit_mode="merge", dry_run=True))
        with patch.dict("os.environ", {"GPT_API_KEY": "key", "STG_LLM_ALLOW_IN_DRY_RUN": "1"}, clear=True):
            self.assertTrue(llm_merge_requested(audit_mode="merge", dry_run=True))


class ProbeOpenAIModelAccessTests(unittest.TestCase):
    def _response(self, status_code: int) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )

    def _client(self, create: object) -> SimpleNamespace:
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        client.with_options = lambda **_kwargs: client
        return client

    def test_probe_uses_responses_api_with_configured_effort(self) -> None:
        calls: list[dict[str, object]] = []

        def _create(**kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=5, output_tokens=1))

        with patch("app.llm.llm_client.get_openai_client", return_value=self._client(_create)):
            probe_openai_model_access(
                provider_name="openai",
                model_name="gpt-5.4",
                timeout_sec=30.0,
                reasoning_effort="high",
            )
        self.assertEqual(1, len(calls))
        self.assertEqual("gpt-5.4", calls[0]["model"])
        self.assertEqual({"effort": "high"}, calls[0]["reasoning"])
        self.assertEqual(16, calls[0]["max_output_tokens"])
        self.assertNotIn("text", calls[0])

    def test_probe_maps_404_to_model_not_found(self) -> None:
        def _create(**_kwargs: object) -> None:
            raise NotFoundError(
                "That model does not exist",
                response=self._response(404),
                body={"param": "id", "type": "invalid_request_error"},
            )

        with patch("app.llm.llm_client.get_openai_client", return_value=self._client(_create)):
            with self.assertRaises(LlmModelConfigurationError) as raised:
                probe_openai_model_access(
                    provider_name="openai",
                    model_name="gpt-5.4",
                    timeout_sec=30.0,
                    reasoning_effort="medium",
                )
        self.assertEqual("openai_model_not_found", raised.exception.reason_code)
        self.assertEqual(404, raised.exception.status_code)

    def test_probe_maps_403_to_access_denied(self) -> None:
        def _create(**_kwargs: object) -> None:
            raise PermissionDeniedError(
                "Project does not have access to model `gpt-5.4`",
                response=self._response(403),
                body={"code": "access_denied", "param": "model", "type": "invalid_request_error"},
            )

        with patch("app.llm.llm_client.get_openai_client", return_value=self._client(_create)):
            with self.assertRaises(LlmModelConfigurationError) as raised:
                probe_openai_model_access(
                    provider_name="openai",
                    model_name="gpt-5.4",
                    timeout_sec=30.0,
                    reasoning_effort="medium",
                )
        self.assertEqual("openai_model_access_denied", raised.exception.reason_code)

    def test_probe_reraises_rate_limit_unchanged(self) -> None:
        def _create(**_kwargs: object) -> None:
            raise RateLimitError(
                "Rate limit reached",
                response=self._response(429),
                body={"code": "rate_limit_exceeded", "type": "requests"},
            )

        with patch("app.llm.llm_client.get_openai_client", return_value=self._client(_create)):
            with self.assertRaises(RateLimitError):
                probe_openai_model_access(
                    provider_name="openai",
                    model_name="gpt-5.4",
                    timeout_sec=30.0,
                    reasoning_effort="medium",
                )


if __name__ == "__main__":
    unittest.main()
