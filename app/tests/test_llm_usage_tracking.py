from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace

from app.llm import llm_client
from app.llm.llm_client import (
    LlmTraceContext,
    _log_llm_request_finish,
    _record_run_local_openai_request,
    get_run_local_openai_usage,
    reset_run_local_openai_usage,
)
from app.llm.llm_usage_tracker import OpenAIRequestUsage, extract_openai_request_usage
from app.llm.model_pricing import estimate_cost_usd, price_for_model
from app.observability.openai_usage import log_run_local_openai_usage


def _sdk_response(
    *,
    model: str,
    input_tokens: int,
    cached: int,
    cache_write: int,
    output_tokens: int,
    reasoning: int,
    response_id: str = "resp_test",
    service_tier: str = "default",
) -> SimpleNamespace:
    """Shape of openai 2.21 Response/ResponseUsage as observed in _HELP/usage_probe.log."""
    return SimpleNamespace(
        id=response_id,
        model=model,
        service_tier=service_tier,
        output_text="391",
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=cache_write),
            output_tokens=output_tokens,
            output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            total_tokens=input_tokens + output_tokens,
        ),
    )


class ExtractUsageTests(unittest.TestCase):
    def test_reads_every_field_from_sdk_shaped_response(self) -> None:
        response = _sdk_response(
            model="gpt-5.4-2026-03-05",
            input_tokens=2272,
            cached=1792,
            cache_write=0,
            output_tokens=42,
            reasoning=35,
            response_id="resp_00e2",
        )
        usage = extract_openai_request_usage(response)
        self.assertEqual(
            OpenAIRequestUsage(
                input_tokens=2272,
                cached_input_tokens=1792,
                cache_write_tokens=0,
                output_tokens=42,
                reasoning_tokens=35,
                total_tokens=2314,
                response_id="resp_00e2",
                served_model="gpt-5.4-2026-03-05",
                service_tier="default",
            ),
            usage,
        )

    def test_reads_dict_payload(self) -> None:
        response = {
            "id": "resp_dict",
            "model": "gpt-5.6-sol",
            "usage": {
                "input_tokens": 2272,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 2269},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2277,
            },
        }
        usage = extract_openai_request_usage(response)
        assert usage is not None
        self.assertEqual(2269, usage.cache_write_tokens)
        self.assertEqual(2277, usage.total_tokens)
        self.assertEqual("gpt-5.6-sol", usage.served_model)

    def test_missing_details_default_to_zero_and_total_is_derived(self) -> None:
        response = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        usage = extract_openai_request_usage(response)
        assert usage is not None
        self.assertEqual((0, 0, 0, 15, "", "", ""), (
            usage.cached_input_tokens,
            usage.cache_write_tokens,
            usage.reasoning_tokens,
            usage.total_tokens,
            usage.response_id,
            usage.served_model,
            usage.service_tier,
        ))

    def test_no_usage_returns_none(self) -> None:
        self.assertIsNone(extract_openai_request_usage(SimpleNamespace(output_text="x")))
        self.assertIsNone(extract_openai_request_usage(SimpleNamespace(usage=SimpleNamespace(input_tokens=None, output_tokens=5))))


class RunLocalAccumulationTests(unittest.TestCase):
    def test_every_successful_response_is_summed(self) -> None:
        reset_run_local_openai_usage()
        first = _sdk_response(model="gpt-5.6-sol", input_tokens=2272, cached=0, cache_write=2269, output_tokens=5, reasoning=0)
        second = _sdk_response(model="gpt-5.6-sol", input_tokens=2272, cached=2269, cache_write=0, output_tokens=40, reasoning=30)
        _record_run_local_openai_request(model_name="gpt-5.6-sol", response=first, request_kind="structured")
        _record_run_local_openai_request(model_name="gpt-5.6-sol", response=second, request_kind="structured")
        state = get_run_local_openai_usage()
        self.assertEqual(2, state.requests_sent)
        self.assertEqual(2, state.usage_reports)
        self.assertEqual(4544, state.input_tokens)
        self.assertEqual(2269, state.cached_input_tokens)
        self.assertEqual(2269, state.cache_write_tokens)
        self.assertEqual(45, state.output_tokens)
        self.assertEqual(30, state.reasoning_tokens)
        self.assertEqual(4589, state.total_tokens)
        self.assertEqual({"gpt-5.6-sol"}, state.served_models)
        self.assertEqual({"default"}, state.service_tiers)
        self.assertTrue(state.tokens_known)
        self.assertTrue(state.cost_known)
        # first: 3*$4 + 2269*$5 + 5*$20 ; second: 3*$4 + 2269*$0.40 + 40*$20  (per 1M)
        self.assertAlmostEqual(0.011457 + 0.0017196, state.estimated_cost_usd, places=6)

    def test_flex_tier_halves_cost_and_is_recorded(self) -> None:
        reset_run_local_openai_usage()
        response = _sdk_response(model="gpt-5.6-sol", input_tokens=2272, cached=0, cache_write=2269, output_tokens=5, reasoning=0, service_tier="flex")
        _record_run_local_openai_request(model_name="gpt-5.6-sol", response=response, request_kind="structured")
        state = get_run_local_openai_usage()
        self.assertEqual({"flex"}, state.service_tiers)
        self.assertAlmostEqual(0.011457 / 2, state.estimated_cost_usd, places=6)

    def test_unknown_served_model_marks_cost_unknown(self) -> None:
        reset_run_local_openai_usage()
        response = _sdk_response(model="gpt-9-unknown", input_tokens=10, cached=0, cache_write=0, output_tokens=5, reasoning=0)
        _record_run_local_openai_request(model_name="gpt-9-unknown", response=response, request_kind="structured")
        self.assertFalse(get_run_local_openai_usage().cost_known)

    def test_response_without_usage_marks_tokens_unknown(self) -> None:
        reset_run_local_openai_usage()
        _record_run_local_openai_request(model_name="gpt-5.2", response=SimpleNamespace(), request_kind="plain")
        state = get_run_local_openai_usage()
        self.assertEqual(1, state.requests_sent)
        self.assertEqual(0, state.usage_reports)
        self.assertFalse(state.tokens_known)

    def test_run_usage_log_line_carries_totals_and_cost(self) -> None:
        reset_run_local_openai_usage()
        response = _sdk_response(model="gpt-5.4-2026-03-05", input_tokens=2272, cached=1792, cache_write=0, output_tokens=42, reasoning=35)
        _record_run_local_openai_request(model_name="gpt-5.4", response=response, request_kind="structured")
        logger = logging.getLogger("usage-tracking-run-line")
        with self.assertLogs(logger, level="INFO") as captured:
            log_run_local_openai_usage(logger, effective_model="gpt-5.4")
        text: str = "\n".join(captured.output)
        self.assertIn("OPENAI RUN USAGE", text)
        self.assertIn("served_models=gpt-5.4-2026-03-05", text)
        self.assertIn("service_tiers=default", text)
        self.assertIn("cached_input_tokens=1792", text)
        self.assertIn("reasoning_tokens=35", text)
        self.assertIn("total_tokens=2314", text)
        self.assertIn("estimated_cost_usd=$0.0023", text)
        self.assertIn("tokens_known=yes", text)

    def test_request_finish_log_line_carries_per_request_usage(self) -> None:
        response = _sdk_response(model="gpt-5.6-sol", input_tokens=2272, cached=2269, cache_write=0, output_tokens=5, reasoning=0, response_id="resp_04e2")
        trace = LlmTraceContext(
            branch_label="merge",
            date_key="010130",
            slot_key="010130_1000",
            language="en",
            provider="openai",
            model_name="gpt-5.6-sol",
            attempt_index=1,
            request_kind="structured",
            source_count=3,
        )
        with self.assertLogs(llm_client.LOGGER.name, level="INFO") as captured:
            _log_llm_request_finish(trace_context=trace, response=response, success=True, elapsed_ms=1200, max_output_hit=False)
        text: str = "\n".join(captured.output)
        self.assertIn("llm_request_finish", text)
        self.assertIn("input_tokens=2272 cached_input_tokens=2269 cache_write_tokens=0 output_tokens=5 reasoning_tokens=0 total_tokens=2277", text)
        self.assertIn("served_model=gpt-5.6-sol response_id=resp_04e2", text)


class PricingTests(unittest.TestCase):
    def test_snapshot_suffix_maps_to_base_price(self) -> None:
        self.assertIs(price_for_model("gpt-5.4-2026-03-05"), price_for_model("gpt-5.4"))
        self.assertIs(price_for_model("gpt-5.6"), price_for_model("gpt-5.6-sol"))
        self.assertIsNone(price_for_model("gpt-9-unknown"))

    def test_cost_formula_matches_probe_numbers(self) -> None:
        first = extract_openai_request_usage(
            _sdk_response(model="gpt-5.6-sol", input_tokens=2272, cached=0, cache_write=2269, output_tokens=5, reasoning=0)
        )
        second = extract_openai_request_usage(
            _sdk_response(model="gpt-5.6-sol", input_tokens=2272, cached=2269, cache_write=0, output_tokens=5, reasoning=0)
        )
        assert first is not None and second is not None
        # 3 plain input * $4 + 2269 cache writes * $5 + 5 output * $20, per 1M tokens
        self.assertAlmostEqual(0.011457, estimate_cost_usd("gpt-5.6-sol", first) or 0.0, places=6)
        # 3 plain input * $4 + 2269 cached * $0.40 + 5 output * $20
        self.assertAlmostEqual(0.0010196, estimate_cost_usd("gpt-5.6-sol", second) or 0.0, places=6)

    def test_service_tier_multiplier(self) -> None:
        usage = extract_openai_request_usage(
            _sdk_response(model="gpt-5.4", input_tokens=1000, cached=0, cache_write=0, output_tokens=1000, reasoning=0)
        )
        assert usage is not None
        standard = estimate_cost_usd("gpt-5.4", usage)
        self.assertAlmostEqual(0.0175, standard or 0.0, places=6)  # 1000*$2.5 + 1000*$15 per 1M
        self.assertAlmostEqual(0.00875, estimate_cost_usd("gpt-5.4", usage, service_tier="flex") or 0.0, places=6)
        self.assertAlmostEqual(0.035, estimate_cost_usd("gpt-5.4", usage, service_tier="fast") or 0.0, places=6)

    def test_unknown_model_has_no_cost(self) -> None:
        usage = extract_openai_request_usage(
            _sdk_response(model="gpt-9-unknown", input_tokens=10, cached=0, cache_write=0, output_tokens=5, reasoning=0)
        )
        assert usage is not None
        self.assertIsNone(estimate_cost_usd("gpt-9-unknown", usage))


if __name__ == "__main__":
    unittest.main()
