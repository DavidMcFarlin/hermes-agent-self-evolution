"""Tests for cost estimation and real token metering."""

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.cost import (
    DEFAULT_RATE_PER_1K,
    CostMeter,
    cost_from_usage,
    estimate_evolution_cost,
    rate_for_model,
)


def _rates():
    return EvolutionConfig().cost_per_1k_tokens


def test_rate_for_model_full_then_suffix_then_default():
    rates = _rates()
    # full name present
    assert rate_for_model("openai/gpt-4.1", rates) == rates["openai/gpt-4.1"]
    # only suffix present resolves via suffix
    assert rate_for_model("azure/gpt-4.1-mini", rates) == rates["gpt-4.1-mini"]
    # unknown model falls back to the conservative default
    assert rate_for_model("some/unknown-model", rates) == DEFAULT_RATE_PER_1K


def test_estimate_scales_with_iterations():
    rates = _rates()
    one = estimate_evolution_cost(1, "openai/gpt-4.1-mini", rates=rates)
    ten = estimate_evolution_cost(10, "openai/gpt-4.1-mini", rates=rates)
    assert ten == round(one * 10, 10) or abs(ten - one * 10) < 1e-9
    assert estimate_evolution_cost(0, "openai/gpt-4.1-mini", rates=rates) == 0.0


def test_cost_from_usage_handles_missing_total_tokens():
    rates = _rates()
    # total_tokens present
    assert cost_from_usage({"openai/gpt-4.1": {"total_tokens": 1000}}, rates) == rates[
        "openai/gpt-4.1"
    ]
    # falls back to prompt+completion when total_tokens missing
    got = cost_from_usage(
        {"openai/gpt-4.1": {"prompt_tokens": 600, "completion_tokens": 400}}, rates
    )
    assert abs(got - rates["openai/gpt-4.1"]) < 1e-9
    assert cost_from_usage({}, rates) == 0.0


def test_cost_meter_aggregates_real_usage():
    meter = CostMeter(EvolutionConfig())
    assert meter.total_cost_usd == 0.0
    assert meter.total_tokens == 0

    with meter.track():
        dspy.settings.usage_tracker.add_usage(
            "openai/gpt-4.1",
            {"prompt_tokens": 600, "completion_tokens": 400, "total_tokens": 1000},
        )
        dspy.settings.usage_tracker.add_usage(
            "openai/gpt-4.1-mini",
            {"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000},
        )

    assert meter.total_tokens == 2000
    rates = _rates()
    expected = rates["openai/gpt-4.1"] + rates["openai/gpt-4.1-mini"]
    assert abs(meter.total_cost_usd - expected) < 1e-9


def test_cost_meter_accumulates_across_multiple_track_blocks():
    meter = CostMeter(EvolutionConfig())
    for _ in range(2):
        with meter.track():
            dspy.settings.usage_tracker.add_usage(
                "openai/gpt-4.1", {"total_tokens": 1000}
            )
    assert meter.total_tokens == 2000
