"""Cost estimation and real token metering for evolution runs.

Two complementary mechanisms live here:

* ``estimate_evolution_cost`` — an *a priori* guess used to gate a run before
  any tokens are spent (a coarse iterations × tokens-per-iteration model).
* ``CostMeter`` — *real* accounting that wraps an evolution run in
  ``dspy.track_usage()`` and converts the observed per-model token counts into
  USD using the rate table on :class:`EvolutionConfig`. This is what feeds the
  stop-loss guard and the post-run cap check, replacing the previous
  ``cost_usd=0.0`` placeholder.

Both paths read their rate table from a single source of truth
(:class:`~evolution.core.config.EvolutionConfig`), so there is no longer a
hardcoded constant duplicated across phase modules.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import dspy

from evolution.core.config import EvolutionConfig

# Conservative default rate (USD / 1K tokens) for any model not in the table.
DEFAULT_RATE_PER_1K = 0.005

# Conservative token budget assumed per optimization iteration when no real
# usage data is available yet (prompt + completion + judge round-trips).
TOKENS_PER_ITERATION = 50_000


def rate_for_model(model: str, rates: Dict[str, float]) -> float:
    """Resolve a per-1K-token USD rate for ``model``.

    Tries the full name first (e.g. ``openai/gpt-4.1-mini``), then the bare
    suffix (``gpt-4.1-mini``), then falls back to :data:`DEFAULT_RATE_PER_1K`.
    """
    if model in rates:
        return rates[model]
    suffix = model.split("/")[-1]
    if suffix in rates:
        return rates[suffix]
    return DEFAULT_RATE_PER_1K


def estimate_evolution_cost(
    iterations: int,
    model: str = "openai/gpt-4.1-mini",
    *,
    rates: Optional[Dict[str, float]] = None,
) -> float:
    """Estimate total cost before running evolution (coarse upper bound)."""
    if rates is None:
        rates = EvolutionConfig().cost_per_1k_tokens
    rate = rate_for_model(model, rates)
    estimated_tokens = max(0, int(iterations)) * TOKENS_PER_ITERATION
    return (estimated_tokens / 1000.0) * rate


def cost_from_usage(
    usage_by_lm: Dict[str, Dict[str, Any]],
    rates: Dict[str, float],
) -> float:
    """Convert a ``UsageTracker.get_total_tokens()`` mapping into USD."""
    total = 0.0
    for model, usage in (usage_by_lm or {}).items():
        tokens = usage.get("total_tokens")
        if not tokens:
            tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        total += (tokens / 1000.0) * rate_for_model(model, rates)
    return total


class CostMeter:
    """Measure real LM spend for an evolution run.

    Usage::

        meter = CostMeter(config)
        with meter.track():
            ... run optimization + holdout eval ...
        print(meter.total_cost_usd, meter.total_tokens)

    The meter relies on dspy's ``track_usage`` context, which records token
    counts for every ``dspy.LM`` call made inside the block (optimizer model,
    eval model, reflection/judge model — all of them).
    """

    def __init__(self, config: EvolutionConfig):
        self._rates = dict(config.cost_per_1k_tokens)
        self._usage_by_lm: Dict[str, Dict[str, Any]] = {}

    @contextmanager
    def track(self) -> Generator["CostMeter", None, None]:
        """Context manager that accumulates LM usage into this meter."""
        with dspy.track_usage() as tracker:
            try:
                yield self
            finally:
                # Merge in case track() is used more than once on one meter.
                observed = tracker.get_total_tokens()
                for model, usage in observed.items():
                    bucket = self._usage_by_lm.setdefault(model, {})
                    for key, value in usage.items():
                        if isinstance(value, (int, float)):
                            bucket[key] = bucket.get(key, 0) + value

    @property
    def usage_by_lm(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._usage_by_lm)

    @property
    def total_tokens(self) -> int:
        total = 0
        for usage in self._usage_by_lm.values():
            total += int(usage.get("total_tokens", 0) or 0)
        return total

    @property
    def total_cost_usd(self) -> float:
        return cost_from_usage(self._usage_by_lm, self._rates)
