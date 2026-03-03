from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class ScenarioEventFn(Protocol):
    def __call__(self, rng: np.random.Generator) -> bool: ...


@dataclass(frozen=True)
class MonteCarloEstimate:
    n: int
    n_events: int
    p_hat: float
    ci95_low: float
    ci95_high: float


def _wilson_ci_95(*, n: int, n_events: int) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0

    z = 1.959963984540054  # 97.5% quantile of N(0,1)
    phat = n_events / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt((phat * (1.0 - phat)) / n + (z * z) / (4.0 * n * n))
        / denom
    )
    low = float(max(0.0, center - half))
    high = float(min(1.0, center + half))
    return low, high


def estimate_event_probability(
    simulate_event: ScenarioEventFn,
    *,
    n_scenarios: int,
    rng: np.random.Generator,
    scenario_rng: Callable[[int], np.random.Generator] | None = None,
) -> MonteCarloEstimate:
    """
    Estimate P(event) via Monte Carlo over `n_scenarios`.

    `simulate_event(rng)` should return True when the scenario is counted as an event.
    """
    n = int(max(n_scenarios, 0))
    n_events = 0
    for i in range(n):
        r = scenario_rng(i) if scenario_rng is not None else rng
        n_events += int(bool(simulate_event(r)))

    p_hat = n_events / n if n > 0 else 0.0
    ci_low, ci_high = _wilson_ci_95(n=n, n_events=n_events)
    return MonteCarloEstimate(
        n=n,
        n_events=n_events,
        p_hat=float(p_hat),
        ci95_low=ci_low,
        ci95_high=ci_high,
    )
