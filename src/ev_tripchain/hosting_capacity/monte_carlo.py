from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class ScenarioEventFn(Protocol):
    def __call__(self, rng: np.random.Generator) -> bool: ...


@dataclass(frozen=True)
class ProbabilityEstimate:
    n_events: int
    p_hat: float
    ci95_low: float
    ci95_high: float


@dataclass(frozen=True)
class HardConstraintProbabilityBreakdown:
    any_limit_exceedance: ProbabilityEstimate
    voltage_limit_exceedance: ProbabilityEstimate
    line_limit_exceedance: ProbabilityEstimate
    trafo_limit_exceedance: ProbabilityEstimate
    solver_failure: ProbabilityEstimate


@dataclass(frozen=True)
class SoftMetricAggregate:
    mean_peak_voltage_deviation_pu: float = 0.0
    max_peak_voltage_deviation_pu: float = 0.0
    mean_peak_network_loss_mw: float = 0.0
    max_peak_network_loss_mw: float = 0.0
    mean_peak_line_loading_percent: float = 0.0
    max_peak_line_loading_percent: float = 0.0
    mean_peak_trafo_loading_percent: float = 0.0
    max_peak_trafo_loading_percent: float = 0.0


@dataclass(frozen=True)
class MonteCarloEstimate:
    n: int
    n_events: int
    p_hat: float
    ci95_low: float
    ci95_high: float
    hard_constraints: HardConstraintProbabilityBreakdown | None = None
    soft_metrics: SoftMetricAggregate | None = None


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


def build_probability_estimate(*, n: int, n_events: int) -> ProbabilityEstimate:
    ci95_low, ci95_high = _wilson_ci_95(n=n, n_events=n_events)
    p_hat = n_events / n if n > 0 else 0.0
    return ProbabilityEstimate(
        n_events=int(n_events),
        p_hat=float(p_hat),
        ci95_low=float(ci95_low),
        ci95_high=float(ci95_high),
    )


def estimate_event_probability(
    simulate_event: ScenarioEventFn,
    *,
    n_scenarios: int,
    rng: np.random.Generator,
    scenario_rng: Callable[[int], np.random.Generator] | None = None,
    early_stop_threshold: float | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 0,
) -> MonteCarloEstimate:
    """
    Estimate P(event) via Monte Carlo over `n_scenarios`.

    `simulate_event(rng)` should return True when the scenario is counted as an event.

    When `early_stop_threshold` is set, stops early once the Wilson CI lower bound
    exceeds the threshold (clearly above target) or the upper bound drops below it
    (clearly safe).
    """
    n = int(max(n_scenarios, 0))
    n_events = 0
    actually_run = n
    for i in range(n):
        r = scenario_rng(i) if scenario_rng is not None else rng
        n_events += int(bool(simulate_event(r)))

        if progress is not None:
            should_report = (i == 0) or (i + 1 == n)
            if progress_every > 0 and (i + 1) % progress_every == 0:
                should_report = True
            if should_report:
                progress(f"scenarios {i + 1}/{n}, events={n_events}")

        # Early stopping: once we have enough evidence, stop running more scenarios.
        # Use a stricter threshold to avoid premature stopping for borderline cases.
        if early_stop_threshold is not None and (i + 1) >= 5:
            ci_lo, ci_hi = _wilson_ci_95(n=i + 1, n_events=n_events)
            if ci_lo > early_stop_threshold * 3:
                # Clearly well above threshold — no point running more
                actually_run = i + 1
                if progress is not None:
                    progress(f"early stop at {actually_run}/{n}: CI lower={ci_lo:.4f} > {early_stop_threshold * 3:.4f}")
                break
            if ci_hi <= early_stop_threshold:
                # Clearly below threshold — further samples cannot change the decision much
                actually_run = i + 1
                if progress is not None:
                    progress(f"early stop at {actually_run}/{n}: CI upper={ci_hi:.4f} <= {early_stop_threshold:.4f}")
                break

    p_hat = n_events / actually_run if actually_run > 0 else 0.0
    ci_low, ci_high = _wilson_ci_95(n=actually_run, n_events=n_events)
    return MonteCarloEstimate(
        n=actually_run,
        n_events=n_events,
        p_hat=float(p_hat),
        ci95_low=ci_low,
        ci95_high=ci_high,
    )
