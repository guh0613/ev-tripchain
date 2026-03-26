from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from pydantic import BaseModel

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.deterministic import run_deterministic_hc
from ev_tripchain.hosting_capacity.evaluate import (
    create_mc_parallel_context,
    estimate_hard_exceedance_probability_mc,
)
from ev_tripchain.hosting_capacity.monte_carlo import MonteCarloEstimate
from ev_tripchain.hosting_capacity.search import binary_search_max_n
from ev_tripchain.hosting_capacity.sensitivity import run_sensitivity_hc
from ev_tripchain.rng import make_rng_for


class ProbabilityPoint(BaseModel):
    n_events: int
    p_hat: float
    ci95_low: float
    ci95_high: float


class HardConstraintBreakdown(BaseModel):
    any_limit_exceedance: ProbabilityPoint
    voltage_limit_exceedance: ProbabilityPoint
    line_limit_exceedance: ProbabilityPoint
    trafo_limit_exceedance: ProbabilityPoint
    solver_failure: ProbabilityPoint


class SoftMetricPoint(BaseModel):
    mean_peak_voltage_deviation_pu: float
    max_peak_voltage_deviation_pu: float
    mean_peak_network_loss_mw: float
    max_peak_network_loss_mw: float
    mean_peak_line_loading_percent: float
    max_peak_line_loading_percent: float
    mean_peak_trafo_loading_percent: float
    max_peak_trafo_loading_percent: float


class RiskPoint(BaseModel):
    n: int
    p_hat: float
    ci95_low: float
    ci95_high: float
    metric: float
    hard_constraints: HardConstraintBreakdown | None = None
    soft_metrics: SoftMetricPoint | None = None


class HostingCapacityResult(BaseModel):
    n_star: int
    base_case_safe: bool
    risk_tolerance: float
    risk_metric: str
    scenarios: int
    common_random_numbers: bool
    risk_curve: list[tuple[int, float]]
    risk_curve_detail: list[RiskPoint]


class ComparisonResult(BaseModel):
    mc: HostingCapacityResult
    deterministic_n_star: int
    deterministic_weakest_bus: int
    deterministic_weakest_voltage: float
    sensitivity_n_star_representative: int
    sensitivity_n_star_uniform: int
    sensitivity_n_star_weakest: int


def run_hosting_capacity(
    cfg: ProjectConfig,
    *,
    progress: Callable[[str], None] | None = None,
    progress_label: str | None = None,
) -> HostingCapacityResult:
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    parallel_ctx = create_mc_parallel_context(cfg)

    est_cache: dict[int, MonteCarloEstimate] = {}
    prefix = f"[{progress_label}] " if progress_label else ""
    if progress is not None and parallel_ctx is not None:
        progress(f"{prefix}parallel workers={parallel_ctx.max_workers}")

    try:
        def risk_at_n(n: int) -> float:
            nn = int(n)
            if nn not in est_cache:
                if progress is not None:
                    progress(f"{prefix}evaluating N={nn}")
                # deterministic per-N to keep binary search stable/reproducible
                rng_n = make_rng_for(int(cfg.seed), nn)
                est_cache[nn] = estimate_hard_exceedance_probability_mc(
                    net,
                    cfg,
                    n=nn,
                    rng=rng_n,
                    progress=(
                        None
                        if progress is None
                        else lambda msg, nn=nn: progress(f"{prefix}N={nn}: {msg}")
                    ),
                    parallel=parallel_ctx,
                )
                if progress is not None:
                    est = est_cache[nn]
                    progress(
                        f"{prefix}N={nn} done: p_hat={est.p_hat:.4f}, "
                        f"ci95=[{est.ci95_low:.4f}, {est.ci95_high:.4f}]"
                    )
            est = est_cache[nn]
            metric = cfg.hosting_capacity.risk_metric
            if metric == "ci95_high":
                return float(est.ci95_high)
            return float(est.p_hat)

        base_case_est = est_cache.get(0)
        if base_case_est is None:
            risk_at_n(0)
            base_case_est = est_cache[0]
        base_case_safe = float(
            base_case_est.ci95_high
            if cfg.hosting_capacity.risk_metric == "ci95_high"
            else base_case_est.p_hat
        ) <= cfg.hosting_capacity.risk_tolerance

        n_star, curve = binary_search_max_n(
            risk_at_n,
            n_max=cfg.hosting_capacity.n_max,
            risk_tolerance=cfg.hosting_capacity.risk_tolerance,
            max_iter=cfg.hosting_capacity.binary_search.max_iter,
            min_step=cfg.hosting_capacity.binary_search.min_step,
            initial_hi=cfg.hosting_capacity.binary_search.initial_hi,
        )

        detail = [
            RiskPoint(
                n=int(n),
                p_hat=float(est_cache[int(n)].p_hat),
                ci95_low=float(est_cache[int(n)].ci95_low),
                ci95_high=float(est_cache[int(n)].ci95_high),
                metric=float(r),
                hard_constraints=(
                    None
                    if est_cache[int(n)].hard_constraints is None
                    else HardConstraintBreakdown.model_validate(
                        asdict(est_cache[int(n)].hard_constraints)
                    )
                ),
                soft_metrics=(
                    None
                    if est_cache[int(n)].soft_metrics is None
                    else SoftMetricPoint.model_validate(
                        asdict(est_cache[int(n)].soft_metrics)
                    )
                ),
            )
            for n, r in curve
        ]
        detail.sort(key=lambda x: x.n)
        return HostingCapacityResult(
            n_star=n_star,
            base_case_safe=base_case_safe,
            risk_tolerance=cfg.hosting_capacity.risk_tolerance,
            risk_metric=cfg.hosting_capacity.risk_metric,
            scenarios=cfg.hosting_capacity.scenarios,
            common_random_numbers=cfg.hosting_capacity.common_random_numbers,
            risk_curve=curve,
            risk_curve_detail=detail,
        )
    finally:
        if parallel_ctx is not None:
            parallel_ctx.executor.shutdown(wait=True, cancel_futures=True)


def run_method_comparison(
    cfg: ProjectConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> ComparisonResult:
    """Run all three hosting capacity methods and return comparison."""
    net_det = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    net_sens = load_case(cfg.case.name, load_scale=cfg.case.load_scale)

    # 1. Monte Carlo (existing)
    if progress is not None:
        progress("[mc] running Monte Carlo hosting capacity")
    mc_result = run_hosting_capacity(cfg, progress=progress, progress_label="mc")
    if progress is not None:
        progress(f"[mc] done: N*={mc_result.n_star}")

    # 2. Deterministic representative template
    if progress is not None:
        progress("[deterministic] running")
    det_result = run_deterministic_hc(net_det, cfg)
    if progress is not None:
        progress(f"[deterministic] done: N*={det_result.n_star}")

    # 3. Voltage sensitivity
    if progress is not None:
        progress("[sensitivity] running")
    sens_result = run_sensitivity_hc(net_sens, cfg)
    if progress is not None:
        progress(
            "[sensitivity] done: "
            f"representative={sens_result.n_star_representative}, "
            f"uniform={sens_result.n_star_uniform}, "
            f"weakest={sens_result.n_star_weakest}"
        )

    return ComparisonResult(
        mc=mc_result,
        deterministic_n_star=det_result.n_star,
        deterministic_weakest_bus=det_result.weakest_bus_id,
        deterministic_weakest_voltage=det_result.weakest_bus_voltage_pu,
        sensitivity_n_star_representative=sens_result.n_star_representative,
        sensitivity_n_star_uniform=sens_result.n_star_uniform,
        sensitivity_n_star_weakest=sens_result.n_star_weakest,
    )
