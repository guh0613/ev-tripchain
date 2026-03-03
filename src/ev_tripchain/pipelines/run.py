from __future__ import annotations

from pydantic import BaseModel

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.evaluate import estimate_violation_probability_mc
from ev_tripchain.hosting_capacity.monte_carlo import MonteCarloEstimate
from ev_tripchain.hosting_capacity.search import binary_search_max_n
from ev_tripchain.rng import make_rng_for


class RiskPoint(BaseModel):
    n: int
    p_hat: float
    ci95_low: float
    ci95_high: float
    metric: float


class HostingCapacityResult(BaseModel):
    n_star: int
    risk_tolerance: float
    risk_metric: str
    scenarios: int
    common_random_numbers: bool
    risk_curve: list[tuple[int, float]]
    risk_curve_detail: list[RiskPoint]


def run_hosting_capacity(cfg: ProjectConfig) -> HostingCapacityResult:
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)

    est_cache: dict[int, MonteCarloEstimate] = {}

    def risk_at_n(n: int) -> float:
        nn = int(n)
        if nn not in est_cache:
            # deterministic per-N to keep binary search stable/reproducible
            rng_n = make_rng_for(int(cfg.seed), nn)
            est_cache[nn] = estimate_violation_probability_mc(net, cfg, n=nn, rng=rng_n)
        est = est_cache[nn]
        metric = cfg.hosting_capacity.risk_metric
        if metric == "ci95_high":
            return float(est.ci95_high)
        return float(est.p_hat)

    n_star, curve = binary_search_max_n(
        risk_at_n,
        n_max=cfg.hosting_capacity.n_max,
        risk_tolerance=cfg.hosting_capacity.risk_tolerance,
        max_iter=cfg.hosting_capacity.binary_search.max_iter,
        min_step=cfg.hosting_capacity.binary_search.min_step,
    )

    detail = [
        RiskPoint(
            n=int(n),
            p_hat=float(est_cache[int(n)].p_hat),
            ci95_low=float(est_cache[int(n)].ci95_low),
            ci95_high=float(est_cache[int(n)].ci95_high),
            metric=float(r),
        )
        for n, r in curve
    ]
    detail.sort(key=lambda x: x.n)
    return HostingCapacityResult(
        n_star=n_star,
        risk_tolerance=cfg.hosting_capacity.risk_tolerance,
        risk_metric=cfg.hosting_capacity.risk_metric,
        scenarios=cfg.hosting_capacity.scenarios,
        common_random_numbers=cfg.hosting_capacity.common_random_numbers,
        risk_curve=curve,
        risk_curve_detail=detail,
    )
