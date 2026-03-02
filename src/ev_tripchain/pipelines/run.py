from __future__ import annotations

from pydantic import BaseModel

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.evaluate import estimate_violation_probability
from ev_tripchain.hosting_capacity.search import binary_search_max_n
from ev_tripchain.rng import make_rng_for


class HostingCapacityResult(BaseModel):
    n_star: int
    risk_tolerance: float
    risk_curve: list[tuple[int, float]]


def run_hosting_capacity(cfg: ProjectConfig) -> HostingCapacityResult:
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)

    def risk_at_n(n: int) -> float:
        # deterministic per-N to keep binary search stable/reproducible
        rng_n = make_rng_for(cfg.seed, int(n))
        return estimate_violation_probability(net, cfg, n=n, rng=rng_n)

    n_star, curve = binary_search_max_n(
        risk_at_n,
        n_max=cfg.hosting_capacity.n_max,
        risk_tolerance=cfg.hosting_capacity.risk_tolerance,
        max_iter=cfg.hosting_capacity.binary_search.max_iter,
        min_step=cfg.hosting_capacity.binary_search.min_step,
    )
    return HostingCapacityResult(
        n_star=n_star, risk_tolerance=cfg.hosting_capacity.risk_tolerance, risk_curve=curve
    )
