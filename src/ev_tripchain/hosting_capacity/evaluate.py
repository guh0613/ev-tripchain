from __future__ import annotations

from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.constraints import check_violations
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.monte_carlo import estimate_event_probability
from ev_tripchain.mobility.profile import build_ev_profile_mw


def _ensure_ev_load_elements(net: Any) -> list[int]:
    import pandapower as pp  # type: ignore

    if "ev_tripchain_kind" not in net.load.columns:
        net.load["ev_tripchain_kind"] = ""

    ev_idx = net.load.index[net.load["ev_tripchain_kind"] == "ev"].tolist()
    if ev_idx:
        return ev_idx

    # one EV load element per bus (except ext_grid bus if it exists)
    ext_buses = set(net.ext_grid.bus.tolist()) if hasattr(net, "ext_grid") else set()
    for bus in net.bus.index.tolist():
        if bus in ext_buses:
            continue
        idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"ev@{bus}")
        net.load.at[idx, "ev_tripchain_kind"] = "ev"
        ev_idx.append(idx)
    return ev_idx


def estimate_violation_probability(
    net: Any,
    cfg: ProjectConfig,
    *,
    n: int,
    rng: np.random.Generator,
) -> float:
    """
    Monte Carlo estimate of violation probability under EV scale N.

    A scenario is counted as 'violating' if ANY time step violates ANY hard constraint.
    """
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)

    def simulate_event(rng_s: np.random.Generator) -> bool:
        profile = build_ev_profile_mw(
            cfg=cfg,
            n_vehicles=n,
            buses=buses,
            n_buses=n_buses,
            rng=rng_s,
        )  # shape: (T, n_buses)

        for t in range(cfg.time.n_steps):
            net.load.loc[ev_idx, "p_mw"] = profile[t, :]
            try:
                run_powerflow(net)
            except Exception:
                return True
            v = check_violations(
                net,
                vmin=cfg.constraints.vmin_pu,
                vmax=cfg.constraints.vmax_pu,
                line_max=cfg.constraints.line_loading_max_percent,
                trafo_max=cfg.constraints.trafo_loading_max_percent,
            )
            if v.any_violation:
                return True
        return False

    est = estimate_event_probability(
        simulate_event, n_scenarios=cfg.hosting_capacity.scenarios, rng=rng
    )
    return est.p_hat
