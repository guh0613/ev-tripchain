from __future__ import annotations

from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.constraints import check_violations
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.monte_carlo import MonteCarloEstimate, estimate_event_probability
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.rng import make_rng_for


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


def _static_voltage_margin_score(
    net: Any,
    *,
    buses: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Compute a static 'grid headroom' score per EV-load column based on base-case voltages.

    This is a lightweight proxy for "node margin" mentioned in the opening report/literature.
    """
    try:
        run_powerflow(net)
        bus_ids = [int(b) for b in np.asarray(buses, dtype=int).reshape(-1).tolist()]
        vm = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float)  # type: ignore[attr-defined]
        margin = np.minimum(vm - float(vmin), float(vmax) - vm)
        margin = np.clip(margin, 0.0, None)
        if not np.isfinite(margin).all():
            raise ValueError("non-finite voltage margin")
        return margin
    except Exception:
        # Fallback: neutral scores (no grid preference).
        return np.ones(int(np.asarray(buses).size), dtype=float)


def _base_case_is_safe(
    net: Any,
    *,
    ev_idx: list[int],
    vmin: float,
    vmax: float,
    line_max: float,
    trafo_max: float,
) -> bool:
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    try:
        run_powerflow(net)
    except Exception:
        return False
    return not check_violations(
        net,
        vmin=vmin,
        vmax=vmax,
        line_max=line_max,
        trafo_max=trafo_max,
    ).any_violation


def estimate_violation_probability_mc(
    net: Any,
    cfg: ProjectConfig,
    *,
    n: int,
    rng: np.random.Generator,
    progress: callable | None = None,
) -> MonteCarloEstimate:
    """
    Monte Carlo estimate of violation probability under EV scale N.

    A scenario is counted as 'violating' if ANY time step violates ANY hard constraint.
    """
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)
    vmin = cfg.constraints.vmin_pu
    vmax = cfg.constraints.vmax_pu
    line_max = cfg.constraints.line_loading_max_percent
    trafo_max = cfg.constraints.trafo_loading_max_percent

    # Ensure we're scoring a "no-EV" base operating point (net is reused across calls).
    base_case_safe = _base_case_is_safe(
        net,
        ev_idx=ev_idx,
        vmin=vmin,
        vmax=vmax,
        line_max=line_max,
        trafo_max=trafo_max,
    )
    if not base_case_safe:
        n_scenarios = int(max(cfg.hosting_capacity.scenarios, 0))
        return MonteCarloEstimate(
            n=n_scenarios,
            n_events=n_scenarios,
            p_hat=1.0,
            ci95_low=1.0,
            ci95_high=1.0,
        )

    bus_score = _static_voltage_margin_score(
        net,
        buses=buses,
        vmin=vmin,
        vmax=vmax,
    )

    def simulate_event(rng_s: np.random.Generator) -> bool:
        profile = build_ev_profile_mw(
            cfg=cfg,
            n_vehicles=n,
            buses=buses,
            n_buses=n_buses,
            bus_score=bus_score,
            rng=rng_s,
        )  # shape: (T, n_buses)

        total_per_step = profile.sum(axis=1)
        # Check time steps with highest total load first (most likely to violate).
        # Skip steps with zero EV load (base case already verified safe).
        nonzero_mask = total_per_step > 1e-9
        if not nonzero_mask.any():
            return False
        nonzero_steps = np.where(nonzero_mask)[0]
        step_order = nonzero_steps[np.argsort(-total_per_step[nonzero_steps])]

        pf_init = "auto"
        for t in step_order:
            net.load.loc[ev_idx, "p_mw"] = profile[t, :]
            try:
                run_powerflow(net, init=pf_init)
                pf_init = "results"  # warm-start subsequent solves
            except Exception:
                pf_init = "auto"
                return True
            v = check_violations(
                net, vmin=vmin, vmax=vmax,
                line_max=line_max, trafo_max=trafo_max,
            )
            if v.any_violation:
                return True
        return False

    scenario_rng = None
    if cfg.hosting_capacity.common_random_numbers:
        scenario_rng = lambda i: make_rng_for(int(cfg.seed), 9103, int(i))

    return estimate_event_probability(
        simulate_event,
        n_scenarios=cfg.hosting_capacity.scenarios,
        rng=rng,
        scenario_rng=scenario_rng,
        early_stop_threshold=cfg.hosting_capacity.risk_tolerance,
        progress=progress,
        progress_every=5,
    )


def estimate_violation_probability(
    net: Any,
    cfg: ProjectConfig,
    *,
    n: int,
    rng: np.random.Generator,
) -> float:
    """Backwards-compatible wrapper that returns p_hat only."""
    return float(estimate_violation_probability_mc(net, cfg, n=n, rng=rng).p_hat)
