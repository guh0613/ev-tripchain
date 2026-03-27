"""Deterministic typical-profile hosting capacity method.

Builds a representative multi-period EV charging template and uses full
power-flow checks to find the maximum safe EV count under that template.
Compared with the previous "all EVs at the weakest bus" extreme assumption,
this is a more meaningful deterministic baseline for method comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.constraints import evaluate_constraints
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.common import ensure_ev_load_elements
from ev_tripchain.hosting_capacity.representative import (
    build_representative_ev_profile,
)
from ev_tripchain.hosting_capacity.search import binary_search_max_n
from ev_tripchain.hosting_capacity.sensitivity import build_voltage_sensitivity_model


@dataclass(frozen=True)
class DeterministicResult:
    n_star: int
    weakest_bus_id: int
    weakest_bus_voltage_pu: float
    risk_curve: list[tuple[int, float]]


def run_deterministic_hc(
    net: Any,
    cfg: ProjectConfig,
) -> DeterministicResult:
    """
    Deterministic hosting capacity on a representative multi-period template.

    The representative template is extracted from a small set of typical EV
    scenarios, then scaled with N. Binary search finds the largest N whose
    full power-flow trajectory stays within all hard constraints.
    """
    ev_idx = ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()

    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    run_powerflow(net)
    vm_base = net.res_bus.vm_pu.to_numpy()

    # Find weakest bus among EV-connectable buses
    bus_ids = np.asarray(buses, dtype=int)
    vm_ev_buses = vm_base[bus_ids]
    weakest_col = int(np.argmin(vm_ev_buses))
    weakest_bus_id = int(bus_ids[weakest_col])
    weakest_voltage = float(vm_ev_buses[weakest_col])

    navigation_voltage_model = None
    if cfg.strategy.name == "navigation" and cfg.strategy.navigation.dynamic_scoring:
        try:
            navigation_voltage_model = build_voltage_sensitivity_model(
                net,
                ev_idx=ev_idx,
                buses=bus_ids,
                vmin=float(cfg.constraints.vmin_pu),
                vmax=float(cfg.constraints.vmax_pu),
                line_max=float(cfg.constraints.line_loading_max_percent),
            )
        except Exception:
            navigation_voltage_model = None

    representative = build_representative_ev_profile(
        net,
        cfg,
        ev_idx=ev_idx,
        buses=bus_ids,
        navigation_voltage_model=navigation_voltage_model,
    )
    per_vehicle_profile = representative.per_vehicle_profile_mw
    total_per_step = per_vehicle_profile.sum(axis=1)
    nonzero_steps = np.where(total_per_step > 1e-12)[0]
    step_order = nonzero_steps[np.argsort(-total_per_step[nonzero_steps])]

    def risk_at_n(n: int) -> float:
        net.load.loc[ev_idx, "p_mw"] = 0.0
        net.load.loc[ev_idx, "q_mvar"] = 0.0
        if int(n) <= 0 or step_order.size == 0:
            return 0.0

        scaled_profile = per_vehicle_profile * float(n)
        pf_init = "auto"
        for t in step_order:
            net.load.loc[ev_idx, "p_mw"] = scaled_profile[t, :]
            try:
                run_powerflow(net, init=pf_init)
                pf_init = "results"
            except Exception:
                return 1.0

            assessment = evaluate_constraints(
                net,
                vmin=cfg.constraints.vmin_pu,
                vmax=cfg.constraints.vmax_pu,
                line_max=cfg.constraints.line_loading_max_percent,
                trafo_max=cfg.constraints.trafo_loading_max_percent,
                nominal_voltage_pu=cfg.constraints.nominal_voltage_pu,
            )
            if assessment.hard.any_exceedance:
                return 1.0
        return 0.0

    n_star, curve = binary_search_max_n(
        risk_at_n,
        n_max=cfg.hosting_capacity.n_max,
        risk_tolerance=cfg.hosting_capacity.risk_tolerance,
        max_iter=cfg.hosting_capacity.binary_search.max_iter,
        min_step=cfg.hosting_capacity.binary_search.min_step,
        initial_hi=cfg.hosting_capacity.binary_search.initial_hi,
    )

    # Clean up
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0

    return DeterministicResult(
        n_star=n_star,
        weakest_bus_id=weakest_bus_id,
        weakest_bus_voltage_pu=weakest_voltage,
        risk_curve=curve,
    )
