"""Deterministic extreme scenario hosting capacity method.

Constructs the worst-case charging scenario (all EVs at the weakest bus)
and uses binary search to find the maximum safe EV count.
This provides a conservative lower bound on hosting capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.constraints import evaluate_constraints
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.search import binary_search_max_n


@dataclass(frozen=True)
class DeterministicResult:
    n_star: int
    weakest_bus_id: int
    weakest_bus_voltage_pu: float
    risk_curve: list[tuple[int, float]]


def _ensure_ev_load_elements(net: Any) -> list[int]:
    import pandapower as pp  # type: ignore

    if "ev_tripchain_kind" not in net.load.columns:
        net.load["ev_tripchain_kind"] = ""

    ev_idx = net.load.index[net.load["ev_tripchain_kind"] == "ev"].tolist()
    if ev_idx:
        return ev_idx

    ext_buses = set(net.ext_grid.bus.tolist()) if hasattr(net, "ext_grid") else set()
    for bus in net.bus.index.tolist():
        if bus in ext_buses:
            continue
        idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"ev@{bus}")
        net.load.at[idx, "ev_tripchain_kind"] = "ev"
        ev_idx.append(idx)
    return ev_idx


def run_deterministic_hc(
    net: Any,
    cfg: ProjectConfig,
) -> DeterministicResult:
    """
    Deterministic extreme scenario hosting capacity.

    All N EVs are assumed to charge simultaneously at the bus with
    the lowest base-case voltage margin. Binary search finds N*.
    """
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()

    # Clear EV loads and find weakest bus
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

    p_mw_per_ev = float(cfg.ev.charge_power_kw) / 1000.0

    def risk_at_n(n: int) -> float:
        # Place all N EVs at the weakest bus
        net.load.loc[ev_idx, "p_mw"] = 0.0
        net.load.loc[ev_idx, "q_mvar"] = 0.0
        net.load.loc[ev_idx[weakest_col], "p_mw"] = n * p_mw_per_ev
        try:
            run_powerflow(net)
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
        return 1.0 if assessment.hard.any_exceedance else 0.0

    n_star, curve = binary_search_max_n(
        risk_at_n,
        n_max=cfg.hosting_capacity.n_max,
        risk_tolerance=cfg.hosting_capacity.risk_tolerance,
        max_iter=cfg.hosting_capacity.binary_search.max_iter,
        min_step=cfg.hosting_capacity.binary_search.min_step,
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
