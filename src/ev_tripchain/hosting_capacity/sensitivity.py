"""Analytical voltage sensitivity hosting capacity method.

Uses linearized voltage sensitivity dV/dP to estimate hosting capacity
without iterative Monte Carlo simulation. Fast but approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.powerflow import run_powerflow


@dataclass(frozen=True)
class SensitivityResult:
    n_star_uniform: int
    n_star_weakest: int
    sensitivity_diagonal: np.ndarray  # dV_i/dP_i for each EV bus
    base_voltage: np.ndarray  # base-case voltage at each EV bus
    voltage_margin: np.ndarray  # V_base - V_min at each EV bus


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


def _compute_sensitivity_matrix(
    net: Any,
    ev_idx: list[int],
    buses: np.ndarray,
    delta_mw: float = 0.01,
) -> np.ndarray:
    """
    Compute voltage sensitivity matrix dV/dP via numerical perturbation.

    Returns matrix S of shape (n_ev_buses, n_ev_buses) where
    S[i, j] = dV_bus_i / dP_bus_j (in pu / MW).
    """
    n = len(ev_idx)
    bus_ids = np.asarray(buses, dtype=int)

    # Base case
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    run_powerflow(net)
    v_base = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float).copy()

    S = np.zeros((n, n), dtype=float)

    for j in range(n):
        # Perturb bus j
        net.load.loc[ev_idx, "p_mw"] = 0.0
        net.load.loc[ev_idx[j], "p_mw"] = delta_mw
        try:
            run_powerflow(net)
            v_perturbed = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float)
            S[:, j] = (v_perturbed - v_base) / delta_mw
        except Exception:
            S[:, j] = -1.0 / max(delta_mw, 1e-6)

    # Reset
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0

    return S


def run_sensitivity_hc(
    net: Any,
    cfg: ProjectConfig,
) -> SensitivityResult:
    """
    Voltage sensitivity hosting capacity estimation.

    Computes dV/dP via numerical perturbation, then estimates the maximum
    EV count under both uniform and worst-bus allocation assumptions.
    """
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)

    # Get base voltages
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    run_powerflow(net)
    bus_ids = np.asarray(buses, dtype=int)
    v_base = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float).copy()

    # Compute sensitivity matrix
    S = _compute_sensitivity_matrix(net, ev_idx, buses)

    p_mw_per_ev = float(cfg.ev.charge_power_kw) / 1000.0
    vmin = float(cfg.constraints.vmin_pu)
    margin = v_base - vmin  # voltage margin at each bus

    # Sensitivity diagonal: dV_i/dP_i (typically negative)
    s_diag = np.diag(S)

    # --- Worst-bus allocation: all EVs at one bus ---
    # Constraint: V_base_i + N * P_ev * S_ii >= V_min for all i
    # => N <= margin_i / (-P_ev * S_ii) for each bus i where S_ii < 0
    n_star_per_bus = np.full(n_buses, float("inf"))
    for i in range(n_buses):
        if s_diag[i] < -1e-12:
            n_star_per_bus[i] = margin[i] / (-p_mw_per_ev * s_diag[i])
    n_star_weakest = int(np.floor(np.min(n_star_per_bus)))
    n_star_weakest = max(0, n_star_weakest)

    # --- Uniform allocation: N/n_buses EVs per bus ---
    # Constraint: V_base_i + (N/n_buses) * P_ev * sum_j S_ij >= V_min for all i
    s_row_sum = S.sum(axis=1)  # sum_j S_ij for each bus i
    n_star_uniform_per_bus = np.full(n_buses, float("inf"))
    for i in range(n_buses):
        if s_row_sum[i] < -1e-12:
            n_star_uniform_per_bus[i] = margin[i] * n_buses / (-p_mw_per_ev * s_row_sum[i])
    n_star_uniform = int(np.floor(np.min(n_star_uniform_per_bus)))
    n_star_uniform = max(0, n_star_uniform)

    return SensitivityResult(
        n_star_uniform=n_star_uniform,
        n_star_weakest=n_star_weakest,
        sensitivity_diagonal=s_diag,
        base_voltage=v_base,
        voltage_margin=margin,
    )
