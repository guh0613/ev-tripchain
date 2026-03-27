"""Analytical voltage sensitivity hosting capacity method.

Uses linearized voltage sensitivity dV/dP to estimate hosting capacity.
Besides the legacy extreme-allocation diagnostics, the main result now
reuses the same representative charging template as the deterministic
baseline so the method has clearer horizontal comparison meaning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.common import ensure_ev_load_elements
from ev_tripchain.hosting_capacity.representative import build_representative_ev_profile


@dataclass(frozen=True)
class SensitivityResult:
    n_star_representative: int
    n_star_uniform: int
    n_star_weakest: int
    sensitivity_diagonal: np.ndarray  # dV_i/dP_i for each EV bus
    base_voltage: np.ndarray  # base-case voltage at each EV bus
    voltage_margin: np.ndarray  # V_base - V_min at each EV bus
    representative_bus_share: np.ndarray


@dataclass(frozen=True)
class VoltageSensitivityModel:
    base_voltage_pu: np.ndarray
    sensitivity_pu_per_mw: np.ndarray
    vmin_pu: float
    vmax_pu: float
    path_incidence: np.ndarray | None = None
    line_capacity_mw: np.ndarray | None = None
    base_line_loading_percent: np.ndarray | None = None
    line_loading_limit_percent: float = 100.0


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


def _build_line_path_incidence(
    net: Any,
    *,
    buses: np.ndarray,
) -> np.ndarray | None:
    if len(getattr(net, "line", ())) == 0 or len(getattr(net, "ext_grid", ())) == 0:
        return None

    bus_ids = np.asarray(buses, dtype=int).reshape(-1)
    if bus_ids.size == 0:
        return np.zeros((0, 0), dtype=float)

    line_df = net.line
    line_index = list(line_df.index.tolist())
    line_pos = {int(idx): pos for pos, idx in enumerate(line_index)}
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for idx in line_index:
        row = line_df.loc[idx]
        if "in_service" in line_df.columns and not bool(row["in_service"]):
            continue
        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        pos = line_pos[int(idx)]
        adjacency.setdefault(fb, []).append((tb, pos))
        adjacency.setdefault(tb, []).append((fb, pos))

    if not adjacency:
        return None

    root_bus = int(net.ext_grid.bus.iloc[0])
    parent_bus: dict[int, int] = {root_bus: root_bus}
    parent_line_pos: dict[int, int] = {}
    queue = [root_bus]
    head = 0
    while head < len(queue):
        bus = queue[head]
        head += 1
        for nb, pos in adjacency.get(bus, []):
            if nb in parent_bus:
                continue
            parent_bus[nb] = bus
            parent_line_pos[nb] = pos
            queue.append(nb)

    n_lines = len(line_index)
    incidence = np.zeros((n_lines, bus_ids.size), dtype=float)
    for col, bus in enumerate(bus_ids.tolist()):
        current = int(bus)
        guard = 0
        while current != root_bus:
            if current not in parent_bus or current not in parent_line_pos:
                return None
            pos = parent_line_pos[current]
            incidence[pos, col] = 1.0
            current = parent_bus[current]
            guard += 1
            if guard > n_lines:
                return None
    return incidence


def _compute_line_capacity_mw(net: Any) -> np.ndarray | None:
    if len(getattr(net, "line", ())) == 0 or "max_i_ka" not in net.line.columns:
        return None

    out = np.zeros(len(net.line), dtype=float)
    for pos, idx in enumerate(net.line.index.tolist()):
        row = net.line.loc[idx]
        max_i_ka = float(row["max_i_ka"])
        if not np.isfinite(max_i_ka) or max_i_ka <= 0.0:
            return None
        from_bus = int(row["from_bus"])
        try:
            vn_kv = float(net.bus.loc[from_bus, "vn_kv"])
        except Exception:
            return None
        if not np.isfinite(vn_kv) or vn_kv <= 0.0:
            return None
        out[pos] = math.sqrt(3.0) * vn_kv * max_i_ka
    return out


def build_voltage_sensitivity_model(
    net: Any,
    *,
    ev_idx: list[int],
    buses: np.ndarray,
    vmin: float,
    vmax: float,
    line_max: float = 100.0,
    delta_mw: float = 0.01,
) -> VoltageSensitivityModel:
    """
    Build a linearized voltage model V ~= V_base + S @ P_ev for navigation guidance.

    The model is evaluated around the EV-free base case so it stays lightweight enough
    to be reused inside Monte Carlo charging decisions.
    """
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    run_powerflow(net)
    bus_ids = np.asarray(buses, dtype=int)
    v_base = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float).copy()
    base_line_loading = None
    if len(getattr(net, "res_line", ())) > 0 and "loading_percent" in net.res_line.columns:
        base_line_loading = net.res_line["loading_percent"].to_numpy(dtype=float).copy()
    sensitivity = _compute_sensitivity_matrix(
        net,
        ev_idx,
        buses,
        delta_mw=float(delta_mw),
    )
    path_incidence = _build_line_path_incidence(net, buses=bus_ids)
    line_capacity_mw = _compute_line_capacity_mw(net)
    return VoltageSensitivityModel(
        base_voltage_pu=v_base,
        sensitivity_pu_per_mw=sensitivity,
        vmin_pu=float(vmin),
        vmax_pu=float(vmax),
        path_incidence=path_incidence,
        line_capacity_mw=line_capacity_mw,
        base_line_loading_percent=base_line_loading,
        line_loading_limit_percent=float(line_max),
    )


def run_sensitivity_hc(
    net: Any,
    cfg: ProjectConfig,
) -> SensitivityResult:
    """
    Voltage sensitivity hosting capacity estimation.

    Computes dV/dP via numerical perturbation. The main result uses a
    representative EV charging template for comparison with the deterministic
    and Monte Carlo methods, while the uniform and worst-bus estimates are
    retained as analytical diagnostics.
    """
    ev_idx = ensure_ev_load_elements(net)
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

    representative_model = VoltageSensitivityModel(
        base_voltage_pu=v_base,
        sensitivity_pu_per_mw=S,
        vmin_pu=float(cfg.constraints.vmin_pu),
        vmax_pu=float(cfg.constraints.vmax_pu),
        path_incidence=_build_line_path_incidence(net, buses=bus_ids),
        line_capacity_mw=_compute_line_capacity_mw(net),
        base_line_loading_percent=(
            net.res_line["loading_percent"].to_numpy(dtype=float).copy()
            if len(getattr(net, "res_line", ())) > 0 and "loading_percent" in net.res_line.columns
            else None
        ),
        line_loading_limit_percent=float(cfg.constraints.line_loading_max_percent),
    )
    navigation_voltage_model = None
    if cfg.strategy.name == "navigation" and cfg.strategy.navigation.dynamic_scoring:
        navigation_voltage_model = representative_model

    representative = build_representative_ev_profile(
        net,
        cfg,
        ev_idx=ev_idx,
        buses=bus_ids,
        navigation_voltage_model=navigation_voltage_model,
    )
    per_vehicle_profile = representative.per_vehicle_profile_mw
    voltage_delta_per_vehicle = per_vehicle_profile @ S.T

    representative_limit = float(cfg.hosting_capacity.n_max)
    lower_margin = v_base - float(cfg.constraints.vmin_pu)
    upper_margin = float(cfg.constraints.vmax_pu) - v_base
    for coeff, low, high in zip(
        voltage_delta_per_vehicle.reshape(-1),
        np.broadcast_to(lower_margin, voltage_delta_per_vehicle.shape).reshape(-1),
        np.broadcast_to(upper_margin, voltage_delta_per_vehicle.shape).reshape(-1),
        strict=True,
    ):
        if coeff < -1e-12:
            representative_limit = min(representative_limit, low / (-coeff))
        elif coeff > 1e-12:
            representative_limit = min(representative_limit, high / coeff)
    n_star_representative = int(np.floor(representative_limit))
    n_star_representative = min(max(n_star_representative, 0), int(cfg.hosting_capacity.n_max))

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
        n_star_representative=n_star_representative,
        n_star_uniform=n_star_uniform,
        n_star_weakest=n_star_weakest,
        sensitivity_diagonal=s_diag,
        base_voltage=v_base,
        voltage_margin=margin,
        representative_bus_share=representative.bus_energy_share,
    )
