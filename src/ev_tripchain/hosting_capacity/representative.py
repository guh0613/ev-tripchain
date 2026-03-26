"""Representative deterministic EV charging profiles for HC baselines.

The thesis proposal describes deterministic hosting-capacity assessment as a
"typical day + multi-period power-flow" procedure. The core probabilistic
pipeline already samples stochastic EV scenarios, so this module collapses a
small set of representative scenarios into one reproducible template profile
that can be shared by deterministic and sensitivity baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.rng import make_rng_for


@dataclass(frozen=True)
class RepresentativeProfile:
    aggregated_profile_mw: np.ndarray
    per_vehicle_profile_mw: np.ndarray
    bus_energy_share: np.ndarray
    sample_vehicles: int
    n_samples: int


def ensure_ev_load_elements(net: Any) -> list[int]:
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


def compute_static_voltage_margin_score(
    net: Any,
    *,
    buses: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    try:
        run_powerflow(net)
        bus_ids = np.asarray(buses, dtype=int).reshape(-1)
        vm = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float)  # type: ignore[attr-defined]
        margin = np.minimum(vm - float(vmin), float(vmax) - vm)
        margin = np.clip(margin, 0.0, None)
        if not np.isfinite(margin).all():
            raise ValueError("non-finite voltage margin")
        return margin
    except Exception:
        return np.ones(int(np.asarray(buses).size), dtype=float)


def _default_sample_vehicles(cfg: ProjectConfig) -> int:
    n_max = max(int(cfg.hosting_capacity.n_max), 1)
    initial_hi = max(int(cfg.hosting_capacity.binary_search.initial_hi), 1)
    base = min(max(initial_hi, 64), 256)
    return max(1, min(base, n_max))


def _default_n_samples(cfg: ProjectConfig) -> int:
    # The main reference discusses multiple "typical days"; we keep four
    # representative scenarios here to stay lightweight while avoiding
    # single-scenario arbitrariness.
    return max(1, min(int(cfg.hosting_capacity.scenarios), 4))


def build_representative_ev_profile(
    net: Any,
    cfg: ProjectConfig,
    *,
    ev_idx: list[int] | None = None,
    buses: np.ndarray | None = None,
    navigation_voltage_model: Any | None = None,
    sample_vehicles: int | None = None,
    n_samples: int | None = None,
) -> RepresentativeProfile:
    if ev_idx is None:
        ev_idx = ensure_ev_load_elements(net)
    if buses is None:
        buses = net.load.loc[ev_idx, "bus"].to_numpy()

    bus_ids = np.asarray(buses, dtype=int).reshape(-1)
    n_buses = int(bus_ids.size)
    sample_n = int(sample_vehicles or _default_sample_vehicles(cfg))
    sample_n = max(sample_n, 1)
    scenario_count = int(n_samples or _default_n_samples(cfg))
    scenario_count = max(scenario_count, 1)

    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    bus_score = compute_static_voltage_margin_score(
        net,
        buses=bus_ids,
        vmin=float(cfg.constraints.vmin_pu),
        vmax=float(cfg.constraints.vmax_pu),
    )

    total_steps = int(cfg.time.total_steps)
    profile_sum = np.zeros((total_steps, n_buses), dtype=float)
    for scenario_idx in range(scenario_count):
        rng = make_rng_for(int(cfg.seed), 7301, scenario_idx)
        profile_sum += build_ev_profile_mw(
            cfg=cfg,
            n_vehicles=sample_n,
            buses=bus_ids,
            n_buses=n_buses,
            bus_score=bus_score,
            navigation_voltage_model=navigation_voltage_model,
            rng=rng,
        )

    aggregated = profile_sum / float(scenario_count)
    per_vehicle = aggregated / float(sample_n)
    energy_by_bus = aggregated.sum(axis=0)
    total_energy = float(energy_by_bus.sum())
    if total_energy > 0.0:
        bus_energy_share = energy_by_bus / total_energy
    elif n_buses > 0:
        bus_energy_share = np.full(n_buses, 1.0 / n_buses, dtype=float)
    else:
        bus_energy_share = np.zeros(0, dtype=float)

    return RepresentativeProfile(
        aggregated_profile_mw=aggregated,
        per_vehicle_profile_mw=per_vehicle,
        bus_energy_share=bus_energy_share,
        sample_vehicles=sample_n,
        n_samples=scenario_count,
    )
