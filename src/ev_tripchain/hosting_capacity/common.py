from __future__ import annotations

from typing import Any

import numpy as np

from ev_tripchain.grid.powerflow import run_powerflow


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
