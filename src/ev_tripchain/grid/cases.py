from __future__ import annotations

import math
from typing import Any


_IEEE33_IMPROVED_LINE_CAPACITY_MW = 6.0
_IEEE33_NODE_TRAFO_CAPACITY_MVA = 6.3
_IEEE33_UPSTREAM_CAPACITY_MVA = 63.0


def _calibrate_ieee33_thermal_limits(net: Any) -> None:
    """
    Calibrate IEEE33 line thermal limits so `loading_percent` is meaningful.

    Aligned to the project's core reference ("improved IEEE33", appendix table A2):
    uniform line capacity of 6 MW on a 12.66 kV system.
    """
    if len(getattr(net, "line", ())) == 0:
        return

    try:
        vn_kv = float(net.bus.vn_kv.median())
    except Exception:
        return

    if not (vn_kv > 0):
        return

    s_mva = float(_IEEE33_IMPROVED_LINE_CAPACITY_MW)
    max_i_ka = s_mva / (math.sqrt(3.0) * vn_kv)
    net.line["max_i_ka"] = max_i_ka


def _annotate_ieee33_node_supply_limits(net: Any) -> None:
    """
    Attach literature-based node supply-transformer ratings to the medium-voltage buses.

    The project currently evaluates an MV feeder model only, without explicit distribution
    transformers at each load point. We therefore store the 6.3 MVA node capacity from the
    thesis reference as bus metadata so constraints can use it as a local supply-capacity
    surrogate.
    """
    if len(getattr(net, "bus", ())) == 0:
        return

    ext_buses = (
        set(getattr(net.ext_grid, "bus", []).tolist())
        if hasattr(net, "ext_grid")
        else set()
    )
    net.bus["ev_tripchain_node_trafo_sn_mva"] = float(_IEEE33_NODE_TRAFO_CAPACITY_MVA)
    if ext_buses:
        net.bus.loc[list(ext_buses), "ev_tripchain_node_trafo_sn_mva"] = math.nan
    net["ev_tripchain_upstream_supply_sn_mva"] = float(_IEEE33_UPSTREAM_CAPACITY_MVA)


def _apply_improved_ieee33_design(net: Any) -> None:
    """
    Apply the literature-backed "improved IEEE33" redesign used by this project.

    References in `docs/references` indicate two reproducible changes that fit the current
    simulation scope:
    1. line thermal ratings are unified to 6 MW (appendix table A2);
    2. the standard five normally-open tie lines are closed to strengthen end-node coupling
       and avoid the raw IEEE33 case becoming voltage-limited too early.

    The same sources also state a 6.3 MVA node transformer capacity and a 63 MVA upstream
    supply capacity. Since the repo models only the feeder itself, those capacities are
    stored as metadata for constraint evaluation.
    """
    _calibrate_ieee33_thermal_limits(net)

    if len(getattr(net, "line", ())) > 0 and "in_service" in net.line.columns:
        tie_mask = ~net.line["in_service"].astype(bool)
        if tie_mask.any():
            net.line.loc[tie_mask, "in_service"] = True
            net.line["ev_tripchain_improved_tie"] = False
            net.line.loc[tie_mask, "ev_tripchain_improved_tie"] = True
        else:
            net.line["ev_tripchain_improved_tie"] = False

    _annotate_ieee33_node_supply_limits(net)
    net["ev_tripchain_case_variant"] = "ieee33_improved"


def load_case(name: str, *, load_scale: float = 1.0) -> Any:
    """
    Load a pandapower network case by name.

    Parameters
    ----------
    name : str
        Case name. Supported:
        - "cigre_mv"/"cigre"
        - "ieee33"/"improved_ieee33" (literature-backed improved IEEE33)
        - "ieee33bw"/"case33bw"/"ieee33_standard" (raw pandapower IEEE33 topology)
        - "simple"/"4bus"
    load_scale : float
        Scale factor applied to base loads (p_mw and q_mvar).
        Values < 1.0 represent moderate-load scenarios with headroom for EV hosting.

    Notes
    -----
    The default "ieee33" case uses the project's literature-backed improved variant rather
    than the raw pandapower feeder. Use "case33bw" or "ieee33_standard" to access the
    original open-tie topology.
    """
    try:
        import pandapower.networks as pn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pandapower is required. Install with: pip install pandapower"
        ) from exc

    name = name.lower()
    if name in {"cigre_mv", "cigre"}:
        net = pn.create_cigre_network_mv(with_der=False)
    elif name in {"ieee33", "ieee_33", "ieee33_improved", "improved_ieee33"}:
        net = pn.case33bw()
        _apply_improved_ieee33_design(net)
    elif name in {"ieee33bw", "case33bw", "ieee33_standard", "raw_ieee33"}:
        net = pn.case33bw()
        _calibrate_ieee33_thermal_limits(net)
        net["ev_tripchain_case_variant"] = "ieee33_standard"
    elif name in {"simple", "4bus"}:
        net = pn.simple_four_bus_system()
    else:
        raise ValueError(f"Unknown case: {name!r}")

    if load_scale != 1.0:
        net.load.p_mw *= load_scale
        net.load.q_mvar *= load_scale

    return net
