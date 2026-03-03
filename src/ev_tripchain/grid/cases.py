from __future__ import annotations

import math
from typing import Any


_IEEE33_IMPROVED_LINE_CAPACITY_MW = 6.0


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


def load_case(name: str, *, load_scale: float = 1.0) -> Any:
    """
    Load a pandapower network case by name.

    Parameters
    ----------
    name : str
        Case name. Supported:
        - "cigre_mv"/"cigre"
        - "ieee33"/"ieee33bw"/"case33bw"
        - "simple"/"4bus"
    load_scale : float
        Scale factor applied to base loads (p_mw and q_mvar).
        Values < 1.0 represent moderate-load scenarios with headroom for EV hosting.

    Notes
    -----
    For IEEE33 (pandapower `case33bw`), this function applies a lightweight thermal-limit
    calibration so that `net.res_line.loading_percent` is not trivially near-zero.
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
    elif name in {"ieee33", "ieee_33", "ieee33bw", "case33bw"}:
        net = pn.case33bw()
        _calibrate_ieee33_thermal_limits(net)
    elif name in {"simple", "4bus"}:
        net = pn.simple_four_bus_system()
    else:
        raise ValueError(f"Unknown case: {name!r}")

    if load_scale != 1.0:
        net.load.p_mw *= load_scale
        net.load.q_mvar *= load_scale

    return net
