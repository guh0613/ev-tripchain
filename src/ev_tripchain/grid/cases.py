from __future__ import annotations

from typing import Any


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
    elif name in {"simple", "4bus"}:
        net = pn.simple_four_bus_system()
    else:
        raise ValueError(f"Unknown case: {name!r}")

    if load_scale != 1.0:
        net.load.p_mw *= load_scale
        net.load.q_mvar *= load_scale

    return net
