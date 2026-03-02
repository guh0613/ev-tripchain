from __future__ import annotations

from typing import Any


def run_powerflow(net: Any) -> None:
    import pandapower as pp  # type: ignore

    pp.runpp(net, algorithm="nr", init="auto", calculate_voltage_angles=False)

