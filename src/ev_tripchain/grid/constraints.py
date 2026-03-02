from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ViolationSummary:
    voltage_violation: bool
    line_overload: bool
    trafo_overload: bool

    @property
    def any_violation(self) -> bool:
        return self.voltage_violation or self.line_overload or self.trafo_overload


def check_violations(
    net: Any, *, vmin: float, vmax: float, line_max: float, trafo_max: float
) -> ViolationSummary:
    vm = net.res_bus.vm_pu.to_numpy()
    voltage_violation = bool(((vm < vmin) | (vm > vmax)).any())

    line_overload = False
    if len(net.res_line) > 0:
        line_loading = net.res_line.loading_percent.to_numpy()
        line_overload = bool((line_loading > line_max).any())

    trafo_overload = False
    if hasattr(net, "res_trafo") and len(net.res_trafo) > 0:
        trafo_loading = net.res_trafo.loading_percent.to_numpy()
        trafo_overload = bool((trafo_loading > trafo_max).any())

    return ViolationSummary(
        voltage_violation=voltage_violation,
        line_overload=line_overload,
        trafo_overload=trafo_overload,
    )
