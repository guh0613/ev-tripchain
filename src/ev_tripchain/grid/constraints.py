from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HardConstraintSummary:
    voltage_lower_exceedance_count: int
    voltage_upper_exceedance_count: int
    line_overload_count: int
    trafo_overload_count: int
    voltage_lower_max_exceedance_pu: float
    voltage_upper_max_exceedance_pu: float
    line_max_exceedance_percent: float
    trafo_max_exceedance_percent: float
    min_voltage_pu: float
    max_voltage_pu: float
    line_loading_peak_percent: float
    trafo_loading_peak_percent: float

    @property
    def voltage_exceedance(self) -> bool:
        return (
            self.voltage_lower_exceedance_count > 0
            or self.voltage_upper_exceedance_count > 0
        )

    @property
    def line_overload(self) -> bool:
        return self.line_overload_count > 0

    @property
    def trafo_overload(self) -> bool:
        return self.trafo_overload_count > 0

    @property
    def any_exceedance(self) -> bool:
        return self.voltage_exceedance or self.line_overload or self.trafo_overload


@dataclass(frozen=True)
class SoftConstraintSummary:
    voltage_deviation_max_pu: float
    voltage_deviation_mean_pu: float
    network_loss_mw: float
    line_loading_peak_percent: float
    trafo_loading_peak_percent: float


@dataclass(frozen=True)
class ConstraintEvaluation:
    hard: HardConstraintSummary
    soft: SoftConstraintSummary


def _optional_percent_array(frame: Any) -> np.ndarray:
    if frame is None or len(frame) == 0 or "loading_percent" not in frame.columns:
        return np.zeros(0, dtype=float)
    return frame["loading_percent"].to_numpy(dtype=float)


def _optional_loss_sum(frame: Any) -> float:
    if frame is None or len(frame) == 0 or "pl_mw" not in frame.columns:
        return 0.0
    return float(frame["pl_mw"].to_numpy(dtype=float).sum())


def evaluate_constraints(
    net: Any,
    *,
    vmin: float,
    vmax: float,
    line_max: float,
    trafo_max: float,
    nominal_voltage_pu: float = 1.0,
) -> ConstraintEvaluation:
    vm = net.res_bus.vm_pu.to_numpy(dtype=float)
    voltage_lower_gap = np.clip(float(vmin) - vm, 0.0, None)
    voltage_upper_gap = np.clip(vm - float(vmax), 0.0, None)

    line_loading = _optional_percent_array(getattr(net, "res_line", None))
    trafo_loading = _optional_percent_array(getattr(net, "res_trafo", None))

    line_gap = np.clip(line_loading - float(line_max), 0.0, None)
    trafo_gap = np.clip(trafo_loading - float(trafo_max), 0.0, None)

    hard = HardConstraintSummary(
        voltage_lower_exceedance_count=int(np.count_nonzero(voltage_lower_gap > 0.0)),
        voltage_upper_exceedance_count=int(np.count_nonzero(voltage_upper_gap > 0.0)),
        line_overload_count=int(np.count_nonzero(line_gap > 0.0)),
        trafo_overload_count=int(np.count_nonzero(trafo_gap > 0.0)),
        voltage_lower_max_exceedance_pu=float(voltage_lower_gap.max()) if vm.size else 0.0,
        voltage_upper_max_exceedance_pu=float(voltage_upper_gap.max()) if vm.size else 0.0,
        line_max_exceedance_percent=float(line_gap.max()) if line_gap.size else 0.0,
        trafo_max_exceedance_percent=float(trafo_gap.max()) if trafo_gap.size else 0.0,
        min_voltage_pu=float(vm.min()) if vm.size else float(nominal_voltage_pu),
        max_voltage_pu=float(vm.max()) if vm.size else float(nominal_voltage_pu),
        line_loading_peak_percent=float(line_loading.max()) if line_loading.size else 0.0,
        trafo_loading_peak_percent=float(trafo_loading.max()) if trafo_loading.size else 0.0,
    )

    voltage_deviation = np.abs(vm - float(nominal_voltage_pu))
    soft = SoftConstraintSummary(
        voltage_deviation_max_pu=float(voltage_deviation.max()) if voltage_deviation.size else 0.0,
        voltage_deviation_mean_pu=float(voltage_deviation.mean()) if voltage_deviation.size else 0.0,
        network_loss_mw=(
            _optional_loss_sum(getattr(net, "res_line", None))
            + _optional_loss_sum(getattr(net, "res_trafo", None))
        ),
        line_loading_peak_percent=hard.line_loading_peak_percent,
        trafo_loading_peak_percent=hard.trafo_loading_peak_percent,
    )

    return ConstraintEvaluation(hard=hard, soft=soft)
