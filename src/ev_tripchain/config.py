from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class CaseConfig(BaseModel):
    name: str = Field(default="cigre_mv", description="pandapower case name")
    load_scale: float = Field(default=0.5, description="Scale factor for base loads (0-1)")


class TimeConfig(BaseModel):
    step_minutes: int = Field(default=15, gt=0)
    n_steps: int = Field(default=96, gt=0)
    n_days: int = Field(
        default=2,
        gt=0,
        description="Continuous simulation days per Monte Carlo scenario.",
    )

    @property
    def day_minutes(self) -> int:
        return int(self.step_minutes) * int(self.n_steps)

    @property
    def total_steps(self) -> int:
        return int(self.n_steps) * int(self.n_days)

    @property
    def total_minutes(self) -> int:
        return int(self.day_minutes) * int(self.n_days)


class BinarySearchConfig(BaseModel):
    max_iter: int = 16
    min_step: int = 1
    initial_hi: int = Field(
        default=128,
        gt=0,
        description="Initial upper bracket for adaptive exponential search before binary refinement.",
    )


class HostingCapacityConfig(BaseModel):
    scenarios: int = 50
    risk_tolerance: float = 0.05
    common_random_numbers: bool = Field(
        default=True,
        description="Use scenario-index RNGs (common random numbers) across N.",
    )
    parallel_workers: int = Field(
        default=1,
        description=(
            "Process workers for Monte Carlo scenario evaluation. "
            "1 disables parallelism; 0 uses up to all available CPU cores."
        ),
    )
    risk_metric: Literal["p_hat", "ci95_high"] = Field(
        default="p_hat",
        description="Risk metric used for N* decision; thesis baseline uses p_hat and reports Wilson CI separately.",
    )
    n_max: int = 2000
    binary_search: BinarySearchConfig = Field(default_factory=BinarySearchConfig)

    @property
    def resolved_parallel_workers(self) -> int:
        requested = int(self.parallel_workers)
        if requested > 0:
            return requested
        return max(1, int(os.cpu_count() or 1))


class HardConstraintsConfig(BaseModel):
    vmin_pu: float = 0.95
    vmax_pu: float = 1.05
    line_loading_max_percent: float = 100.0
    trafo_loading_max_percent: float = 100.0


class SoftConstraintsConfig(BaseModel):
    nominal_voltage_pu: float = Field(
        default=1.0,
        description="Reference voltage used for soft metrics such as voltage deviation.",
    )


class ConstraintsConfig(BaseModel):
    hard: HardConstraintsConfig = Field(default_factory=HardConstraintsConfig)
    soft: SoftConstraintsConfig = Field(default_factory=SoftConstraintsConfig)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_flat_shape(cls, data: object) -> object:
        if data is None or not isinstance(data, dict):
            return data
        if "hard" in data or "soft" in data:
            return data

        hard_keys = {
            "vmin_pu",
            "vmax_pu",
            "line_loading_max_percent",
            "trafo_loading_max_percent",
        }
        soft_keys = {"nominal_voltage_pu"}

        hard = {k: data[k] for k in hard_keys if k in data}
        soft = {k: data[k] for k in soft_keys if k in data}
        other = {
            k: v
            for k, v in data.items()
            if k not in hard_keys and k not in soft_keys
        }
        return {**other, "hard": hard, "soft": soft}

    @property
    def vmin_pu(self) -> float:
        return float(self.hard.vmin_pu)

    @property
    def vmax_pu(self) -> float:
        return float(self.hard.vmax_pu)

    @property
    def line_loading_max_percent(self) -> float:
        return float(self.hard.line_loading_max_percent)

    @property
    def trafo_loading_max_percent(self) -> float:
        return float(self.hard.trafo_loading_max_percent)

    @property
    def nominal_voltage_pu(self) -> float:
        return float(self.soft.nominal_voltage_pu)


class StartTimeComponent(BaseModel):
    weight: float
    mean: str  # "HH:MM"
    std_minutes: int


class EVConfig(BaseModel):
    charge_power_kw: float = 7.2
    sessions_per_vehicle_mean: float = 1.0
    duration_minutes_mean: float = 120
    duration_minutes_std: float = 40
    start_time_mix: list[StartTimeComponent] = Field(default_factory=list)


class OrderedStrategyConfig(BaseModel):
    window_start: str = "22:00"
    window_end: str = "06:00"
    random_delay: bool = Field(
        default=True,
        description="Spread charging starts uniformly within window to avoid synchronous peak.",
    )


class NavigationStrategyConfig(BaseModel):
    candidate_k: int = 5
    distance_limit_m: float | None = Field(
        default=None,
        description="Max navigation distance (meters). None disables distance filtering.",
    )
    distance_beta: float = Field(
        default=1.0,
        description="Distance penalty exponent in navigation (weight ~ 1/d^beta).",
    )
    dynamic_scoring: bool = Field(
        default=True,
        description="Update bus scores based on accumulated load to achieve spatial dispersion.",
    )
    dynamic_safety_buffer_pu: float = Field(
        default=0.002,
        ge=0.0,
        description="Preferred minimum voltage headroom used by dynamic navigation scoring.",
    )
    dynamic_voltage_penalty_window_pu: float = Field(
        default=0.006,
        gt=0.0,
        description=(
            "Soft-penalty window for dynamic voltage scoring. "
            "Larger values make the voltage factor less brittle near the limit."
        ),
    )
    path_congestion_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Weight of upstream-path congestion in navigation scoring. "
            "0 disables the feeder-sharing penalty."
        ),
    )


class StrategyConfig(BaseModel):
    name: Literal["uncontrolled", "nearest", "navigation", "ordered"] = "uncontrolled"
    ordered: OrderedStrategyConfig = Field(default_factory=OrderedStrategyConfig)
    navigation: NavigationStrategyConfig = Field(default_factory=NavigationStrategyConfig)


class TripChainConfig(BaseModel):
    n_zones: int = 50
    other_stops_mean: float = 1.2

    first_departure_mean: str = "07:30"
    first_departure_std_minutes: int = 35

    work_duration_mean_minutes: int = 8 * 60
    work_duration_std_minutes: int = 45

    other_dwell_mean_minutes: int = 60
    other_dwell_std_minutes: int = 30

    travel_minutes_per_km: float = 2.2
    distance_km_mean: float = 8.0
    distance_km_std: float = 4.0


class SOCConfig(BaseModel):
    battery_capacity_kwh: float = 60.0
    consumption_kwh_per_km: float = 0.18

    initial_soc_mean: float = 0.7
    initial_soc_std: float = 0.15

    soc_min: float = 0.0
    soc_max: float = 1.0

    charge_efficiency: float = 0.92
    charge_trigger_soc: float = 0.3
    charge_purposes: list[str] = Field(default_factory=lambda: ["home", "work"])


class MappingConfig(BaseModel):
    policy: Literal["random_onehot", "from_pairs"] = "random_onehot"
    n_nodes: int = 50
    node_bus_pairs: list[tuple[int, int]] = Field(default_factory=list)


class MobilityConfig(BaseModel):
    model: Literal["synthetic_sessions", "tripchain_soc"] = "synthetic_sessions"
    trip_chain: TripChainConfig = Field(default_factory=TripChainConfig)
    soc: SOCConfig = Field(default_factory=SOCConfig)
    mapping: MappingConfig = Field(default_factory=MappingConfig)


class ProjectConfig(BaseModel):
    seed: int = 42
    case: CaseConfig = Field(default_factory=CaseConfig)
    time: TimeConfig = Field(default_factory=TimeConfig)
    hosting_capacity: HostingCapacityConfig = Field(default_factory=HostingCapacityConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    ev: EVConfig = Field(default_factory=EVConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    mobility: MobilityConfig = Field(default_factory=MobilityConfig)

    @model_validator(mode="after")
    def _validate_tripchain_mapping_shape(self) -> ProjectConfig:
        if self.mobility.model != "tripchain_soc":
            return self

        n_zones = int(self.mobility.trip_chain.n_zones)
        n_nodes = int(self.mobility.mapping.n_nodes)
        if n_zones != n_nodes:
            raise ValueError(
                "mobility.trip_chain.n_zones must match mobility.mapping.n_nodes "
                f"for tripchain_soc (got {n_zones} vs {n_nodes})."
            )
        return self


def load_config(path: Path) -> ProjectConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProjectConfig.model_validate(data)
