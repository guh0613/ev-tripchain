from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class CaseConfig(BaseModel):
    name: str = Field(default="cigre_mv", description="pandapower case name")
    load_scale: float = Field(default=0.5, description="Scale factor for base loads (0-1)")


class TimeConfig(BaseModel):
    step_minutes: int = 15
    n_steps: int = 96


class BinarySearchConfig(BaseModel):
    max_iter: int = 16
    min_step: int = 1


class HostingCapacityConfig(BaseModel):
    scenarios: int = 50
    risk_tolerance: float = 0.05
    n_max: int = 2000
    binary_search: BinarySearchConfig = Field(default_factory=BinarySearchConfig)


class ConstraintsConfig(BaseModel):
    vmin_pu: float = 0.95
    vmax_pu: float = 1.05
    line_loading_max_percent: float = 100.0
    trafo_loading_max_percent: float = 100.0


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


class NavigationStrategyConfig(BaseModel):
    candidate_k: int = 5


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


def load_config(path: Path) -> ProjectConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ProjectConfig.model_validate(data)
