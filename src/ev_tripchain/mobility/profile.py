from __future__ import annotations

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.mobility.mapping import (
    build_mapping_from_node_bus_pairs,
    build_random_onehot_mapping,
)
from ev_tripchain.mobility.soc import SOCEvolutionParams
from ev_tripchain.mobility.spatial import build_spatial_distance_model
from ev_tripchain.mobility.synthetic import build_ev_profile_mw as build_ev_profile_mw_sessions
from ev_tripchain.mobility.tripchain_profile import build_ev_profile_mw_tripchain
from ev_tripchain.mobility.tripchain_sampling import TripChainSamplingParams


def _parse_hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.strip().split(":")
    return int(hh) * 60 + int(mm)


def build_ev_profile_mw(
    *,
    cfg: ProjectConfig,
    n_vehicles: int,
    buses: np.ndarray,
    n_buses: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build aggregated EV charging profile for a single scenario.

    Output shape: (T, n_buses) in MW, aligned with the EV load elements order in
    `net.load.loc[ev_idx]`.
    """
    if cfg.mobility.model == "synthetic_sessions":
        return build_ev_profile_mw_sessions(
            cfg=cfg, n_vehicles=n_vehicles, buses=buses, n_buses=n_buses, rng=rng
        )

    # trip-chain + SOC model
    bus_ids = np.asarray(buses, dtype=int)
    n_nodes = int(cfg.mobility.mapping.n_nodes)

    # mapping is treated as system data: deterministic w.r.t cfg.seed (not per-scenario RNG)
    rng_map = np.random.default_rng(np.random.SeedSequence([int(cfg.seed), 701]))
    if cfg.mobility.mapping.policy == "from_pairs":
        mapping = build_mapping_from_node_bus_pairs(
            n_nodes=n_nodes, bus_ids=bus_ids, node_bus_pairs=cfg.mobility.mapping.node_bus_pairs
        )
    else:
        mapping = build_random_onehot_mapping(n_nodes=n_nodes, bus_ids=bus_ids, rng=rng_map)

    tc_cfg = cfg.mobility.trip_chain
    trip_params = TripChainSamplingParams(
        n_zones=n_nodes,
        day_minutes=int(cfg.time.step_minutes) * int(cfg.time.n_steps),
        other_stops_mean=float(tc_cfg.other_stops_mean),
        first_departure_mean=tc_cfg.first_departure_mean,
        first_departure_std_minutes=int(tc_cfg.first_departure_std_minutes),
        work_duration_mean_minutes=int(tc_cfg.work_duration_mean_minutes),
        work_duration_std_minutes=int(tc_cfg.work_duration_std_minutes),
        other_dwell_mean_minutes=int(tc_cfg.other_dwell_mean_minutes),
        other_dwell_std_minutes=int(tc_cfg.other_dwell_std_minutes),
        travel_minutes_per_km=float(tc_cfg.travel_minutes_per_km),
        distance_km_mean=float(tc_cfg.distance_km_mean),
        distance_km_std=float(tc_cfg.distance_km_std),
    )

    soc_cfg = cfg.mobility.soc
    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=float(soc_cfg.battery_capacity_kwh),
        consumption_kwh_per_km=float(soc_cfg.consumption_kwh_per_km),
        initial_soc_mean=float(soc_cfg.initial_soc_mean),
        initial_soc_std=float(soc_cfg.initial_soc_std),
        soc_min=float(soc_cfg.soc_min),
        soc_max=float(soc_cfg.soc_max),
        charge_power_kw=float(cfg.ev.charge_power_kw),
        charge_efficiency=float(soc_cfg.charge_efficiency),
        charge_trigger_soc=float(soc_cfg.charge_trigger_soc),
        charge_purposes=tuple(soc_cfg.charge_purposes),
    )

    ordered_window: tuple[int, int] | None = None
    if cfg.strategy.name == "ordered":
        ordered_window = (
            _parse_hhmm_to_minutes(cfg.strategy.ordered.window_start),
            _parse_hhmm_to_minutes(cfg.strategy.ordered.window_end),
        )

    bus_distance_m: np.ndarray | None = None
    candidate_bus_idx: np.ndarray | None = None
    if cfg.strategy.name in {"nearest", "navigation"}:
        spatial_model = build_spatial_distance_model(
            buses=bus_ids,
            n_buses=int(n_buses),
        )
        bus_distance_m = spatial_model.dist_m
        candidate_bus_idx = spatial_model.candidate_bus_idx

    prof = build_ev_profile_mw_tripchain(
        n_vehicles=n_vehicles,
        step_minutes=int(cfg.time.step_minutes),
        n_steps=int(cfg.time.n_steps),
        mapping=mapping,
        trip_params=trip_params,
        soc_params=soc_params,
        strategy_name=cfg.strategy.name,
        ordered_window=ordered_window,
        navigation_candidate_k=int(cfg.strategy.navigation.candidate_k),
        bus_distance_m=bus_distance_m,
        candidate_bus_idx=candidate_bus_idx,
        rng=rng,
    )
    # safety: align to expected columns
    if prof.shape != (int(cfg.time.n_steps), int(n_buses)):
        raise ValueError("Profile shape mismatch.")
    return prof
