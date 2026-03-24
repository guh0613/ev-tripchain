from __future__ import annotations

import numpy as np

from ev_tripchain.mobility.mapping import NodeBusMapping
from ev_tripchain.mobility.soc import (
    ChargingDecision,
    SOCEvolutionParams,
    simulate_soc_and_bus_profile,
)
from ev_tripchain.mobility.spatial import choose_spatial_target_bus
from ev_tripchain.mobility.trip_chain import Stop, TripChain
from ev_tripchain.mobility.tripchain_sampling import (
    TripChainSamplingParams,
    sample_daily_trip_chain,
)


def _is_in_window(minute: int, *, window_start: int, window_end: int) -> bool:
    m = int(minute)
    ws = int(window_start)
    we = int(window_end)
    if ws == we:
        return True
    if ws < we:
        return ws <= m < we
    return (m >= ws) or (m < we)


def _next_window_start(
    minute: int,
    *,
    window_start: int,
    window_end: int,
    day_minutes: int,
) -> int:
    """
    Return the earliest minute >= `minute` that lies in the charging window.
    """
    m = int(minute)
    ws = int(window_start)
    we = int(window_end)
    t_day = int(day_minutes)
    if _is_in_window(m, window_start=ws, window_end=we):
        return m

    if ws == we:
        return m

    # Window does not cross midnight: [ws, we)
    if ws < we:
        if m < ws:
            return ws
        # next day's window start
        return t_day + ws

    # Window crosses midnight, e.g. 22:00-06:00
    # Outside-window region is [we, ws).
    if we <= m < ws:
        return ws
    return m


def build_ev_profile_mw_tripchain(
    *,
    n_vehicles: int,
    step_minutes: int,
    n_steps: int,
    mapping: NodeBusMapping,
    trip_params: TripChainSamplingParams,
    soc_params: SOCEvolutionParams,
    strategy_name: str = "uncontrolled",
    ordered_window: tuple[int, int] | None = None,
    navigation_candidate_k: int = 5,
    navigation_distance_limit_m: float | None = None,
    navigation_distance_beta: float = 1.0,
    bus_distance_m: np.ndarray | None = None,
    candidate_bus_idx: np.ndarray | None = None,
    bus_score: np.ndarray | None = None,
    ordered_random_delay: bool = False,
    dynamic_bus_score: bool = False,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build aggregated charging profile (MW) by simulating per-vehicle trip chains + SOC.

    The charging location is determined by the activity (stop zone) and mapped to a bus column
    via the provided `mapping` matrix.
    """
    n_vehicles = int(max(n_vehicles, 0))
    if n_vehicles == 0:
        return np.zeros((int(n_steps), int(mapping.n_buses)), dtype=float)

    if step_minutes <= 0 or n_steps <= 0:
        raise ValueError("step_minutes and n_steps must be positive.")

    day_minutes = int(step_minutes) * int(n_steps)
    zone_to_bus_col = mapping.node_to_bus_col()

    if strategy_name not in {"uncontrolled", "ordered", "nearest", "navigation"}:
        raise ValueError(f"Unknown strategy_name: {strategy_name!r}")
    if strategy_name in {"nearest", "navigation"}:
        if bus_distance_m is None:
            raise ValueError("bus_distance_m is required for nearest/navigation strategies.")
        if bus_distance_m.shape != (mapping.n_buses, mapping.n_buses):
            raise ValueError("bus_distance_m shape mismatch.")
        if candidate_bus_idx is not None:
            c = np.asarray(candidate_bus_idx, dtype=int).reshape(-1)
            if np.any((c < 0) | (c >= mapping.n_buses)):
                raise ValueError("candidate_bus_idx out of bounds.")
    if strategy_name == "ordered":
        if ordered_window is None:
            raise ValueError("ordered_window is required for ordered strategy.")
        ws, we = int(ordered_window[0]), int(ordered_window[1])
    else:
        ws, we = 0, 0

    profile = np.zeros((int(n_steps), int(mapping.n_buses)), dtype=float)
    peak_per_bus = np.zeros(int(mapping.n_buses), dtype=float)
    p_mw_max = float(soc_params.charge_power_kw) / 1000.0

    def simulate_vehicle(tc: TripChain) -> None:
        def decide_charge(
            *,
            stop_index: int,
            stop: Stop,
            arrival_minute: int,
            departure_minute: int,
            soc_at_arrival: float,
            needed_kwh: float,
            minutes_needed: int,
            rng: np.random.Generator,
        ) -> ChargingDecision | None:
            del stop_index, soc_at_arrival, needed_kwh

            zone = int(stop.zone)
            if zone < 0 or zone >= mapping.n_nodes:
                return None

            charge_start = int(arrival_minute)
            if strategy_name == "ordered":
                charge_start = _next_window_start(
                    charge_start,
                    window_start=ws,
                    window_end=we,
                    day_minutes=day_minutes,
                )
                if ordered_random_delay and charge_start < int(departure_minute):
                    latest_start = int(departure_minute) - int(minutes_needed)
                    if latest_start > charge_start:
                        charge_start = int(rng.integers(charge_start, latest_start + 1))
            if charge_start >= int(departure_minute):
                return None

            src_bcol = int(zone_to_bus_col[zone])
            bcol = src_bcol
            if strategy_name in {"nearest", "navigation"} and mapping.n_buses > 1:
                effective_score = bus_score
                if dynamic_bus_score and strategy_name == "navigation":
                    load_penalty = 1.0 / (1.0 + peak_per_bus / max(p_mw_max, 1e-9))
                    effective_score = load_penalty if bus_score is None else bus_score * load_penalty
                bcol = choose_spatial_target_bus(
                    src_bus_col=src_bcol,
                    strategy_name=strategy_name,
                    dist_m=bus_distance_m,
                    candidate_bus_idx=candidate_bus_idx,
                    navigation_candidate_k=navigation_candidate_k,
                    navigation_distance_limit_m=navigation_distance_limit_m,
                    navigation_distance_beta=navigation_distance_beta,
                    candidate_bus_score=effective_score,
                    rng=rng,
                )
            return ChargingDecision(start_minute=charge_start, bus_col=bcol)

        _, vehicle_p_kw = simulate_soc_and_bus_profile(
            tc,
            soc_params,
            step_minutes=int(step_minutes),
            n_steps=int(n_steps),
            n_buses=int(mapping.n_buses),
            charging_decision_fn=decide_charge,
            rng=rng,
        )
        vehicle_profile_mw = vehicle_p_kw / 1000.0
        profile[:, :] += vehicle_profile_mw
        if dynamic_bus_score and strategy_name == "navigation":
            # Use the aggregated profile seen so far, not the single-vehicle profile.
            # Otherwise the load penalty saturates after the first assignment and
            # dynamic navigation becomes effectively indistinguishable from static scoring.
            peak_per_bus[:] = np.maximum(peak_per_bus, profile.max(axis=0))

    for _ in range(n_vehicles):
        tc = sample_daily_trip_chain(trip_params, rng=rng)
        if tc.stops[-1].departure_minute > day_minutes:
            # defensively clip to horizon if caller passes longer trip params.
            clipped_stops: list[Stop] = []
            clipped_legs: list[float] = []
            for i, st in enumerate(tc.stops):
                arr = int(max(0, min(st.arrival_minute, day_minutes)))
                dep = int(max(arr, min(st.departure_minute, day_minutes)))
                if clipped_stops:
                    arr = max(arr, clipped_stops[-1].departure_minute)
                    dep = max(dep, arr)
                clipped_stops.append(
                    Stop(
                        zone=st.zone,
                        arrival_minute=arr,
                        departure_minute=dep,
                        purpose=st.purpose,
                    )
                )
                if i < len(tc.leg_distance_km):
                    clipped_legs.append(float(tc.leg_distance_km[i]))
                if dep >= day_minutes:
                    break

            if len(clipped_stops) < 2:
                s0 = clipped_stops[0]
                clipped_stops.append(
                    Stop(
                        zone=s0.zone,
                        arrival_minute=s0.departure_minute,
                        departure_minute=day_minutes,
                        purpose=s0.purpose,
                    )
                )
                clipped_legs.append(0.0)

            tc = TripChain(
                stops=clipped_stops,
                leg_distance_km=clipped_legs[: len(clipped_stops) - 1],
            )
        simulate_vehicle(tc)

    return profile
