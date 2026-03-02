from __future__ import annotations

import numpy as np

from ev_tripchain.mobility.mapping import NodeBusMapping
from ev_tripchain.mobility.soc import SOCEvolutionParams, sample_initial_soc
from ev_tripchain.mobility.spatial import choose_spatial_target_bus
from ev_tripchain.mobility.trip_chain import Stop, TripChain
from ev_tripchain.mobility.tripchain_sampling import (
    TripChainSamplingParams,
    sample_daily_trip_chain,
)


def _overlap_minutes(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(float(a0), float(b0))
    hi = min(float(a1), float(b1))
    return max(0.0, hi - lo)


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
    bus_distance_m: np.ndarray | None = None,
    candidate_bus_idx: np.ndarray | None = None,
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

    p_mw_max = float(soc_params.charge_power_kw) / 1000.0
    profile = np.zeros((int(n_steps), int(mapping.n_buses)), dtype=float)

    def simulate_vehicle(tc: TripChain) -> None:
        soc = float(sample_initial_soc(soc_params, rng=rng))
        soc = float(np.clip(soc, soc_params.soc_min, soc_params.soc_max))
        charge_purpose_set = set(soc_params.charge_purposes)

        # optional charging at the initial stop (e.g. home overnight before first departure)
        for i_stop, st in enumerate(tc.stops):
            # apply travel consumption at arrival (skip first stop)
            if i_stop > 0:
                dist_km = float(tc.leg_distance_km[i_stop - 1])
                consume_kwh = dist_km * float(soc_params.consumption_kwh_per_km)
                soc -= consume_kwh / float(soc_params.battery_capacity_kwh)
                soc = float(np.clip(soc, soc_params.soc_min, soc_params.soc_max))

            if st.departure_minute <= st.arrival_minute:
                continue

            if st.purpose not in charge_purpose_set:
                continue
            if i_stop == 0 and not soc_params.allow_initial_stop_charging:
                continue
            if soc > float(soc_params.charge_trigger_soc):
                continue
            if soc_params.charge_power_kw <= 0:
                continue

            needed_kwh = float(soc_params.battery_capacity_kwh) * (float(soc_params.soc_max) - soc)
            if needed_kwh <= 0:
                continue

            # charging start may be delayed by "ordered" strategy.
            charge_start = int(st.arrival_minute)
            if strategy_name == "ordered":
                charge_start = _next_window_start(
                    charge_start,
                    window_start=ws,
                    window_end=we,
                    day_minutes=day_minutes,
                )
            if charge_start >= int(st.departure_minute):
                continue

            available_min = int(st.departure_minute - charge_start)
            if available_min <= 0:
                continue

            max_batt_kwh = (
                float(soc_params.charge_power_kw)
                * (available_min / 60.0)
                * float(soc_params.charge_efficiency)
            )
            batt_kwh = min(needed_kwh, max_batt_kwh)
            if batt_kwh <= 0:
                continue

            # compute actual charging duration (may be less than dwell)
            charge_minutes = (
                batt_kwh
                / (float(soc_params.charge_power_kw) * float(soc_params.charge_efficiency))
            ) * 60.0
            charge_minutes = float(min(float(available_min), max(0.0, charge_minutes)))
            if charge_minutes <= 0.0:
                continue

            charge_start_f = float(charge_start)
            charge_end = float(min(charge_start_f + charge_minutes, float(st.departure_minute)))

            zone = int(st.zone)
            if zone < 0 or zone >= mapping.n_nodes:
                continue
            src_bcol = int(zone_to_bus_col[zone])
            bcol = src_bcol
            if strategy_name in {"nearest", "navigation"} and mapping.n_buses > 1:
                bcol = choose_spatial_target_bus(
                    src_bus_col=src_bcol,
                    strategy_name=strategy_name,
                    dist_m=bus_distance_m,
                    candidate_bus_idx=candidate_bus_idx,
                    navigation_candidate_k=navigation_candidate_k,
                    rng=rng,
                )

            k0 = int(np.floor(charge_start_f / float(step_minutes)))
            k1 = int(np.ceil(charge_end / float(step_minutes)))
            k0 = max(0, k0)
            k1 = min(int(n_steps), k1)

            for k in range(k0, k1):
                t0 = float(k * int(step_minutes))
                t1 = float(t0 + int(step_minutes))
                ov = _overlap_minutes(t0, t1, charge_start_f, charge_end)
                if ov:
                    profile[k, bcol] += p_mw_max * (ov / float(step_minutes))

            # update SOC after charging
            soc += batt_kwh / float(soc_params.battery_capacity_kwh)
            soc = float(np.clip(soc, soc_params.soc_min, soc_params.soc_max))

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
