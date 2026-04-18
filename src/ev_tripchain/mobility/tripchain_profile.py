from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ev_tripchain.mobility.mapping import NodeBusMapping
from ev_tripchain.mobility.soc import (
    ChargingDecision,
    SOCEvolutionParams,
    simulate_soc_and_bus_profile,
)
from ev_tripchain.mobility.spatial import (
    choose_spatial_target_bus,
    compute_session_step_weights,
)
from ev_tripchain.mobility.trip_chain import Stop, TripChain
from ev_tripchain.mobility.tripchain_sampling import (
    TripChainSamplingParams,
    sample_anchor_zones,
    sample_daily_trip_chain,
)

if TYPE_CHECKING:
    from ev_tripchain.hosting_capacity.sensitivity import VoltageSensitivityModel


def _minute_of_day(minute: int, *, day_minutes: int) -> int:
    t_day = int(day_minutes)
    if t_day <= 0:
        raise ValueError("day_minutes must be positive.")
    return int(minute) % t_day


def _is_in_window(
    minute: int,
    *,
    window_start: int,
    window_end: int,
    day_minutes: int,
) -> bool:
    m = _minute_of_day(minute, day_minutes=day_minutes)
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
    day_idx = m // t_day
    m_local = _minute_of_day(m, day_minutes=t_day)
    if _is_in_window(m, window_start=ws, window_end=we, day_minutes=t_day):
        return m

    if ws == we:
        return m

    # Window does not cross midnight: [ws, we)
    if ws < we:
        if m_local < ws:
            return day_idx * t_day + ws
        # next day's window start
        return (day_idx + 1) * t_day + ws

    # Window crosses midnight, e.g. 22:00-06:00
    # Outside-window region is [we, ws).
    if we <= m_local < ws:
        return day_idx * t_day + ws
    return m


def _offset_trip_chain(trip_chain: TripChain, *, minute_offset: int) -> TripChain:
    offset = int(minute_offset)
    if offset == 0:
        return trip_chain
    return TripChain(
        stops=[
            Stop(
                zone=st.zone,
                arrival_minute=int(st.arrival_minute) + offset,
                departure_minute=int(st.departure_minute) + offset,
                purpose=st.purpose,
            )
            for st in trip_chain.stops
        ],
        leg_distance_km=[float(d) for d in trip_chain.leg_distance_km],
    )


def _clip_trip_chain_to_horizon(trip_chain: TripChain, *, day_minutes: int) -> TripChain:
    if trip_chain.stops[-1].departure_minute <= int(day_minutes):
        return trip_chain

    clipped_stops: list[Stop] = []
    clipped_legs: list[float] = []
    for i, st in enumerate(trip_chain.stops):
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
        if i < len(trip_chain.leg_distance_km):
            clipped_legs.append(float(trip_chain.leg_distance_km[i]))
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

    return TripChain(
        stops=clipped_stops,
        leg_distance_km=clipped_legs[: len(clipped_stops) - 1],
    )


def _append_daily_trip_chain(
    base: TripChain | None,
    extra: TripChain,
) -> TripChain:
    if base is None:
        return extra

    stops = list(base.stops)
    legs = [float(d) for d in base.leg_distance_km]
    last_stop = stops[-1]
    first_stop = extra.stops[0]

    if last_stop.departure_minute > first_stop.arrival_minute:
        raise ValueError("Daily trip chains overlap in time.")

    if last_stop.zone == first_stop.zone and last_stop.purpose == first_stop.purpose:
        # Merge the day boundary into one continuous stop so charging can span midnight.
        stops[-1] = replace(
            last_stop,
            departure_minute=max(last_stop.departure_minute, first_stop.departure_minute),
        )
        stops.extend(extra.stops[1:])
        legs.extend(float(d) for d in extra.leg_distance_km)
        return TripChain(stops=stops, leg_distance_km=legs)

    # When independently sampled days land in different end/start states, keep the
    # continuous time axis by inserting a zero-distance boundary leg instead of
    # discarding the cross-day carry-over state outright.
    stops.append(first_stop)
    legs.append(0.0)
    stops.extend(extra.stops[1:])
    legs.extend(float(d) for d in extra.leg_distance_km)
    return TripChain(stops=stops, leg_distance_km=legs)


def _sample_continuous_trip_chain(
    *,
    n_days: int,
    trip_params: TripChainSamplingParams,
    rng: np.random.Generator,
    home_zone: int,
    work_zone: int,
) -> TripChain:
    continuous_trip_chain: TripChain | None = None
    day_minutes = int(trip_params.day_minutes)

    for day_idx in range(int(n_days)):
        daily_trip_chain = sample_daily_trip_chain(
            trip_params,
            rng=rng,
            home_zone=home_zone,
            work_zone=work_zone,
        )
        daily_trip_chain = _clip_trip_chain_to_horizon(
            daily_trip_chain,
            day_minutes=day_minutes,
        )
        daily_trip_chain = _offset_trip_chain(
            daily_trip_chain,
            minute_offset=day_idx * day_minutes,
        )
        continuous_trip_chain = _append_daily_trip_chain(
            continuous_trip_chain,
            daily_trip_chain,
        )

    if continuous_trip_chain is None:
        raise ValueError("n_days must be positive.")
    return continuous_trip_chain


def build_ev_profile_mw_tripchain(
    *,
    n_vehicles: int,
    step_minutes: int,
    n_steps: int,
    n_days: int,
    mapping: NodeBusMapping,
    trip_params: TripChainSamplingParams,
    soc_params: SOCEvolutionParams,
    strategy_name: str = "uncontrolled",
    ordered_window: tuple[int, int] | None = None,
    navigation_candidate_k: int = 5,
    navigation_distance_limit_m: float | None = None,
    navigation_distance_beta: float = 1.0,
    navigation_dynamic_safety_buffer_pu: float = 0.002,
    navigation_dynamic_voltage_penalty_window_pu: float = 0.006,
    navigation_path_congestion_weight: float = 0.35,
    navigation_softmax_temperature: float = 0.35,
    navigation_disable_voltage_factor: bool = False,
    bus_distance_m: np.ndarray | None = None,
    candidate_bus_idx: np.ndarray | None = None,
    bus_score: np.ndarray | None = None,
    navigation_voltage_model: VoltageSensitivityModel | None = None,
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
    total_steps = int(n_steps) * int(n_days)
    if n_vehicles == 0:
        return np.zeros((total_steps, int(mapping.n_buses)), dtype=float)

    if step_minutes <= 0 or n_steps <= 0 or n_days <= 0:
        raise ValueError("step_minutes, n_steps and n_days must be positive.")

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

    profile = np.zeros((total_steps, int(mapping.n_buses)), dtype=float)
    p_mw_max = float(soc_params.charge_power_kw) / 1000.0

    def simulate_vehicle(*, home_zone: int, work_zone: int) -> None:
        # In continuous multi-day mode, the scenario starts from an already-parked state,
        # so the day-0 initial stop should also be eligible for overnight charging.
        allow_initial_stop_charging = bool(n_days > 1)
        reserved_profile = np.zeros_like(profile)

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
            charge_end = min(int(departure_minute), charge_start + int(minutes_needed))
            if charge_end <= charge_start:
                return None

            src_bcol = int(zone_to_bus_col[zone])
            bcol = src_bcol
            if strategy_name in {"nearest", "navigation"} and mapping.n_buses > 1:
                session_step_idx, session_step_weight = compute_session_step_weights(
                    start_minute=charge_start,
                    end_minute=charge_end,
                    step_minutes=int(step_minutes),
                    n_steps=total_steps,
                )
                dynamic_context_enabled = (
                    strategy_name == "navigation"
                    and dynamic_bus_score
                    and session_step_idx.size > 0
                )
                bcol = choose_spatial_target_bus(
                    src_bus_col=src_bcol,
                    strategy_name=strategy_name,
                    dist_m=bus_distance_m,
                    candidate_bus_idx=candidate_bus_idx,
                    navigation_candidate_k=navigation_candidate_k,
                    navigation_distance_limit_m=navigation_distance_limit_m,
                    navigation_distance_beta=navigation_distance_beta,
                    candidate_bus_score=bus_score,
                    session_step_idx=session_step_idx if dynamic_context_enabled else None,
                    session_step_weight=session_step_weight if dynamic_context_enabled else None,
                    scheduled_load_mw=(profile + reserved_profile) if dynamic_context_enabled else None,
                    candidate_charge_power_mw=p_mw_max if dynamic_context_enabled else None,
                    voltage_model=navigation_voltage_model if dynamic_context_enabled else None,
                    dynamic_safety_buffer_pu=navigation_dynamic_safety_buffer_pu,
                    dynamic_voltage_penalty_window_pu=(
                        navigation_dynamic_voltage_penalty_window_pu
                    ),
                    path_congestion_weight=navigation_path_congestion_weight,
                    softmax_temperature=navigation_softmax_temperature,
                    disable_voltage_factor=navigation_disable_voltage_factor,
                    rng=rng,
                )
                if dynamic_context_enabled:
                    reserved_profile[session_step_idx, bcol] += session_step_weight * p_mw_max
            return ChargingDecision(start_minute=charge_start, bus_col=bcol)

        trip_chain = _sample_continuous_trip_chain(
            n_days=int(n_days),
            trip_params=trip_params,
            rng=rng,
            home_zone=home_zone,
            work_zone=work_zone,
        )
        sim_soc_params = replace(
            soc_params,
            allow_initial_stop_charging=allow_initial_stop_charging,
        )
        _, vehicle_p_kw = simulate_soc_and_bus_profile(
            trip_chain,
            sim_soc_params,
            step_minutes=int(step_minutes),
            n_steps=total_steps,
            n_buses=int(mapping.n_buses),
            charging_decision_fn=decide_charge,
            rng=rng,
        )
        profile[:, :] += vehicle_p_kw / 1000.0

    for _ in range(n_vehicles):
        home_zone, work_zone = sample_anchor_zones(trip_params, rng=rng)
        simulate_vehicle(home_zone=home_zone, work_zone=work_zone)

    return profile
