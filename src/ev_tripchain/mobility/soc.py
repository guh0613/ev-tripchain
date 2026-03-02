from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ev_tripchain.mobility.trip_chain import TripChain


def _overlap_minutes(a0: int, a1: int, b0: int, b1: int) -> int:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    return max(0, hi - lo)


def _trunc01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


@dataclass(frozen=True)
class SOCEvolutionParams:
    battery_capacity_kwh: float = 60.0
    consumption_kwh_per_km: float = 0.18

    initial_soc_mean: float = 0.7
    initial_soc_std: float = 0.15

    soc_min: float = 0.0
    soc_max: float = 1.0

    charge_power_kw: float = 7.2
    charge_efficiency: float = 0.92
    charge_trigger_soc: float = 0.3
    charge_purposes: tuple[str, ...] = ("home", "work")
    allow_initial_stop_charging: bool = False


def sample_initial_soc(params: SOCEvolutionParams, *, rng: np.random.Generator) -> float:
    if params.initial_soc_std <= 0:
        return _trunc01(params.initial_soc_mean)
    x = float(rng.normal(params.initial_soc_mean, params.initial_soc_std))
    return _trunc01(x)


def simulate_soc_and_charging_profile(
    trip_chain: TripChain,
    params: SOCEvolutionParams,
    *,
    step_minutes: int,
    n_steps: int,
    rng: np.random.Generator,
    initial_soc: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate SOC evolution and charging power (grid-side, kW) on a fixed time grid.

    Returns:
    - soc: shape (n_steps + 1,), SOC at step boundaries.
    - p_kw: shape (n_steps,), average charging power within each step.
    """
    if step_minutes <= 0 or n_steps <= 0:
        raise ValueError("step_minutes and n_steps must be positive.")
    if params.battery_capacity_kwh <= 0:
        raise ValueError("battery_capacity_kwh must be positive.")
    if params.charge_efficiency <= 0 or params.charge_efficiency > 1.0:
        raise ValueError("charge_efficiency must be in (0, 1].")

    day_minutes = step_minutes * n_steps
    if trip_chain.stops[-1].departure_minute > day_minutes:
        raise ValueError("TripChain exceeds simulation horizon.")

    soc0 = _trunc01(initial_soc) if initial_soc is not None else sample_initial_soc(params, rng=rng)
    soc0 = float(np.clip(soc0, params.soc_min, params.soc_max))

    # Minute-level simulation to ensure a charging session, once triggered at arrival,
    # continues until full (or until the dwell ends).
    soc = np.empty(n_steps + 1, dtype=float)
    p_kw = np.zeros(n_steps, dtype=float)
    soc[0] = soc0

    cap = float(params.battery_capacity_kwh)
    energy_kwh = cap * float(soc0)

    # Per-minute travel consumption (battery-side, kWh per minute)
    d_energy_travel = np.zeros(int(day_minutes), dtype=float)
    for i in range(trip_chain.n_legs):
        dep = int(trip_chain.stops[i].departure_minute)
        arr = int(trip_chain.stops[i + 1].arrival_minute)
        dep = max(0, min(dep, day_minutes))
        arr = max(0, min(arr, day_minutes))
        if arr <= dep:
            continue
        e_kwh = float(trip_chain.leg_distance_km[i]) * float(params.consumption_kwh_per_km)
        d_energy_travel[dep:arr] -= e_kwh / float(arr - dep)

    arrivals: dict[int, int] = {int(st.arrival_minute): i for i, st in enumerate(trip_chain.stops)}
    charge_purpose_set = set(params.charge_purposes)

    charging_end_minute = -1
    batt_charge_kwh_per_min = (
        float(params.charge_power_kw) * float(params.charge_efficiency)
    ) / 60.0
    grid_charge_kw = float(params.charge_power_kw)

    for m in range(int(day_minutes)):
        # travel
        energy_kwh += float(d_energy_travel[m])
        energy_kwh = float(np.clip(energy_kwh, cap * params.soc_min, cap * params.soc_max))

        # start a charging session at stop arrivals (if not already charging)
        if m in arrivals and charging_end_minute <= m:
            i_stop = int(arrivals[m])
            st = trip_chain.stops[i_stop]
            if params.allow_initial_stop_charging or i_stop != 0:
                if (
                    st.purpose in charge_purpose_set
                    and grid_charge_kw > 0
                    and batt_charge_kwh_per_min > 0
                ):
                    soc_at_arrival = energy_kwh / cap
                    if soc_at_arrival <= float(params.charge_trigger_soc):
                        needed_kwh = cap * (float(params.soc_max) - soc_at_arrival)
                        if needed_kwh > 0:
                            minutes_needed = int(np.ceil(needed_kwh / batt_charge_kwh_per_min))
                            charging_end_minute = min(int(st.departure_minute), m + minutes_needed)

        # charging
        if charging_end_minute > m:
            energy_kwh += batt_charge_kwh_per_min
            energy_kwh = float(np.clip(energy_kwh, cap * params.soc_min, cap * params.soc_max))
            k = m // int(step_minutes)
            if 0 <= k < int(n_steps):
                p_kw[k] += grid_charge_kw / float(step_minutes)

        # record SOC at step boundary
        if (m + 1) % int(step_minutes) == 0:
            k_next = (m + 1) // int(step_minutes)
            soc[k_next] = float(np.clip(energy_kwh / cap, params.soc_min, params.soc_max))

    return soc, p_kw
