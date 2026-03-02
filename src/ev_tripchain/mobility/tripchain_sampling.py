from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ev_tripchain.mobility.trip_chain import Stop, TripChain


def _parse_hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.strip().split(":")
    return int(hh) * 60 + int(mm)


def _trunc_normal_int(
    rng: np.random.Generator, *, mean: float, std: float, low: int, high: int
) -> int:
    if std <= 0:
        return int(np.clip(round(mean), low, high))
    x = float(rng.normal(mean, std))
    return int(np.clip(round(x), low, high))


def _lognormal_pos(rng: np.random.Generator, *, mean: float, std: float) -> float:
    """
    Sample a positive value with roughly (mean, std) on the original scale.
    """
    m = max(float(mean), 1e-6)
    s = max(float(std), 1e-6)
    # moment matching: for lognormal, mu/sigma on log scale
    phi = np.sqrt(s * s + m * m)
    mu = np.log(m * m / phi)
    sigma = np.sqrt(np.log((phi * phi) / (m * m)))
    return float(rng.lognormal(mean=mu, sigma=sigma))


@dataclass(frozen=True)
class TripChainSamplingParams:
    n_zones: int = 50
    other_stops_mean: float = 1.2  # expected number of "other" stops after work

    day_minutes: int = 24 * 60

    first_departure_mean: str = "07:30"
    first_departure_std_minutes: int = 35

    work_duration_mean_minutes: int = 8 * 60
    work_duration_std_minutes: int = 45

    other_dwell_mean_minutes: int = 60
    other_dwell_std_minutes: int = 30

    # travel time per km, used to convert distance -> time (coarse)
    travel_minutes_per_km: float = 2.2

    # trip distance distribution
    distance_km_mean: float = 8.0
    distance_km_std: float = 4.0


def sample_daily_trip_chain(
    params: TripChainSamplingParams,
    *,
    rng: np.random.Generator,
) -> TripChain:
    """
    Sample a simple daily trip chain: home -> work -> other* -> home.

    Returns a time-ordered TripChain within [0, day_minutes].
    """
    if params.n_zones < 2:
        raise ValueError("n_zones must be >= 2.")

    t_end = int(params.day_minutes)
    home_zone = int(rng.integers(0, params.n_zones))
    work_zone = int(rng.integers(0, params.n_zones - 1))
    if work_zone >= home_zone:
        work_zone += 1

    # home stop (start of day)
    dep_home = _trunc_normal_int(
        rng,
        mean=_parse_hhmm_to_minutes(params.first_departure_mean),
        std=float(params.first_departure_std_minutes),
        low=0,
        high=t_end - 1,
    )

    stops: list[Stop] = [
        Stop(zone=home_zone, arrival_minute=0, departure_minute=dep_home, purpose="home")
    ]
    leg_distance_km: list[float] = []

    def add_leg_to(
        *,
        dest_zone: int,
        dest_purpose: str,
        dwell_mean: int,
        dwell_std: int,
    ) -> None:
        nonlocal stops, leg_distance_km
        prev = stops[-1]
        dist_km = _lognormal_pos(rng, mean=params.distance_km_mean, std=params.distance_km_std)
        travel_min = max(1, int(round(dist_km * float(params.travel_minutes_per_km))))
        arr = min(prev.departure_minute + travel_min, t_end)

        # dwell at destination
        dwell = _trunc_normal_int(rng, mean=dwell_mean, std=float(dwell_std), low=0, high=t_end)
        dep = min(arr + dwell, t_end)

        stops.append(
            Stop(
                zone=int(dest_zone),
                arrival_minute=int(arr),
                departure_minute=int(dep),
                purpose=dest_purpose,
            )
        )
        leg_distance_km.append(float(dist_km))

    # to work
    add_leg_to(
        dest_zone=work_zone,
        dest_purpose="work",
        dwell_mean=int(params.work_duration_mean_minutes),
        dwell_std=int(params.work_duration_std_minutes),
    )

    # after-work "other" stops
    n_other = int(rng.poisson(max(params.other_stops_mean, 0.0)))
    for _ in range(n_other):
        if stops[-1].departure_minute >= t_end:
            break
        other_zone = int(rng.integers(0, params.n_zones))
        add_leg_to(
            dest_zone=other_zone,
            dest_purpose="other",
            dwell_mean=int(params.other_dwell_mean_minutes),
            dwell_std=int(params.other_dwell_std_minutes),
        )

    # final back home
    if stops[-1].zone != home_zone and stops[-1].departure_minute < t_end:
        add_leg_to(dest_zone=home_zone, dest_purpose="home", dwell_mean=0, dwell_std=0)

    # force last stop to extend to end-of-day for convenience
    last = stops[-1]
    if last.departure_minute < t_end:
        stops[-1] = Stop(
            zone=last.zone,
            arrival_minute=last.arrival_minute,
            departure_minute=t_end,
            purpose=last.purpose,
        )

    # ensure at least 2 stops
    if len(stops) < 2:
        stops.append(
            Stop(
                zone=home_zone,
                arrival_minute=dep_home,
                departure_minute=t_end,
                purpose="home",
            )
        )
        leg_distance_km.append(0.0)

    return TripChain(stops=stops, leg_distance_km=leg_distance_km)


def sample_trip_chains(
    params: TripChainSamplingParams,
    *,
    n_vehicles: int,
    rng: np.random.Generator,
) -> list[TripChain]:
    return [sample_daily_trip_chain(params, rng=rng) for _ in range(int(max(n_vehicles, 0)))]
