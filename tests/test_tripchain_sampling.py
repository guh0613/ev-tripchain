import numpy as np

from ev_tripchain.mobility.tripchain_sampling import (
    TripChainSamplingParams,
    sample_daily_trip_chain,
)


def test_sample_daily_trip_chain_is_time_ordered() -> None:
    rng = np.random.default_rng(0)
    params = TripChainSamplingParams(n_zones=10, other_stops_mean=2.0)
    tc = sample_daily_trip_chain(params, rng=rng)

    assert len(tc.stops) >= 2
    assert len(tc.leg_distance_km) == len(tc.stops) - 1
    assert tc.stops[0].arrival_minute == 0
    assert tc.stops[-1].departure_minute == params.day_minutes

    for s in tc.stops:
        assert 0 <= s.arrival_minute <= s.departure_minute <= params.day_minutes

    for i in range(len(tc.stops) - 1):
        assert tc.stops[i].departure_minute <= tc.stops[i + 1].arrival_minute
        assert tc.leg_distance_km[i] >= 0.0
