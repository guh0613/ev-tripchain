import numpy as np

import ev_tripchain.mobility.tripchain_profile as tripchain_profile
from ev_tripchain.mobility.mapping import NodeBusMapping
from ev_tripchain.mobility.soc import SOCEvolutionParams
from ev_tripchain.mobility.trip_chain import Stop, TripChain
from ev_tripchain.mobility.tripchain_sampling import TripChainSamplingParams


def test_tripchain_multiday_profile_carries_charging_across_midnight(monkeypatch) -> None:
    tc = TripChain(
        stops=[
            Stop(zone=0, arrival_minute=0, departure_minute=8 * 60, purpose="home"),
            Stop(zone=1, arrival_minute=9 * 60, departure_minute=17 * 60, purpose="work"),
            Stop(zone=0, arrival_minute=23 * 60, departure_minute=24 * 60, purpose="home"),
        ],
        leg_distance_km=[60.0, 60.0],
    )

    def fake_sample_anchor_zones(
        params: TripChainSamplingParams, *, rng: np.random.Generator
    ) -> tuple[int, int]:
        del params, rng
        return 0, 1

    def fake_sample_daily_trip_chain(
        params: TripChainSamplingParams,
        *,
        rng: np.random.Generator,
        home_zone: int | None = None,
        work_zone: int | None = None,
    ) -> TripChain:
        del params, rng, home_zone, work_zone
        return tc

    monkeypatch.setattr(tripchain_profile, "sample_anchor_zones", fake_sample_anchor_zones)
    monkeypatch.setattr(tripchain_profile, "sample_daily_trip_chain", fake_sample_daily_trip_chain)

    mapping = NodeBusMapping(matrix=np.eye(2, dtype=float), bus_ids=np.array([0, 1], dtype=int))
    trip_params = TripChainSamplingParams(n_zones=2)
    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=50.0,
        consumption_kwh_per_km=0.2,
        initial_soc_mean=0.55,
        initial_soc_std=0.0,
        charge_power_kw=10.0,
        charge_efficiency=1.0,
        charge_trigger_soc=0.3,
        charge_purposes=("home",),
    )

    prof = tripchain_profile.build_ev_profile_mw_tripchain(
        n_vehicles=1,
        step_minutes=15,
        n_steps=96,
        n_days=2,
        mapping=mapping,
        trip_params=trip_params,
        soc_params=soc_params,
        strategy_name="ordered",
        ordered_window=(22 * 60, 6 * 60),
        rng=np.random.default_rng(7),
    )

    day_steps = 96
    late_day1 = prof[23 * 4 : day_steps, 0]
    early_day2 = prof[day_steps : day_steps + 6 * 4, 0]

    assert prof.shape == (2 * day_steps, 2)
    assert np.isclose(prof[: 8 * 4, 0].sum(), 0.0)
    assert late_day1.sum() > 0.0
    assert early_day2.sum() > 0.0


def test_tripchain_multiday_profile_does_not_cut_an_active_session_at_midnight(
    monkeypatch,
) -> None:
    tc = TripChain(
        stops=[
            Stop(zone=0, arrival_minute=0, departure_minute=8 * 60, purpose="home"),
            Stop(zone=1, arrival_minute=9 * 60, departure_minute=17 * 60, purpose="work"),
            Stop(zone=0, arrival_minute=23 * 60 + 30, departure_minute=24 * 60, purpose="home"),
        ],
        leg_distance_km=[60.0, 60.0],
    )

    def fake_sample_anchor_zones(
        params: TripChainSamplingParams, *, rng: np.random.Generator
    ) -> tuple[int, int]:
        del params, rng
        return 0, 1

    def fake_sample_daily_trip_chain(
        params: TripChainSamplingParams,
        *,
        rng: np.random.Generator,
        home_zone: int | None = None,
        work_zone: int | None = None,
    ) -> TripChain:
        del params, rng, home_zone, work_zone
        return tc

    monkeypatch.setattr(tripchain_profile, "sample_anchor_zones", fake_sample_anchor_zones)
    monkeypatch.setattr(tripchain_profile, "sample_daily_trip_chain", fake_sample_daily_trip_chain)

    mapping = NodeBusMapping(matrix=np.eye(2, dtype=float), bus_ids=np.array([0, 1], dtype=int))
    trip_params = TripChainSamplingParams(n_zones=2)
    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=50.0,
        consumption_kwh_per_km=0.2,
        initial_soc_mean=0.75,
        initial_soc_std=0.0,
        charge_power_kw=10.0,
        charge_efficiency=1.0,
        charge_trigger_soc=0.3,
        charge_purposes=("home",),
    )

    prof = tripchain_profile.build_ev_profile_mw_tripchain(
        n_vehicles=1,
        step_minutes=15,
        n_steps=96,
        n_days=2,
        mapping=mapping,
        trip_params=trip_params,
        soc_params=soc_params,
        rng=np.random.default_rng(11),
    )

    day_steps = 96
    assert prof.shape == (2 * day_steps, 2)
    assert prof[95, 0] > 0.0
    assert prof[96, 0] > 0.0
    assert prof[day_steps : day_steps + 12, 0].sum() > 0.0


def test_next_window_start_wraps_by_day() -> None:
    day_minutes = 24 * 60
    minute = 2 * day_minutes + 9 * 60
    assert tripchain_profile._next_window_start(
        minute,
        window_start=22 * 60,
        window_end=6 * 60,
        day_minutes=day_minutes,
    ) == 2 * day_minutes + 22 * 60

    in_window = 2 * day_minutes + 2 * 60
    assert tripchain_profile._next_window_start(
        in_window,
        window_start=22 * 60,
        window_end=6 * 60,
        day_minutes=day_minutes,
    ) == in_window
