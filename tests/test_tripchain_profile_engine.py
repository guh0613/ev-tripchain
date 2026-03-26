import numpy as np
from ev_tripchain.config import ProjectConfig

import ev_tripchain.mobility.tripchain_profile as tripchain_profile
from ev_tripchain.mobility.mapping import NodeBusMapping
from ev_tripchain.mobility.soc import SOCEvolutionParams
from ev_tripchain.mobility.trip_chain import Stop, TripChain
from ev_tripchain.mobility.tripchain_sampling import TripChainSamplingParams
from ev_tripchain.mobility.profile import build_ev_profile_mw


def test_tripchain_profile_uses_shared_soc_simulation_kernel(monkeypatch) -> None:
    tc = TripChain(
        stops=[
            Stop(zone=0, arrival_minute=0, departure_minute=60, purpose="home"),
            Stop(zone=1, arrival_minute=90, departure_minute=180, purpose="work"),
            Stop(zone=0, arrival_minute=210, departure_minute=24 * 60, purpose="home"),
        ],
        leg_distance_km=[10.0, 10.0],
    )

    calls: list[tuple[int, int, int]] = []

    def fake_sample_daily_trip_chain(
        params: TripChainSamplingParams,
        *,
        rng: np.random.Generator,
        home_zone: int | None = None,
        work_zone: int | None = None,
    ) -> TripChain:
        del params, rng, home_zone, work_zone
        return tc

    def fake_simulate_soc_and_bus_profile(
        trip_chain: TripChain,
        params: SOCEvolutionParams,
        *,
        step_minutes: int,
        n_steps: int,
        n_buses: int,
        charging_decision_fn,
        rng: np.random.Generator,
        initial_soc: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del trip_chain, params, charging_decision_fn, rng, initial_soc
        calls.append((step_minutes, n_steps, n_buses))
        p_kw = np.zeros((n_steps, n_buses), dtype=float)
        p_kw[0, 0] = 3.6
        p_kw[1, 1] = 7.2
        p_kw[n_steps // 2, 0] = 3.6
        return np.ones(n_steps + 1, dtype=float), p_kw

    monkeypatch.setattr(tripchain_profile, "sample_daily_trip_chain", fake_sample_daily_trip_chain)
    monkeypatch.setattr(
        tripchain_profile,
        "simulate_soc_and_bus_profile",
        fake_simulate_soc_and_bus_profile,
    )

    mapping = NodeBusMapping(matrix=np.eye(2, dtype=float), bus_ids=np.array([0, 1], dtype=int))
    trip_params = TripChainSamplingParams(n_zones=2)
    soc_params = SOCEvolutionParams(charge_trigger_soc=1.0, charge_purposes=("home", "work"))

    prof = tripchain_profile.build_ev_profile_mw_tripchain(
        n_vehicles=2,
        step_minutes=15,
        n_steps=96,
        n_days=2,
        mapping=mapping,
        trip_params=trip_params,
        soc_params=soc_params,
        rng=np.random.default_rng(1),
    )

    assert calls == [(15, 192, 2), (15, 192, 2)]
    assert np.isclose(prof[0, 0], 2 * 3.6 / 1000.0)
    assert np.isclose(prof[1, 1], 2 * 7.2 / 1000.0)
    assert np.isclose(prof[96, 0], 2 * 3.6 / 1000.0)


def test_build_ev_profile_mw_passes_trip_chain_n_zones_to_sampler(monkeypatch) -> None:
    captured: dict[str, int] = {}

    def fake_build_ev_profile_mw_tripchain(**kwargs) -> np.ndarray:
        captured["n_zones"] = int(kwargs["trip_params"].n_zones)
        return np.zeros((int(kwargs["n_steps"]) * int(kwargs["n_days"]), kwargs["mapping"].n_buses))

    monkeypatch.setattr(
        "ev_tripchain.mobility.profile.build_ev_profile_mw_tripchain",
        fake_build_ev_profile_mw_tripchain,
    )

    cfg = ProjectConfig(
        time={"step_minutes": 15, "n_steps": 96, "n_days": 2},
        mobility={
            "model": "tripchain_soc",
            "mapping": {"policy": "random_onehot", "n_nodes": 7},
            "trip_chain": {"n_zones": 7},
        },
    )

    prof = build_ev_profile_mw(
        cfg=cfg,
        n_vehicles=3,
        buses=np.array([0, 1], dtype=int),
        n_buses=2,
        rng=np.random.default_rng(1),
    )

    assert prof.shape == (cfg.time.total_steps, 2)
    assert captured["n_zones"] == 7
