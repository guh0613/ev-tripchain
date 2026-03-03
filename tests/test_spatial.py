import numpy as np

from ev_tripchain.mobility.spatial import (
    build_spatial_distance_model,
    choose_spatial_target_bus,
)


def test_ieee33_table_a1_is_loaded_when_station_nodes_exist() -> None:
    buses = np.arange(1, 34, dtype=int)
    model = build_spatial_distance_model(buses=buses, n_buses=buses.size)

    assert model.candidate_bus_idx.tolist() == [0, 5, 8, 12, 15, 19, 29]
    assert np.allclose(
        model.dist_m[0, model.candidate_bus_idx],
        np.array([0, 900, 700, 1300, 1700, 900, 1500], dtype=float),
    )
    assert np.allclose(
        model.dist_m[32, model.candidate_bus_idx],
        np.array([200, 1700, 1000, 1200, 800, 900, 1600], dtype=float),
    )


def test_ieee33_table_a1_supports_pandapower_bus_ids_without_slack() -> None:
    # Common pipeline layout: ev-load elements are created on all buses except ext_grid,
    # which yields pandapower-style bus indices [1..32] (slack bus 0 is excluded).
    buses = np.arange(1, 33, dtype=int)
    model = build_spatial_distance_model(buses=buses, n_buses=buses.size)

    # Station at node 1 is excluded because the slack bus is not part of `buses`.
    assert model.candidate_bus_idx.tolist() == [4, 7, 11, 14, 18, 28]
    assert np.allclose(
        model.dist_m[0, model.candidate_bus_idx],
        np.array([700, 900, 1500, 1600, 700, 1400], dtype=float),
    )
    assert np.allclose(
        model.dist_m[31, model.candidate_bus_idx],
        np.array([1700, 1000, 1200, 800, 900, 1600], dtype=float),
    )


def test_fallback_distance_model_is_deterministic() -> None:
    buses = np.array([10, 22, 35], dtype=int)
    model = build_spatial_distance_model(buses=buses, n_buses=buses.size)

    assert model.candidate_bus_idx.tolist() == [0, 1, 2]
    assert np.allclose(
        model.dist_m,
        np.array(
            [
                [0.0, 1000.0, 2000.0],
                [1000.0, 0.0, 1000.0],
                [2000.0, 1000.0, 0.0],
            ],
            dtype=float,
        ),
    )


def test_choose_spatial_target_bus_ignores_inf_distances() -> None:
    dist_m = np.array(
        [
            [0.0, np.inf, 300.0, np.inf],
            [np.inf, 0.0, np.inf, 100.0],
            [300.0, np.inf, 0.0, np.inf],
            [np.inf, 100.0, np.inf, 0.0],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(123)

    nearest = choose_spatial_target_bus(
        src_bus_col=0,
        strategy_name="nearest",
        dist_m=dist_m,
        candidate_bus_idx=np.array([1, 3], dtype=int),
        navigation_candidate_k=5,
        rng=rng,
    )
    assert nearest == 0  # no finite candidate among [1, 3]

    navigation = choose_spatial_target_bus(
        src_bus_col=1,
        strategy_name="navigation",
        dist_m=dist_m,
        candidate_bus_idx=np.array([0, 2, 3], dtype=int),
        navigation_candidate_k=2,
        rng=rng,
    )
    assert navigation == 3
