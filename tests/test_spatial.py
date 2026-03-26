import numpy as np

from ev_tripchain.hosting_capacity.sensitivity import VoltageSensitivityModel
from ev_tripchain.mobility.spatial import (
    build_spatial_distance_model,
    choose_spatial_target_bus,
    compute_session_step_weights,
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


def test_nearest_strategy_can_stay_on_current_bus() -> None:
    dist_m = np.array(
        [
            [0.0, 100.0, 250.0],
            [100.0, 0.0, 50.0],
            [250.0, 50.0, 0.0],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(0)

    nearest = choose_spatial_target_bus(
        src_bus_col=1,
        strategy_name="nearest",
        dist_m=dist_m,
        candidate_bus_idx=np.array([0, 1, 2], dtype=int),
        navigation_candidate_k=3,
        rng=rng,
    )
    assert nearest == 1


def test_compute_session_step_weights_keeps_partial_step_overlap() -> None:
    step_idx, step_weight = compute_session_step_weights(
        start_minute=10,
        end_minute=40,
        step_minutes=15,
        n_steps=4,
    )

    assert step_idx.tolist() == [0, 1, 2]
    assert np.allclose(step_weight, np.array([5 / 15, 1.0, 10 / 15], dtype=float))


def test_navigation_uses_temporal_overlap_to_avoid_loaded_candidate() -> None:
    dist_m = np.array(
        [
            [0.0, 100.0, 120.0],
            [100.0, 0.0, 60.0],
            [120.0, 60.0, 0.0],
        ],
        dtype=float,
    )
    scheduled_load_mw = np.zeros((2, 3), dtype=float)
    scheduled_load_mw[:, 1] = 0.05
    rng = np.random.default_rng(7)

    target = choose_spatial_target_bus(
        src_bus_col=0,
        strategy_name="navigation",
        dist_m=dist_m,
        candidate_bus_idx=np.array([1, 2], dtype=int),
        navigation_candidate_k=2,
        candidate_bus_score=np.ones(3, dtype=float),
        session_step_idx=np.array([0, 1], dtype=int),
        session_step_weight=np.array([1.0, 1.0], dtype=float),
        scheduled_load_mw=scheduled_load_mw,
        candidate_charge_power_mw=0.0072,
        rng=rng,
    )

    assert target == 2


def test_navigation_uses_voltage_headroom_to_filter_risky_candidate() -> None:
    dist_m = np.array(
        [
            [0.0, 100.0, 120.0],
            [100.0, 0.0, 60.0],
            [120.0, 60.0, 0.0],
        ],
        dtype=float,
    )
    voltage_model = VoltageSensitivityModel(
        base_voltage_pu=np.array([0.99, 0.953, 0.97], dtype=float),
        sensitivity_pu_per_mw=np.array(
            [
                [-0.01, -0.01, -0.01],
                [-0.01, -0.25, -0.02],
                [-0.01, -0.02, -0.05],
            ],
            dtype=float,
        ),
        vmin_pu=0.95,
        vmax_pu=1.05,
    )
    rng = np.random.default_rng(11)

    target = choose_spatial_target_bus(
        src_bus_col=0,
        strategy_name="navigation",
        dist_m=dist_m,
        candidate_bus_idx=np.array([1, 2], dtype=int),
        navigation_candidate_k=2,
        candidate_bus_score=np.ones(3, dtype=float),
        session_step_idx=np.array([0], dtype=int),
        session_step_weight=np.array([1.0], dtype=float),
        scheduled_load_mw=np.zeros((1, 3), dtype=float),
        candidate_charge_power_mw=0.02,
        voltage_model=voltage_model,
        rng=rng,
    )

    assert target == 2


def test_navigation_soft_voltage_barrier_keeps_near_limit_candidate_competitive() -> None:
    dist_m = np.array(
        [
            [0.0, 250.0, 20.0],
            [250.0, 0.0, 60.0],
            [20.0, 60.0, 0.0],
        ],
        dtype=float,
    )
    voltage_model = VoltageSensitivityModel(
        base_voltage_pu=np.array([0.99, 0.9534, 0.9527], dtype=float),
        sensitivity_pu_per_mw=np.array(
            [
                [-0.01, -0.01, -0.01],
                [-0.01, -0.06, -0.02],
                [-0.01, -0.02, -0.04],
            ],
            dtype=float,
        ),
        vmin_pu=0.95,
        vmax_pu=1.05,
    )
    rng = np.random.default_rng(19)

    target = choose_spatial_target_bus(
        src_bus_col=0,
        strategy_name="navigation",
        dist_m=dist_m,
        candidate_bus_idx=np.array([1, 2], dtype=int),
        navigation_candidate_k=2,
        candidate_bus_score=np.ones(3, dtype=float),
        session_step_idx=np.array([0], dtype=int),
        session_step_weight=np.array([1.0], dtype=float),
        scheduled_load_mw=np.zeros((1, 3), dtype=float),
        candidate_charge_power_mw=0.02,
        voltage_model=voltage_model,
        dynamic_safety_buffer_pu=0.002,
        dynamic_voltage_penalty_window_pu=0.006,
        rng=rng,
    )

    assert target == 2


def test_navigation_uses_upstream_path_congestion_to_avoid_shared_feeder() -> None:
    dist_m = np.array(
        [
            [0.0, 80.0, 80.0],
            [80.0, 0.0, 100.0],
            [80.0, 100.0, 0.0],
        ],
        dtype=float,
    )
    voltage_model = VoltageSensitivityModel(
        base_voltage_pu=np.array([0.99, 0.98, 0.98], dtype=float),
        sensitivity_pu_per_mw=np.array(
            [
                [-0.01, -0.01, -0.01],
                [-0.01, -0.02, -0.01],
                [-0.01, -0.01, -0.02],
            ],
            dtype=float,
        ),
        vmin_pu=0.95,
        vmax_pu=1.05,
        path_incidence=np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        line_capacity_mw=np.array([0.2, 0.2], dtype=float),
        base_line_loading_percent=np.array([85.0, 20.0], dtype=float),
        line_loading_limit_percent=100.0,
    )
    scheduled_load_mw = np.zeros((1, 3), dtype=float)
    scheduled_load_mw[0, 1] = 0.04
    rng = np.random.default_rng(23)

    target = choose_spatial_target_bus(
        src_bus_col=0,
        strategy_name="navigation",
        dist_m=dist_m,
        candidate_bus_idx=np.array([1, 2], dtype=int),
        navigation_candidate_k=2,
        candidate_bus_score=np.ones(3, dtype=float),
        session_step_idx=np.array([0], dtype=int),
        session_step_weight=np.array([1.0], dtype=float),
        scheduled_load_mw=scheduled_load_mw,
        candidate_charge_power_mw=0.02,
        voltage_model=voltage_model,
        path_congestion_weight=1.0,
        rng=rng,
    )

    assert target == 2
