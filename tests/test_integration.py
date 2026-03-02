from pathlib import Path

import numpy as np

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.evaluate import (
    _ensure_ev_load_elements,
    estimate_violation_probability,
)
from ev_tripchain.mobility.profile import build_ev_profile_mw


def test_evaluate_violation_probability_with_simple_case(tmp_path: Path) -> None:
    """Integration test: estimate_violation_probability on the simple 4-bus case."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 99
case:
  name: simple
  load_scale: 0.3
time:
  step_minutes: 15
  n_steps: 96
hosting_capacity:
  scenarios: 5
  risk_tolerance: 0.05
constraints:
  vmin_pu: 0.95
  vmax_pu: 1.05
  line_loading_max_percent: 100.0
  trafo_loading_max_percent: 100.0
ev:
  charge_power_kw: 7.2
  sessions_per_vehicle_mean: 1.0
  duration_minutes_mean: 120
  duration_minutes_std: 40
strategy:
  name: uncontrolled
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)

    rng = np.random.default_rng(cfg.seed)
    prob = estimate_violation_probability(net, cfg, n=0, rng=rng)
    # with 0 EVs and reduced base load, there should be no violations
    assert prob == 0.0


def test_evaluate_high_n_causes_violations(tmp_path: Path) -> None:
    """With many EVs on a small network, violations should occur."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 99
case:
  name: simple
  load_scale: 0.3
time:
  step_minutes: 15
  n_steps: 96
hosting_capacity:
  scenarios: 5
  risk_tolerance: 0.05
constraints:
  vmin_pu: 0.95
  vmax_pu: 1.05
  line_loading_max_percent: 100.0
  trafo_loading_max_percent: 100.0
ev:
  charge_power_kw: 50.0
  sessions_per_vehicle_mean: 1.0
  duration_minutes_mean: 300
  duration_minutes_std: 10
strategy:
  name: uncontrolled
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)

    rng = np.random.default_rng(cfg.seed)
    prob = estimate_violation_probability(net, cfg, n=500, rng=rng)
    # 500 EVs at 50kW each on a small network should trigger violations
    assert prob > 0.0


def test_profile_dispatcher_sessions(tmp_path: Path) -> None:
    """build_ev_profile_mw dispatches correctly for synthetic_sessions model."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 42
case:
  name: simple
  load_scale: 0.5
ev:
  charge_power_kw: 7.2
  sessions_per_vehicle_mean: 1.0
strategy:
  name: uncontrolled
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)

    rng = np.random.default_rng(42)
    profile = build_ev_profile_mw(cfg=cfg, n_vehicles=50, buses=buses, n_buses=n_buses, rng=rng)
    assert profile.shape == (cfg.time.n_steps, n_buses)
    assert profile.sum() > 0


def test_profile_dispatcher_tripchain(tmp_path: Path) -> None:
    """build_ev_profile_mw dispatches correctly for tripchain_soc model."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 42
case:
  name: simple
  load_scale: 0.5
ev:
  charge_power_kw: 7.2
strategy:
  name: uncontrolled
mobility:
  model: tripchain_soc
  mapping:
    policy: random_onehot
    n_nodes: 10
  trip_chain:
    n_zones: 10
    distance_km_mean: 20.0
    distance_km_std: 10.0
  soc:
    charge_trigger_soc: 0.5
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)

    rng = np.random.default_rng(42)
    profile = build_ev_profile_mw(cfg=cfg, n_vehicles=100, buses=buses, n_buses=n_buses, rng=rng)
    assert profile.shape == (cfg.time.n_steps, n_buses)
    assert profile.sum() > 0


def test_load_case_with_scale() -> None:
    """load_case with load_scale < 1 produces reduced loads."""
    net_full = load_case("simple", load_scale=1.0)
    net_half = load_case("simple", load_scale=0.5)
    # half-scale loads should be roughly half of full
    ratio = net_half.load.p_mw.sum() / net_full.load.p_mw.sum()
    assert abs(ratio - 0.5) < 0.01


def test_load_case_ieee33_alias_and_scale() -> None:
    """load_case supports IEEE33 aliases and applies load scaling."""
    net_full = load_case("ieee33", load_scale=1.0)
    net_half = load_case("case33bw", load_scale=0.5)

    assert len(net_full.bus) == 33
    ratio = net_half.load.p_mw.sum() / net_full.load.p_mw.sum()
    assert abs(ratio - 0.5) < 0.01


def test_synthetic_nearest_is_not_identical_to_uncontrolled(tmp_path: Path) -> None:
    """Nearest strategy should produce different spatial profile from uncontrolled."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 42
case:
  name: simple
  load_scale: 0.5
ev:
  charge_power_kw: 7.2
  sessions_per_vehicle_mean: 1.5
  duration_minutes_mean: 120
  duration_minutes_std: 20
strategy:
  name: uncontrolled
""",
        encoding="utf-8",
    )
    cfg_u = load_config(cfg_yaml)
    cfg_n = ProjectConfig.model_validate(
        {**cfg_u.model_dump(), "strategy": {**cfg_u.model_dump()["strategy"], "name": "nearest"}}
    )

    net = load_case(cfg_u.case.name, load_scale=cfg_u.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)

    rng_u = np.random.default_rng(123)
    rng_n = np.random.default_rng(123)
    p_u = build_ev_profile_mw(cfg=cfg_u, n_vehicles=200, buses=buses, n_buses=n_buses, rng=rng_u)
    p_n = build_ev_profile_mw(cfg=cfg_n, n_vehicles=200, buses=buses, n_buses=n_buses, rng=rng_n)

    assert not np.allclose(p_u, p_n)


def test_tripchain_ordered_strategy_changes_profile(tmp_path: Path) -> None:
    """Trip-chain model should apply strategy and produce different profile."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 42
case:
  name: simple
  load_scale: 0.5
time:
  step_minutes: 15
  n_steps: 96
ev:
  charge_power_kw: 7.2
strategy:
  name: uncontrolled
mobility:
  model: tripchain_soc
  mapping:
    policy: random_onehot
    n_nodes: 10
  trip_chain:
    n_zones: 10
    distance_km_mean: 20.0
    distance_km_std: 10.0
  soc:
    charge_trigger_soc: 1.0
    charge_purposes: ["home", "work", "other"]
""",
        encoding="utf-8",
    )
    cfg_u = load_config(cfg_yaml)
    cfg_o = ProjectConfig.model_validate(
        {
            **cfg_u.model_dump(),
            "strategy": {
                **cfg_u.model_dump()["strategy"],
                "name": "ordered",
                "ordered": {"window_start": "22:00", "window_end": "06:00"},
            },
        }
    )

    net = load_case(cfg_u.case.name, load_scale=cfg_u.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)

    rng_u = np.random.default_rng(123)
    rng_o = np.random.default_rng(123)
    p_u = build_ev_profile_mw(cfg=cfg_u, n_vehicles=120, buses=buses, n_buses=n_buses, rng=rng_u)
    p_o = build_ev_profile_mw(cfg=cfg_o, n_vehicles=120, buses=buses, n_buses=n_buses, rng=rng_o)

    assert p_u.sum() > 0.0
    assert p_o.sum() > 0.0
    assert not np.allclose(p_u, p_o)


def test_tripchain_profile_supports_non_24h_horizon(tmp_path: Path) -> None:
    """Trip-chain profile should run with configurable shorter simulation horizon."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 42
case:
  name: simple
  load_scale: 0.5
time:
  step_minutes: 15
  n_steps: 24
ev:
  charge_power_kw: 7.2
strategy:
  name: uncontrolled
mobility:
  model: tripchain_soc
  mapping:
    policy: random_onehot
    n_nodes: 10
  trip_chain:
    n_zones: 10
  soc:
    charge_trigger_soc: 0.5
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)

    rng = np.random.default_rng(123)
    p = build_ev_profile_mw(cfg=cfg, n_vehicles=40, buses=buses, n_buses=n_buses, rng=rng)
    assert p.shape == (cfg.time.n_steps, n_buses)
    assert np.isfinite(p).all()
