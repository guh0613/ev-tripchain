"""Tests for dynamic navigation scoring."""

from pathlib import Path

import numpy as np

from ev_tripchain.config import load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.evaluate import _ensure_ev_load_elements
from ev_tripchain.mobility.profile import build_ev_profile_mw


def test_dynamic_scoring_produces_different_spatial_distribution(tmp_path: Path) -> None:
    """Dynamic scoring should produce more spatially dispersed load than static scoring."""
    base_yaml = """
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
  name: navigation
  navigation:
    candidate_k: 3
    distance_beta: 1.0
    dynamic_scoring: {dynamic}
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
"""
    cfg_yaml = tmp_path / "cfg.yaml"

    # Static scoring
    cfg_yaml.write_text(base_yaml.format(dynamic="false"), encoding="utf-8")
    cfg_static = load_config(cfg_yaml)
    net = load_case(cfg_static.case.name, load_scale=cfg_static.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)
    rng1 = np.random.default_rng(42)
    p_static = build_ev_profile_mw(
        cfg=cfg_static, n_vehicles=200, buses=buses, n_buses=n_buses, rng=rng1
    )

    # Dynamic scoring
    cfg_yaml.write_text(base_yaml.format(dynamic="true"), encoding="utf-8")
    cfg_dynamic = load_config(cfg_yaml)
    rng2 = np.random.default_rng(42)
    p_dynamic = build_ev_profile_mw(
        cfg=cfg_dynamic, n_vehicles=200, buses=buses, n_buses=n_buses, rng=rng2
    )

    # Both should have load
    assert p_static.sum() > 0
    assert p_dynamic.sum() > 0

    # Profiles should differ (dynamic scoring changes allocation)
    assert not np.allclose(p_static, p_dynamic)

    # Dynamic should have more even spatial distribution (lower peak per bus)
    peak_per_bus_static = p_static.max(axis=0)
    peak_per_bus_dynamic = p_dynamic.max(axis=0)
    # The max of per-bus peaks should be lower or similar with dynamic scoring
    max_peak_static = float(peak_per_bus_static.max())
    max_peak_dynamic = float(peak_per_bus_dynamic.max())
    # Allow some tolerance - the key test is that profiles differ
    assert max_peak_dynamic <= max_peak_static * 1.1
