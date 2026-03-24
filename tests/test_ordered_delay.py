"""Tests for the improved ordered charging strategy with random delay."""

from pathlib import Path

import numpy as np

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.evaluate import _ensure_ev_load_elements
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.mobility.synthetic import _apply_ordered_window


def test_ordered_window_random_delay_spreads_starts() -> None:
    """With random_delay=True, starts should be spread across the window, not clustered."""
    rng = np.random.default_rng(42)
    # All vehicles start outside the window (e.g. 18:00 = 1080 min)
    starts = np.full(500, 1080, dtype=int)

    # Without random delay: all should be at window_start (22:00 = 1320)
    out_no_delay = _apply_ordered_window(
        starts, window_start=1320, window_end=360, random_delay=False
    )
    assert np.all(out_no_delay == 1320)

    # With random delay: should be spread across [22:00, 06:00)
    out_delay = _apply_ordered_window(
        starts, window_start=1320, window_end=360, random_delay=True, rng=rng
    )
    # Not all at window_start
    assert not np.all(out_delay == 1320)
    # Check they're within the window [22:00-24:00) or [00:00-06:00)
    in_window = (out_delay >= 1320) | (out_delay < 360)
    assert in_window.all()
    # Standard deviation should be non-trivial (spread out)
    assert float(np.std(out_delay)) > 30


def test_ordered_window_no_midnight_random_delay() -> None:
    """Test random delay for a window that doesn't cross midnight."""
    rng = np.random.default_rng(42)
    starts = np.full(300, 500, dtype=int)  # 8:20 AM, outside [10:00, 16:00)

    out = _apply_ordered_window(
        starts, window_start=600, window_end=960, random_delay=True, rng=rng
    )
    assert np.all((out >= 600) & (out < 960))
    assert float(np.std(out)) > 20


def test_tripchain_ordered_random_delay_reduces_peak(tmp_path: Path) -> None:
    """Random delay in ordered charging should produce lower peak than synchronous start."""
    cfg_yaml = tmp_path / "cfg.yaml"
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
  name: ordered
  ordered:
    window_start: "22:00"
    window_end: "06:00"
    random_delay: {delay}
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
    # Without delay
    cfg_yaml.write_text(base_yaml.format(delay="false"), encoding="utf-8")
    cfg_no = load_config(cfg_yaml)
    net = load_case(cfg_no.case.name, load_scale=cfg_no.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)
    rng1 = np.random.default_rng(42)
    p_no = build_ev_profile_mw(cfg=cfg_no, n_vehicles=200, buses=buses, n_buses=n_buses, rng=rng1)

    # With delay
    cfg_yaml.write_text(base_yaml.format(delay="true"), encoding="utf-8")
    cfg_yes = load_config(cfg_yaml)
    rng2 = np.random.default_rng(42)
    p_yes = build_ev_profile_mw(cfg=cfg_yes, n_vehicles=200, buses=buses, n_buses=n_buses, rng=rng2)

    peak_no = float(p_no.sum(axis=1).max())
    peak_yes = float(p_yes.sum(axis=1).max())

    # With delay should have lower or equal peak (spread out)
    assert peak_yes <= peak_no or np.isclose(peak_yes, peak_no, rtol=0.05)
    # Both should have non-zero load
    assert p_no.sum() > 0
    assert p_yes.sum() > 0
