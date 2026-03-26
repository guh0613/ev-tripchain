from pathlib import Path

import numpy as np

from ev_tripchain.config import load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.evaluate import _ensure_ev_load_elements
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.pipelines.run import run_hosting_capacity


def test_multiday_ordered_tripchain_profile_has_overnight_load_with_real_sampler(
    tmp_path: Path,
) -> None:
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
  n_days: 2
ev:
  charge_power_kw: 7.2
strategy:
  name: ordered
  ordered:
    window_start: "22:00"
    window_end: "06:00"
    random_delay: true
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
    initial_soc_mean: 0.55
    initial_soc_std: 0.0
    charge_trigger_soc: 1.0
    charge_purposes: ["home", "work", "other"]
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()

    profile = build_ev_profile_mw(
        cfg=cfg,
        n_vehicles=200,
        buses=buses,
        n_buses=len(ev_idx),
        rng=np.random.default_rng(cfg.seed),
    )

    steps_per_day = int(cfg.time.n_steps)
    late_day1 = float(profile[22 * 4 : steps_per_day, :].sum())
    early_day2 = float(profile[steps_per_day : steps_per_day + 6 * 4, :].sum())

    assert profile.shape == (cfg.time.total_steps, len(ev_idx))
    assert late_day1 > 0.0
    assert early_day2 > 0.0


def test_run_hosting_capacity_supports_multiday_tripchain_end_to_end(tmp_path: Path) -> None:
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        """
seed: 7
case:
  name: simple
  load_scale: 0.4
time:
  step_minutes: 15
  n_steps: 96
  n_days: 2
hosting_capacity:
  scenarios: 6
  risk_tolerance: 0.05
  common_random_numbers: true
  parallel_workers: 1
  n_max: 60
  binary_search:
    max_iter: 8
    min_step: 2
constraints:
  hard:
    vmin_pu: 0.95
    vmax_pu: 1.05
    line_loading_max_percent: 100.0
    trafo_loading_max_percent: 100.0
  soft:
    nominal_voltage_pu: 1.0
ev:
  charge_power_kw: 7.2
strategy:
  name: ordered
  ordered:
    window_start: "22:00"
    window_end: "06:00"
    random_delay: true
mobility:
  model: tripchain_soc
  mapping:
    policy: random_onehot
    n_nodes: 10
  trip_chain:
    n_zones: 10
    distance_km_mean: 18.0
    distance_km_std: 8.0
  soc:
    initial_soc_mean: 0.6
    initial_soc_std: 0.05
    charge_trigger_soc: 0.5
    charge_purposes: ["home", "work"]
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_yaml)

    result = run_hosting_capacity(cfg)

    assert result.n_star >= 0
    assert result.scenarios == 6
    assert result.base_case_safe is True
    assert len(result.risk_curve_detail) > 0
    assert result.risk_curve_detail[0].n == 0
    assert all(0.0 <= point.p_hat <= 1.0 for point in result.risk_curve_detail)
    assert all(point.soft_metrics is not None for point in result.risk_curve_detail)
