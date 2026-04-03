from pathlib import Path

import pytest
from pydantic import ValidationError

from ev_tripchain.config import ProjectConfig, load_config


def test_load_config_minimal(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 1\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.seed == 1
    assert cfg.time.n_steps > 0
    assert cfg.time.total_steps == cfg.time.n_steps * cfg.time.n_days


def test_default_risk_metric_matches_report_definition() -> None:
    cfg = ProjectConfig()
    assert cfg.hosting_capacity.risk_metric == "p_hat"
    assert cfg.hosting_capacity.parallel_workers == 1
    assert cfg.hosting_capacity.binary_search.initial_hi == 128


def test_repo_configs_keep_p_hat_for_hosting_capacity_decision() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for rel in ("configs/example.yaml", "configs/tripchain_soc.yaml"):
        cfg = load_config(repo_root / rel)
        assert cfg.hosting_capacity.risk_metric == "p_hat"
        assert cfg.hosting_capacity.parallel_workers == 0
        assert cfg.hosting_capacity.binary_search.initial_hi == 128


def test_tripchain_config_enables_final_home_charging_rule() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "configs/tripchain_soc.yaml")
    assert cfg.mobility.soc.final_home_charge_enabled is True
    assert cfg.mobility.soc.final_home_target_soc == 0.9


def test_parallel_workers_auto_resolution_is_positive() -> None:
    cfg = ProjectConfig(hosting_capacity={"parallel_workers": 0})
    assert cfg.hosting_capacity.resolved_parallel_workers >= 1


def test_constraints_support_legacy_flat_shape() -> None:
    cfg = ProjectConfig(
        constraints={
            "vmin_pu": 0.94,
            "vmax_pu": 1.06,
            "line_loading_max_percent": 95.0,
            "trafo_loading_max_percent": 90.0,
            "nominal_voltage_pu": 1.01,
        }
    )
    assert cfg.constraints.hard.vmin_pu == 0.94
    assert cfg.constraints.hard.vmax_pu == 1.06
    assert cfg.constraints.hard.line_loading_max_percent == 95.0
    assert cfg.constraints.hard.trafo_loading_max_percent == 90.0
    assert cfg.constraints.soft.nominal_voltage_pu == 1.01


def test_tripchain_soc_requires_matching_zone_and_mapping_counts() -> None:
    with pytest.raises(ValidationError, match="mobility.trip_chain.n_zones must match"):
        ProjectConfig(
            mobility={
                "model": "tripchain_soc",
                "trip_chain": {"n_zones": 7},
                "mapping": {"n_nodes": 5},
            }
        )
