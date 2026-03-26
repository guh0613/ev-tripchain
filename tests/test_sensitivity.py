"""Tests for voltage sensitivity hosting capacity method."""

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.sensitivity import run_sensitivity_hc


def test_sensitivity_hc_simple_case() -> None:
    """Sensitivity method should produce valid results on simple case."""
    cfg = ProjectConfig(
        seed=42,
        case={"name": "simple", "load_scale": 0.3},
        ev={"charge_power_kw": 7.2},
    )
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    result = run_sensitivity_hc(net, cfg)

    assert result.n_star_representative >= 0
    assert result.n_star_uniform >= 0
    assert result.n_star_weakest >= 0
    # Uniform allocation should allow more EVs than worst-bus allocation
    assert result.n_star_uniform >= result.n_star_weakest
    # Sensitivity diagonal should be negative (adding load drops voltage)
    assert np.all(result.sensitivity_diagonal <= 0)
    # Base voltages should be reasonable
    assert np.all(result.base_voltage > 0.9)
    assert np.all(result.voltage_margin >= 0)
    assert np.isclose(result.representative_bus_share.sum(), 1.0)


def test_sensitivity_hc_ieee33() -> None:
    """Sensitivity method should work on IEEE 33 system."""
    cfg = ProjectConfig(
        seed=42,
        case={"name": "ieee33", "load_scale": 0.55},
        ev={"charge_power_kw": 7.2},
    )
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    result = run_sensitivity_hc(net, cfg)

    assert result.n_star_representative >= 0
    assert result.n_star_uniform >= 0
    assert result.n_star_weakest >= 0
    assert result.sensitivity_diagonal.shape[0] == 32  # 33 buses minus ext_grid
    assert result.representative_bus_share.shape[0] == 32
    # Weakest bus should have smallest margin
    weakest_idx = int(np.argmin(result.voltage_margin))
    assert result.voltage_margin[weakest_idx] >= 0
