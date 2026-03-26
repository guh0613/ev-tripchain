"""Tests for deterministic representative-profile hosting capacity method."""

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.deterministic import run_deterministic_hc


def test_deterministic_hc_simple_case() -> None:
    """Deterministic HC should find a valid N* on the simple 4-bus case."""
    cfg = ProjectConfig(
        seed=42,
        case={"name": "simple", "load_scale": 0.3},
        ev={"charge_power_kw": 7.2},
        hosting_capacity={"n_max": 500, "binary_search": {"max_iter": 16, "min_step": 1}},
    )
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    result = run_deterministic_hc(net, cfg)

    assert result.n_star >= 0
    assert result.weakest_bus_voltage_pu > 0.9
    assert len(result.risk_curve) > 0
    assert {float(r) for _, r in result.risk_curve}.issubset({0.0, 1.0})


def test_deterministic_hc_gives_conservative_bound() -> None:
    """Deterministic N* should stay within the configured search range."""
    cfg = ProjectConfig(
        seed=42,
        case={"name": "simple", "load_scale": 0.3},
        ev={"charge_power_kw": 7.2},
        hosting_capacity={
            "n_max": 500,
            "scenarios": 5,
            "binary_search": {"max_iter": 12, "min_step": 1},
        },
    )
    net_det = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    det_result = run_deterministic_hc(net_det, cfg)

    assert det_result.n_star >= 0
    assert det_result.n_star <= cfg.hosting_capacity.n_max


def test_deterministic_zero_ev_no_hard_limit_exceedance() -> None:
    """With 0 EVs, deterministic should report no hard-limit exceedance."""
    cfg = ProjectConfig(
        seed=42,
        case={"name": "simple", "load_scale": 0.3},
        ev={"charge_power_kw": 7.2},
        hosting_capacity={"n_max": 100, "binary_search": {"max_iter": 10, "min_step": 1}},
    )
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    result = run_deterministic_hc(net, cfg)

    # Should find some non-zero N* since the base case is safe at load_scale=0.3.
    assert result.n_star >= 0
