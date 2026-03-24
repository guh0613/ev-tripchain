import numpy as np

from ev_tripchain.grid.cases import load_case
from ev_tripchain.grid.constraints import evaluate_constraints
from ev_tripchain.grid.powerflow import run_powerflow


def test_evaluate_constraints_reports_soft_metrics_for_safe_snapshot() -> None:
    net = load_case("simple", load_scale=0.3)
    run_powerflow(net)

    assessment = evaluate_constraints(
        net,
        vmin=0.95,
        vmax=1.05,
        line_max=100.0,
        trafo_max=100.0,
        nominal_voltage_pu=1.0,
    )

    assert not assessment.hard.any_exceedance
    assert assessment.soft.voltage_deviation_max_pu >= 0.0
    assert assessment.soft.voltage_deviation_mean_pu >= 0.0
    assert assessment.soft.network_loss_mw >= 0.0


def test_evaluate_constraints_detects_voltage_limit_exceedance() -> None:
    net = load_case("simple", load_scale=1.0)
    run_powerflow(net)

    assessment = evaluate_constraints(
        net,
        vmin=0.999,
        vmax=1.05,
        line_max=100.0,
        trafo_max=100.0,
        nominal_voltage_pu=1.0,
    )

    assert assessment.hard.voltage_exceedance
    assert assessment.hard.voltage_lower_exceedance_count > 0
    assert assessment.hard.voltage_lower_max_exceedance_pu > 0.0
    assert np.isfinite(assessment.hard.min_voltage_pu)
