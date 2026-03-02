import numpy as np

from ev_tripchain.hosting_capacity.monte_carlo import estimate_event_probability


def test_estimate_event_probability_matches_frequency() -> None:
    rng = np.random.default_rng(123)

    def event(r: np.random.Generator) -> bool:
        # deterministic given RNG stream
        return bool(r.random() < 0.2)

    est = estimate_event_probability(event, n_scenarios=1000, rng=rng)
    assert est.n == 1000
    assert 0.15 <= est.p_hat <= 0.25
    assert 0.0 <= est.ci95_low <= est.p_hat <= est.ci95_high <= 1.0

