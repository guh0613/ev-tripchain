from ev_tripchain.hosting_capacity.search import binary_search_max_n


def test_binary_search_max_n_returns_max_feasible() -> None:
    # risk is monotone increasing
    def risk(n: int) -> float:
        return n / 10.0

    n_star, curve = binary_search_max_n(
        risk,
        n_max=100,
        risk_tolerance=0.35,
        max_iter=20,
        min_step=1,
    )
    assert n_star == 3
    assert curve

