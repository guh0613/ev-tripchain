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


def test_binary_search_max_n_handles_unsafe_base_case() -> None:
    def risk(n: int) -> float:
        return 0.2 if n == 0 else 1.0

    n_star, curve = binary_search_max_n(
        risk,
        n_max=100,
        risk_tolerance=0.05,
        max_iter=20,
        min_step=1,
    )

    assert n_star == 0
    assert curve == [(0, 0.2)]


def test_binary_search_max_n_returns_n_max_when_all_safe() -> None:
    def risk(n: int) -> float:
        return 0.01

    n_star, curve = binary_search_max_n(
        risk,
        n_max=12,
        risk_tolerance=0.05,
        max_iter=4,
        min_step=1,
    )

    assert n_star == 12
    assert curve == [(0, 0.01), (12, 0.01)]
