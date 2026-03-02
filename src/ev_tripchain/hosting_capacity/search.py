from __future__ import annotations

from collections.abc import Callable


def binary_search_max_n(
    risk_at_n: Callable[[int], float],
    *,
    n_max: int,
    risk_tolerance: float,
    max_iter: int,
    min_step: int,
) -> tuple[int, list[tuple[int, float]]]:
    """
    Search N* = max{N: risk(N) <= epsilon} using risk monotonicity.

    Returns (n_star, sampled_curve) where curve contains (N, risk(N)) pairs visited.
    """
    lo = 0
    hi = n_max
    curve: list[tuple[int, float]] = []

    while (hi - lo) > min_step and len(curve) < max_iter:
        mid = (lo + hi) // 2
        r = float(risk_at_n(mid))
        curve.append((mid, r))
        if r <= risk_tolerance:
            lo = mid
        else:
            hi = mid

    # ensure boundary points are included
    for n in {lo, hi}:
        if not any(x == n for x, _ in curve):
            curve.append((n, float(risk_at_n(n))))

    curve.sort(key=lambda x: x[0])
    n_star = max((n for n, r in curve if r <= risk_tolerance), default=0)
    return n_star, curve

