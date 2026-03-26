from __future__ import annotations

from collections.abc import Callable


def binary_search_max_n(
    risk_at_n: Callable[[int], float],
    *,
    n_max: int,
    risk_tolerance: float,
    max_iter: int,
    min_step: int,
    initial_hi: int = 128,
) -> tuple[int, list[tuple[int, float]]]:
    """
    Search N* = max{N: risk(N) <= epsilon} using risk monotonicity.

    Returns (n_star, sampled_curve) where curve contains (N, risk(N)) pairs visited.
    """
    sampled: dict[int, float] = {}

    def eval_risk(n: int) -> float:
        nn = int(n)
        if nn not in sampled:
            sampled[nn] = float(risk_at_n(nn))
        return sampled[nn]

    lo = 0
    hi_cap = int(max(n_max, 0))

    r0 = eval_risk(0)
    if r0 > risk_tolerance:
        return 0, sorted(sampled.items())

    if hi_cap == 0:
        return 0, sorted(sampled.items())

    hi = min(max(int(initial_hi), 1), hi_cap)
    r_hi = eval_risk(hi)

    while r_hi <= risk_tolerance and hi < hi_cap:
        lo = hi
        next_hi = min(hi_cap, hi * 2)
        if next_hi == hi:
            break
        hi = next_hi
        r_hi = eval_risk(hi)

    if r_hi <= risk_tolerance:
        return hi, sorted(sampled.items())

    iterations = 0
    while (hi - lo) > min_step and iterations < max_iter:
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break
        r_mid = eval_risk(mid)
        if r_mid <= risk_tolerance:
            lo = mid
        else:
            hi = mid
        iterations += 1

    # Verify the full final bracket so we do not skip the true boundary.
    for n in range(lo + 1, hi):
        eval_risk(n)

    curve = sorted(sampled.items())
    n_star = max((n for n, r in curve if r <= risk_tolerance), default=0)
    return n_star, curve
