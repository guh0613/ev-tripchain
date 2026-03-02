from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Huang Mengqi (2023), Appendix Table A1:
# distance from each IEEE 33 node to 7 EV charging stations (meters).
_IEEE33_EVCS_NODE_IDS = np.array([1, 6, 9, 13, 16, 20, 30], dtype=int)
_IEEE33_NODE_TO_EVCS_DISTANCE_M = np.array(
    [
        [0, 900, 700, 1300, 1700, 900, 1500],
        [150, 700, 900, 1500, 1600, 700, 1400],
        [500, 600, 1000, 1700, 1500, 300, 1300],
        [900, 700, 1100, 1800, 1500, 200, 1400],
        [700, 300, 700, 1300, 1300, 500, 1200],
        [900, 0, 300, 1200, 1200, 900, 600],
        [1000, 300, 200, 700, 1000, 1200, 200],
        [1000, 500, 200, 600, 1200, 1300, 500],
        [700, 300, 0, 700, 1300, 1200, 200],
        [1000, 900, 300, 500, 1500, 1500, 900],
        [1100, 1000, 500, 200, 1500, 1700, 900],
        [1300, 1100, 700, 200, 700, 1700, 700],
        [1300, 1200, 700, 0, 900, 1800, 900],
        [1500, 1500, 1100, 300, 900, 1800, 1100],
        [1700, 1600, 1100, 700, 300, 1700, 700],
        [1700, 1600, 1200, 1000, 0, 1500, 700],
        [1600, 1300, 1400, 1200, 300, 1500, 500],
        [200, 1300, 1000, 1800, 1900, 900, 1700],
        [300, 1000, 1400, 2000, 1800, 500, 1700],
        [900, 1300, 1500, 2100, 1800, 0, 1700],
        [1000, 1400, 1600, 2200, 1900, 200, 1800],
        [200, 700, 700, 1200, 1800, 900, 1200],
        [600, 700, 300, 1000, 1700, 1100, 1300],
        [900, 700, 200, 1500, 1500, 1200, 1200],
        [1000, 300, 1000, 1300, 1300, 700, 1000],
        [1100, 200, 1200, 1200, 1200, 900, 700],
        [1200, 300, 1300, 1000, 900, 1000, 600],
        [1300, 400, 1300, 900, 700, 1100, 400],
        [1200, 300, 700, 900, 900, 1200, 200],
        [1500, 600, 700, 900, 700, 1300, 0],
        [1600, 900, 1000, 700, 600, 1400, 200],
        [1700, 1000, 1100, 700, 500, 1500, 400],
        [200, 1700, 1000, 1200, 800, 900, 1600],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SpatialDistanceModel:
    dist_m: np.ndarray  # shape (n_buses, n_buses)
    candidate_bus_idx: np.ndarray  # shape (n_candidates,), bus-column indices


def _build_ieee33_station_distance_model(bus_ids: np.ndarray) -> SpatialDistanceModel | None:
    n = int(bus_ids.size)
    if n < len(_IEEE33_EVCS_NODE_IDS):
        return None

    # Accept either 1-based bus ids (1..33) or 0-based bus ids (0..32).
    if np.all((bus_ids >= 1) & (bus_ids <= 33)):
        node_ids = bus_ids.copy()
    elif np.all((bus_ids >= 0) & (bus_ids <= 32)):
        node_ids = bus_ids + 1
    else:
        return None

    # Avoid false positives on non-IEEE33 cases: all 7 charging-station nodes must exist.
    if not np.isin(_IEEE33_EVCS_NODE_IDS, node_ids).all():
        return None

    station_to_col: dict[int, int] = {}
    for col, node_id in enumerate(node_ids.tolist()):
        if node_id in _IEEE33_EVCS_NODE_IDS:
            station_to_col[node_id] = col

    candidate_bus_idx = np.array(
        [station_to_col[int(node_id)] for node_id in _IEEE33_EVCS_NODE_IDS],
        dtype=int,
    )

    dist_m = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist_m, 0.0)

    row_idx = node_ids - 1
    for station_node, station_col in zip(
        _IEEE33_EVCS_NODE_IDS.tolist(),
        candidate_bus_idx.tolist(),
    ):
        station_k = int(np.where(_IEEE33_EVCS_NODE_IDS == station_node)[0][0])
        dist_m[:, station_col] = _IEEE33_NODE_TO_EVCS_DISTANCE_M[row_idx, station_k]

    return SpatialDistanceModel(dist_m=dist_m, candidate_bus_idx=candidate_bus_idx)


def _build_fallback_distance_model(n_buses: int) -> SpatialDistanceModel:
    # Deterministic surrogate for non-IEEE33 cases: no random geometry.
    order = np.arange(int(n_buses), dtype=float)
    dist_m = np.abs(order[:, None] - order[None, :]) * 1000.0
    candidate_bus_idx = np.arange(int(n_buses), dtype=int)
    return SpatialDistanceModel(dist_m=dist_m, candidate_bus_idx=candidate_bus_idx)


def build_spatial_distance_model(*, buses: np.ndarray, n_buses: int) -> SpatialDistanceModel:
    bus_ids = np.asarray(buses, dtype=int).reshape(-1)
    n = int(n_buses)
    if bus_ids.size != n:
        raise ValueError("buses size mismatch with n_buses.")
    if n < 0:
        raise ValueError("n_buses must be non-negative.")
    if n == 0:
        return SpatialDistanceModel(
            dist_m=np.zeros((0, 0), dtype=float),
            candidate_bus_idx=np.zeros((0,), dtype=int),
        )

    model = _build_ieee33_station_distance_model(bus_ids)
    if model is not None:
        return model
    return _build_fallback_distance_model(n)


def choose_spatial_target_bus(
    *,
    src_bus_col: int,
    strategy_name: str,
    dist_m: np.ndarray,
    candidate_bus_idx: np.ndarray | None,
    navigation_candidate_k: int,
    rng: np.random.Generator,
) -> int:
    src = int(src_bus_col)
    if dist_m.shape[0] <= 1:
        return src

    if candidate_bus_idx is None:
        candidates = np.arange(dist_m.shape[1], dtype=int)
    else:
        candidates = np.asarray(candidate_bus_idx, dtype=int).reshape(-1)

    candidates = candidates[candidates != src]
    if candidates.size == 0:
        return src

    d = dist_m[src, candidates]
    finite_mask = np.isfinite(d)
    if not finite_mask.any():
        return src

    candidates = candidates[finite_mask]
    d = d[finite_mask]
    ranked = candidates[np.argsort(d, kind="stable")]
    if ranked.size == 0:
        return src

    if strategy_name == "nearest":
        return int(ranked[0])

    k = max(int(navigation_candidate_k), 1)
    k = min(k, int(ranked.size))
    return int(rng.choice(ranked[:k]))
