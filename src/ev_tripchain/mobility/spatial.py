from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ev_tripchain.hosting_capacity.sensitivity import VoltageSensitivityModel


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


def compute_session_step_weights(
    *,
    start_minute: int,
    end_minute: int,
    step_minutes: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a minute-domain charging interval into step indices and fractional occupancy.
    """
    step = int(step_minutes)
    total_steps = int(n_steps)
    if step <= 0:
        raise ValueError("step_minutes must be positive.")
    if total_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    if total_steps == 0:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
        )

    total_minutes = step * total_steps
    start = max(0, min(int(start_minute), total_minutes))
    end = max(0, min(int(end_minute), total_minutes))
    if end <= start:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=float),
        )

    first_step = start // step
    last_step = min(total_steps, int(np.ceil(end / step)))
    step_idx: list[int] = []
    step_weight: list[float] = []
    for k in range(first_step, last_step):
        lo = max(start, k * step)
        hi = min(end, (k + 1) * step)
        overlap = hi - lo
        if overlap <= 0:
            continue
        step_idx.append(k)
        step_weight.append(float(overlap) / float(step))
    return np.asarray(step_idx, dtype=int), np.asarray(step_weight, dtype=float)


def _build_ieee33_station_distance_model(bus_ids: np.ndarray) -> SpatialDistanceModel | None:
    n = int(bus_ids.size)
    if n < len(_IEEE33_EVCS_NODE_IDS):
        return None

    bus_ids = np.asarray(bus_ids, dtype=int).reshape(-1)

    # Accept either 1-based node ids (1..33) or 0-based pandapower indices (0..32).
    #
    # Note: In our pipelines we often exclude the ext_grid bus (typically node 1 / index 0),
    # which yields an ambiguous set like {1..32}. In that case we treat it as 0-based and
    # map to node ids {2..33}.
    if np.any(bus_ids == 0):
        node_ids = bus_ids + 1
    elif np.any(bus_ids == 33):
        node_ids = bus_ids.copy()
    elif np.all((bus_ids >= 1) & (bus_ids <= 32)):
        # Ambiguous: could be 1-based subset or 0-based without 0.
        if n == 32 and np.array_equal(np.sort(bus_ids), np.arange(1, 33, dtype=int)):
            node_ids = bus_ids + 1
        else:
            node_ids = bus_ids.copy()
    elif np.all((bus_ids >= 1) & (bus_ids <= 33)):
        node_ids = bus_ids.copy()
    elif np.all((bus_ids >= 0) & (bus_ids <= 32)):
        node_ids = bus_ids + 1
    else:
        return None

    # Avoid false positives on non-IEEE33 cases: require the core station nodes to exist.
    required_station_nodes = np.array([6, 9, 13, 16, 20, 30], dtype=int)
    if not np.isin(required_station_nodes, node_ids).all():
        return None

    node_set = set(node_ids.tolist())
    present_station_nodes = [x for x in _IEEE33_EVCS_NODE_IDS.tolist() if x in node_set]
    if len(present_station_nodes) < required_station_nodes.size:
        return None

    station_to_col: dict[int, int] = {}
    for col, node_id in enumerate(node_ids.tolist()):
        if node_id in _IEEE33_EVCS_NODE_IDS:
            station_to_col[node_id] = col

    candidate_bus_idx = np.array(
        [station_to_col[int(node_id)] for node_id in present_station_nodes],
        dtype=int,
    )

    dist_m = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist_m, 0.0)

    row_idx = node_ids - 1
    for station_node, station_col in zip(
        present_station_nodes,
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


def _normalize_positive(values: np.ndarray, *, floor: float = 0.2) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    x = np.maximum(x, 0.0)
    xmax = float(np.max(x))
    if not np.isfinite(xmax) or xmax <= 0.0:
        return np.ones_like(x, dtype=float)
    x = x / xmax
    floor = float(min(max(floor, 0.0), 1.0))
    return floor + (1.0 - floor) * x


def _normalize_span(values: np.ndarray, *, floor: float = 0.2) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    finite = np.isfinite(x)
    if not finite.any():
        return np.ones_like(x, dtype=float)
    xmin = float(np.min(x[finite]))
    xmax = float(np.max(x[finite]))
    if xmax - xmin <= 1e-12:
        return np.ones_like(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    y[finite] = (x[finite] - xmin) / (xmax - xmin)
    floor = float(min(max(floor, 0.0), 1.0))
    return floor + (1.0 - floor) * y


def _soft_barrier(
    values: np.ndarray,
    *,
    buffer: float,
    window: float,
    floor: float = 0.03,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x
    scale = float(max(window, 1e-9))
    z = np.clip((x - float(buffer)) / scale, -60.0, 60.0)
    y = 1.0 / (1.0 + np.exp(-z))
    floor = float(min(max(floor, 0.0), 1.0))
    return floor + (1.0 - floor) * y


def choose_spatial_target_bus(
    *,
    src_bus_col: int,
    strategy_name: str,
    dist_m: np.ndarray,
    candidate_bus_idx: np.ndarray | None,
    navigation_candidate_k: int,
    navigation_distance_limit_m: float | None = None,
    navigation_distance_beta: float = 1.0,
    candidate_bus_score: np.ndarray | None = None,
    session_step_idx: np.ndarray | None = None,
    session_step_weight: np.ndarray | None = None,
    scheduled_load_mw: np.ndarray | None = None,
    candidate_charge_power_mw: float | None = None,
    voltage_model: VoltageSensitivityModel | None = None,
    dynamic_safety_buffer_pu: float = 0.002,
    dynamic_voltage_penalty_window_pu: float = 0.006,
    path_congestion_weight: float = 0.35,
    softmax_temperature: float = 0.35,
    disable_voltage_factor: bool = False,
    rng: np.random.Generator,
) -> int:
    """
    Select a charging bus using either pure distance (`nearest`) or multi-factor navigation.

    Navigation first respects reachable stations, then blends static grid headroom,
    current time-overlap congestion, and optional linearized voltage-risk filtering.
    """
    src = int(src_bus_col)
    if dist_m.shape[0] <= 1:
        return src

    if candidate_bus_idx is None:
        candidates = np.arange(dist_m.shape[1], dtype=int)
    else:
        candidates = np.asarray(candidate_bus_idx, dtype=int).reshape(-1)
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
    ranked_k = ranked[:k]
    d_k = dist_m[src, ranked_k].astype(float, copy=False)

    dlim = navigation_distance_limit_m
    if dlim is not None:
        dlim = float(dlim)
        if dlim > 0.0:
            within = d_k <= dlim
            if within.any():
                ranked_k = ranked_k[within]
                d_k = d_k[within]

    beta = float(max(0.0, navigation_distance_beta))
    if beta == 0.0:
        w_dist = np.ones_like(d_k, dtype=float)
    else:
        w_dist = 1.0 / np.power(np.maximum(d_k, 1.0), beta)
    w_dist = _normalize_positive(w_dist)

    if candidate_bus_score is None:
        w_score = np.ones_like(w_dist, dtype=float)
    else:
        s = np.asarray(candidate_bus_score, dtype=float).reshape(-1)
        if s.size != dist_m.shape[0]:
            raise ValueError("candidate_bus_score size mismatch with dist_m.")
        w_score = _normalize_positive(s[ranked_k])

    w_temporal = np.ones_like(w_dist, dtype=float)
    w_voltage = np.ones_like(w_dist, dtype=float)
    w_path = np.ones_like(w_dist, dtype=float)
    dynamic_context_ready = (
        scheduled_load_mw is not None
        and session_step_idx is not None
        and session_step_weight is not None
        and candidate_charge_power_mw is not None
    )
    if dynamic_context_ready:
        load_mw = np.asarray(scheduled_load_mw, dtype=float)
        if load_mw.ndim != 2 or load_mw.shape[1] != dist_m.shape[0]:
            raise ValueError("scheduled_load_mw shape mismatch with dist_m.")

        step_idx = np.asarray(session_step_idx, dtype=int).reshape(-1)
        step_weight = np.asarray(session_step_weight, dtype=float).reshape(-1)
        if step_idx.size != step_weight.size:
            raise ValueError("session_step_idx and session_step_weight size mismatch.")

        valid_mask = (
            (step_idx >= 0)
            & (step_idx < load_mw.shape[0])
            & np.isfinite(step_weight)
            & (step_weight > 0.0)
        )
        step_idx = step_idx[valid_mask]
        step_weight = step_weight[valid_mask]
        if step_idx.size > 0:
            session_load = load_mw[np.ix_(step_idx, ranked_k)]
            p_ref = float(max(candidate_charge_power_mw, 1e-9))
            avg_ratio = np.average(
                session_load / p_ref,
                axis=0,
                weights=step_weight,
            )
            peak_ratio = session_load.max(axis=0) / p_ref
            temporal_metric = 1.0 / (1.0 + avg_ratio + 0.5 * peak_ratio)
            w_temporal = _normalize_positive(temporal_metric)

            if voltage_model is not None:
                base_v = np.asarray(voltage_model.base_voltage_pu, dtype=float).reshape(-1)
                sens = np.asarray(voltage_model.sensitivity_pu_per_mw, dtype=float)
                if base_v.size != dist_m.shape[0]:
                    raise ValueError("voltage_model base_voltage_pu size mismatch with dist_m.")
                if sens.shape != dist_m.shape:
                    raise ValueError("voltage_model sensitivity_pu_per_mw shape mismatch.")

                existing_margin = (
                    base_v[None, :]
                    + load_mw[np.ix_(step_idx, np.arange(dist_m.shape[0], dtype=int))] @ sens.T
                    - float(voltage_model.vmin_pu)
                )
                delta = sens[:, ranked_k]
                step_increment = step_weight * float(candidate_charge_power_mw)
                candidate_margin = (
                    existing_margin[:, :, None]
                    + step_increment[:, None, None] * delta[None, :, :]
                )
                system_margin = candidate_margin.min(axis=(0, 1))
                local_delta = delta[ranked_k, np.arange(ranked_k.size)]
                local_margin = (
                    existing_margin[:, ranked_k]
                    + step_increment[:, None] * local_delta[None, :]
                ).min(axis=0)
                combined_margin = np.minimum(system_margin, local_margin)

                if not disable_voltage_factor:
                    keep_mask = np.isfinite(combined_margin)
                    if np.any(combined_margin >= 0.0):
                        keep_mask &= combined_margin >= 0.0
                    if keep_mask.any() and not keep_mask.all():
                        ranked_k = ranked_k[keep_mask]
                        d_k = d_k[keep_mask]
                        w_dist = w_dist[keep_mask]
                        w_score = w_score[keep_mask]
                        w_temporal = w_temporal[keep_mask]
                        combined_margin = combined_margin[keep_mask]
                w_voltage = np.ones(ranked_k.size, dtype=float)
                w_path = np.ones(ranked_k.size, dtype=float)

                if not disable_voltage_factor:
                    safety_buffer = float(max(dynamic_safety_buffer_pu, 0.0))
                    voltage_window = float(max(dynamic_voltage_penalty_window_pu, 1e-6))
                    w_voltage = _soft_barrier(
                        combined_margin,
                        buffer=0.0,
                        window=voltage_window,
                    ) * _soft_barrier(
                        combined_margin,
                        buffer=safety_buffer,
                        window=voltage_window,
                    )

                path_incidence = getattr(voltage_model, "path_incidence", None)
                line_capacity_mw = getattr(voltage_model, "line_capacity_mw", None)
                base_line_loading_percent = getattr(
                    voltage_model,
                    "base_line_loading_percent",
                    None,
                )
                if (
                    path_congestion_weight > 0.0
                    and path_incidence is not None
                    and line_capacity_mw is not None
                    and base_line_loading_percent is not None
                ):
                    path_inc = np.asarray(path_incidence, dtype=float)
                    line_cap = np.asarray(line_capacity_mw, dtype=float).reshape(-1)
                    base_line = np.asarray(base_line_loading_percent, dtype=float).reshape(-1)
                    if (
                        path_inc.ndim == 2
                        and path_inc.shape[1] == dist_m.shape[0]
                        and path_inc.shape[0] == line_cap.size
                        and base_line.size == line_cap.size
                        and np.all(np.isfinite(line_cap))
                        and np.all(line_cap > 0.0)
                    ):
                        current_path_load = load_mw[np.ix_(step_idx, np.arange(dist_m.shape[0], dtype=int))] @ path_inc.T
                        candidate_path = path_inc[:, ranked_k].T
                        path_overlap_metric = np.ones(ranked_k.size, dtype=float)
                        path_margin = np.full(
                            ranked_k.size,
                            float(getattr(voltage_model, "line_loading_limit_percent", 100.0)),
                            dtype=float,
                        )
                        line_limit = float(getattr(voltage_model, "line_loading_limit_percent", 100.0))
                        for idx, upstream_mask in enumerate(candidate_path > 0.0):
                            if not upstream_mask.any():
                                continue
                            upstream_load = current_path_load[:, upstream_mask]
                            mean_ratio = np.average(
                                upstream_load / p_ref,
                                axis=0,
                                weights=step_weight,
                            )
                            peak_ratio = upstream_load.max(axis=0) / p_ref
                            path_overlap_metric[idx] = 1.0 / (
                                1.0 + float(mean_ratio.max()) + 0.5 * float(peak_ratio.max())
                            )

                            predicted_line_loading = (
                                base_line[upstream_mask][None, :]
                                + 100.0
                                * (
                                    upstream_load
                                    + step_increment[:, None]
                                )
                                / line_cap[upstream_mask][None, :]
                            )
                            path_margin[idx] = float(line_limit - predicted_line_loading.max())

                        raw_path = _normalize_positive(path_overlap_metric) * _soft_barrier(
                            path_margin,
                            buffer=12.0,
                            window=8.0,
                        )
                        raw_path = np.clip(raw_path, 1e-6, None)
                        w_path = np.power(raw_path, float(path_congestion_weight))

    w = (w_score + 1e-6) * w_dist
    w *= w_temporal * w_voltage * w_path
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        return int(ranked_k[0])

    if dynamic_context_ready:
        logits = np.log(np.maximum(w, 1e-12))
        logits = logits - float(np.max(logits))
        scaled = logits / float(max(softmax_temperature, 1e-6))
        exp_logits = np.exp(scaled - float(np.max(scaled)))
        p = exp_logits / float(np.sum(exp_logits))
    else:
        p = w / w_sum
    return int(rng.choice(ranked_k, p=p))
