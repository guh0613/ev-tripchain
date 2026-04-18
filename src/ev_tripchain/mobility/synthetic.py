from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.mobility.spatial import (
    build_spatial_distance_model,
    choose_spatial_target_bus,
    compute_session_step_weights,
)

if TYPE_CHECKING:
    from ev_tripchain.hosting_capacity.sensitivity import VoltageSensitivityModel


def _parse_hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.strip().split(":")
    return int(hh) * 60 + int(mm)


def _sample_start_minutes(cfg: ProjectConfig, *, size: int, rng: np.random.Generator) -> np.ndarray:
    if not cfg.ev.start_time_mix:
        # default: around 20:00
        mean = 20 * 60
        std = 90
        return np.clip(rng.normal(mean, std, size=size).round().astype(int), 0, 24 * 60 - 1)

    weights = np.array([c.weight for c in cfg.ev.start_time_mix], dtype=float)
    weights = weights / weights.sum()
    comp = rng.choice(len(cfg.ev.start_time_mix), size=size, p=weights)

    out = np.empty(size, dtype=int)
    for i, c in enumerate(cfg.ev.start_time_mix):
        mask = comp == i
        if not mask.any():
            continue
        mean = _parse_hhmm_to_minutes(c.mean)
        out[mask] = np.clip(
            rng.normal(mean, c.std_minutes, size=int(mask.sum())).round().astype(int),
            0,
            24 * 60 - 1,
        )
    return out


def _apply_ordered_window(
    start_minute: np.ndarray,
    *,
    window_start: int,
    window_end: int,
    random_delay: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Shift start times into a charging window. Supports windows that cross midnight.

    When random_delay is True, vehicles outside the window are uniformly distributed
    across the window instead of all clustering at window_start.
    """
    start = start_minute.copy()
    if window_start <= window_end:
        in_window = (start >= window_start) & (start < window_end)
        out_mask = ~in_window
        if random_delay and rng is not None:
            n_out = int(out_mask.sum())
            if n_out > 0:
                start[out_mask] = rng.integers(window_start, max(window_start + 1, window_end), size=n_out)
        else:
            start[out_mask] = window_start
        return start

    # crossing midnight, e.g., 22:00-06:00
    in_window = (start >= window_start) | (start < window_end)
    out_mask = ~in_window
    if random_delay and rng is not None:
        n_out = int(out_mask.sum())
        if n_out > 0:
            window_len = (24 * 60 - window_start) + window_end
            offsets = rng.integers(0, max(1, window_len), size=n_out)
            start[out_mask] = (window_start + offsets) % (24 * 60)
    else:
        start[out_mask] = window_start
    return start


def build_ev_profile_mw(
    *,
    cfg: ProjectConfig,
    n_vehicles: int,
    buses: np.ndarray,
    n_buses: int,
    bus_score: np.ndarray | None = None,
    navigation_voltage_model: VoltageSensitivityModel | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build aggregated EV charging profile for a single scenario.

    Returns array of shape (T, n_buses) in MW, aligned with `buses` ordering.
    """
    step = cfg.time.step_minutes
    n_days = int(cfg.time.n_days)
    t_steps = int(cfg.time.total_steps)
    minutes_per_day = int(cfg.time.day_minutes)

    if n_vehicles <= 0:
        return np.zeros((t_steps, n_buses), dtype=float)

    # Sample sequentially per vehicle so common-random-number runs keep a stable prefix
    # when N changes, which makes the scenario risk much less noisy under binary search.
    lam = max(cfg.ev.sessions_per_vehicle_mean, 0.0)
    spatial_model = None
    if cfg.strategy.name in {"nearest", "navigation"}:
        spatial_model = build_spatial_distance_model(buses=buses, n_buses=int(n_buses))

    p_kw = float(cfg.ev.charge_power_kw)
    p_mw = p_kw / 1000.0
    prof = np.zeros((t_steps, n_buses), dtype=float)
    if p_mw <= 0.0:
        return prof

    for _ in range(int(n_vehicles)):
        for day_idx in range(n_days):
            n_sessions = int(rng.poisson(lam))
            for _ in range(n_sessions):
                home_bus_idx = int(rng.integers(0, n_buses))
                start_min = int(_sample_start_minutes(cfg, size=1, rng=rng)[0])
                dur_min = int(
                    np.clip(
                        round(float(rng.normal(cfg.ev.duration_minutes_mean, cfg.ev.duration_minutes_std))),
                        step,
                        minutes_per_day,
                    )
                )

                if cfg.strategy.name == "ordered":
                    ws = _parse_hhmm_to_minutes(cfg.strategy.ordered.window_start)
                    we = _parse_hhmm_to_minutes(cfg.strategy.ordered.window_end)
                    start_min = int(
                        _apply_ordered_window(
                            np.array([start_min], dtype=int),
                            window_start=ws,
                            window_end=we,
                            random_delay=cfg.strategy.ordered.random_delay,
                            rng=rng,
                        )[0]
                    )

                target_bus_idx = home_bus_idx
                start_abs_min = day_idx * minutes_per_day + start_min
                end_abs_min = min(start_abs_min + dur_min, minutes_per_day * n_days)
                session_step_idx, session_step_weight = compute_session_step_weights(
                    start_minute=start_abs_min,
                    end_minute=end_abs_min,
                    step_minutes=int(step),
                    n_steps=t_steps,
                )
                if spatial_model is not None:
                    dynamic_context_enabled = (
                        cfg.strategy.name == "navigation"
                        and cfg.strategy.navigation.dynamic_scoring
                        and session_step_idx.size > 0
                    )
                    target_bus_idx = choose_spatial_target_bus(
                        src_bus_col=home_bus_idx,
                        strategy_name=cfg.strategy.name,
                        dist_m=spatial_model.dist_m,
                        candidate_bus_idx=spatial_model.candidate_bus_idx,
                        navigation_candidate_k=int(cfg.strategy.navigation.candidate_k),
                        navigation_distance_limit_m=cfg.strategy.navigation.distance_limit_m,
                        navigation_distance_beta=float(cfg.strategy.navigation.distance_beta),
                        candidate_bus_score=bus_score,
                        session_step_idx=session_step_idx if dynamic_context_enabled else None,
                        session_step_weight=session_step_weight if dynamic_context_enabled else None,
                        scheduled_load_mw=prof if dynamic_context_enabled else None,
                        candidate_charge_power_mw=p_mw if dynamic_context_enabled else None,
                        voltage_model=navigation_voltage_model if dynamic_context_enabled else None,
                        dynamic_safety_buffer_pu=float(
                            cfg.strategy.navigation.dynamic_safety_buffer_pu
                        ),
                        dynamic_voltage_penalty_window_pu=float(
                            cfg.strategy.navigation.dynamic_voltage_penalty_window_pu
                        ),
                        path_congestion_weight=float(
                            cfg.strategy.navigation.path_congestion_weight
                        ),
                        softmax_temperature=float(
                            cfg.strategy.navigation.softmax_temperature
                        ),
                        disable_voltage_factor=bool(
                            cfg.strategy.navigation.disable_voltage_factor
                        ),
                        rng=rng,
                    )

                if session_step_idx.size == 0:
                    continue
                prof[session_step_idx, target_bus_idx] += session_step_weight * p_mw

    return prof
