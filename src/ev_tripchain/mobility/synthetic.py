from __future__ import annotations

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.mobility.spatial import (
    build_spatial_distance_model,
    choose_spatial_target_bus,
)


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
    start_minute: np.ndarray, *, window_start: int, window_end: int
) -> np.ndarray:
    """
    Shift start times into a charging window. Supports windows that cross midnight.
    """
    start = start_minute.copy()
    if window_start <= window_end:
        in_window = (start >= window_start) & (start < window_end)
        start[~in_window] = window_start
        return start

    # crossing midnight, e.g., 22:00-06:00
    in_window = (start >= window_start) | (start < window_end)
    start[~in_window] = window_start
    return start


def build_ev_profile_mw(
    *,
    cfg: ProjectConfig,
    n_vehicles: int,
    buses: np.ndarray,
    n_buses: int,
    bus_score: np.ndarray | None = None,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Build aggregated EV charging profile for a single scenario.

    Returns array of shape (T, n_buses) in MW, aligned with `buses` ordering.
    """
    step = cfg.time.step_minutes
    t_steps = cfg.time.n_steps
    minutes_per_day = step * t_steps

    if n_vehicles <= 0:
        return np.zeros((t_steps, n_buses), dtype=float)

    # sessions per vehicle ~ Poisson(mean)
    lam = max(cfg.ev.sessions_per_vehicle_mean, 0.0)
    sessions = rng.poisson(lam, size=n_vehicles)
    total_sessions = int(sessions.sum())
    if total_sessions == 0:
        return np.zeros((t_steps, n_buses), dtype=float)

    # assign each session to a "home" bus index (uniform for now)
    home_bus_idx = rng.integers(0, n_buses, size=total_sessions)

    start_min = _sample_start_minutes(cfg, size=total_sessions, rng=rng)
    dur_min = rng.normal(
        cfg.ev.duration_minutes_mean,
        cfg.ev.duration_minutes_std,
        size=total_sessions,
    )
    dur_min = np.clip(dur_min, step, minutes_per_day).round().astype(int)

    if cfg.strategy.name == "ordered":
        ws = _parse_hhmm_to_minutes(cfg.strategy.ordered.window_start)
        we = _parse_hhmm_to_minutes(cfg.strategy.ordered.window_end)
        start_min = _apply_ordered_window(start_min, window_start=ws, window_end=we)

    target_bus_idx = home_bus_idx.copy()
    spatial_model = None
    if cfg.strategy.name in {"nearest", "navigation"}:
        spatial_model = build_spatial_distance_model(buses=buses, n_buses=int(n_buses))
        for i in range(total_sessions):
            target_bus_idx[i] = choose_spatial_target_bus(
                src_bus_col=int(home_bus_idx[i]),
                strategy_name=cfg.strategy.name,
                dist_m=spatial_model.dist_m,
                candidate_bus_idx=spatial_model.candidate_bus_idx,
                navigation_candidate_k=int(cfg.strategy.navigation.candidate_k),
                navigation_distance_limit_m=cfg.strategy.navigation.distance_limit_m,
                navigation_distance_beta=float(cfg.strategy.navigation.distance_beta),
                candidate_bus_score=bus_score,
                rng=rng,
            )

    # build time series
    p_kw = float(cfg.ev.charge_power_kw)
    p_mw = p_kw / 1000.0
    prof = np.zeros((t_steps, n_buses), dtype=float)

    start_step = (start_min // step).astype(int)
    dur_steps = np.maximum(1, (dur_min + step - 1) // step)

    for s in range(total_sessions):
        b = int(target_bus_idx[s])
        t0 = int(start_step[s])
        dt = int(dur_steps[s])
        for tt in range(t0, min(t0 + dt, t_steps)):
            prof[tt, b] += p_mw

    return prof
