"""Generate all Chapter 2 figures for the thesis.

Usage:
    uv run python scripts/gen_ch2_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.figure import Figure

# ── Style ────────────────────────────────────────────────────
matplotlib.use("Agg")

COLORS = {
    "home": "#4C78A8",
    "work": "#F58518",
    "other": "#54A24B",
    "travel": "#94a3b8",
    "charge": "#E45756",
    "primary": "#4C78A8",
    "secondary": "#E45756",
    "success": "#54A24B",
    "warning": "#F58518",
    "purple": "#B279A2",
    "gray": "#64748b",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Songti SC", "STHeiti", "SimSong", "serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "figure.constrained_layout.use": True,
        }
    )


def _save(fig: Figure, outdir: Path, name: str) -> Path:
    path = outdir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {path}")
    return path


def _load_tripchain_cfg(path: str = "configs/tripchain_soc.yaml"):
    from ev_tripchain.config import load_config

    return load_config(Path(path))


def _sampling_and_soc_params_from_cfg(cfg):
    from ev_tripchain.mobility.soc import SOCEvolutionParams
    from ev_tripchain.mobility.tripchain_sampling import TripChainSamplingParams

    tc_cfg = cfg.mobility.trip_chain
    trip_params = TripChainSamplingParams(
        n_zones=int(tc_cfg.n_zones),
        other_stops_mean=float(tc_cfg.other_stops_mean),
        first_departure_mean=tc_cfg.first_departure_mean,
        first_departure_std_minutes=int(tc_cfg.first_departure_std_minutes),
        work_duration_mean_minutes=int(tc_cfg.work_duration_mean_minutes),
        work_duration_std_minutes=int(tc_cfg.work_duration_std_minutes),
        other_dwell_mean_minutes=int(tc_cfg.other_dwell_mean_minutes),
        other_dwell_std_minutes=int(tc_cfg.other_dwell_std_minutes),
        travel_minutes_per_km=float(tc_cfg.travel_minutes_per_km),
        distance_km_mean=float(tc_cfg.distance_km_mean),
        distance_km_std=float(tc_cfg.distance_km_std),
    )

    soc_cfg = cfg.mobility.soc
    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=float(soc_cfg.battery_capacity_kwh),
        consumption_kwh_per_km=float(soc_cfg.consumption_kwh_per_km),
        initial_soc_mean=float(soc_cfg.initial_soc_mean),
        initial_soc_std=float(soc_cfg.initial_soc_std),
        soc_min=float(soc_cfg.soc_min),
        soc_max=float(soc_cfg.soc_max),
        charge_power_kw=float(cfg.ev.charge_power_kw),
        charge_efficiency=float(soc_cfg.charge_efficiency),
        charge_trigger_soc=float(soc_cfg.charge_trigger_soc),
        charge_purposes=tuple(soc_cfg.charge_purposes),
        allow_initial_stop_charging=True,
        final_home_charge_enabled=bool(soc_cfg.final_home_charge_enabled),
        final_home_target_soc=float(soc_cfg.final_home_target_soc),
    )
    return trip_params, soc_params


def _extract_midwindow_profile(
    cfg,
    *,
    n_vehicles: int,
    strategy: dict[str, object] | None = None,
    warmup_days: int = 1,
    window_days: int = 2,
    lookahead_days: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    from ev_tripchain.reporting.report import _profile_total_kw

    cfg_ext = cfg.model_copy(deep=True)
    cfg_ext.time.n_days = int(warmup_days + window_days + lookahead_days)
    total_kw = _profile_total_kw(
        cfg_ext,
        strategy if strategy is not None else {"name": cfg_ext.strategy.name},
        n_vehicles=n_vehicles,
    )

    steps_per_day = int(cfg.time.n_steps)
    start = int(warmup_days) * steps_per_day
    stop = start + int(window_days) * steps_per_day
    window_kw = total_kw[start:stop]
    hours = np.arange(window_kw.shape[0]) * (cfg.time.step_minutes / 60.0)
    return hours, window_kw


def _clip_interval_hours(
    start_minute: int,
    end_minute: int,
    *,
    window_start_minute: int,
    window_end_minute: int,
) -> tuple[float, float] | None:
    start = max(int(start_minute), int(window_start_minute))
    end = min(int(end_minute), int(window_end_minute))
    if end <= start:
        return None
    offset = int(window_start_minute)
    return (start - offset) / 60.0, (end - offset) / 60.0


# ── Fig 2-1: Trip chain structure (conceptual) ──────────────

def fig_2_1_trip_chain_structure() -> Figure:
    """Conceptual timeline of a typical daily trip chain."""
    fig, ax = plt.subplots(figsize=(11, 3.0))

    segments = [
        (0, 7.5, "home", "家(H)", COLORS["home"]),
        (7.5, 8.0, "travel", "通勤", COLORS["travel"]),
        (8.0, 16.7, "work", "工作(W)", COLORS["work"]),
        (16.7, 17.2, "travel", "出行₁", COLORS["travel"]),
        (17.2, 18.0, "other", "购物(O)", COLORS["other"]),
        (18.0, 18.4, "travel", "返程", COLORS["travel"]),
        (18.4, 24.0, "home", "家(H)", COLORS["home"]),
    ]

    y0, h = 0.3, 0.4
    for t0, t1, kind, label, color in segments:
        width = t1 - t0
        rect = mpatches.FancyBboxPatch(
            (t0, y0), width, h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="white", alpha=0.85, linewidth=1.5,
        )
        ax.add_patch(rect)
        if width > 0.8:
            ax.text(
                (t0 + t1) / 2, y0 + h / 2, label,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if kind != "travel" else "#1e293b",
            )

    # Arrows between travel segments
    arrow_y = y0 + h + 0.08
    for t0, t1, kind, label, color in segments:
        if kind == "travel":
            mid = (t0 + t1) / 2
            ax.annotate(
                "", xy=(t1, arrow_y), xytext=(t0, arrow_y),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8),
            )
            ax.text(mid, arrow_y + 0.06, label, ha="center", va="bottom", fontsize=8.5, color="#475569")

    # Random variable annotations below
    annotations = [
        (3.75, "$t_{dep}$\n首次出发时刻"),
        (12.35, "$\\tau_{work}$\n工作驻留时长"),
        (17.6, "$\\tau_{other}$\n停靠时长"),
    ]
    for x, txt in annotations:
        ax.annotate(
            txt, xy=(x, y0 - 0.02), xytext=(x, y0 - 0.22),
            ha="center", va="top", fontsize=8.5, color="#475569",
            arrowprops=dict(arrowstyle="-[", color="#94a3b8", lw=1),
        )

    # Distance annotations
    dist_pairs = [(7.5, 8.0, "$d_1$"), (16.7, 17.2, "$d_2$"), (18.0, 18.4, "$d_3$")]
    for t0, t1, txt in dist_pairs:
        mid = (t0 + t1) / 2
        ax.text(mid, y0 - 0.05, txt, ha="center", va="top", fontsize=8, color="#64748b")

    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("时刻（小时）")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("典型日内出行链结构示意图")
    return fig


# ── Fig 2-2: Parameter distributions (data) ─────────────────

def fig_2_2_distributions() -> Figure:
    """4-panel histogram: departure time, daily km, work duration, stop count."""
    from ev_tripchain.mobility.tripchain_sampling import (
        TripChainSamplingParams,
        sample_daily_trip_chain,
    )

    rng = np.random.default_rng(42)
    params = TripChainSamplingParams(
        n_zones=50,
        other_stops_mean=1.0,
        first_departure_mean="07:30",
        first_departure_std_minutes=35,
        work_duration_mean_minutes=520,
        work_duration_std_minutes=45,
        other_dwell_mean_minutes=50,
        other_dwell_std_minutes=25,
        travel_minutes_per_km=1.8,
        distance_km_mean=20.0,
        distance_km_std=10.0,
    )

    n_samples = 20000
    dep_hours = np.empty(n_samples)
    daily_km = np.empty(n_samples)
    work_dur = np.empty(n_samples)
    n_other = np.empty(n_samples, dtype=int)

    for i in range(n_samples):
        tc = sample_daily_trip_chain(params, rng=rng)
        dep_hours[i] = tc.stops[0].departure_minute / 60.0
        daily_km[i] = sum(tc.leg_distance_km)
        # work duration = departure from work - arrival at work
        if len(tc.stops) >= 2:
            work_dur[i] = (tc.stops[1].departure_minute - tc.stops[1].arrival_minute) / 60.0
        else:
            work_dur[i] = 0
        # count "other" stops
        n_other[i] = sum(1 for s in tc.stops if s.purpose == "other")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.hist(dep_hours, bins=48, density=True, color=COLORS["primary"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("首次出发时间（小时）")
    ax.set_ylabel("概率密度")
    ax.set_title("(a) 首次出发时间分布")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.axvline(x=7.5, color=COLORS["secondary"], linestyle="--", linewidth=1, alpha=0.8, label="$\\mu_t$=7:30")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.hist(daily_km, bins=60, density=True, color=COLORS["warning"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("日行驶里程（km）")
    ax.set_ylabel("概率密度")
    ax.set_title("(b) 日行驶里程分布")
    ax.set_xlim(0, min(120, np.percentile(daily_km, 99.5)))
    median_km = np.median(daily_km)
    ax.axvline(x=median_km, color=COLORS["secondary"], linestyle="--", linewidth=1, alpha=0.8, label=f"中位数={median_km:.1f}km")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.hist(work_dur, bins=50, density=True, color=COLORS["success"], alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("工作驻留时长（小时）")
    ax.set_ylabel("概率密度")
    ax.set_title("(c) 工作驻留时长分布")
    ax.set_xlim(0, min(16, np.percentile(work_dur, 99.5)))
    ax.axvline(x=520/60, color=COLORS["secondary"], linestyle="--", linewidth=1, alpha=0.8, label="$\\mu_w$=8.67h")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    max_stops = int(n_other.max())
    bins_edge = np.arange(-0.5, max_stops + 1.5, 1)
    ax.hist(n_other, bins=bins_edge, density=True, color=COLORS["purple"], alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("中间停靠次数")
    ax.set_ylabel("概率")
    ax.set_title("(d) 中间停靠次数分布")
    ax.set_xticks(range(0, min(max_stops + 1, 8)))
    mean_stops = n_other.mean()
    ax.axvline(x=mean_stops, color=COLORS["secondary"], linestyle="--", linewidth=1, alpha=0.8, label=f"$\\bar{{n}}$={mean_stops:.2f}")
    ax.legend(fontsize=9)

    fig.suptitle(f"出行链关键参数概率分布（{n_samples}辆车采样）", fontsize=14, y=1.01)
    return fig


# ── Fig 2-3: Multi-day framework (conceptual) ───────────────

def fig_2_3_multiday_framework() -> Figure:
    """Conceptual diagram: two-day continuous simulation with cross-midnight charging."""
    fig, ax = plt.subplots(figsize=(11, 4.0))

    y_day1, y_day2 = 0.6, 0.15
    h = 0.2

    # Day 1 background
    ax.add_patch(mpatches.FancyBboxPatch((0, y_day1), 24, h, boxstyle="round,pad=0.02",
                 facecolor="#e2e8f0", edgecolor="#94a3b8", linewidth=1))
    ax.text(-0.5, y_day1 + h/2, "Day 1", ha="right", va="center", fontsize=11, fontweight="bold")

    # Day 2 background
    ax.add_patch(mpatches.FancyBboxPatch((24, y_day2), 24, h, boxstyle="round,pad=0.02",
                 facecolor="#e2e8f0", edgecolor="#94a3b8", linewidth=1))
    ax.text(23.5, y_day2 + h/2, "Day 2", ha="right", va="center", fontsize=11, fontweight="bold")

    # Day 1 activities (simplified)
    d1_segs = [
        (0, 7.5, COLORS["home"], "H"), (7.5, 8, COLORS["travel"], ""),
        (8, 17, COLORS["work"], "W"), (17, 17.5, COLORS["travel"], ""),
        (17.5, 18, COLORS["other"], "O"), (18, 18.5, COLORS["travel"], ""),
        (18.5, 22, COLORS["home"], "H"),
    ]
    for t0, t1, c, label in d1_segs:
        ax.add_patch(plt.Rectangle((t0, y_day1 + 0.02), t1 - t0, h - 0.04,
                     facecolor=c, alpha=0.8, edgecolor="white", linewidth=0.5))
        if label and (t1 - t0) > 1:
            ax.text((t0+t1)/2, y_day1 + h/2, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    # Cross-midnight charging block (the key feature!)
    charge_start, charge_end = 22, 30  # 22:00 day1 to 06:00 day2
    # Part on day 1 row
    ax.add_patch(plt.Rectangle((22, y_day1 + 0.02), 2, h - 0.04,
                 facecolor=COLORS["charge"], alpha=0.85, edgecolor="white", linewidth=0.5))
    ax.text(23, y_day1 + h/2, "充电", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    # Part on day 2 row
    ax.add_patch(plt.Rectangle((24, y_day2 + 0.02), 6, h - 0.04,
                 facecolor=COLORS["charge"], alpha=0.85, edgecolor="white", linewidth=0.5))
    ax.text(27, y_day2 + h/2, "充电（续）", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    # Day 2 remaining activities
    d2_segs = [
        (30, 31.5, COLORS["home"], "H"), (31.5, 32, COLORS["travel"], ""),
        (32, 41, COLORS["work"], "W"), (41, 41.5, COLORS["travel"], ""),
        (41.5, 48, COLORS["home"], "H"),
    ]
    for t0, t1, c, label in d2_segs:
        ax.add_patch(plt.Rectangle((t0, y_day2 + 0.02), t1 - t0, h - 0.04,
                     facecolor=c, alpha=0.8, edgecolor="white", linewidth=0.5))
        if label and (t1 - t0) > 1:
            ax.text((t0+t1)/2, y_day2 + h/2, label, ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    # Curved arrow connecting day1 end to day2 start
    ax.annotate(
        "", xy=(24, y_day2 + h), xytext=(24, y_day1),
        arrowprops=dict(arrowstyle="->", color=COLORS["charge"], lw=2.5,
                        connectionstyle="arc3,rad=-0.3"),
    )

    # Midnight line
    ax.axvline(x=24, color="#475569", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(24, y_day1 + h + 0.12, "午夜(24h)", ha="center", va="bottom",
            fontsize=9, color="#475569", fontstyle="italic")

    # Single-day truncation annotation
    ax.annotate(
        "单日仿真截断点", xy=(24, y_day1 + h + 0.03), xytext=(20, y_day1 + h + 0.22),
        fontsize=9, color=COLORS["secondary"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"], lw=1.2),
    )

    # Charging window annotation
    ax.annotate(
        "有序充电窗口 22:00—06:00\n多日仿真完整覆盖", xy=(26, y_day2 + h + 0.05),
        xytext=(33, y_day1 + h + 0.15),
        fontsize=9, color=COLORS["success"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=1.2),
    )

    ax.set_xlim(-2, 49)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48])
    ax.set_xticklabels(["0", "4", "8", "12", "16", "20", "24/0", "4", "8", "12", "16", "20", "24"])
    ax.set_xlabel("时刻（小时）")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("多日连续仿真框架示意图（2天）")
    return fig


# ── Fig 2-4: SOC evolution (data, 2-panel) ──────────────────

def fig_2_4_soc_evolution() -> Figure:
    """Two-panel figure: upper=SOC curve, lower=charging power (2-day)."""
    from ev_tripchain.mobility.soc import SOCEvolutionParams, simulate_soc_and_charging_profile
    from ev_tripchain.mobility.tripchain_profile import _sample_continuous_trip_chain
    from ev_tripchain.mobility.tripchain_sampling import TripChainSamplingParams

    rng = np.random.default_rng(43)
    trip_params = TripChainSamplingParams(
        n_zones=50,
        other_stops_mean=1.0,
        first_departure_mean="07:30",
        first_departure_std_minutes=35,
        work_duration_mean_minutes=520,
        work_duration_std_minutes=45,
        other_dwell_mean_minutes=50,
        other_dwell_std_minutes=25,
        travel_minutes_per_km=1.8,
        distance_km_mean=20.0,
        distance_km_std=10.0,
    )
    tc = _sample_continuous_trip_chain(
        n_days=2, trip_params=trip_params, rng=rng,
        home_zone=5, work_zone=12,
    )
    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=60.0,
        consumption_kwh_per_km=0.18,
        initial_soc_mean=0.60,
        initial_soc_std=0.0,
        charge_power_kw=7.2,
        charge_efficiency=0.92,
        charge_trigger_soc=0.3,
        charge_purposes=("home", "work"),
        allow_initial_stop_charging=True,
        final_home_charge_enabled=True,
        final_home_target_soc=0.9,
    )

    step_min = 15
    n_steps = 192  # 2 days × 96
    soc, p_kw = simulate_soc_and_charging_profile(
        tc, soc_params, step_minutes=step_min, n_steps=n_steps, rng=rng, initial_soc=0.60,
    )
    hours = np.arange(n_steps + 1) * (step_min / 60.0)
    total_hours = float(hours[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 1]})

    # Upper: SOC curve
    ax1.plot(hours, soc, color=COLORS["primary"], linewidth=2.2, label="SOC")
    ax1.set_ylabel("荷电状态（SOC）")
    ax1.set_ylim(-0.02, 1.05)
    ax1.set_xlim(0, total_hours)
    ax1.set_xticks(np.arange(0, total_hours + 1, 4))
    ax1.axhline(y=0.3, color=COLORS["secondary"], linestyle="--", linewidth=1, alpha=0.6, label="充电触发阈值 r=0.3")
    ax1.set_title("典型单车SOC演化曲线（2天连续仿真）")
    ax1.grid(True, alpha=0.2)

    # Shade charging periods on SOC plot
    p_step = np.arange(len(p_kw))
    charging_mask = p_kw > 1e-6
    in_charge = False
    charge_start = 0
    for i in range(len(p_kw)):
        if charging_mask[i] and not in_charge:
            charge_start = i
            in_charge = True
        elif not charging_mask[i] and in_charge:
            t0 = charge_start * (step_min / 60.0)
            t1 = i * (step_min / 60.0)
            ax1.axvspan(t0, t1, color=COLORS["success"], alpha=0.12)
            in_charge = False
    if in_charge:
        t0 = charge_start * (step_min / 60.0)
        t1 = len(p_kw) * (step_min / 60.0)
        ax1.axvspan(t0, t1, color=COLORS["success"], alpha=0.12)

    # Shade driving periods (SOC decreasing segments) approximately
    soc_diff = np.diff(soc)
    in_drive = False
    drive_start = 0
    for i in range(len(soc_diff)):
        if soc_diff[i] < -1e-4 and not in_drive:
            drive_start = i
            in_drive = True
        elif soc_diff[i] >= -1e-4 and in_drive:
            t0 = drive_start * (step_min / 60.0)
            t1 = i * (step_min / 60.0)
            ax1.axvspan(t0, t1, color=COLORS["warning"], alpha=0.10)
            in_drive = False
    if in_drive:
        t0 = drive_start * (step_min / 60.0)
        t1 = len(soc_diff) * (step_min / 60.0)
        ax1.axvspan(t0, t1, color=COLORS["warning"], alpha=0.10)

    # Legend
    ax1.legend(
        handles=[
            ax1.get_lines()[0],  # SOC line
            ax1.get_lines()[1],  # threshold line
            mpatches.Patch(color=COLORS["success"], alpha=0.3, label="充电时段"),
            mpatches.Patch(color=COLORS["warning"], alpha=0.3, label="行驶消耗段"),
        ],
        loc="lower left", fontsize=9,
    )

    # Midnight lines
    for boundary in np.arange(24, total_hours, 24):
        ax1.axvline(boundary, color=COLORS["gray"], linestyle=":", linewidth=0.9, alpha=0.5)
        ax2.axvline(boundary, color=COLORS["gray"], linestyle=":", linewidth=0.9, alpha=0.5)

    # Lower: charging power
    step_hours = np.arange(len(p_kw)) * (step_min / 60.0)
    ax2.fill_between(step_hours, 0, p_kw, color=COLORS["charge"], alpha=0.7, step="mid")
    ax2.set_ylabel("充电功率（kW）")
    ax2.set_xlabel("时刻（小时）")
    ax2.set_xlim(0, total_hours)
    ax2.set_xticks(np.arange(0, total_hours + 1, 4))
    ax2.set_ylim(0, max(p_kw.max() * 1.15, 1))
    ax2.grid(True, alpha=0.2)

    return fig


# ── Fig 2-5: Traffic-grid coupling (conceptual) ─────────────

def fig_2_5_coupling_mapping() -> Figure:
    """Conceptual diagram: zone-to-bus mapping with IEEE 33 topology sketch."""
    fig, (ax_zone, ax_net) = plt.subplots(1, 2, figsize=(12, 5.5),
                                           gridspec_kw={"width_ratios": [1, 1.5]})

    # Left: Functional zones
    zone_data = [
        (0.3, 0.8, "居住区\nk=1,2,3", COLORS["home"]),
        (0.7, 0.8, "居住区\nk=4,5", COLORS["home"]),
        (0.5, 0.5, "商业区\nk=6,7", COLORS["other"]),
        (0.3, 0.2, "工作区\nk=8,9,10", COLORS["work"]),
        (0.7, 0.2, "工作区\nk=11,...", COLORS["work"]),
    ]
    for x, y, label, color in zone_data:
        circle = plt.Circle((x, y), 0.12, facecolor=color, edgecolor="white",
                            alpha=0.7, linewidth=2)
        ax_zone.add_patch(circle)
        ax_zone.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold", color="white")

    ax_zone.set_xlim(0, 1)
    ax_zone.set_ylim(0, 1)
    ax_zone.set_aspect("equal")
    ax_zone.set_title("城市功能分区", fontsize=12)
    ax_zone.axis("off")

    # Right: IEEE 33 simplified topology (linear main feeder + branches)
    # Main feeder: bus 0-17 (horizontal)
    # Branch 1: bus 18-21 from bus 1
    # Branch 2: bus 22-24 from bus 2
    # Branch 3: bus 25-32 from bus 5

    bus_pos = {}
    # Main feeder (top row, horizontal)
    for i in range(18):
        bus_pos[i] = (0.05 + i * 0.052, 0.75)
    # Branch from bus 1 (downward)
    for i, b in enumerate([18, 19, 20, 21]):
        bus_pos[b] = (bus_pos[1][0], 0.75 - (i + 1) * 0.12)
    # Branch from bus 2
    for i, b in enumerate([22, 23, 24]):
        bus_pos[b] = (bus_pos[2][0] + 0.04, 0.75 - (i + 1) * 0.12)
    # Branch from bus 5
    for i, b in enumerate([25, 26, 27, 28, 29, 30, 31, 32]):
        bus_pos[b] = (bus_pos[5][0] + 0.03 + i * 0.052, 0.33)

    # Edges (simplified)
    main_edges = [(i, i+1) for i in range(17)]
    branch_edges = [
        (1, 18), (18, 19), (19, 20), (20, 21),
        (2, 22), (22, 23), (23, 24),
        (5, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),
    ]
    all_edges = main_edges + branch_edges

    for (a, b) in all_edges:
        if a in bus_pos and b in bus_pos:
            ax_net.plot([bus_pos[a][0], bus_pos[b][0]], [bus_pos[a][1], bus_pos[b][1]],
                       color="#64748b", linewidth=1.2, zorder=1)

    # EV charging station buses (from Huang Mengqi 2024)
    evcs_buses = {0, 5, 8, 12, 15, 19, 29}  # 0-indexed (bus 1,6,9,13,16,20,30)

    for b, (x, y) in bus_pos.items():
        if b in evcs_buses:
            ax_net.plot(x, y, "s", color=COLORS["success"], markersize=9, zorder=3, markeredgecolor="white", markeredgewidth=1)
        else:
            ax_net.plot(x, y, "o", color=COLORS["primary"], markersize=5, zorder=3, markeredgecolor="white", markeredgewidth=0.5)

    # Label a few key buses
    for b in [0, 5, 8, 12, 15, 17, 19, 29, 32]:
        if b in bus_pos:
            x, y = bus_pos[b]
            offset_y = 0.05 if bus_pos[b][1] > 0.5 else -0.05
            ax_net.text(x, y + offset_y, str(b + 1), ha="center", va="center", fontsize=7, color="#475569")

    ax_net.set_xlim(-0.05, 1.05)
    ax_net.set_ylim(0.05, 1.0)
    ax_net.set_title("IEEE 33节点配电网拓扑", fontsize=12)
    ax_net.axis("off")

    # Legend for charging stations
    ax_net.plot([], [], "s", color=COLORS["success"], markersize=8, label="充电站母线")
    ax_net.plot([], [], "o", color=COLORS["primary"], markersize=5, label="普通母线")
    ax_net.legend(loc="upper right", fontsize=9)

    # Mapping arrows between the two subplots (in figure coordinates)
    fig.patches.extend([
        mpatches.FancyArrowPatch(
            (0.42, 0.7), (0.52, 0.78),
            arrowstyle="->", mutation_scale=15,
            color="#94a3b8", linestyle="--", linewidth=1.5,
            transform=fig.transFigure,
        ),
        mpatches.FancyArrowPatch(
            (0.42, 0.45), (0.52, 0.55),
            arrowstyle="->", mutation_scale=15,
            color="#94a3b8", linestyle="--", linewidth=1.5,
            transform=fig.transFigure,
        ),
        mpatches.FancyArrowPatch(
            (0.42, 0.25), (0.52, 0.38),
            arrowstyle="->", mutation_scale=15,
            color="#94a3b8", linestyle="--", linewidth=1.5,
            transform=fig.transFigure,
        ),
    ])

    # φ(k)→n label
    fig.text(0.47, 0.55, "$\\varphi(k) \\to n$", ha="center", va="center",
             fontsize=13, color="#475569", fontweight="bold",
             transform=fig.transFigure)

    fig.suptitle("交通—配电网耦合映射示意图", fontsize=14, y=1.01)
    return fig


# ── Fig 2-6: Aggregate charging load (data) ─────────────────

def fig_2_6_charging_load() -> Figure:
    """Aggregate charging load using an interior 48 h window of an extended simulation."""
    cfg = _load_tripchain_cfg()
    n_vehicles = 1500
    hours, total_kw = _extract_midwindow_profile(cfg, n_vehicles=n_vehicles)
    total_hours = float(hours[-1]) + cfg.time.step_minutes / 60.0

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hours, total_kw, color=COLORS["purple"], linewidth=2.0)
    ax.fill_between(hours, 0, total_kw, color=COLORS["purple"], alpha=0.15)
    ax.set_xlim(0, total_hours)
    ax.set_xticks(np.arange(0, total_hours + 1, 4))
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("总充电功率（kW）")
    ax.set_title(f"大规模EV聚合充电负荷时序曲线（N={n_vehicles}，扩展仿真中间48h）")
    ax.grid(True, alpha=0.2)

    # Midnight line
    for boundary in np.arange(24, total_hours, 24):
        ax.axvline(boundary, color=COLORS["gray"], linestyle=":", linewidth=1.2, alpha=0.6)
        ax.text(boundary + 0.3, ax.get_ylim()[1] * 0.95, "午夜", fontsize=8, color=COLORS["gray"])

    # Peak annotation
    peak_idx = int(np.argmax(total_kw))
    peak_hour = hours[peak_idx]
    peak_val = total_kw[peak_idx]
    ax.annotate(
        f"峰值 {peak_val:.0f} kW\n({peak_hour:.1f}h)",
        xy=(peak_hour, peak_val), xytext=(peak_hour + 3, peak_val * 0.85),
        fontsize=9, color=COLORS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"], lw=1.2),
    )

    return fig


# ── Fig 2-7: Model comparison (data) ────────────────────────

def fig_2_7_model_comparison() -> Figure:
    """Trip-chain vs session model load profiles on a shared interior 48 h window."""
    cfg_tc = _load_tripchain_cfg("configs/tripchain_soc.yaml")
    cfg_sess = _load_tripchain_cfg("configs/example.yaml")
    n_vehicles = 1000
    hours, total_tc_kw = _extract_midwindow_profile(cfg_tc, n_vehicles=n_vehicles)
    _, total_sess_kw = _extract_midwindow_profile(cfg_sess, n_vehicles=n_vehicles)
    total_hours = float(hours[-1]) + cfg_tc.time.step_minutes / 60.0

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hours, total_sess_kw, color=COLORS["secondary"], linewidth=2.0, label="会话式模型")
    ax.plot(hours, total_tc_kw, color=COLORS["primary"], linewidth=2.0, label="出行链+SOC模型")
    ax.fill_between(hours, total_sess_kw, total_tc_kw, alpha=0.08, color=COLORS["gray"])
    ax.set_xlim(0, total_hours)
    ax.set_xticks(np.arange(0, total_hours + 1, 4))
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("总充电功率（kW）")
    ax.set_title(f"两种充电负荷模型对比（N={n_vehicles}，中间48h窗口）")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)

    for boundary in np.arange(24, total_hours, 24):
        ax.axvline(boundary, color=COLORS["gray"], linestyle=":", linewidth=0.9, alpha=0.5)

    # Peak annotations
    sess_peak = float(np.max(total_sess_kw))
    tc_peak = float(np.max(total_tc_kw))
    ax.text(
        0.98, 0.95,
        f"会话式峰值: {sess_peak:.0f} kW\n出行链峰值: {tc_peak:.0f} kW\n差异: {((sess_peak - tc_peak) / tc_peak * 100):.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor=COLORS["gray"], boxstyle="round,pad=0.4"),
    )

    return fig


# ── Fig 2-8: Activity Gantt chart (data) ────────────────────

def fig_2_8_activity_gantt() -> Figure:
    """Gantt-style activity timeline on an interior 48 h window using thesis config."""
    from ev_tripchain.mobility.tripchain_profile import _sample_continuous_trip_chain
    from ev_tripchain.mobility.tripchain_sampling import sample_anchor_zones
    from ev_tripchain.mobility.soc import ChargingDecision, simulate_soc_and_bus_profile

    cfg = _load_tripchain_cfg()
    params, soc_params = _sampling_and_soc_params_from_cfg(cfg)

    warmup_days = 1
    window_days = 2
    lookahead_days = 1
    total_days = warmup_days + window_days + lookahead_days
    step_minutes = int(cfg.time.step_minutes)
    n_steps_per_day = int(cfg.time.n_steps)
    total_steps = n_steps_per_day * total_days
    window_start_minute = warmup_days * int(params.day_minutes)
    window_end_minute = window_start_minute + window_days * int(params.day_minutes)
    n_display = 6

    purpose_colors = {
        "home": COLORS["home"],
        "work": COLORS["work"],
        "other": COLORS["other"],
    }

    rng = np.random.default_rng(cfg.seed)
    candidates: list[dict[str, object]] = []
    for _ in range(80):
        home_zone, work_zone = sample_anchor_zones(params, rng=rng)
        tc = _sample_continuous_trip_chain(
            n_days=total_days, trip_params=params, rng=rng,
            home_zone=home_zone, work_zone=work_zone,
        )
        charging_periods: list[tuple[float, float, str]] = []

        def decide_charge(
            *, stop_index, stop, arrival_minute, departure_minute,
            soc_at_arrival, needed_kwh, minutes_needed, rng,
            _cp=charging_periods,
        ):
            del stop_index, soc_at_arrival, needed_kwh, rng
            start = int(arrival_minute)
            end = min(int(departure_minute), start + int(minutes_needed))
            clipped = _clip_interval_hours(
                start,
                end,
                window_start_minute=window_start_minute,
                window_end_minute=window_end_minute,
            )
            if clipped is not None:
                _cp.append((clipped[0], clipped[1], stop.purpose))
            return ChargingDecision(start_minute=start, bus_col=0)

        simulate_soc_and_bus_profile(
            tc, soc_params,
            step_minutes=step_minutes, n_steps=total_steps, n_buses=1,
            charging_decision_fn=decide_charge, rng=rng,
        )
        charge_purposes = {purpose for _, _, purpose in charging_periods}
        if not charge_purposes:
            category = "none"
        elif charge_purposes == {"home"}:
            category = "home_only"
        elif charge_purposes == {"work"}:
            category = "work_only"
        else:
            category = "home_work"
        candidates.append(
            {
                "tc": tc,
                "charging_periods": list(charging_periods),
                "category": category,
            }
        )

    category_order = ["home_only", "work_only", "home_work", "none"]
    grouped = {
        category: [candidate for candidate in candidates if candidate["category"] == category]
        for category in category_order
    }
    selected: list[dict[str, object]] = []
    while len(selected) < n_display and any(grouped[cat] for cat in category_order):
        for category in category_order:
            if grouped[category] and len(selected) < n_display:
                selected.append(grouped[category].pop(0))
    if len(selected) < n_display:
        for candidate in candidates:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= n_display:
                break
    selected = selected[:n_display]

    fig, axes = plt.subplots(n_display, 1, figsize=(12, 1.5 * n_display + 1), sharex=True)

    for v_idx, candidate in enumerate(selected):
        ax = axes[v_idx]
        tc = candidate["tc"]
        charging_periods = candidate["charging_periods"]

        for stop in tc.stops:
            clipped = _clip_interval_hours(
                stop.arrival_minute,
                stop.departure_minute,
                window_start_minute=window_start_minute,
                window_end_minute=window_end_minute,
            )
            if clipped is None:
                continue
            t0_h, t1_h = clipped
            color = purpose_colors.get(stop.purpose, COLORS["gray"])
            ax.barh(0, t1_h - t0_h, left=t0_h, height=0.6,
                    color=color, edgecolor="white", linewidth=0.3, alpha=0.8)

        for i in range(len(tc.stops) - 1):
            clipped = _clip_interval_hours(
                tc.stops[i].departure_minute,
                tc.stops[i + 1].arrival_minute,
                window_start_minute=window_start_minute,
                window_end_minute=window_end_minute,
            )
            if clipped is None:
                continue
            t0_h, t1_h = clipped
            if t1_h > t0_h + 0.01:
                ax.barh(0, t1_h - t0_h, left=t0_h, height=0.6,
                        color=COLORS["travel"], edgecolor="white", linewidth=0.3, alpha=0.6)

        for (c0, c1, _purpose) in charging_periods:
            ax.barh(0, c1 - c0, left=c0, height=0.6,
                    color=COLORS["charge"], alpha=0.55, edgecolor=COLORS["charge"],
                    linewidth=1.5)

        ax.set_yticks([0])
        ax.set_yticklabels([f"车辆{v_idx + 1}"], fontsize=9)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, 48)
        ax.axvline(24, color=COLORS["gray"], linestyle=":", linewidth=0.8, alpha=0.5)

        if v_idx == 0:
            ax.set_title("多辆车日内出行—充电时空轨迹（中间48h窗口，按配置采样）")

    axes[-1].set_xticks(np.arange(0, 49, 4))
    axes[-1].set_xlabel("时刻（小时）")

    legend_patches = [
        mpatches.Patch(color=COLORS["home"], label="家庭(H)", alpha=0.8),
        mpatches.Patch(color=COLORS["work"], label="工作(W)", alpha=0.8),
        mpatches.Patch(color=COLORS["other"], label="其他(O)", alpha=0.8),
        mpatches.Patch(color=COLORS["travel"], label="行驶", alpha=0.6),
        mpatches.Patch(color=COLORS["charge"], label="充电", alpha=0.55),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncols=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    return fig


# ── Fig 2-9: Spatial distribution (data) ────────────────────

def fig_2_9_spatial_distribution() -> Figure:
    """Per-bus peak charging power bar chart."""
    from ev_tripchain.config import load_config
    from ev_tripchain.grid.cases import load_case
    from ev_tripchain.hosting_capacity.common import ensure_ev_load_elements
    from ev_tripchain.mobility.profile import build_ev_profile_mw

    cfg = load_config(Path("configs/tripchain_soc.yaml"))
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()

    rng = np.random.default_rng(42)
    prof = build_ev_profile_mw(
        cfg=cfg, n_vehicles=1500, buses=buses, n_buses=len(ev_idx), rng=rng,
    )
    # prof shape: (total_steps, n_buses) in MW
    peak_kw = prof.max(axis=0) * 1000  # MW -> kW
    mean_kw = prof.mean(axis=0) * 1000

    bus_labels = [str(int(b)) for b in buses]
    n_buses = len(buses)
    x = np.arange(n_buses)

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_width = 0.4
    bars_peak = ax.bar(x - bar_width/2, peak_kw, bar_width,
                       color=COLORS["secondary"], alpha=0.8, label="峰值充电功率", edgecolor="white")
    bars_mean = ax.bar(x + bar_width/2, mean_kw, bar_width,
                       color=COLORS["primary"], alpha=0.8, label="平均充电功率", edgecolor="white")

    ax.set_xlabel("母线编号")
    ax.set_ylabel("充电功率（kW）")
    ax.set_title("充电负荷母线级空间分布（N=1500）")
    ax.set_xticks(x)
    ax.set_xticklabels(bus_labels, fontsize=8, rotation=45)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.2)

    # Highlight top 3 buses
    top3_idx = np.argsort(peak_kw)[-3:]
    for idx in top3_idx:
        ax.annotate(
            f"{peak_kw[idx]:.0f}kW",
            xy=(idx - bar_width/2, peak_kw[idx]),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=8, fontweight="bold", color=COLORS["secondary"],
        )

    return fig


# ── Main ─────────────────────────────────────────────────────

FIGURE_MAP = {
    "fig_2_1": ("图2-1 典型出行链结构示意图", fig_2_1_trip_chain_structure),
    "fig_2_2": ("图2-2 出行链关键参数概率分布", fig_2_2_distributions),
    "fig_2_3": ("图2-3 多日连续仿真框架示意图", fig_2_3_multiday_framework),
    "fig_2_4": ("图2-4 典型单车SOC演化曲线", fig_2_4_soc_evolution),
    "fig_2_5": ("图2-5 交通-配电网耦合映射示意图", fig_2_5_coupling_mapping),
    "fig_2_6": ("图2-6 大规模EV聚合充电负荷时序曲线", fig_2_6_charging_load),
    "fig_2_7": ("图2-7 两种充电负荷模型对比", fig_2_7_model_comparison),
    "fig_2_8": ("图2-8 多辆车日内出行-充电时空轨迹", fig_2_8_activity_gantt),
    "fig_2_9": ("图2-9 充电负荷母线级空间分布", fig_2_9_spatial_distribution),
}


def main() -> None:
    _apply_style()
    outdir = Path("output/ch2_figures")
    outdir.mkdir(parents=True, exist_ok=True)

    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "--only":
        only = set(sys.argv[2].split(","))

    total = 0
    for key, (desc, func) in FIGURE_MAP.items():
        if only and key not in only:
            continue
        print(f"[{key}] {desc} ...")
        try:
            fig = func()
            _save(fig, outdir, key)
            total += 1
        except Exception as e:
            print(f"  !! Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone: {total}/{len(FIGURE_MAP) if not only else len(only)} figures -> {outdir}/")


if __name__ == "__main__":
    main()
