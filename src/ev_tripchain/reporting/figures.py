"""Data-driven figure generators for thesis.

Every function takes pre-computed data as input and returns a matplotlib Figure.
No computation, no I/O — pure visualization.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ev_tripchain.reporting.style import COLORS, METHOD_LABELS, STRATEGY_LABELS


# ──────────────────────────────────────────────────────────
# Fig 1: Input distribution histograms
# ──────────────────────────────────────────────────────────
def fig_input_distributions(
    dep_hours: np.ndarray,
    daily_km: np.ndarray,
) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax0, ax1 = axes

    ax0.hist(dep_hours, bins=48, density=True, color=COLORS["primary"], alpha=0.85)
    ax0.set_xlabel("出发时间（小时）")
    ax0.set_ylabel("概率密度")
    ax0.set_title("首次出发时间分布")
    ax0.set_xlim(0, 24)
    ax0.set_xticks(np.arange(0, 25, 4))

    ax1.hist(daily_km, bins=50, density=True, color=COLORS["warning"], alpha=0.85)
    ax1.set_xlabel("日行驶里程（km）")
    ax1.set_ylabel("概率密度")
    ax1.set_title("日行驶里程分布")

    return fig


# ──────────────────────────────────────────────────────────
# Fig 2: Single vehicle SOC evolution
# ──────────────────────────────────────────────────────────
def fig_soc_evolution(
    hours: np.ndarray,
    soc: np.ndarray,
    p_kw: np.ndarray,
    step_minutes: int = 15,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    ax.plot(hours, soc, color=COLORS["success"], linewidth=2.2)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("荷电状态（SOC）")
    ax.set_title("单车24小时SOC演化曲线")

    charging_steps = np.where(p_kw > 1e-9)[0]
    if charging_steps.size:
        t0 = charging_steps.min() * (step_minutes / 60.0)
        t1 = (charging_steps.max() + 1) * (step_minutes / 60.0)
        ax.axvspan(t0, t1, color=COLORS["success"], alpha=0.10, label="充电时段")
        ax.legend(loc="lower right")

    return fig


# ──────────────────────────────────────────────────────────
# Fig 3: Aggregate charging load curve
# ──────────────────────────────────────────────────────────
def fig_charging_load(
    hours: np.ndarray,
    total_kw: np.ndarray,
    n_vehicles: int,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.2))

    ax.plot(hours, total_kw, color=COLORS["purple"], linewidth=2.2)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("总充电功率（kW）")
    ax.set_title(f"无序充电总负荷曲线，N={n_vehicles}")

    return fig


# ──────────────────────────────────────────────────────────
# Fig 4: Risk curve (N vs violation probability)
# ──────────────────────────────────────────────────────────
def fig_risk_curve(
    risk_points: list[dict[str, Any]],
    n_star: int,
    risk_tolerance: float,
) -> Figure:
    """Plot risk curve with CI bands.

    risk_points: list of dicts with keys {n, p_hat, ci95_low, ci95_high}.
    """
    pts = sorted(risk_points, key=lambda x: x["n"])
    ns = [p["n"] for p in pts]
    p_hats = [p["p_hat"] for p in pts]
    ci_lo = [p["ci95_low"] for p in pts]
    ci_hi = [p["ci95_high"] for p in pts]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.fill_between(ns, ci_lo, ci_hi, color=COLORS["secondary"], alpha=0.15, label="95% CI")
    ax.plot(ns, p_hats, "o-", color=COLORS["secondary"], linewidth=2, markersize=5, label="$\\hat{\\pi}(N)$")
    ax.axhline(y=risk_tolerance, color=COLORS["primary"], linestyle="--", linewidth=1.5, label=f"$\\varepsilon$ = {risk_tolerance}")
    if n_star > 0:
        ax.axvline(x=n_star, color=COLORS["success"], linestyle=":", linewidth=1.5, label=f"$N^*$ = {n_star}")

    ax.set_xlabel("接入电动汽车数量（N）")
    ax.set_ylabel("违规概率 $\\hat{\\pi}(N)$")
    ax.set_title("风险曲线：电动汽车概率承载力")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    return fig


# ──────────────────────────────────────────────────────────
# Fig 5: Bus voltage profile (24h, all buses)
# ──────────────────────────────────────────────────────────
def fig_bus_voltage_profile(
    hours: np.ndarray,
    all_vm: np.ndarray,
    n_vehicles: int,
    vmin: float = 0.95,
    vmax: float = 1.05,
) -> Figure:
    """all_vm: shape (n_steps, n_buses)."""
    fig, ax = plt.subplots(figsize=(10.5, 5.0))

    for b in range(all_vm.shape[1]):
        ax.plot(hours, all_vm[:, b], linewidth=0.8, alpha=0.7)

    ax.axhline(y=vmin, color="red", linestyle="--", linewidth=1.2, label=f"$V_{{min}}$ = {vmin} p.u.")
    ax.axhline(y=vmax, color="red", linestyle="--", linewidth=1.2, label=f"$V_{{max}}$ = {vmax} p.u.")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("电压（p.u.）")
    ax.set_title(f"各母线24小时电压剖面，N={n_vehicles}")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    return fig


# ──────────────────────────────────────────────────────────
# Fig 6: Model comparison (session vs tripchain)
# ──────────────────────────────────────────────────────────
def fig_model_comparison(
    hours: np.ndarray,
    total_sess_kw: np.ndarray,
    total_tc_kw: np.ndarray,
    n_vehicles: int,
) -> Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    ax.plot(hours, total_sess_kw, color=COLORS["primary"], linewidth=2.2, label="会话式模型")
    ax.plot(hours, total_tc_kw, color=COLORS["secondary"], linewidth=2.2, label="出行链+SOC模型")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("总充电功率（kW）")
    ax.set_title(f"两种充电负荷模型对比，N={n_vehicles}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# ──────────────────────────────────────────────────────────
# Fig 7: Ordered charging — with/without random delay
# ──────────────────────────────────────────────────────────
def fig_ordered_delay(
    hours: np.ndarray,
    p_uncontrolled: np.ndarray,
    p_no_delay: np.ndarray,
    p_with_delay: np.ndarray,
    n_vehicles: int,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(hours, p_uncontrolled, color=COLORS["gray"], linewidth=1.2, label="无序充电")
    ax.plot(hours, p_no_delay, color=COLORS["ordered_no_delay"], linewidth=1.6, label="有序充电（无延迟）")
    ax.plot(hours, p_with_delay, color=COLORS["ordered_delay"], linewidth=1.6, label="有序充电（随机延迟）")
    ax.axvspan(22, 24, alpha=0.08, color="blue", label="充电窗口")
    ax.axvspan(0, 6, alpha=0.08, color="blue")
    ax.set_xlabel("时刻 (h)")
    ax.set_ylabel("聚合充电功率 (kW)")
    ax.set_title(f"有序充电随机延迟效果对比（{n_vehicles}辆EV）")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig


# ──────────────────────────────────────────────────────────
# Fig 8: Strategy comparison (N* bar chart)
# ──────────────────────────────────────────────────────────
def fig_strategy_comparison(
    tc_results: dict[str, int],
    sess_results: dict[str, int],
    case_label: str = "IEEE 33",
    load_scale: float = 0.55,
    charge_kw: float = 7.2,
) -> Figure:
    """tc_results / sess_results: {strategy_key: n_star}."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    strategy_colors = [
        COLORS["uncontrolled"], COLORS["ordered_no_delay"], COLORS["ordered_delay"],
        COLORS["nearest"], COLORS["navigation_static"], COLORS["navigation_dynamic"],
    ]

    # Trip-chain
    labels_tc = [STRATEGY_LABELS.get(k, k) for k in tc_results]
    vals_tc = list(tc_results.values())
    colors_tc = strategy_colors[: len(vals_tc)]
    bars1 = ax1.bar(labels_tc, vals_tc, color=colors_tc, edgecolor="white", width=0.65)
    ax1.bar_label(bars1, fontsize=10, fontweight="bold", padding=3)
    ax1.set_ylabel("$N^*$（最大EV数量）")
    ax1.set_title("出行链 + SOC 模型")
    ax1.set_ylim(0, max(vals_tc) * 1.18 if vals_tc else 10)
    ax1.grid(axis="y", alpha=0.3)
    if vals_tc:
        ax1.axhline(y=vals_tc[0], color=COLORS["gray"], linestyle="--", alpha=0.4, linewidth=0.8)

    # Session
    labels_s = [STRATEGY_LABELS.get(k, k) for k in sess_results]
    vals_s = list(sess_results.values())
    colors_s = strategy_colors[: len(vals_s)]
    bars2 = ax2.bar(labels_s, vals_s, color=colors_s, edgecolor="white", width=0.55)
    ax2.bar_label(bars2, fontsize=10, fontweight="bold", padding=3)
    ax2.set_ylabel("$N^*$（最大EV数量）")
    ax2.set_title("会话模型")
    ax2.set_ylim(0, max(vals_s) * 1.22 if vals_s else 10)
    ax2.grid(axis="y", alpha=0.3)
    if vals_s:
        ax2.axhline(y=vals_s[0], color=COLORS["gray"], linestyle="--", alpha=0.4, linewidth=0.8)

    fig.suptitle(
        f"充电策略对比（{case_label}, $\\lambda$={load_scale}, $P_{{ch}}$={charge_kw} kW）",
        fontsize=13,
        y=1.02,
    )
    return fig


# ──────────────────────────────────────────────────────────
# Fig 9: Method comparison (deterministic / sensitivity / MC)
# ──────────────────────────────────────────────────────────
def fig_method_comparison(
    method_results: dict[str, int],
    method_times: dict[str, float] | None = None,
) -> Figure:
    """method_results: {method_key: n_star}, method_times: {method_key: seconds}."""
    method_colors = [COLORS.get(k, COLORS["gray"]) for k in method_results]
    labels = [METHOD_LABELS.get(k, k) for k in method_results]
    n_stars = list(method_results.values())

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    bars = ax1.bar(labels, n_stars, color=method_colors, edgecolor="white", width=0.55)
    ax1.bar_label(bars, fontsize=11, fontweight="bold", padding=3)
    ax1.set_ylabel("$N^*$（最大EV数量）")
    ax1.set_title("承载力评估方法对比")
    ax1.set_ylim(0, max(n_stars) * 1.25 if n_stars else 10)
    ax1.grid(axis="y", alpha=0.3)

    if method_times:
        ax2 = ax1.twinx()
        times = [method_times.get(k, 0) for k in method_results]
        ax2.plot(labels, times, "ko--", markersize=6, linewidth=1.2, label="计算时间")
        ax2.set_ylabel("计算时间 (s)")
        ax2.set_ylim(0, max(times) * 1.4 if times else 1)
        ax2.legend(loc="center right", fontsize=9)

    return fig


# ──────────────────────────────────────────────────────────
# Fig 10: Voltage sensitivity (margin + diagonal)
# ──────────────────────────────────────────────────────────
def fig_voltage_sensitivity(
    bus_ids: np.ndarray,
    voltage_margin: np.ndarray,
    sensitivity_diagonal: np.ndarray,
    case_label: str = "IEEE 33",
    load_scale: float = 0.55,
) -> Figure:
    n = len(bus_ids)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

    ax1.bar(range(n), voltage_margin * 1000, color=COLORS["primary"], alpha=0.8, width=0.7)
    ax1.set_ylabel("电压裕度\n($V_{base} - V_{min}$) [mpu]")
    ax1.set_title(f"各母线基线电压裕度与灵敏度分布（{case_label}, $\\lambda$={load_scale}）")
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(y=0, color="red", linewidth=0.8)

    ax2.bar(range(n), sensitivity_diagonal * 1000, color=COLORS["secondary"], alpha=0.8, width=0.7)
    ax2.set_ylabel("$dV_i/dP_i$\n[mpu/MW]")
    ax2.set_xlabel("母线编号")
    labels = [str(b) for b in bus_ids]
    ax2.set_xticks(range(0, n, 2))
    ax2.set_xticklabels([labels[i] for i in range(0, n, 2)], fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    return fig


# ──────────────────────────────────────────────────────────
# Fig 11: Parameter sweep heatmap
# ──────────────────────────────────────────────────────────
def fig_parameter_sweep(
    load_scales: list[float],
    charge_powers: list[float],
    n_star_grid: np.ndarray,
) -> Figure:
    """n_star_grid: shape (len(load_scales), len(charge_powers))."""
    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(n_star_grid, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(charge_powers)))
    ax.set_xticklabels([f"{p:.1f}" for p in charge_powers])
    ax.set_yticks(range(len(load_scales)))
    ax.set_yticklabels([f"{s:.2f}" for s in load_scales])
    ax.set_xlabel("充电功率 (kW)")
    ax.set_ylabel("基础负荷比例 $\\lambda$")
    ax.set_title("参数敏感性：$N^*$ 随充电功率与负荷比例变化")

    for i in range(len(load_scales)):
        for j in range(len(charge_powers)):
            ax.text(j, i, str(int(n_star_grid[i, j])), ha="center", va="center", fontweight="bold", fontsize=11)

    fig.colorbar(im, ax=ax, label="$N^*$")
    return fig
