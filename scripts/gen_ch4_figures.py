"""Generate all data figures for thesis Chapter 4.

Usage:
    uv run python scripts/gen_ch4_figures.py              # all figures
    uv run python scripts/gen_ch4_figures.py 3 4 5        # lightweight only
    uv run python scripts/gen_ch4_figures.py 6 7 8        # heavy (MC) only
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.common import (
    compute_static_voltage_margin_score,
    ensure_ev_load_elements,
)
from ev_tripchain.hosting_capacity.sensitivity import (
    VoltageSensitivityModel,
    build_voltage_sensitivity_model,
)
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.reporting.style import COLORS, apply_style

OUT_DIR = Path("output/ch4_figures")
CFG_TC_PATH = Path("configs/tripchain_soc.yaml")
CFG_SESS_PATH = Path("configs/example.yaml")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    _log(f"  -> {path}")
    return path


def _set_panel_xlabel(
    ax: plt.Axes,
    xlabel: str,
    panel_caption: str,
    *,
    labelpad: float = 8,
) -> None:
    """Place the panel label below the x-axis without excessive subplot spacing."""
    label = f"{xlabel}\n{panel_caption}" if xlabel else panel_caption
    ax.set_xlabel(label, labelpad=labelpad)


def _save_json(data: dict, name: str) -> Path:
    path = OUT_DIR / f"{name}.json"

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_default) + "\n",
        encoding="utf-8",
    )
    _log(f"  -> {path}")
    return path


def _with_overrides(cfg: ProjectConfig, updates: dict) -> ProjectConfig:
    def _merge(base: dict, upd: dict) -> dict:
        merged = dict(base)
        for k, v in upd.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    return ProjectConfig.model_validate(_merge(cfg.model_dump(), updates))


def _load_case_context(cfg: ProjectConfig):
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    return net, ev_idx, buses


def _compute_nav_context(cfg: ProjectConfig, net, ev_idx, buses):
    """Compute bus_score and voltage model for navigation strategy."""
    hard_cfg = cfg.constraints.hard
    bus_score = compute_static_voltage_margin_score(
        net, buses=buses,
        vmin=float(hard_cfg.vmin_pu),
        vmax=float(hard_cfg.vmax_pu),
    )
    voltage_model: VoltageSensitivityModel | None = None
    if cfg.strategy.name == "navigation" and cfg.strategy.navigation.dynamic_scoring:
        try:
            voltage_model = build_voltage_sensitivity_model(
                net, ev_idx=ev_idx, buses=buses,
                vmin=float(hard_cfg.vmin_pu),
                vmax=float(hard_cfg.vmax_pu),
                line_max=float(hard_cfg.line_loading_max_percent),
            )
        except Exception:
            voltage_model = None
    return bus_score, voltage_model


def _build_profile(cfg: ProjectConfig, strategy: dict, n_vehicles: int, seed_offset: int = 0):
    """Build bus-level EV profile (T, n_buses) in MW."""
    cfg_mod = _with_overrides(cfg, {"strategy": strategy})
    net, ev_idx, buses = _load_case_context(cfg_mod)

    # Compute navigation context if needed
    bus_score = None
    voltage_model = None
    if strategy.get("name") in ("navigation", "nearest"):
        bus_score, voltage_model = _compute_nav_context(cfg_mod, net, ev_idx, buses)

    rng = np.random.default_rng(cfg.seed + seed_offset)
    prof = build_ev_profile_mw(
        cfg=cfg_mod,
        n_vehicles=n_vehicles,
        buses=buses,
        n_buses=len(ev_idx),
        bus_score=bus_score,
        navigation_voltage_model=voltage_model,
        rng=rng,
    )
    return prof, buses


# ── Fig 4-3: Ordered charging delay comparison ────────────────────


def gen_fig_4_3(cfg: ProjectConfig) -> dict:
    _log("[Fig 4-3] Ordered charging delay comparison ...")
    n_vehicles = 500

    p_unc, _ = _build_profile(cfg, {"name": "uncontrolled"}, n_vehicles)
    p_no, _ = _build_profile(cfg, {"name": "ordered", "ordered": {"random_delay": False}}, n_vehicles)
    p_yes, _ = _build_profile(cfg, {"name": "ordered", "ordered": {"random_delay": True}}, n_vehicles)

    # Convert to total kW
    total_unc = p_unc.sum(axis=1) * 1000
    total_no = p_no.sum(axis=1) * 1000
    total_yes = p_yes.sum(axis=1) * 1000

    step_min = cfg.time.step_minutes
    hours = np.arange(total_unc.shape[0]) * (step_min / 60.0)
    total_hours = hours[-1] + step_min / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.2))

    # Upper panel: full 2-day view
    for arr, label, color, ls in [
        (total_unc, "无序充电", COLORS["uncontrolled"], "-"),
        (total_no, "有序充电（无延迟）", COLORS["ordered_no_delay"], "--"),
        (total_yes, "有序充电（随机延迟）", COLORS["ordered_delay"], "-"),
    ]:
        ax1.plot(hours, arr, color=color, linestyle=ls, linewidth=1.8, label=label)

    ax1.axvline(24.0, color=COLORS["gray"], linestyle=":", linewidth=0.9, alpha=0.5)
    # Shade charging window (22:00-06:00 on each day)
    for day in range(cfg.time.n_days):
        ws = day * 24 + 22
        we = day * 24 + 30  # 06:00 next day
        if we > total_hours:
            we = total_hours
        ax1.axvspan(ws, we, alpha=0.08, color=COLORS["primary"])

    ax1.set_xlim(0, total_hours)
    ax1.set_xticks(np.arange(0, total_hours + 1e-9, 4))
    _set_panel_xlabel(ax1, "时刻（小时）", "(a) 两日完整充电功率时序")
    ax1.set_ylabel("总充电功率（kW）")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Lower panel: overnight window zoom (Day1 22:00 to Day2 06:00)
    zoom_start, zoom_end = 22.0, 30.0
    mask = (hours >= zoom_start) & (hours < zoom_end)
    hours_zoom = hours[mask]

    for arr, label, color, ls in [
        (total_unc, "无序充电", COLORS["uncontrolled"], "-"),
        (total_no, "有序充电（无延迟）", COLORS["ordered_no_delay"], "--"),
        (total_yes, "有序充电（随机延迟）", COLORS["ordered_delay"], "-"),
    ]:
        ax2.plot(hours_zoom, arr[mask], color=color, linestyle=ls, linewidth=1.8, label=label)

    ax2.axvline(24.0, color=COLORS["gray"], linestyle=":", linewidth=0.9, alpha=0.5,
                label="午夜分界")
    ax2.axvspan(zoom_start, zoom_end, alpha=0.08, color=COLORS["primary"])
    ax2.set_xlim(zoom_start, zoom_end)
    ax2.set_xticks(np.arange(zoom_start, zoom_end + 1e-9, 1))
    xlabels = [f"{int(h % 24):02d}:00" for h in np.arange(zoom_start, zoom_end + 1e-9, 1)]
    ax2.set_xticklabels(xlabels, fontsize=8)
    _set_panel_xlabel(ax2, "时刻", "(b) 跨午夜充电窗口放大视图")
    ax2.set_ylabel("总充电功率（kW）")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"有序充电随机延迟效果对比（出行链模型，{n_vehicles}辆车，2天连续仿真）",
        fontsize=13,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.08)

    _save(fig, "fig_4_3")

    summary = {
        "n_vehicles": n_vehicles,
        "peak_uncontrolled_kw": float(np.max(total_unc)),
        "peak_ordered_no_delay_kw": float(np.max(total_no)),
        "peak_ordered_with_delay_kw": float(np.max(total_yes)),
        "energy_uncontrolled_kwh": float(np.sum(total_unc) * step_min / 60.0),
        "energy_ordered_no_delay_kwh": float(np.sum(total_no) * step_min / 60.0),
        "energy_ordered_with_delay_kwh": float(np.sum(total_yes) * step_min / 60.0),
    }
    _save_json(summary, "fig_4_3_data")
    return summary


# ── Fig 4-4: Navigation bus-level load distribution ──────────────


def gen_fig_4_4(cfg: ProjectConfig) -> dict:
    _log("[Fig 4-4] Navigation bus-level load distribution ...")
    n_vehicles = 500

    prof_unc, buses = _build_profile(cfg, {"name": "uncontrolled"}, n_vehicles)
    prof_nav, _ = _build_profile(
        cfg,
        {"name": "navigation", "navigation": {"dynamic_scoring": True}},
        n_vehicles,
    )

    # Peak load at each bus (kW)
    peak_unc = np.max(prof_unc, axis=0) * 1000
    peak_nav = np.max(prof_nav, axis=0) * 1000

    n_bus = len(buses)
    x = np.arange(n_bus)
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    bars1 = ax.bar(x - width / 2, peak_unc, width, color=COLORS["uncontrolled"],
                   alpha=0.85, label="无序充电")
    bars2 = ax.bar(x + width / 2, peak_nav, width, color=COLORS["success"],
                   alpha=0.85, label="导航策略")

    ax.set_xlabel("母线编号")
    ax.set_ylabel("峰值充电功率（kW）")
    ax.set_title(f"导航策略前后母线级峰值充电负荷空间分布（{n_vehicles}辆车）")
    labels = [str(b) for b in buses]
    ax.set_xticks(x[::2])
    ax.set_xticklabels([labels[i] for i in range(0, n_bus, 2)], fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate the highest uncontrolled bus
    max_unc_idx = int(np.argmax(peak_unc))
    ax.annotate(
        f"Bus {buses[max_unc_idx]}\n{peak_unc[max_unc_idx]:.0f} kW",
        xy=(max_unc_idx - width / 2, peak_unc[max_unc_idx]),
        xytext=(max_unc_idx + 3, peak_unc[max_unc_idx] * 0.95),
        fontsize=9, color=COLORS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
    )

    _save(fig, "fig_4_4")

    summary = {
        "n_vehicles": n_vehicles,
        "bus_ids": buses.tolist(),
        "peak_uncontrolled_kw": peak_unc.tolist(),
        "peak_navigation_kw": peak_nav.tolist(),
        "max_peak_unc_bus": int(buses[max_unc_idx]),
        "max_peak_unc_kw": float(peak_unc[max_unc_idx]),
        "max_peak_nav_kw": float(peak_nav[int(np.argmax(peak_nav))]),
        "std_unc": float(np.std(peak_unc)),
        "std_nav": float(np.std(peak_nav)),
    }
    _save_json(summary, "fig_4_4_data")
    return summary


# ── Fig 4-5: Navigation voltage improvement ─────────────────────


def gen_fig_4_5(cfg: ProjectConfig, n_vehicles: int = 130, seed_offset: int = 208) -> dict:
    _log(f"[Fig 4-5] Navigation voltage profile at critical moment (N={n_vehicles}) ...")

    strategies = {
        "uncontrolled": {"name": "uncontrolled"},
        "navigation": {"name": "navigation", "navigation": {"dynamic_scoring": True}},
    }

    # Store per-strategy: full voltage timeseries and the profile for the critical timestep
    vm_timeseries: dict[str, np.ndarray] = {}
    profiles: dict[str, np.ndarray] = {}

    for key, strategy in strategies.items():
        cfg_mod = _with_overrides(cfg, {"strategy": strategy})
        net, ev_idx, buses = _load_case_context(cfg_mod)

        bus_score = None
        voltage_model = None
        if strategy.get("name") in ("navigation", "nearest"):
            bus_score, voltage_model = _compute_nav_context(cfg_mod, net, ev_idx, buses)

        rng = np.random.default_rng(cfg.seed + seed_offset)
        prof = build_ev_profile_mw(
            cfg=cfg_mod,
            n_vehicles=n_vehicles,
            buses=buses,
            n_buses=len(ev_idx),
            bus_score=bus_score,
            navigation_voltage_model=voltage_model,
            rng=rng,
        )

        n_steps = prof.shape[0]
        n_bus_total = len(net.bus)
        vm_all = np.ones((n_steps, n_bus_total), dtype=float)
        pf_init = "auto"
        for t in range(n_steps):
            net.load.loc[ev_idx, "p_mw"] = prof[t, :]
            try:
                run_powerflow(net, init=pf_init)
                pf_init = "results"
            except Exception:
                pf_init = "auto"
            vm_all[t, :] = net.res_bus.vm_pu.to_numpy()

        vm_timeseries[key] = vm_all
        profiles[key] = prof

    # Find critical timestep: the moment when uncontrolled voltage is at its global minimum
    vm_unc_all = vm_timeseries["uncontrolled"]
    # Global min voltage across all buses and timesteps for uncontrolled
    min_per_step = np.min(vm_unc_all, axis=1)  # min voltage at each timestep
    critical_step = int(np.argmin(min_per_step))

    step_minutes = cfg.time.step_minutes
    critical_hour = (critical_step * step_minutes) / 60.0
    critical_hh = int(critical_hour) % 24
    critical_mm = int((critical_hour % 1) * 60)
    # For multi-day, show day + time
    critical_day = int(critical_hour) // 24 + 1
    time_label = f"第{critical_day}天 {critical_hh:02d}:{critical_mm:02d}"

    # Extract voltage profiles at the critical timestep (exclude slack bus 0)
    vm_unc = vm_unc_all[critical_step, 1:]
    vm_nav = vm_timeseries["navigation"][critical_step, 1:]
    bus_ids = np.arange(1, vm_unc_all.shape[1])
    vmin = cfg.constraints.vmin_pu
    delta_v = (vm_nav - vm_unc) * 1000  # mpu

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 7.0),
                                   height_ratios=[3, 2], sharex=True)

    # Upper panel: voltage profile at critical moment
    ax.plot(bus_ids, vm_unc, "o-", color=COLORS["uncontrolled"],
            linewidth=1.8, markersize=4, label="无序充电")
    ax.plot(bus_ids, vm_nav, "s-", color=COLORS["success"],
            linewidth=1.8, markersize=4, label="导航策略")
    ax.axhline(y=vmin, color="red", linestyle="--", linewidth=1.2,
               label=f"$V_{{\\min}}$ = {vmin} p.u.")

    ax.set_ylabel("母线电压（p.u.）")
    ax.set_xlim(0.5, len(bus_ids) + 0.5)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Mark buses violating Vmin
    violated_unc = bus_ids[vm_unc < vmin]
    if len(violated_unc) > 0:
        ax.scatter(violated_unc, vm_unc[vm_unc < vmin],
                   color=COLORS["secondary"], s=60, zorder=5, marker="x",
                   linewidths=2, label=f"越限母线（无序，{len(violated_unc)}条）")
        ax.legend(loc="lower left")

    # Lower panel: ΔV bar chart
    bar_colors = [COLORS["success"] if d >= 0 else COLORS["primary"] for d in delta_v]
    ax2.bar(bus_ids, delta_v, color=bar_colors, width=0.7, alpha=0.75)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    _set_panel_xlabel(ax, "", "(a) 关键时刻全网电压剖面")
    _set_panel_xlabel(ax2, "母线编号", "(b) 导航策略电压改善量")
    ax2.set_ylabel("$\\Delta V$ (mpu)")
    ax2.set_xticks(range(1, len(bus_ids) + 1, 2))
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotate the bus with largest improvement
    best_idx = int(np.argmax(delta_v))
    best_bus = bus_ids[best_idx]
    ax2.annotate(
        f"Bus {best_bus}: {delta_v[best_idx]:+.1f} mpu",
        xy=(best_bus, delta_v[best_idx]),
        xytext=(best_bus - 8, delta_v[best_idx] + max(abs(delta_v)) * 0.25),
        fontsize=9, color=COLORS["success"],
        arrowprops=dict(arrowstyle="->", color=COLORS["success"]),
    )

    fig.suptitle(
        f"关键时刻全网电压剖面对比（$N = {n_vehicles}$，{time_label}）",
        fontsize=13,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, hspace=0.08)

    _save(fig, "fig_4_5")

    summary = {
        "n_vehicles": n_vehicles,
        "critical_step": critical_step,
        "critical_time": time_label,
        "vm_unc_at_critical": vm_unc.tolist(),
        "vm_nav_at_critical": vm_nav.tolist(),
        "delta_v_mpu": delta_v.tolist(),
        "n_violated_unc": int(len(violated_unc)),
        "n_violated_nav": int(np.sum(vm_nav < vmin)),
        "global_min_unc": float(np.min(vm_unc)),
        "global_min_nav": float(np.min(vm_nav)),
        "max_improvement_mpu": float(np.max(delta_v)),
        "max_improvement_bus": int(best_bus),
    }
    _save_json(summary, "fig_4_5_data")
    return summary


# ── Fig 4-6: Strategy N* comparison (dual model) ────────────────


def _plot_fig_4_6(
    cfg_tc: ProjectConfig,
    tc_results: dict[str, int],
    sess_results: dict[str, int],
) -> None:
    strategy_labels = {
        "uncontrolled": "无序\n充电",
        "ordered_no_delay": "有序\n（无延迟）",
        "ordered_with_delay": "有序\n（延迟）",
        "nearest": "就近\n充电",
        "navigation_static": "导航\n（静态）",
        "navigation_dynamic": "导航\n（动态）",
    }

    tc_keys = [
        "uncontrolled",
        "ordered_no_delay",
        "ordered_with_delay",
        "nearest",
        "navigation_static",
        "navigation_dynamic",
    ]
    sess_keys = ["uncontrolled", "ordered_no_delay", "ordered_with_delay", "nearest"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.0), sharey=False)

    # Trip chain model
    tc_vals = [tc_results[k] for k in tc_keys]
    tc_colors = [
        COLORS["uncontrolled"], COLORS["ordered_no_delay"], COLORS["ordered_delay"],
        COLORS["nearest"], COLORS["navigation_static"], COLORS["navigation_dynamic"],
    ]
    bars1 = ax1.bar(
        [strategy_labels[k] for k in tc_keys], tc_vals,
        color=tc_colors, edgecolor="white", width=0.6,
    )
    ax1.bar_label(bars1, fontsize=10, fontweight="bold", padding=3)
    ax1.set_ylabel("$N^*$（最大EV数量）")
    ax1.set_title("出行链模型")
    ax1.set_ylim(0, max(tc_vals) * 1.25 if tc_vals else 10)
    ax1.grid(axis="y", alpha=0.3)

    # Session model
    sess_vals = [sess_results[k] for k in sess_keys]
    sess_colors = [
        COLORS["uncontrolled"], COLORS["ordered_no_delay"],
        COLORS["ordered_delay"], COLORS["nearest"],
    ]
    bars2 = ax2.bar(
        [strategy_labels[k] for k in sess_keys], sess_vals,
        color=sess_colors, edgecolor="white", width=0.6,
    )
    ax2.bar_label(bars2, fontsize=10, fontweight="bold", padding=3)
    ax2.set_ylabel("$N^*$（最大EV数量）")
    ax2.set_title("会话模型")
    ax2.set_ylim(0, max(sess_vals) * 1.25 if sess_vals else 10)
    ax2.grid(axis="y", alpha=0.3)

    _set_panel_xlabel(ax1, "", "(a) 出行链模型承载力对比", labelpad=12)
    _set_panel_xlabel(ax2, "", "(b) 会话模型承载力对比", labelpad=12)

    fig.suptitle(
        f"充电策略承载力对比（改进IEEE 33，$\\lambda$={cfg_tc.case.load_scale}，"
        f"$P_{{ch}}$={cfg_tc.ev.charge_power_kw} kW）",
        fontsize=13,
    )
    fig.set_constrained_layout_pads(w_pad=0.08, h_pad=0.08, wspace=0.08)

    _save(fig, "fig_4_6")


def gen_fig_4_6(cfg_tc: ProjectConfig, cfg_sess: ProjectConfig) -> dict:
    _log("[Fig 4-6] Strategy N* comparison (dual model, HEAVY) ...")
    from ev_tripchain.pipelines.run import run_hosting_capacity

    tc_strategies = {
        "uncontrolled": {"name": "uncontrolled"},
        "ordered_no_delay": {"name": "ordered", "ordered": {"random_delay": False}},
        "ordered_with_delay": {"name": "ordered", "ordered": {"random_delay": True}},
        "nearest": {"name": "nearest"},
        "navigation_static": {"name": "navigation", "navigation": {"dynamic_scoring": False}},
        "navigation_dynamic": {"name": "navigation", "navigation": {"dynamic_scoring": True}},
    }

    sess_strategies = {
        "uncontrolled": {"name": "uncontrolled"},
        "ordered_no_delay": {"name": "ordered", "ordered": {"random_delay": False}},
        "ordered_with_delay": {"name": "ordered", "ordered": {"random_delay": True}},
        "nearest": {"name": "nearest"},
    }

    def _eval_group(scope: str, cfg: ProjectConfig, strategies: dict) -> dict[str, int]:
        results = {}
        for key, strategy in strategies.items():
            cfg_mod = _with_overrides(cfg, {"strategy": strategy})
            _log(f"  [{scope}] {key:28s} running...")
            result = run_hosting_capacity(cfg_mod, progress=_log, progress_label=f"{scope}/{key}")
            results[key] = result.n_star
            _log(f"  [{scope}] {key:28s} N* = {result.n_star}")
        return results

    tc_results = _eval_group("tripchain", cfg_tc, tc_strategies)
    sess_results = _eval_group("session", cfg_sess, sess_strategies)

    _plot_fig_4_6(cfg_tc, tc_results, sess_results)

    data = {"tc_results": tc_results, "sess_results": sess_results}
    _save_json(data, "fig_4_6_data")
    return data


# ── Fig 4-7: Parameter sensitivity heatmap ───────────────────────


def gen_fig_4_7(cfg: ProjectConfig) -> dict:
    _log("[Fig 4-7] Parameter sensitivity heatmap (HEAVY) ...")
    from ev_tripchain.pipelines.run import run_hosting_capacity

    load_scales = [0.5, 0.7, 0.9, 1.0]
    charge_powers = [3.7, 7.2, 11.0, 22.0]

    grid = np.zeros((len(load_scales), len(charge_powers)), dtype=int)
    weakest_bus = np.zeros_like(grid)
    min_voltage = np.zeros((len(load_scales), len(charge_powers)), dtype=float)

    for i, ls in enumerate(load_scales):
        for j, cp in enumerate(charge_powers):
            cfg_mod = _with_overrides(cfg, {
                "case": {"load_scale": ls},
                "ev": {"charge_power_kw": cp},
            })
            _log(f"  [sweep] lambda={ls}, P={cp}kW running...")
            r = run_hosting_capacity(cfg_mod, progress=_log, progress_label=f"sweep/{ls}/{cp}")
            grid[i, j] = r.n_star
            _log(f"  [sweep] lambda={ls}, P={cp}kW -> N*={r.n_star}")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="YlOrRd_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="$N^*$")

    # Annotate cells
    for i in range(len(load_scales)):
        for j in range(len(charge_powers)):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if grid[i, j] < np.median(grid) else "black")

    ax.set_xticks(range(len(charge_powers)))
    ax.set_xticklabels([f"{cp} kW" for cp in charge_powers])
    ax.set_yticks(range(len(load_scales)))
    ax.set_yticklabels([f"$\\lambda$ = {ls}" for ls in load_scales])
    ax.set_xlabel("充电功率 $P_{ch}$")
    ax.set_ylabel("基础负荷缩放因子 $\\lambda$")
    ax.set_title("参数敏感性分析：$N^*$ 随充电功率与负荷水平的变化")

    _save(fig, "fig_4_7")

    data = {
        "load_scales": load_scales,
        "charge_powers": charge_powers,
        "n_star_grid": grid.tolist(),
    }
    _save_json(data, "fig_4_7_data")
    return data


# ── Fig 4-8: MC convergence verification ─────────────────────────


def gen_fig_4_8(cfg: ProjectConfig) -> dict:
    _log("[Fig 4-8] MC convergence verification (HEAVY) ...")
    from ev_tripchain.pipelines.run import run_hosting_capacity

    scenario_counts = [5, 10, 15, 20, 30, 50]
    n_repeats = 3  # repeat with different seeds for error bars
    results = np.zeros((len(scenario_counts), n_repeats), dtype=int)

    for i, s in enumerate(scenario_counts):
        for r in range(n_repeats):
            cfg_mod = _with_overrides(cfg, {
                "seed": cfg.seed + r * 1000,
                "hosting_capacity": {"scenarios": s},
            })
            _log(f"  [convergence] S={s}, repeat={r+1}/{n_repeats} running...")
            result = run_hosting_capacity(cfg_mod, progress=_log,
                                          progress_label=f"conv/S{s}/r{r}")
            results[i, r] = result.n_star
            _log(f"  [convergence] S={s}, repeat={r+1} -> N*={result.n_star}")

    means = results.mean(axis=1)
    stds = results.std(axis=1)
    mins = results.min(axis=1)
    maxs = results.max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(scenario_counts, means, yerr=[means - mins, maxs - means],
                fmt="o-", color=COLORS["primary"], linewidth=2, markersize=7,
                capsize=5, capthick=1.5, label="$N^*$ 均值±范围")
    ax.fill_between(scenario_counts, means - stds, means + stds,
                    alpha=0.15, color=COLORS["primary"])

    # Reference line at S=20
    ref_val = means[scenario_counts.index(20)] if 20 in scenario_counts else means[-1]
    ax.axhline(y=ref_val, color=COLORS["secondary"], linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"$S=20$ 参考值: {ref_val:.0f}")

    ax.set_xlabel("场景数 $S$")
    ax.set_ylabel("$N^*$（最大EV数量）")
    ax.set_title("蒙特卡洛评估收敛性验证（出行链模型，无序充电）")
    ax.set_xticks(scenario_counts)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    _save(fig, "fig_4_8")

    data = {
        "scenario_counts": scenario_counts,
        "results": results.tolist(),
        "means": means.tolist(),
        "stds": stds.tolist(),
    }
    _save_json(data, "fig_4_8_data")
    return data


# ── Main ──────────────────────────────────────────────────────────


FIGURE_MAP = {
    3: ("fig_4_3", "有序充电随机延迟对比", False),
    4: ("fig_4_4", "导航策略母线级负荷分布", False),
    5: ("fig_4_5", "导航策略电压改善", False),
    6: ("fig_4_6", "充电策略N*对比（双模型）", True),
    7: ("fig_4_7", "参数敏感性热力图", True),
    8: ("fig_4_8", "MC收敛性验证", True),
}


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg_tc = load_config(CFG_TC_PATH)
    cfg_sess = load_config(CFG_SESS_PATH)

    # Parse which figures to generate
    if len(sys.argv) > 1:
        fig_ids = [int(x) for x in sys.argv[1:]]
    else:
        fig_ids = sorted(FIGURE_MAP.keys())

    t_start = time.time()

    for fid in fig_ids:
        if fid not in FIGURE_MAP:
            _log(f"Unknown figure id: {fid}, skipping")
            continue

        name, desc, heavy = FIGURE_MAP[fid]
        _log(f"\n{'='*60}")
        _log(f"[Fig 4-{fid}] {desc} {'(HEAVY)' if heavy else ''}")
        _log(f"{'='*60}")

        t0 = time.time()
        try:
            if fid == 3:
                gen_fig_4_3(cfg_tc)
            elif fid == 4:
                gen_fig_4_4(cfg_tc)
            elif fid == 5:
                gen_fig_4_5(cfg_tc)
            elif fid == 6:
                gen_fig_4_6(cfg_tc, cfg_sess)
            elif fid == 7:
                gen_fig_4_7(cfg_tc)
            elif fid == 8:
                gen_fig_4_8(cfg_tc)

            elapsed = time.time() - t0
            _log(f"  Done in {elapsed:.1f}s")
        except Exception as e:
            _log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    total = time.time() - t_start
    _log(f"\nAll Chapter 4 figures generated in {total:.1f}s")
    _log(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
