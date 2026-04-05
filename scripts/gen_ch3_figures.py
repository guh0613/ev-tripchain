"""Generate all data figures for thesis Chapter 3.

Usage:
    uv run python scripts/gen_ch3_figures.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ev_tripchain.config import load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.common import ensure_ev_load_elements
from ev_tripchain.hosting_capacity.deterministic import run_deterministic_hc
from ev_tripchain.hosting_capacity.sensitivity import run_sensitivity_hc
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.pipelines.run import run_hosting_capacity
from ev_tripchain.reporting.style import COLORS, apply_style

OUT_DIR = Path("output/ch3_figures")
CFG_PATH = Path("configs/tripchain_soc.yaml")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    _log(f"  -> {path}")
    return path


# ── Fig 3-2: Standard vs Improved IEEE33 baseline voltage ──────────


def gen_fig_3_2() -> None:
    _log("[Fig 3-2] Standard vs Improved IEEE33 baseline voltage ...")

    # Standard case (radial, open tie switches)
    net_std = load_case("case33bw", load_scale=1.0)
    run_powerflow(net_std)
    vm_std = net_std.res_bus.vm_pu.to_numpy()

    # Improved case (closed ties, unified thermal limits)
    net_imp = load_case("ieee33", load_scale=1.0)
    run_powerflow(net_imp)
    vm_imp = net_imp.res_bus.vm_pu.to_numpy()

    n_bus = len(vm_std)
    bus_ids = np.arange(n_bus)

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.plot(bus_ids, vm_std, "o-", color=COLORS["secondary"], linewidth=1.8,
            markersize=4, label="标准IEEE 33（辐射状）")
    ax.plot(bus_ids, vm_imp, "s-", color=COLORS["primary"], linewidth=1.8,
            markersize=4, label="改进IEEE 33（弱环网）")
    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1.2,
               label="$V_{\\min}$ = 0.95 p.u.")
    ax.set_xlabel("母线编号")
    ax.set_ylabel("电压幅值（p.u.）")
    ax.set_title("标准与改进IEEE 33节点系统基线电压分布对比")
    ax.set_xlim(-0.5, n_bus - 0.5)
    ax.set_xticks(range(0, n_bus, 2))
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Annotate minimum voltage points
    idx_std_min = int(np.argmin(vm_std))
    idx_imp_min = int(np.argmin(vm_imp))
    ax.annotate(
        f"Bus {idx_std_min}\n{vm_std[idx_std_min]:.3f} p.u.",
        xy=(idx_std_min, vm_std[idx_std_min]),
        xytext=(idx_std_min + 2, vm_std[idx_std_min] - 0.015),
        fontsize=9, color=COLORS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
    )
    ax.annotate(
        f"Bus {idx_imp_min}\n{vm_imp[idx_imp_min]:.3f} p.u.",
        xy=(idx_imp_min, vm_imp[idx_imp_min]),
        xytext=(idx_imp_min + 2, vm_imp[idx_imp_min] + 0.008),
        fontsize=9, color=COLORS["primary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["primary"]),
    )

    _save(fig, "fig_3_2")


# ── Fig 3-6: Risk probability curve ────────────────────────────────


def gen_fig_3_6(cfg) -> dict:
    _log("[Fig 3-6] Risk probability curve ...")
    result = run_hosting_capacity(cfg, progress=_log, progress_label="fig3-6")

    pts = sorted(result.risk_curve_detail, key=lambda x: x.n)
    ns = [p.n for p in pts]
    p_hats = [p.p_hat for p in pts]
    ci_lo = [p.ci95_low for p in pts]
    ci_hi = [p.ci95_high for p in pts]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.fill_between(ns, ci_lo, ci_hi, color=COLORS["secondary"], alpha=0.15,
                    label="95% Wilson CI")
    ax.plot(ns, p_hats, "o-", color=COLORS["secondary"], linewidth=2,
            markersize=5, label="$\\hat{\\pi}(N)$")
    ax.axhline(y=result.risk_tolerance, color=COLORS["primary"],
               linestyle="--", linewidth=1.5,
               label=f"$\\varepsilon$ = {result.risk_tolerance}")
    if result.n_star > 0:
        ax.axvline(x=result.n_star, color=COLORS["success"],
                   linestyle=":", linewidth=1.5,
                   label=f"$N^*$ = {result.n_star}")
    ax.set_xlabel("接入电动汽车数量（$N$）")
    ax.set_ylabel("硬约束越限概率 $\\hat{\\pi}(N)$")
    ax.set_title("风险曲线：电动汽车概率承载力（出行链模型，无序充电）")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    _save(fig, "fig_3_6")

    # Return data for fig 3-7
    return {
        "n_star": result.n_star,
        "risk_tolerance": result.risk_tolerance,
        "detail": result.risk_curve_detail,
    }


# ── Fig 3-7: Per-constraint violation probability breakdown ────────


def gen_fig_3_7(risk_data: dict) -> None:
    _log("[Fig 3-7] Per-constraint violation probability breakdown ...")

    pts = sorted(risk_data["detail"], key=lambda x: x.n)
    ns = [p.n for p in pts]

    any_p = []
    volt_p = []
    line_p = []
    trafo_p = []
    for p in pts:
        if p.hard_constraints is not None:
            any_p.append(p.hard_constraints.any_limit_exceedance.p_hat)
            volt_p.append(p.hard_constraints.voltage_limit_exceedance.p_hat)
            line_p.append(p.hard_constraints.line_limit_exceedance.p_hat)
            trafo_p.append(p.hard_constraints.trafo_limit_exceedance.p_hat)
        else:
            any_p.append(p.p_hat)
            volt_p.append(p.p_hat)
            line_p.append(0.0)
            trafo_p.append(0.0)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(ns, any_p, "o-", color=COLORS["gray"], linewidth=2,
            markersize=5, label="总越限概率")
    ax.plot(ns, volt_p, "s--", color=COLORS["secondary"], linewidth=1.5,
            markersize=4, label="电压越限概率")
    ax.plot(ns, line_p, "^--", color=COLORS["warning"], linewidth=1.5,
            markersize=4, label="线路过载概率")
    ax.plot(ns, trafo_p, "d--", color=COLORS["purple"], linewidth=1.5,
            markersize=4, label="变压器过载概率")

    eps = risk_data["risk_tolerance"]
    ax.axhline(y=eps, color=COLORS["primary"], linestyle="--",
               linewidth=1.2, label=f"$\\varepsilon$ = {eps}")
    if risk_data["n_star"] > 0:
        ax.axvline(x=risk_data["n_star"], color=COLORS["success"],
                   linestyle=":", linewidth=1.2,
                   label=f"$N^*$ = {risk_data['n_star']}")

    ax.set_xlabel("接入电动汽车数量（$N$）")
    ax.set_ylabel("越限概率")
    ax.set_title("分项约束越限概率分解（出行链模型，无序充电）")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    _save(fig, "fig_3_7")


# ── Fig 3-8: Voltage margin and sensitivity distribution ───────────


def gen_fig_3_8(cfg) -> None:
    _log("[Fig 3-8] Voltage margin and sensitivity distribution ...")

    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    sens = run_sensitivity_hc(net, cfg)

    ev_idx = ensure_ev_load_elements(
        load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    )
    net2 = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ensure_ev_load_elements(net2)
    bus_ids = net2.load.loc[ev_idx, "bus"].to_numpy()
    n = len(bus_ids)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)

    margin_mpu = sens.voltage_margin * 1000
    ax1.bar(range(n), margin_mpu, color=COLORS["primary"], alpha=0.8, width=0.7)
    ax1.set_ylabel("电压裕度\n$(V_{base} - V_{min})$ [mpu]")
    ax1.set_title("各母线基线电压裕度与灵敏度分布（改进IEEE 33）")
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(y=0, color="red", linewidth=0.8)

    # Highlight weakest buses
    weakest_idx = int(np.argmin(margin_mpu))
    ax1.bar(weakest_idx, margin_mpu[weakest_idx], color=COLORS["secondary"],
            alpha=0.9, width=0.7)
    ax1.annotate(
        f"Bus {bus_ids[weakest_idx]}\n{margin_mpu[weakest_idx]:.1f} mpu",
        xy=(weakest_idx, margin_mpu[weakest_idx]),
        xytext=(weakest_idx + 3, margin_mpu[weakest_idx] + 2),
        fontsize=9, color=COLORS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
    )

    diag_mpu = sens.sensitivity_diagonal * 1000
    ax2.bar(range(n), diag_mpu, color=COLORS["secondary"], alpha=0.8, width=0.7)
    ax2.set_ylabel("$dV_i/dP_i$\n[mpu/MW]")
    ax2.set_xlabel("母线编号")
    labels = [str(b) for b in bus_ids]
    ax2.set_xticks(range(0, n, 2))
    ax2.set_xticklabels([labels[i] for i in range(0, n, 2)], fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Highlight most sensitive bus
    most_sensitive = int(np.argmin(diag_mpu))  # most negative = most sensitive
    ax2.bar(most_sensitive, diag_mpu[most_sensitive], color=COLORS["primary"],
            alpha=0.9, width=0.7)

    _save(fig, "fig_3_8")


# ── Fig 3-9: Three method comparison ──────────────────────────────


def gen_fig_3_9(cfg) -> None:
    _log("[Fig 3-9] Three method comparison ...")

    results: dict[str, int] = {}
    times: dict[str, float] = {}

    # Deterministic
    t0 = time.time()
    _log("  [deterministic] running ...")
    det = run_deterministic_hc(
        load_case(cfg.case.name, load_scale=cfg.case.load_scale), cfg
    )
    times["deterministic"] = time.time() - t0
    results["deterministic"] = det.n_star
    _log(f"  [deterministic] N* = {det.n_star}  ({times['deterministic']:.1f}s)")

    # Sensitivity (3 variants)
    t0 = time.time()
    _log("  [sensitivity] running ...")
    sens = run_sensitivity_hc(
        load_case(cfg.case.name, load_scale=cfg.case.load_scale), cfg
    )
    t_sens = time.time() - t0
    times["sensitivity_weakest"] = t_sens
    times["sensitivity_representative"] = t_sens
    results["sensitivity_weakest"] = sens.n_star_weakest
    results["sensitivity_representative"] = sens.n_star_representative
    _log(
        f"  [sensitivity] weakest={sens.n_star_weakest}, "
        f"repr={sens.n_star_representative}  ({t_sens:.1f}s)"
    )

    # Monte Carlo
    t0 = time.time()
    _log("  [monte_carlo] running ...")
    mc = run_hosting_capacity(cfg, progress=_log, progress_label="fig3-9/mc")
    times["monte_carlo"] = time.time() - t0
    results["monte_carlo"] = mc.n_star
    _log(f"  [monte_carlo] N* = {mc.n_star}  ({times['monte_carlo']:.1f}s)")

    # Plot
    method_colors = {
        "sensitivity_weakest": COLORS["secondary"],
        "deterministic": COLORS["warning"],
        "sensitivity_representative": "#f59e0b",
        "monte_carlo": COLORS["success"],
    }
    method_labels = {
        "sensitivity_weakest": "灵敏度法\n（最薄弱母线）",
        "deterministic": "确定性法\n（典型模板）",
        "sensitivity_representative": "灵敏度法\n（典型模板）",
        "monte_carlo": "蒙特卡洛法",
    }

    ordered_keys = ["sensitivity_weakest", "deterministic",
                    "sensitivity_representative", "monte_carlo"]
    labels = [method_labels[k] for k in ordered_keys]
    n_stars = [results[k] for k in ordered_keys]
    colors = [method_colors[k] for k in ordered_keys]
    time_vals = [times[k] for k in ordered_keys]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars = ax1.bar(labels, n_stars, color=colors, edgecolor="white", width=0.55)
    ax1.bar_label(bars, fontsize=11, fontweight="bold", padding=3)
    ax1.set_ylabel("$N^*$（最大EV数量）")
    ax1.set_title("承载力评估方法对比（改进IEEE 33，出行链模型，无序充电）")
    ax1.set_ylim(0, max(n_stars) * 1.25 if n_stars else 10)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(labels, time_vals, "ko--", markersize=6, linewidth=1.2,
             label="计算时间")
    ax2.set_ylabel("计算时间（s）")
    ax2.set_ylim(0, max(time_vals) * 1.4 if time_vals else 1)
    ax2.legend(loc="center right", fontsize=9)

    _save(fig, "fig_3_9")

    # Save data
    data_path = OUT_DIR / "method_comparison.json"
    json_data = {
        "results": results,
        "times": times,
        "sensitivity_uniform": sens.n_star_uniform,
    }
    data_path.write_text(
        json.dumps(json_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _log(f"  -> {data_path}")


# ── Fig 3-10: Bus voltage profiles at N=N* ────────────────────────


def gen_fig_3_10(cfg, n_star: int) -> None:
    _log(f"[Fig 3-10] Bus voltage profiles at N={n_star} ...")

    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()

    rng = np.random.default_rng(cfg.seed + 100)
    prof = build_ev_profile_mw(
        cfg=cfg,
        n_vehicles=n_star,
        buses=buses,
        n_buses=len(ev_idx),
        rng=rng,
    )

    n_steps = prof.shape[0]
    n_bus_total = len(net.bus)
    all_vm = np.zeros((n_steps, n_bus_total), dtype=float)
    pf_init = "auto"
    for t in range(n_steps):
        net.load.loc[ev_idx, "p_mw"] = prof[t, :]
        try:
            run_powerflow(net, init=pf_init)
            pf_init = "results"
        except Exception:
            pf_init = "auto"
        all_vm[t, :] = net.res_bus.vm_pu.to_numpy()

    step_min = cfg.time.step_minutes
    hours = np.arange(n_steps) * (step_min / 60.0)
    total_hours = hours[-1] + step_min / 60.0
    vmin = cfg.constraints.vmin_pu
    vmax = cfg.constraints.vmax_pu

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for b in range(n_bus_total):
        ax.plot(hours, all_vm[:, b], linewidth=0.8, alpha=0.65)

    ax.axhline(y=vmin, color="red", linestyle="--", linewidth=1.2,
               label=f"$V_{{\\min}}$ = {vmin} p.u.")
    ax.axhline(y=vmax, color="red", linestyle="--", linewidth=1.2,
               label=f"$V_{{\\max}}$ = {vmax} p.u.")

    # Mark midnight boundaries
    for boundary in np.arange(24.0, total_hours, 24.0):
        ax.axvline(boundary, color=COLORS["gray"], linestyle=":",
                   linewidth=0.9, alpha=0.5)

    ax.set_xlim(0, total_hours)
    tick_step = 4
    ax.set_xticks(np.arange(0, total_hours + 1e-9, tick_step))
    ax.set_xlabel("时刻（小时）")
    ax.set_ylabel("电压幅值（p.u.）")

    n_days = cfg.time.n_days
    horizon_label = f"{n_days}天（{int(total_hours)}小时）" if n_days > 1 else f"{int(total_hours)}小时"
    ax.set_title(
        f"各母线连续电压剖面（{horizon_label}，$N = {n_star}$，出行链模型，无序充电）"
    )
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Annotate minimum voltage
    min_vm = all_vm.min()
    min_pos = np.unravel_index(all_vm.argmin(), all_vm.shape)
    min_hour = hours[min_pos[0]]
    min_bus = min_pos[1]
    ax.annotate(
        f"Bus {min_bus}: {min_vm:.4f} p.u.",
        xy=(min_hour, min_vm),
        xytext=(min_hour + 3, min_vm + 0.005),
        fontsize=9, color=COLORS["secondary"],
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
    )

    _save(fig, "fig_3_10")


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config(CFG_PATH)
    t_start = time.time()

    # Fig 3-2: baseline voltage comparison (fast)
    gen_fig_3_2()

    # Fig 3-8: sensitivity analysis (fast)
    gen_fig_3_8(cfg)

    # Fig 3-6 + 3-7: risk curve + constraint breakdown (heavy - runs MC search)
    risk_data = gen_fig_3_6(cfg)
    gen_fig_3_7(risk_data)

    # Fig 3-9: method comparison (heavy - runs all 3 methods)
    gen_fig_3_9(cfg)

    # Fig 3-10: voltage profiles at N* (uses N* from fig 3-6)
    gen_fig_3_10(cfg, n_star=risk_data["n_star"])

    elapsed = time.time() - t_start
    _log(f"\nAll Chapter 3 figures generated in {elapsed:.1f}s")
    _log(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
