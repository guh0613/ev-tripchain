"""Weak-link analysis: identify bottleneck buses and lines at N=N*."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.evaluate import _ensure_ev_load_elements
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.pipelines.run import run_hosting_capacity
from ev_tripchain.rng import make_rng_for


def analyse_bottlenecks(
    cfg: ProjectConfig, n_vehicles: int, n_scenarios: int = 20
) -> dict:
    """Run scenarios at given N and collect per-bus/line statistics."""
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)
    n_steps = cfg.time.n_steps
    n_all_buses = len(net.bus)
    n_lines = len(net.line)

    vmin_tracker = np.ones((n_scenarios, n_all_buses), dtype=float) * 999.0
    vmax_tracker = np.zeros((n_scenarios, n_all_buses), dtype=float)
    line_max_tracker = np.zeros((n_scenarios, n_lines), dtype=float)

    for s in range(n_scenarios):
        rng_s = make_rng_for(cfg.seed, n_vehicles, s)
        profile = build_ev_profile_mw(
            cfg=cfg, n_vehicles=n_vehicles, buses=buses, n_buses=n_buses, rng=rng_s
        )
        for t in range(n_steps):
            net.load.loc[ev_idx, "p_mw"] = profile[t, :]
            try:
                run_powerflow(net)
            except Exception:
                continue
            vm = net.res_bus.vm_pu.to_numpy()
            vmin_tracker[s, :] = np.minimum(vmin_tracker[s, :], vm)
            vmax_tracker[s, :] = np.maximum(vmax_tracker[s, :], vm)
            if n_lines > 0:
                ll = net.res_line.loading_percent.to_numpy()
                line_max_tracker[s, :] = np.maximum(line_max_tracker[s, :], ll)

    # aggregate across scenarios
    avg_vmin = vmin_tracker.mean(axis=0)
    worst_vmin = vmin_tracker.min(axis=0)
    avg_vmax = vmax_tracker.mean(axis=0)
    avg_line_max = line_max_tracker.mean(axis=0)
    worst_line_max = line_max_tracker.max(axis=0)

    # identify bottleneck buses (sorted by worst voltage)
    bus_ids = net.bus.index.tolist()
    bus_ranking = sorted(
        zip(bus_ids, worst_vmin.tolist(), avg_vmin.tolist()),
        key=lambda x: x[1],
    )

    # identify bottleneck lines (sorted by worst loading)
    line_ids = net.line.index.tolist()
    line_ranking = sorted(
        zip(line_ids, worst_line_max.tolist(), avg_line_max.tolist()),
        key=lambda x: -x[1],
    )

    return {
        "n_vehicles": n_vehicles,
        "n_scenarios": n_scenarios,
        "bottleneck_buses": [
            {"bus": int(b), "worst_vmin_pu": round(v, 4), "avg_vmin_pu": round(a, 4)}
            for b, v, a in bus_ranking[:5]
        ],
        "bottleneck_lines": [
            {"line": int(l), "worst_loading_pct": round(v, 2), "avg_loading_pct": round(a, 2)}
            for l, v, a in line_ranking[:5]
        ],
        "all_bus_worst_vmin": {int(b): round(v, 4) for b, v in zip(bus_ids, worst_vmin.tolist())},
        "all_line_worst_loading": {int(l): round(v, 2) for l, v in zip(line_ids, worst_line_max.tolist())},
    }


def main() -> None:
    # trip-chain model at current N*
    cfg_tc = load_config(Path("configs/tripchain_soc.yaml"))
    hc_tc = run_hosting_capacity(cfg_tc)
    print(f"=== Trip-chain model: bottleneck analysis at N*={hc_tc.n_star} ===")
    result_tc = analyse_bottlenecks(cfg_tc, n_vehicles=hc_tc.n_star, n_scenarios=15)

    # session model at current N*
    cfg_sess = load_config(Path("configs/example.yaml"))
    hc_sess = run_hosting_capacity(cfg_sess)
    print(f"\n=== Session model: bottleneck analysis at N*={hc_sess.n_star} ===")
    result_sess = analyse_bottlenecks(cfg_sess, n_vehicles=hc_sess.n_star, n_scenarios=15)

    outdir = Path("docs/results")
    outdir.mkdir(parents=True, exist_ok=True)

    combined = {"tripchain_soc": result_tc, "session_based": result_sess}
    out = outdir / "bottleneck_analysis.json"
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")

    # print summary
    labels = [
        (f"Trip-chain (N*={hc_tc.n_star})", result_tc),
        (f"Session (N*={hc_sess.n_star})", result_sess),
    ]
    for label, res in labels:
        print(f"\n--- {label} ---")
        print("Top-5 bottleneck buses (lowest voltage):")
        for b in res["bottleneck_buses"]:
            flag = " *** VIOLATION" if b["worst_vmin_pu"] < 0.95 else ""
            print(f"  Bus {b['bus']}: worst Vmin={b['worst_vmin_pu']:.4f} avg={b['avg_vmin_pu']:.4f}{flag}")
        print("Top-5 bottleneck lines (highest loading):")
        for l in res["bottleneck_lines"]:
            flag = " *** OVERLOAD" if l["worst_loading_pct"] > 100 else ""
            print(f"  Line {l['line']}: worst={l['worst_loading_pct']:.1f}% avg={l['avg_loading_pct']:.1f}%{flag}")


if __name__ == "__main__":
    main()
