"""Report orchestration: run analyses and generate all figures + tables.

Usage from CLI:
    uv run ev-tripchain report -c configs/tripchain_soc.yaml -o output/
    uv run ev-tripchain report -c configs/tripchain_soc.yaml -o output/ --only 1,4,8
"""

from __future__ import annotations

from collections.abc import Callable
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.grid.cases import load_case
from ev_tripchain.hosting_capacity.common import ensure_ev_load_elements
from ev_tripchain.reporting import figures, tables
from ev_tripchain.reporting.style import apply_style

# Figure registry: id -> (name, description, heavy?)
FIGURE_REGISTRY: dict[int, tuple[str, str, bool]] = {
    1: ("input_distributions", "输入分布（出发时间/日行驶里程）", False),
    2: ("soc_evolution", "单车SOC演化曲线", False),
    3: ("charging_load", "聚合充电负荷曲线（连续时序）", False),
    4: ("risk_curve", "风险曲线 N vs π(N)", True),
    5: ("bus_voltage_profile", "各母线连续电压剖面", False),
    6: ("model_comparison", "两种负荷模型连续时序对比", False),
    7: ("ordered_delay", "有序充电随机延迟对比（含跨午夜窗口）", False),
    8: ("strategy_comparison", "充电策略N*对比", True),
    9: ("method_comparison", "评估方法对比", True),
    10: ("voltage_sensitivity", "电压灵敏度分析", False),
    11: ("parameter_sweep", "参数敏感性热力图", True),
}


def _save_fig(fig: plt.Figure, outdir: Path, name: str, fmt: str = "png") -> Path:
    path = outdir / f"{name}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_json(data: Any, path: Path) -> Path:
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return path


def _log(message: str) -> None:
    print(message, flush=True)


FigureGenerator = Callable[
    [ProjectConfig, ProjectConfig, Path, Path, Path, str],
    Path,
]


def _merge_nested(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _with_overrides(cfg: ProjectConfig, updates: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig.model_validate(_merge_nested(cfg.model_dump(), updates))


def _load_case_context(cfg: ProjectConfig) -> tuple[Any, list[int], np.ndarray]:
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    return net, ev_idx, buses


def _evaluate_strategy_group(
    *,
    scope: str,
    cfg: ProjectConfig,
    strategies: dict[str, dict[str, Any]],
) -> dict[str, int]:
    from ev_tripchain.pipelines.run import run_hosting_capacity

    results: dict[str, int] = {}
    for key, strategy in strategies.items():
        cfg_mod = _with_overrides(cfg, {"strategy": strategy})
        _log(f"  [{scope}] {key:28s}  running...")
        result = run_hosting_capacity(cfg_mod, progress=_log, progress_label=f"{scope}/{key}")
        results[key] = result.n_star
        _log(f"  [{scope}] {key:28s}  N* = {result.n_star}")
    return results


# ----------------------------------------------------------------------
# Individual analysis functions (compute data for figures)
# ----------------------------------------------------------------------


def analyse_input_distributions(cfg: ProjectConfig, n_samples: int = 20000) -> dict:
    from ev_tripchain.mobility.tripchain_sampling import (
        TripChainSamplingParams,
        sample_daily_trip_chain,
    )

    rng = np.random.default_rng(cfg.seed)
    tc_cfg = cfg.mobility.trip_chain
    params = TripChainSamplingParams(
        n_zones=tc_cfg.n_zones,
        other_stops_mean=tc_cfg.other_stops_mean,
        first_departure_mean=tc_cfg.first_departure_mean,
        first_departure_std_minutes=tc_cfg.first_departure_std_minutes,
        work_duration_mean_minutes=tc_cfg.work_duration_mean_minutes,
        work_duration_std_minutes=tc_cfg.work_duration_std_minutes,
        other_dwell_mean_minutes=tc_cfg.other_dwell_mean_minutes,
        other_dwell_std_minutes=tc_cfg.other_dwell_std_minutes,
        travel_minutes_per_km=tc_cfg.travel_minutes_per_km,
        distance_km_mean=tc_cfg.distance_km_mean,
        distance_km_std=tc_cfg.distance_km_std,
    )

    dep_min = np.empty(n_samples, dtype=float)
    daily_km = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        tc = sample_daily_trip_chain(params, rng=rng)
        dep_min[i] = float(tc.stops[0].departure_minute)
        daily_km[i] = float(np.sum(tc.leg_distance_km))

    return {"dep_hours": dep_min / 60.0, "daily_km": daily_km}


def analyse_soc_evolution(cfg: ProjectConfig) -> dict:
    from ev_tripchain.mobility.soc import SOCEvolutionParams, simulate_soc_and_charging_profile
    from ev_tripchain.mobility.trip_chain import Stop, TripChain

    rng = np.random.default_rng(cfg.seed + 1)
    tc = TripChain(
        stops=[
            Stop(zone=0, arrival_minute=0, departure_minute=8 * 60, purpose="home"),
            Stop(zone=1, arrival_minute=8 * 60 + 30, departure_minute=17 * 60 + 30, purpose="work"),
            Stop(zone=0, arrival_minute=18 * 60, departure_minute=24 * 60, purpose="home"),
        ],
        leg_distance_km=[45.0, 45.0],
    )

    soc_cfg = cfg.mobility.soc
    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=soc_cfg.battery_capacity_kwh,
        consumption_kwh_per_km=soc_cfg.consumption_kwh_per_km,
        initial_soc_mean=0.55,
        initial_soc_std=0.0,
        charge_power_kw=cfg.ev.charge_power_kw,
        charge_efficiency=soc_cfg.charge_efficiency,
        charge_trigger_soc=0.30,
        charge_purposes=tuple(soc_cfg.charge_purposes),
        allow_initial_stop_charging=False,
    )

    step_minutes = cfg.time.step_minutes
    n_steps = cfg.time.n_steps
    soc, p_kw = simulate_soc_and_charging_profile(
        tc, soc_params, step_minutes=step_minutes, n_steps=n_steps, rng=rng, initial_soc=0.55,
    )
    hours = np.arange(n_steps + 1) * (step_minutes / 60.0)
    return {"hours": hours, "soc": soc, "p_kw": p_kw, "step_minutes": step_minutes}


def _profile_total_kw(cfg: ProjectConfig, strategy: dict, n_vehicles: int = 500) -> np.ndarray:
    """Build total charging load curve (kW) for a given strategy."""
    from ev_tripchain.mobility.profile import build_ev_profile_mw

    cfg_mod = _with_overrides(cfg, {"strategy": strategy})
    net, ev_idx, buses = _load_case_context(cfg_mod)
    rng = np.random.default_rng(cfg.seed)
    prof = build_ev_profile_mw(
        cfg=cfg_mod,
        n_vehicles=n_vehicles,
        buses=buses,
        n_buses=len(ev_idx),
        rng=rng,
    )
    return prof.sum(axis=1) * 1000  # MW -> kW


def analyse_charging_load(cfg: ProjectConfig, n_vehicles: int = 1500) -> dict:
    total_kw = _profile_total_kw(cfg, {"name": "uncontrolled"}, n_vehicles)
    hours = np.arange(total_kw.shape[0]) * (cfg.time.step_minutes / 60.0)
    return {"hours": hours, "total_kw": total_kw, "n_vehicles": n_vehicles}


def analyse_risk_curve(cfg: ProjectConfig) -> dict:
    from ev_tripchain.pipelines.run import run_hosting_capacity

    result = run_hosting_capacity(cfg)
    return {
        "n_star": result.n_star,
        "risk_tolerance": result.risk_tolerance,
        "risk_points": [p.model_dump() for p in result.risk_curve_detail],
    }


def analyse_bus_voltage_profile(cfg: ProjectConfig, n_vehicles: int = 1500) -> dict:
    from ev_tripchain.grid.powerflow import run_powerflow
    from ev_tripchain.mobility.profile import build_ev_profile_mw

    rng = np.random.default_rng(cfg.seed + 10)
    net, ev_idx, buses = _load_case_context(cfg)

    prof = build_ev_profile_mw(
        cfg=cfg,
        n_vehicles=n_vehicles,
        buses=buses,
        n_buses=len(ev_idx),
        rng=rng,
    )

    n_steps = prof.shape[0]
    all_vm = np.zeros((n_steps, len(net.bus)), dtype=float)
    for t in range(n_steps):
        net.load.loc[ev_idx, "p_mw"] = prof[t, :]
        run_powerflow(net)
        all_vm[t, :] = net.res_bus.vm_pu.to_numpy()

    hours = np.arange(n_steps) * (cfg.time.step_minutes / 60.0)
    return {"hours": hours, "all_vm": all_vm, "n_vehicles": n_vehicles}


def analyse_model_comparison(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    n_vehicles: int = 1000,
) -> dict:
    total_tc_kw = _profile_total_kw(cfg_tc, {"name": "uncontrolled"}, n_vehicles)
    total_sess_kw = _profile_total_kw(cfg_sess, {"name": "uncontrolled"}, n_vehicles)
    hours = np.arange(total_tc_kw.shape[0]) * (cfg_tc.time.step_minutes / 60.0)
    return {
        "hours": hours,
        "total_sess_kw": total_sess_kw,
        "total_tc_kw": total_tc_kw,
        "n_vehicles": n_vehicles,
    }


def analyse_ordered_delay(cfg: ProjectConfig, n_vehicles: int = 500) -> dict:
    p_unc = _profile_total_kw(cfg, {"name": "uncontrolled"}, n_vehicles)
    p_no = _profile_total_kw(
        cfg,
        {"name": "ordered", "ordered": {"random_delay": False}},
        n_vehicles,
    )
    p_yes = _profile_total_kw(
        cfg,
        {"name": "ordered", "ordered": {"random_delay": True}},
        n_vehicles,
    )
    hours = np.arange(p_unc.shape[0]) * (cfg.time.step_minutes / 60.0)
    step_hours = float(cfg.time.step_minutes) / 60.0
    hour_of_day = np.mod(hours, 24.0)
    overnight_mask = (hour_of_day >= 22.0) | (hour_of_day < 6.0)

    def _summary(profile_kw: np.ndarray) -> dict[str, float]:
        return {
            "peak_kw": float(np.max(profile_kw)),
            "energy_kwh": float(np.sum(profile_kw) * step_hours),
            "overnight_energy_kwh": float(np.sum(profile_kw[overnight_mask]) * step_hours),
        }

    model_label = (
        "出行链 + SOC 模型"
        if cfg.mobility.model == "tripchain_soc"
        else "会话模型"
    )
    return {
        "hours": hours,
        "p_uncontrolled": p_unc,
        "p_no_delay": p_no,
        "p_with_delay": p_yes,
        "n_vehicles": n_vehicles,
        "model_label": model_label,
        "n_days": int(cfg.time.n_days),
        "summary": {
            "uncontrolled": _summary(p_unc),
            "ordered_no_delay": _summary(p_no),
            "ordered_with_delay": _summary(p_yes),
        },
    }


def analyse_strategies(cfg_tc: ProjectConfig, cfg_sess: ProjectConfig) -> dict:
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

    return {
        "tc_results": _evaluate_strategy_group(
            scope="tripchain",
            cfg=cfg_tc,
            strategies=tc_strategies,
        ),
        "sess_results": _evaluate_strategy_group(
            scope="session",
            cfg=cfg_sess,
            strategies=sess_strategies,
        ),
    }


def analyse_methods(cfg: ProjectConfig) -> dict:
    from ev_tripchain.grid.cases import load_case
    from ev_tripchain.hosting_capacity.deterministic import run_deterministic_hc
    from ev_tripchain.hosting_capacity.sensitivity import run_sensitivity_hc
    from ev_tripchain.pipelines.run import run_hosting_capacity

    results: dict[str, int] = {}
    times: dict[str, float] = {}

    t0 = time.time()
    _log("  [method] deterministic running...")
    det = run_deterministic_hc(load_case(cfg.case.name, load_scale=cfg.case.load_scale), cfg)
    times["deterministic"] = time.time() - t0
    results["deterministic"] = det.n_star
    _log(f"  [method] deterministic done: N* = {det.n_star}")

    t0 = time.time()
    _log("  [method] sensitivity running...")
    sens = run_sensitivity_hc(load_case(cfg.case.name, load_scale=cfg.case.load_scale), cfg)
    times["sensitivity_representative"] = time.time() - t0
    results["sensitivity_representative"] = sens.n_star_representative
    _log(
        "  [method] sensitivity done: "
        f"representative = {sens.n_star_representative}, "
        f"weakest = {sens.n_star_weakest}, "
        f"uniform = {sens.n_star_uniform}"
    )

    t0 = time.time()
    _log("  [method] monte_carlo running...")
    mc = run_hosting_capacity(cfg, progress=_log, progress_label="method/monte_carlo")
    times["monte_carlo"] = time.time() - t0
    results["monte_carlo"] = mc.n_star
    _log(f"  [method] monte_carlo done: N* = {mc.n_star}")

    return {"method_results": results, "method_times": times}


def analyse_voltage_sensitivity(cfg: ProjectConfig) -> dict:
    from ev_tripchain.hosting_capacity.sensitivity import run_sensitivity_hc

    net, _, buses = _load_case_context(cfg)
    sens = run_sensitivity_hc(net, cfg)
    return {
        "bus_ids": buses,
        "voltage_margin": sens.voltage_margin,
        "sensitivity_diagonal": sens.sensitivity_diagonal,
    }


def analyse_parameter_sweep(
    cfg: ProjectConfig,
    load_scales: list[float] | None = None,
    charge_powers: list[float] | None = None,
) -> dict:
    from ev_tripchain.pipelines.run import run_hosting_capacity

    if load_scales is None:
        load_scales = [0.5, 0.7, 0.9]
    if charge_powers is None:
        charge_powers = [3.7, 7.2, 11.0, 22.0]

    grid = np.zeros((len(load_scales), len(charge_powers)), dtype=int)
    for i, ls in enumerate(load_scales):
        for j, cp in enumerate(charge_powers):
            cfg_mod = _with_overrides(
                cfg,
                {
                    "case": {"load_scale": ls},
                    "ev": {"charge_power_kw": cp},
                },
            )
            _log(f"  [sweep] lambda={ls}, P={cp}kW  running...")
            r = run_hosting_capacity(cfg_mod, progress=_log, progress_label=f"sweep/{ls}/{cp}")
            grid[i, j] = r.n_star
            _log(f"  [sweep] lambda={ls}, P={cp}kW  ->  N*={r.n_star}")

    return {"load_scales": load_scales, "charge_powers": charge_powers, "n_star_grid": grid}


def _figure_filename(fid: int) -> str:
    return f"{fid:02d}_{FIGURE_REGISTRY[fid][0]}"


def _generate_input_distributions_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess, tbl_dir, data_dir
    data = analyse_input_distributions(cfg_tc)
    fig = figures.fig_input_distributions(data["dep_hours"], data["daily_km"])
    return _save_fig(fig, fig_dir, _figure_filename(1), fmt)


def _generate_soc_evolution_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess, tbl_dir, data_dir
    data = analyse_soc_evolution(cfg_tc)
    fig = figures.fig_soc_evolution(data["hours"], data["soc"], data["p_kw"], data["step_minutes"])
    return _save_fig(fig, fig_dir, _figure_filename(2), fmt)


def _generate_charging_load_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess, tbl_dir, data_dir
    data = analyse_charging_load(cfg_tc)
    fig = figures.fig_charging_load(data["hours"], data["total_kw"], data["n_vehicles"])
    return _save_fig(fig, fig_dir, _figure_filename(3), fmt)


def _generate_risk_curve_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess
    data = analyse_risk_curve(cfg_tc)
    fig = figures.fig_risk_curve(data["risk_points"], data["n_star"], data["risk_tolerance"])
    tables.export_risk_curve(tbl_dir / "risk_curve.csv", data["risk_points"], data["n_star"])
    _save_json(data, data_dir / "risk_curve.json")
    return _save_fig(fig, fig_dir, _figure_filename(4), fmt)


def _generate_bus_voltage_profile_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess, tbl_dir, data_dir
    data = analyse_bus_voltage_profile(cfg_tc)
    fig = figures.fig_bus_voltage_profile(
        data["hours"],
        data["all_vm"],
        data["n_vehicles"],
        vmin=cfg_tc.constraints.vmin_pu,
        vmax=cfg_tc.constraints.vmax_pu,
    )
    return _save_fig(fig, fig_dir, _figure_filename(5), fmt)


def _generate_model_comparison_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del tbl_dir, data_dir
    data = analyse_model_comparison(cfg_tc, cfg_sess)
    fig = figures.fig_model_comparison(
        data["hours"],
        data["total_sess_kw"],
        data["total_tc_kw"],
        data["n_vehicles"],
    )
    return _save_fig(fig, fig_dir, _figure_filename(6), fmt)


def _generate_ordered_delay_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess
    data = analyse_ordered_delay(cfg_tc)
    fig = figures.fig_ordered_delay(
        data["hours"],
        data["p_uncontrolled"],
        data["p_no_delay"],
        data["p_with_delay"],
        data["n_vehicles"],
        model_label=data["model_label"],
        summary=data["summary"],
    )
    tables.export_ordered_delay(
        tbl_dir / "ordered_delay.csv",
        data["hours"],
        data["p_uncontrolled"],
        data["p_no_delay"],
        data["p_with_delay"],
    )
    _save_json(data, data_dir / "ordered_delay.json")
    return _save_fig(fig, fig_dir, _figure_filename(7), fmt)


def _generate_strategy_comparison_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    data = analyse_strategies(cfg_tc, cfg_sess)
    fig = figures.fig_strategy_comparison(
        data["tc_results"],
        data["sess_results"],
        case_label=cfg_tc.case.name.upper().replace("IEEE", "IEEE "),
        load_scale=cfg_tc.case.load_scale,
        charge_kw=cfg_tc.ev.charge_power_kw,
    )
    tables.export_strategy_comparison(
        tbl_dir / "strategy_comparison.csv",
        data["tc_results"],
        data["sess_results"],
    )
    _save_json(data, data_dir / "strategy_comparison.json")
    return _save_fig(fig, fig_dir, _figure_filename(8), fmt)


def _generate_method_comparison_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess
    data = analyse_methods(cfg_tc)
    fig = figures.fig_method_comparison(data["method_results"], data["method_times"])
    tables.export_method_comparison(
        tbl_dir / "method_comparison.csv",
        data["method_results"],
        data["method_times"],
    )
    _save_json(data, data_dir / "method_comparison.json")
    return _save_fig(fig, fig_dir, _figure_filename(9), fmt)


def _generate_voltage_sensitivity_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess, tbl_dir, data_dir
    data = analyse_voltage_sensitivity(cfg_tc)
    fig = figures.fig_voltage_sensitivity(
        data["bus_ids"],
        data["voltage_margin"],
        data["sensitivity_diagonal"],
        case_label=cfg_tc.case.name.upper().replace("IEEE", "IEEE "),
        load_scale=cfg_tc.case.load_scale,
    )
    return _save_fig(fig, fig_dir, _figure_filename(10), fmt)


def _generate_parameter_sweep_figure(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    del cfg_sess
    data = analyse_parameter_sweep(cfg_tc)
    fig = figures.fig_parameter_sweep(
        data["load_scales"],
        data["charge_powers"],
        data["n_star_grid"],
    )
    tables.export_parameter_sweep(
        tbl_dir / "parameter_sweep.csv",
        data["load_scales"],
        data["charge_powers"],
        data["n_star_grid"],
    )
    _save_json(
        {
            "load_scales": data["load_scales"],
            "charge_powers": data["charge_powers"],
            "n_star_grid": data["n_star_grid"].tolist(),
        },
        data_dir / "parameter_sweep.json",
    )
    return _save_fig(fig, fig_dir, _figure_filename(11), fmt)


FIGURE_GENERATORS: dict[int, FigureGenerator] = {
    1: _generate_input_distributions_figure,
    2: _generate_soc_evolution_figure,
    3: _generate_charging_load_figure,
    4: _generate_risk_curve_figure,
    5: _generate_bus_voltage_profile_figure,
    6: _generate_model_comparison_figure,
    7: _generate_ordered_delay_figure,
    8: _generate_strategy_comparison_figure,
    9: _generate_method_comparison_figure,
    10: _generate_voltage_sensitivity_figure,
    11: _generate_parameter_sweep_figure,
}


# ----------------------------------------------------------------------
# Main report generator
# ----------------------------------------------------------------------


def generate_report(
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig | None = None,
    output_dir: Path = Path("output"),
    figure_ids: list[int] | None = None,
    fmt: str = "png",
) -> dict[int, Path]:
    """Generate selected (or all) figures and tables.

    Args:
        cfg_tc: Primary config (tripchain model).
        cfg_sess: Session model config. If None, loads configs/example.yaml.
        output_dir: Root output directory.
        figure_ids: Which figures to generate (default: all).
        fmt: Image format (png, pdf, svg).

    Returns:
        Mapping of figure_id -> saved file path.
    """
    apply_style()

    if cfg_sess is None:
        cfg_sess = load_config(Path("configs/example.yaml"))

    fig_dir = output_dir / "figures"
    tbl_dir = output_dir / "tables"
    data_dir = output_dir / "data"
    for d in (fig_dir, tbl_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    ids = figure_ids if figure_ids is not None else sorted(FIGURE_REGISTRY.keys())
    saved: dict[int, Path] = {}

    for fid in ids:
        if fid not in FIGURE_REGISTRY:
            _log(f"  Unknown figure id: {fid}, skipping")
            continue

        name, desc, heavy = FIGURE_REGISTRY[fid]
        _log(f"[{fid:2d}] {desc} ...")
        t0 = time.time()

        try:
            path = _generate_one(fid, name, cfg_tc, cfg_sess, fig_dir, tbl_dir, data_dir, fmt)
            saved[fid] = path
            elapsed = time.time() - t0
            _log(f"     -> {path.name}  ({elapsed:.1f}s)")
        except Exception as e:
            _log(f"     !! Error: {e}")

    _log(f"\nDone. {len(saved)}/{len(ids)} figures saved to {fig_dir}/")
    return saved


def _generate_one(
    fid: int,
    name: str,
    cfg_tc: ProjectConfig,
    cfg_sess: ProjectConfig,
    fig_dir: Path,
    tbl_dir: Path,
    data_dir: Path,
    fmt: str,
) -> Path:
    """Generate a single figure (+ associated table if applicable)."""
    del name
    generator = FIGURE_GENERATORS.get(fid)
    if generator is None:
        raise ValueError(f"Unknown figure id: {fid}")
    return generator(cfg_tc, cfg_sess, fig_dir, tbl_dir, data_dir, fmt)
