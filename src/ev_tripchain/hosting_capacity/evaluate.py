from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.cases import load_case
from ev_tripchain.grid.constraints import check_violations
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.monte_carlo import (
    MonteCarloEstimate,
    _wilson_ci_95,
    estimate_event_probability,
)
from ev_tripchain.mobility.profile import build_ev_profile_mw
from ev_tripchain.rng import make_rng_for


@dataclass(frozen=True)
class MCParallelContext:
    max_workers: int
    executor: ProcessPoolExecutor


@dataclass(frozen=True)
class _MCStaticContext:
    ev_idx: tuple[int, ...]
    buses: np.ndarray
    n_buses: int
    bus_score: np.ndarray
    vmin: float
    vmax: float
    line_max: float
    trafo_max: float


@dataclass
class _MCWorkerState:
    cfg: ProjectConfig
    net: Any
    ctx: _MCStaticContext | None


_MC_WORKER_STATE: _MCWorkerState | None = None


def _ensure_ev_load_elements(net: Any) -> list[int]:
    import pandapower as pp  # type: ignore

    if "ev_tripchain_kind" not in net.load.columns:
        net.load["ev_tripchain_kind"] = ""

    ev_idx = net.load.index[net.load["ev_tripchain_kind"] == "ev"].tolist()
    if ev_idx:
        return ev_idx

    # one EV load element per bus (except ext_grid bus if it exists)
    ext_buses = set(net.ext_grid.bus.tolist()) if hasattr(net, "ext_grid") else set()
    for bus in net.bus.index.tolist():
        if bus in ext_buses:
            continue
        idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"ev@{bus}")
        net.load.at[idx, "ev_tripchain_kind"] = "ev"
        ev_idx.append(idx)
    return ev_idx


def _static_voltage_margin_score(
    net: Any,
    *,
    buses: np.ndarray,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Compute a static 'grid headroom' score per EV-load column based on base-case voltages.

    This is a lightweight proxy for "node margin" mentioned in the opening report/literature.
    """
    try:
        run_powerflow(net)
        bus_ids = [int(b) for b in np.asarray(buses, dtype=int).reshape(-1).tolist()]
        vm = net.res_bus.loc[bus_ids, "vm_pu"].to_numpy(dtype=float)  # type: ignore[attr-defined]
        margin = np.minimum(vm - float(vmin), float(vmax) - vm)
        margin = np.clip(margin, 0.0, None)
        if not np.isfinite(margin).all():
            raise ValueError("non-finite voltage margin")
        return margin
    except Exception:
        # Fallback: neutral scores (no grid preference).
        return np.ones(int(np.asarray(buses).size), dtype=float)


def _base_case_is_safe(
    net: Any,
    *,
    ev_idx: list[int],
    vmin: float,
    vmax: float,
    line_max: float,
    trafo_max: float,
) -> bool:
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    try:
        run_powerflow(net)
    except Exception:
        return False
    return not check_violations(
        net,
        vmin=vmin,
        vmax=vmax,
        line_max=line_max,
        trafo_max=trafo_max,
    ).any_violation


def _prepare_mc_static_context(net: Any, cfg: ProjectConfig) -> _MCStaticContext | None:
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)
    vmin = cfg.constraints.vmin_pu
    vmax = cfg.constraints.vmax_pu
    line_max = cfg.constraints.line_loading_max_percent
    trafo_max = cfg.constraints.trafo_loading_max_percent

    base_case_safe = _base_case_is_safe(
        net,
        ev_idx=ev_idx,
        vmin=vmin,
        vmax=vmax,
        line_max=line_max,
        trafo_max=trafo_max,
    )
    if not base_case_safe:
        return None

    bus_score = _static_voltage_margin_score(
        net,
        buses=buses,
        vmin=vmin,
        vmax=vmax,
    )
    return _MCStaticContext(
        ev_idx=tuple(int(x) for x in ev_idx),
        buses=buses,
        n_buses=n_buses,
        bus_score=bus_score,
        vmin=float(vmin),
        vmax=float(vmax),
        line_max=float(line_max),
        trafo_max=float(trafo_max),
    )


def _simulate_event_on_net(
    *,
    net: Any,
    cfg: ProjectConfig,
    ctx: _MCStaticContext,
    n: int,
    rng_s: np.random.Generator,
) -> bool:
    ev_idx = list(ctx.ev_idx)
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    profile = build_ev_profile_mw(
        cfg=cfg,
        n_vehicles=n,
        buses=ctx.buses,
        n_buses=ctx.n_buses,
        bus_score=ctx.bus_score,
        rng=rng_s,
    )  # shape: (T, n_buses)

    total_per_step = profile.sum(axis=1)
    nonzero_mask = total_per_step > 1e-9
    if not nonzero_mask.any():
        return False
    nonzero_steps = np.where(nonzero_mask)[0]
    step_order = nonzero_steps[np.argsort(-total_per_step[nonzero_steps])]

    pf_init = "auto"
    for t in step_order:
        net.load.loc[ev_idx, "p_mw"] = profile[t, :]
        try:
            run_powerflow(net, init=pf_init)
            pf_init = "results"
        except Exception:
            pf_init = "auto"
            return True
        v = check_violations(
            net,
            vmin=ctx.vmin,
            vmax=ctx.vmax,
            line_max=ctx.line_max,
            trafo_max=ctx.trafo_max,
        )
        if v.any_violation:
            return True
    return False


def _init_mc_worker(cfg_data: dict[str, Any]) -> None:
    global _MC_WORKER_STATE
    cfg = ProjectConfig.model_validate(cfg_data)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ctx = _prepare_mc_static_context(net, cfg)
    _MC_WORKER_STATE = _MCWorkerState(cfg=cfg, net=net, ctx=ctx)


def _mc_worker_simulate(task: tuple[int, int]) -> bool:
    state = _MC_WORKER_STATE
    if state is None:
        raise RuntimeError("MC worker state is not initialized.")
    if state.ctx is None:
        return True

    n, scenario_idx = (int(task[0]), int(task[1]))
    rng_s = make_rng_for(int(state.cfg.seed), 9103, int(scenario_idx))
    return _simulate_event_on_net(
        net=state.net,
        cfg=state.cfg,
        ctx=state.ctx,
        n=n,
        rng_s=rng_s,
    )


def create_mc_parallel_context(cfg: ProjectConfig) -> MCParallelContext | None:
    hc = cfg.hosting_capacity
    if not hc.common_random_numbers:
        return None
    n_scenarios = int(max(hc.scenarios, 0))
    if n_scenarios < 2:
        return None
    max_workers = min(int(hc.resolved_parallel_workers), n_scenarios)
    if max_workers <= 1:
        return None
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_mc_worker,
        initargs=(cfg.model_dump(),),
    )
    return MCParallelContext(max_workers=max_workers, executor=executor)


def _estimate_event_probability_parallel(
    *,
    ctx: MCParallelContext,
    cfg: ProjectConfig,
    n: int,
    progress: Callable[[str], None] | None,
) -> MonteCarloEstimate:
    total = int(max(cfg.hosting_capacity.scenarios, 0))
    if total <= 0:
        return MonteCarloEstimate(n=0, n_events=0, p_hat=0.0, ci95_low=0.0, ci95_high=1.0)

    n_events = 0
    executed = 0
    batch = max(1, int(ctx.max_workers))

    while executed < total:
        stop = min(total, executed + batch)
        tasks = [(int(n), i) for i in range(executed, stop)]
        for hit in ctx.executor.map(_mc_worker_simulate, tasks):
            n_events += int(bool(hit))
        executed = stop

        if progress is not None:
            progress(f"scenarios {executed}/{total}, violations={n_events}")

        if cfg.hosting_capacity.risk_tolerance is not None and executed >= 5:
            ci_lo, ci_hi = _wilson_ci_95(n=executed, n_events=n_events)
            threshold = float(cfg.hosting_capacity.risk_tolerance)
            if ci_lo > threshold * 3:
                if progress is not None:
                    progress(f"early stop at {executed}/{total}: CI lower={ci_lo:.4f} > {threshold * 3:.4f}")
                break
            if ci_hi <= threshold:
                if progress is not None:
                    progress(f"early stop at {executed}/{total}: CI upper={ci_hi:.4f} <= {threshold:.4f}")
                break

    p_hat = n_events / executed if executed > 0 else 0.0
    ci_low, ci_high = _wilson_ci_95(n=executed, n_events=n_events)
    return MonteCarloEstimate(
        n=executed,
        n_events=n_events,
        p_hat=float(p_hat),
        ci95_low=float(ci_low),
        ci95_high=float(ci_high),
    )


def estimate_violation_probability_mc(
    net: Any,
    cfg: ProjectConfig,
    *,
    n: int,
    rng: np.random.Generator,
    progress: Callable[[str], None] | None = None,
    parallel: MCParallelContext | None = None,
) -> MonteCarloEstimate:
    """
    Monte Carlo estimate of violation probability under EV scale N.

    A scenario is counted as 'violating' if ANY time step violates ANY hard constraint.
    """
    static_ctx = _prepare_mc_static_context(net, cfg)
    if static_ctx is None:
        n_scenarios = int(max(cfg.hosting_capacity.scenarios, 0))
        return MonteCarloEstimate(
            n=n_scenarios,
            n_events=n_scenarios,
            p_hat=1.0,
            ci95_low=1.0,
            ci95_high=1.0,
        )

    if parallel is not None and int(n) > 0:
        return _estimate_event_probability_parallel(
            ctx=parallel,
            cfg=cfg,
            n=int(n),
            progress=progress,
        )

    def simulate_event(rng_s: np.random.Generator) -> bool:
        return _simulate_event_on_net(
            net=net,
            cfg=cfg,
            ctx=static_ctx,
            n=int(n),
            rng_s=rng_s,
        )

    scenario_rng = None
    if cfg.hosting_capacity.common_random_numbers:
        scenario_rng = lambda i: make_rng_for(int(cfg.seed), 9103, int(i))

    return estimate_event_probability(
        simulate_event,
        n_scenarios=cfg.hosting_capacity.scenarios,
        rng=rng,
        scenario_rng=scenario_rng,
        early_stop_threshold=cfg.hosting_capacity.risk_tolerance,
        progress=progress,
        progress_every=5,
    )


def estimate_violation_probability(
    net: Any,
    cfg: ProjectConfig,
    *,
    n: int,
    rng: np.random.Generator,
) -> float:
    """Backwards-compatible wrapper that returns p_hat only."""
    return float(estimate_violation_probability_mc(net, cfg, n=n, rng=rng).p_hat)
