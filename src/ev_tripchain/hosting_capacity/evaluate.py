from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from ev_tripchain.config import ProjectConfig
from ev_tripchain.grid.cases import load_case
from ev_tripchain.grid.constraints import ConstraintEvaluation, evaluate_constraints
from ev_tripchain.grid.powerflow import run_powerflow
from ev_tripchain.hosting_capacity.monte_carlo import (
    HardConstraintProbabilityBreakdown,
    MonteCarloEstimate,
    SoftMetricAggregate,
    build_probability_estimate,
)
from ev_tripchain.hosting_capacity.sensitivity import (
    VoltageSensitivityModel,
    build_voltage_sensitivity_model,
)
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
    navigation_voltage_model: VoltageSensitivityModel | None
    vmin: float
    vmax: float
    line_max: float
    trafo_max: float
    nominal_voltage_pu: float


@dataclass(frozen=True)
class ScenarioHardConstraintFlags:
    voltage_limit_exceedance: bool = False
    line_limit_exceedance: bool = False
    trafo_limit_exceedance: bool = False
    solver_failure: bool = False

    @property
    def any_limit_exceedance(self) -> bool:
        return (
            self.voltage_limit_exceedance
            or self.line_limit_exceedance
            or self.trafo_limit_exceedance
            or self.solver_failure
        )


@dataclass(frozen=True)
class ScenarioSoftMetricSummary:
    peak_voltage_deviation_pu: float = 0.0
    peak_network_loss_mw: float = 0.0
    peak_line_loading_percent: float = 0.0
    peak_trafo_loading_percent: float = 0.0


@dataclass(frozen=True)
class ScenarioEvaluation:
    hard: ScenarioHardConstraintFlags
    soft: ScenarioSoftMetricSummary


@dataclass
class _MCWorkerState:
    cfg: ProjectConfig
    net: Any
    ctx: _MCStaticContext | None


@dataclass
class _MCAccumulator:
    executed: int = 0
    any_limit_n_events: int = 0
    voltage_n_events: int = 0
    line_n_events: int = 0
    trafo_n_events: int = 0
    solver_failure_n_events: int = 0
    sum_peak_voltage_deviation_pu: float = 0.0
    max_peak_voltage_deviation_pu: float = 0.0
    sum_peak_network_loss_mw: float = 0.0
    max_peak_network_loss_mw: float = 0.0
    sum_peak_line_loading_percent: float = 0.0
    max_peak_line_loading_percent: float = 0.0
    sum_peak_trafo_loading_percent: float = 0.0
    max_peak_trafo_loading_percent: float = 0.0

    def add(self, scenario: ScenarioEvaluation) -> None:
        self.executed += 1

        hard = scenario.hard
        soft = scenario.soft
        self.any_limit_n_events += int(hard.any_limit_exceedance)
        self.voltage_n_events += int(hard.voltage_limit_exceedance)
        self.line_n_events += int(hard.line_limit_exceedance)
        self.trafo_n_events += int(hard.trafo_limit_exceedance)
        self.solver_failure_n_events += int(hard.solver_failure)

        self.sum_peak_voltage_deviation_pu += float(soft.peak_voltage_deviation_pu)
        self.max_peak_voltage_deviation_pu = max(
            self.max_peak_voltage_deviation_pu,
            float(soft.peak_voltage_deviation_pu),
        )
        self.sum_peak_network_loss_mw += float(soft.peak_network_loss_mw)
        self.max_peak_network_loss_mw = max(
            self.max_peak_network_loss_mw,
            float(soft.peak_network_loss_mw),
        )
        self.sum_peak_line_loading_percent += float(soft.peak_line_loading_percent)
        self.max_peak_line_loading_percent = max(
            self.max_peak_line_loading_percent,
            float(soft.peak_line_loading_percent),
        )
        self.sum_peak_trafo_loading_percent += float(soft.peak_trafo_loading_percent)
        self.max_peak_trafo_loading_percent = max(
            self.max_peak_trafo_loading_percent,
            float(soft.peak_trafo_loading_percent),
        )

    def to_estimate(self) -> MonteCarloEstimate:
        any_limit = build_probability_estimate(
            n=self.executed,
            n_events=self.any_limit_n_events,
        )
        hard_constraints = HardConstraintProbabilityBreakdown(
            any_limit_exceedance=any_limit,
            voltage_limit_exceedance=build_probability_estimate(
                n=self.executed,
                n_events=self.voltage_n_events,
            ),
            line_limit_exceedance=build_probability_estimate(
                n=self.executed,
                n_events=self.line_n_events,
            ),
            trafo_limit_exceedance=build_probability_estimate(
                n=self.executed,
                n_events=self.trafo_n_events,
            ),
            solver_failure=build_probability_estimate(
                n=self.executed,
                n_events=self.solver_failure_n_events,
            ),
        )
        soft_metrics = SoftMetricAggregate(
            mean_peak_voltage_deviation_pu=(
                self.sum_peak_voltage_deviation_pu / self.executed
                if self.executed > 0
                else 0.0
            ),
            max_peak_voltage_deviation_pu=self.max_peak_voltage_deviation_pu,
            mean_peak_network_loss_mw=(
                self.sum_peak_network_loss_mw / self.executed
                if self.executed > 0
                else 0.0
            ),
            max_peak_network_loss_mw=self.max_peak_network_loss_mw,
            mean_peak_line_loading_percent=(
                self.sum_peak_line_loading_percent / self.executed
                if self.executed > 0
                else 0.0
            ),
            max_peak_line_loading_percent=self.max_peak_line_loading_percent,
            mean_peak_trafo_loading_percent=(
                self.sum_peak_trafo_loading_percent / self.executed
                if self.executed > 0
                else 0.0
            ),
            max_peak_trafo_loading_percent=self.max_peak_trafo_loading_percent,
        )
        return MonteCarloEstimate(
            n=self.executed,
            n_events=any_limit.n_events,
            p_hat=any_limit.p_hat,
            ci95_low=any_limit.ci95_low,
            ci95_high=any_limit.ci95_high,
            hard_constraints=hard_constraints,
            soft_metrics=soft_metrics,
        )


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
    Compute a static grid-headroom score per EV-load column based on base-case voltages.

    This is a lightweight proxy for node margin mentioned in the opening report/literature.
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


def _evaluate_base_case_constraints(
    net: Any,
    *,
    ev_idx: list[int],
    vmin: float,
    vmax: float,
    line_max: float,
    trafo_max: float,
    nominal_voltage_pu: float,
) -> ConstraintEvaluation | None:
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    try:
        run_powerflow(net)
    except Exception:
        return None
    return evaluate_constraints(
        net,
        vmin=vmin,
        vmax=vmax,
        line_max=line_max,
        trafo_max=trafo_max,
        nominal_voltage_pu=nominal_voltage_pu,
    )


def _prepare_mc_static_context(net: Any, cfg: ProjectConfig) -> _MCStaticContext | None:
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_buses = len(ev_idx)
    hard_cfg = cfg.constraints.hard
    soft_cfg = cfg.constraints.soft

    base_case = _evaluate_base_case_constraints(
        net,
        ev_idx=ev_idx,
        vmin=float(hard_cfg.vmin_pu),
        vmax=float(hard_cfg.vmax_pu),
        line_max=float(hard_cfg.line_loading_max_percent),
        trafo_max=float(hard_cfg.trafo_loading_max_percent),
        nominal_voltage_pu=float(soft_cfg.nominal_voltage_pu),
    )
    if base_case is None or base_case.hard.any_exceedance:
        return None

    bus_score = _static_voltage_margin_score(
        net,
        buses=buses,
        vmin=float(hard_cfg.vmin_pu),
        vmax=float(hard_cfg.vmax_pu),
    )
    navigation_voltage_model: VoltageSensitivityModel | None = None
    if cfg.strategy.name == "navigation" and cfg.strategy.navigation.dynamic_scoring:
        try:
            navigation_voltage_model = build_voltage_sensitivity_model(
                net,
                ev_idx=ev_idx,
                buses=buses,
                vmin=float(hard_cfg.vmin_pu),
                vmax=float(hard_cfg.vmax_pu),
                line_max=float(hard_cfg.line_loading_max_percent),
            )
        except Exception:
            navigation_voltage_model = None
    return _MCStaticContext(
        ev_idx=tuple(int(x) for x in ev_idx),
        buses=buses,
        n_buses=n_buses,
        bus_score=bus_score,
        navigation_voltage_model=navigation_voltage_model,
        vmin=float(hard_cfg.vmin_pu),
        vmax=float(hard_cfg.vmax_pu),
        line_max=float(hard_cfg.line_loading_max_percent),
        trafo_max=float(hard_cfg.trafo_loading_max_percent),
        nominal_voltage_pu=float(soft_cfg.nominal_voltage_pu),
    )


def _finalize_scenario_evaluation(
    *,
    hard: ScenarioHardConstraintFlags,
    peak_voltage_deviation_pu: float,
    peak_network_loss_mw: float,
    peak_line_loading_percent: float,
    peak_trafo_loading_percent: float,
) -> ScenarioEvaluation:
    return ScenarioEvaluation(
        hard=hard,
        soft=ScenarioSoftMetricSummary(
            peak_voltage_deviation_pu=float(peak_voltage_deviation_pu),
            peak_network_loss_mw=float(peak_network_loss_mw),
            peak_line_loading_percent=float(peak_line_loading_percent),
            peak_trafo_loading_percent=float(peak_trafo_loading_percent),
        ),
    )


def _simulate_scenario_on_net(
    *,
    net: Any,
    cfg: ProjectConfig,
    ctx: _MCStaticContext,
    n: int,
    rng_s: np.random.Generator,
) -> ScenarioEvaluation:
    ev_idx = list(ctx.ev_idx)
    net.load.loc[ev_idx, "p_mw"] = 0.0
    net.load.loc[ev_idx, "q_mvar"] = 0.0
    profile = build_ev_profile_mw(
        cfg=cfg,
        n_vehicles=n,
        buses=ctx.buses,
        n_buses=ctx.n_buses,
        bus_score=ctx.bus_score,
        navigation_voltage_model=ctx.navigation_voltage_model,
        rng=rng_s,
    )  # shape: (T, n_buses)

    total_per_step = profile.sum(axis=1)
    nonzero_mask = total_per_step > 1e-9
    if not nonzero_mask.any():
        return ScenarioEvaluation(
            hard=ScenarioHardConstraintFlags(),
            soft=ScenarioSoftMetricSummary(),
        )
    nonzero_steps = np.where(nonzero_mask)[0]
    step_order = nonzero_steps[np.argsort(-total_per_step[nonzero_steps])]

    hard = ScenarioHardConstraintFlags()
    peak_voltage_deviation_pu = 0.0
    peak_network_loss_mw = 0.0
    peak_line_loading_percent = 0.0
    peak_trafo_loading_percent = 0.0

    pf_init = "auto"
    for t in step_order:
        net.load.loc[ev_idx, "p_mw"] = profile[t, :]
        try:
            run_powerflow(net, init=pf_init)
            pf_init = "results"
        except Exception:
            return _finalize_scenario_evaluation(
                hard=ScenarioHardConstraintFlags(
                    voltage_limit_exceedance=hard.voltage_limit_exceedance,
                    line_limit_exceedance=hard.line_limit_exceedance,
                    trafo_limit_exceedance=hard.trafo_limit_exceedance,
                    solver_failure=True,
                ),
                peak_voltage_deviation_pu=peak_voltage_deviation_pu,
                peak_network_loss_mw=peak_network_loss_mw,
                peak_line_loading_percent=peak_line_loading_percent,
                peak_trafo_loading_percent=peak_trafo_loading_percent,
            )

        assessment = evaluate_constraints(
            net,
            vmin=ctx.vmin,
            vmax=ctx.vmax,
            line_max=ctx.line_max,
            trafo_max=ctx.trafo_max,
            nominal_voltage_pu=ctx.nominal_voltage_pu,
        )
        peak_voltage_deviation_pu = max(
            peak_voltage_deviation_pu,
            float(assessment.soft.voltage_deviation_max_pu),
        )
        peak_network_loss_mw = max(
            peak_network_loss_mw,
            float(assessment.soft.network_loss_mw),
        )
        peak_line_loading_percent = max(
            peak_line_loading_percent,
            float(assessment.soft.line_loading_peak_percent),
        )
        peak_trafo_loading_percent = max(
            peak_trafo_loading_percent,
            float(assessment.soft.trafo_loading_peak_percent),
        )
        hard = ScenarioHardConstraintFlags(
            voltage_limit_exceedance=(
                hard.voltage_limit_exceedance or assessment.hard.voltage_exceedance
            ),
            line_limit_exceedance=(
                hard.line_limit_exceedance or assessment.hard.line_overload
            ),
            trafo_limit_exceedance=(
                hard.trafo_limit_exceedance or assessment.hard.trafo_overload
            ),
            solver_failure=hard.solver_failure,
        )
        if assessment.hard.any_exceedance:
            return _finalize_scenario_evaluation(
                hard=hard,
                peak_voltage_deviation_pu=peak_voltage_deviation_pu,
                peak_network_loss_mw=peak_network_loss_mw,
                peak_line_loading_percent=peak_line_loading_percent,
                peak_trafo_loading_percent=peak_trafo_loading_percent,
            )

    return _finalize_scenario_evaluation(
        hard=hard,
        peak_voltage_deviation_pu=peak_voltage_deviation_pu,
        peak_network_loss_mw=peak_network_loss_mw,
        peak_line_loading_percent=peak_line_loading_percent,
        peak_trafo_loading_percent=peak_trafo_loading_percent,
    )


def _unsafe_base_case_estimate(
    *,
    n_scenarios: int,
    base_case: ConstraintEvaluation | None,
) -> MonteCarloEstimate:
    total = int(max(n_scenarios, 0))
    any_limit_n_events = total
    voltage_n_events = total if base_case is not None and base_case.hard.voltage_exceedance else 0
    line_n_events = total if base_case is not None and base_case.hard.line_overload else 0
    trafo_n_events = total if base_case is not None and base_case.hard.trafo_overload else 0
    solver_failure_n_events = total if base_case is None else 0

    any_limit = build_probability_estimate(n=total, n_events=any_limit_n_events)
    hard_constraints = HardConstraintProbabilityBreakdown(
        any_limit_exceedance=any_limit,
        voltage_limit_exceedance=build_probability_estimate(
            n=total,
            n_events=voltage_n_events,
        ),
        line_limit_exceedance=build_probability_estimate(
            n=total,
            n_events=line_n_events,
        ),
        trafo_limit_exceedance=build_probability_estimate(
            n=total,
            n_events=trafo_n_events,
        ),
        solver_failure=build_probability_estimate(
            n=total,
            n_events=solver_failure_n_events,
        ),
    )
    soft_metrics = SoftMetricAggregate(
        mean_peak_voltage_deviation_pu=(
            float(base_case.soft.voltage_deviation_max_pu)
            if base_case is not None
            else 0.0
        ),
        max_peak_voltage_deviation_pu=(
            float(base_case.soft.voltage_deviation_max_pu)
            if base_case is not None
            else 0.0
        ),
        mean_peak_network_loss_mw=(
            float(base_case.soft.network_loss_mw)
            if base_case is not None
            else 0.0
        ),
        max_peak_network_loss_mw=(
            float(base_case.soft.network_loss_mw)
            if base_case is not None
            else 0.0
        ),
        mean_peak_line_loading_percent=(
            float(base_case.soft.line_loading_peak_percent)
            if base_case is not None
            else 0.0
        ),
        max_peak_line_loading_percent=(
            float(base_case.soft.line_loading_peak_percent)
            if base_case is not None
            else 0.0
        ),
        mean_peak_trafo_loading_percent=(
            float(base_case.soft.trafo_loading_peak_percent)
            if base_case is not None
            else 0.0
        ),
        max_peak_trafo_loading_percent=(
            float(base_case.soft.trafo_loading_peak_percent)
            if base_case is not None
            else 0.0
        ),
    )
    return MonteCarloEstimate(
        n=total,
        n_events=any_limit.n_events,
        p_hat=any_limit.p_hat,
        ci95_low=any_limit.ci95_low,
        ci95_high=any_limit.ci95_high,
        hard_constraints=hard_constraints,
        soft_metrics=soft_metrics,
    )


def _init_mc_worker(cfg_data: dict[str, Any]) -> None:
    global _MC_WORKER_STATE
    cfg = ProjectConfig.model_validate(cfg_data)
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ctx = _prepare_mc_static_context(net, cfg)
    _MC_WORKER_STATE = _MCWorkerState(cfg=cfg, net=net, ctx=ctx)


def _mc_worker_simulate(task: tuple[int, int]) -> ScenarioEvaluation:
    state = _MC_WORKER_STATE
    if state is None:
        raise RuntimeError("MC worker state is not initialized.")
    if state.ctx is None:
        ev_idx = _ensure_ev_load_elements(state.net)
        base_case = _evaluate_base_case_constraints(
            state.net,
            ev_idx=ev_idx,
            vmin=float(state.cfg.constraints.hard.vmin_pu),
            vmax=float(state.cfg.constraints.hard.vmax_pu),
            line_max=float(state.cfg.constraints.hard.line_loading_max_percent),
            trafo_max=float(state.cfg.constraints.hard.trafo_loading_max_percent),
            nominal_voltage_pu=float(state.cfg.constraints.soft.nominal_voltage_pu),
        )
        if base_case is None:
            return ScenarioEvaluation(
                hard=ScenarioHardConstraintFlags(solver_failure=True),
                soft=ScenarioSoftMetricSummary(),
            )
        return ScenarioEvaluation(
            hard=ScenarioHardConstraintFlags(
                voltage_limit_exceedance=base_case.hard.voltage_exceedance,
                line_limit_exceedance=base_case.hard.line_overload,
                trafo_limit_exceedance=base_case.hard.trafo_overload,
            ),
            soft=ScenarioSoftMetricSummary(
                peak_voltage_deviation_pu=float(base_case.soft.voltage_deviation_max_pu),
                peak_network_loss_mw=float(base_case.soft.network_loss_mw),
                peak_line_loading_percent=float(base_case.soft.line_loading_peak_percent),
                peak_trafo_loading_percent=float(base_case.soft.trafo_loading_peak_percent),
            ),
        )

    n, scenario_idx = (int(task[0]), int(task[1]))
    rng_s = make_rng_for(int(state.cfg.seed), 9103, int(scenario_idx))
    return _simulate_scenario_on_net(
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


def _estimate_hard_exceedance_probability_parallel(
    *,
    ctx: MCParallelContext,
    cfg: ProjectConfig,
    n: int,
    progress: Callable[[str], None] | None,
) -> MonteCarloEstimate:
    total = int(max(cfg.hosting_capacity.scenarios, 0))
    acc = _MCAccumulator()
    batch = max(1, int(ctx.max_workers))

    while acc.executed < total:
        stop = min(total, acc.executed + batch)
        tasks = [(int(n), i) for i in range(acc.executed, stop)]
        for scenario in ctx.executor.map(_mc_worker_simulate, tasks):
            acc.add(scenario)

        if progress is not None:
            progress(
                f"scenarios {acc.executed}/{total}, "
                f"hard_limit_exceedances={acc.any_limit_n_events}"
            )

        if cfg.hosting_capacity.risk_tolerance is not None and acc.executed >= 5:
            any_limit = build_probability_estimate(
                n=acc.executed,
                n_events=acc.any_limit_n_events,
            )
            threshold = float(cfg.hosting_capacity.risk_tolerance)
            if any_limit.ci95_low > threshold * 3:
                if progress is not None:
                    progress(
                        "early stop at "
                        f"{acc.executed}/{total}: "
                        f"CI lower={any_limit.ci95_low:.4f} > {threshold * 3:.4f}"
                    )
                break
            if any_limit.ci95_high <= threshold:
                if progress is not None:
                    progress(
                        "early stop at "
                        f"{acc.executed}/{total}: "
                        f"CI upper={any_limit.ci95_high:.4f} <= {threshold:.4f}"
                    )
                break

    return acc.to_estimate()


def _estimate_hard_exceedance_probability_serial(
    *,
    net: Any,
    cfg: ProjectConfig,
    ctx: _MCStaticContext,
    n: int,
    rng: np.random.Generator,
    progress: Callable[[str], None] | None,
) -> MonteCarloEstimate:
    total = int(max(cfg.hosting_capacity.scenarios, 0))
    acc = _MCAccumulator()
    scenario_rng = None
    if cfg.hosting_capacity.common_random_numbers:
        scenario_rng = lambda i: make_rng_for(int(cfg.seed), 9103, int(i))

    for i in range(total):
        rng_s = scenario_rng(i) if scenario_rng is not None else rng
        scenario = _simulate_scenario_on_net(
            net=net,
            cfg=cfg,
            ctx=ctx,
            n=int(n),
            rng_s=rng_s,
        )
        acc.add(scenario)

        if progress is not None:
            should_report = (i == 0) or (i + 1 == total)
            if (i + 1) % 5 == 0:
                should_report = True
            if should_report:
                progress(
                    f"scenarios {i + 1}/{total}, "
                    f"hard_limit_exceedances={acc.any_limit_n_events}"
                )

        if cfg.hosting_capacity.risk_tolerance is not None and acc.executed >= 5:
            any_limit = build_probability_estimate(
                n=acc.executed,
                n_events=acc.any_limit_n_events,
            )
            threshold = float(cfg.hosting_capacity.risk_tolerance)
            if any_limit.ci95_low > threshold * 3:
                if progress is not None:
                    progress(
                        "early stop at "
                        f"{acc.executed}/{total}: "
                        f"CI lower={any_limit.ci95_low:.4f} > {threshold * 3:.4f}"
                    )
                break
            if any_limit.ci95_high <= threshold:
                if progress is not None:
                    progress(
                        "early stop at "
                        f"{acc.executed}/{total}: "
                        f"CI upper={any_limit.ci95_high:.4f} <= {threshold:.4f}"
                    )
                break

    return acc.to_estimate()


def estimate_hard_exceedance_probability_mc(
    net: Any,
    cfg: ProjectConfig,
    *,
    n: int,
    rng: np.random.Generator,
    progress: Callable[[str], None] | None = None,
    parallel: MCParallelContext | None = None,
) -> MonteCarloEstimate:
    """
    Monte Carlo estimate of hard-constraint exceedance probability under EV scale N.

    A scenario is counted as an exceedance scenario if any time step exceeds any
    hard limit, or if power flow fails to converge.
    """
    static_ctx = _prepare_mc_static_context(net, cfg)
    if static_ctx is None:
        ev_idx = _ensure_ev_load_elements(net)
        base_case = _evaluate_base_case_constraints(
            net,
            ev_idx=ev_idx,
            vmin=float(cfg.constraints.hard.vmin_pu),
            vmax=float(cfg.constraints.hard.vmax_pu),
            line_max=float(cfg.constraints.hard.line_loading_max_percent),
            trafo_max=float(cfg.constraints.hard.trafo_loading_max_percent),
            nominal_voltage_pu=float(cfg.constraints.soft.nominal_voltage_pu),
        )
        return _unsafe_base_case_estimate(
            n_scenarios=cfg.hosting_capacity.scenarios,
            base_case=base_case,
        )

    if parallel is not None and int(n) > 0:
        return _estimate_hard_exceedance_probability_parallel(
            ctx=parallel,
            cfg=cfg,
            n=int(n),
            progress=progress,
        )

    return _estimate_hard_exceedance_probability_serial(
        net=net,
        cfg=cfg,
        ctx=static_ctx,
        n=int(n),
        rng=rng,
        progress=progress,
    )


def estimate_hard_exceedance_probability(
    net: Any,
    cfg: ProjectConfig,
    *,
    n: int,
    rng: np.random.Generator,
) -> float:
    """Convenience wrapper that returns the main hard-limit p_hat only."""
    return float(
        estimate_hard_exceedance_probability_mc(net, cfg, n=n, rng=rng).p_hat
    )


# Local import to avoid a heavier mobility dependency during module import.
from ev_tripchain.mobility.profile import build_ev_profile_mw  # noqa: E402
