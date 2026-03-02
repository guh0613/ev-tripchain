from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _ensure_outdir() -> Path:
    outdir = Path("docs") / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def plot_input_histograms(*, seed: int = 1, n_samples: int = 20000) -> Path:
    import matplotlib.pyplot as plt

    from ev_tripchain.mobility.tripchain_sampling import (
        TripChainSamplingParams,
        sample_daily_trip_chain,
    )

    rng = np.random.default_rng(seed)
    params = TripChainSamplingParams(
        n_zones=50,
        other_stops_mean=0.8,
        first_departure_mean="07:30",
        first_departure_std_minutes=35,
        work_duration_mean_minutes=8 * 60,
        work_duration_std_minutes=45,
        other_dwell_mean_minutes=50,
        other_dwell_std_minutes=25,
        travel_minutes_per_km=2.0,
        distance_km_mean=20.0,
        distance_km_std=10.0,
    )

    dep_min = np.empty(int(n_samples), dtype=float)
    daily_km = np.empty(int(n_samples), dtype=float)
    for i in range(int(n_samples)):
        tc = sample_daily_trip_chain(params, rng=rng)
        dep_min[i] = float(tc.stops[0].departure_minute)
        daily_km[i] = float(np.sum(tc.leg_distance_km))

    dep_h = dep_min / 60.0

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    ax[0].hist(dep_h, bins=48, density=True, color="#4C78A8", alpha=0.85)
    ax[0].set_xlabel("Departure time (hour)")
    ax[0].set_ylabel("Density")
    ax[0].set_title("Departure time distribution")
    ax[0].set_xlim(0, 24)
    ax[0].set_xticks(np.arange(0, 25, 4))

    ax[1].hist(daily_km, bins=50, density=True, color="#F58518", alpha=0.85)
    ax[1].set_xlabel("Daily distance (km)")
    ax[1].set_ylabel("Density")
    ax[1].set_title("Daily mileage distribution")

    out = _ensure_outdir() / "01_input_histograms.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_single_vehicle_soc(*, seed: int = 2) -> Path:
    import matplotlib.pyplot as plt

    from ev_tripchain.mobility.soc import SOCEvolutionParams, simulate_soc_and_charging_profile
    from ev_tripchain.mobility.trip_chain import Stop, TripChain

    rng = np.random.default_rng(seed)
    tc = TripChain(
        stops=[
            Stop(zone=0, arrival_minute=0, departure_minute=8 * 60, purpose="home"),
            Stop(zone=1, arrival_minute=8 * 60 + 30, departure_minute=17 * 60 + 30, purpose="work"),
            Stop(zone=0, arrival_minute=18 * 60, departure_minute=24 * 60, purpose="home"),
        ],
        leg_distance_km=[25.0, 25.0],
    )

    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=60.0,
        consumption_kwh_per_km=0.22,
        initial_soc_mean=0.75,
        initial_soc_std=0.0,
        charge_power_kw=7.2,
        charge_efficiency=0.92,
        charge_trigger_soc=0.60,
        charge_purposes=("home",),
        allow_initial_stop_charging=False,
    )

    step_minutes = 15
    n_steps = 96
    soc, p_kw = simulate_soc_and_charging_profile(
        tc,
        soc_params,
        step_minutes=step_minutes,
        n_steps=n_steps,
        rng=rng,
        initial_soc=0.75,
    )

    hours = np.arange(n_steps + 1) * (step_minutes / 60.0)
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    ax.plot(hours, soc, color="#54A24B", linewidth=2.2)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("SOC")
    ax.set_title("Single-vehicle SOC over 24 hours")

    charging_steps = np.where(p_kw > 1e-9)[0]
    if charging_steps.size:
        t0 = charging_steps.min() * (step_minutes / 60.0)
        t1 = (charging_steps.max() + 1) * (step_minutes / 60.0)
        ax.axvspan(t0, t1, color="#54A24B", alpha=0.10, label="Charging")
        ax.legend(loc="lower right")

    out = _ensure_outdir() / "02_single_vehicle_soc.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_total_charging_load(*, seed: int = 3, n_vehicles: int = 1500) -> Path:
    import matplotlib.pyplot as plt

    from ev_tripchain.mobility.mapping import NodeBusMapping
    from ev_tripchain.mobility.soc import SOCEvolutionParams
    from ev_tripchain.mobility.tripchain_profile import build_ev_profile_mw_tripchain
    from ev_tripchain.mobility.tripchain_sampling import TripChainSamplingParams

    rng = np.random.default_rng(seed)
    step_minutes = 15
    n_steps = 96

    n_nodes = 50
    mapping = NodeBusMapping(
        matrix=np.ones((n_nodes, 1), dtype=float),
        bus_ids=np.array([0], dtype=int),
    )

    trip_params = TripChainSamplingParams(
        n_zones=n_nodes,
        other_stops_mean=0.8,
        first_departure_mean="07:30",
        first_departure_std_minutes=35,
        work_duration_mean_minutes=8 * 60,
        work_duration_std_minutes=45,
        other_dwell_mean_minutes=50,
        other_dwell_std_minutes=25,
        travel_minutes_per_km=2.0,
        distance_km_mean=20.0,
        distance_km_std=10.0,
    )

    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=60.0,
        consumption_kwh_per_km=0.18,
        initial_soc_mean=0.75,
        initial_soc_std=0.12,
        charge_power_kw=7.2,
        charge_efficiency=0.92,
        charge_trigger_soc=0.50,
        charge_purposes=("home", "work"),
        allow_initial_stop_charging=False,
    )

    prof_mw = build_ev_profile_mw_tripchain(
        n_vehicles=int(n_vehicles),
        step_minutes=step_minutes,
        n_steps=n_steps,
        mapping=mapping,
        trip_params=trip_params,
        soc_params=soc_params,
        rng=rng,
    )
    total_kw = prof_mw[:, 0] * 1000.0

    hours = np.arange(n_steps) * (step_minutes / 60.0)
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    ax.plot(hours, total_kw, color="#B279A2", linewidth=2.2)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Total charging power (kW)")
    ax.set_title(f"Total charging load (uncontrolled), N={int(n_vehicles)}")

    out = _ensure_outdir() / "03_total_charging_load.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_risk_curve(*, result_path: str = "docs/results/tripchain_soc_result.json") -> Path:
    """Plot N vs violation probability (risk curve) from a pipeline result JSON."""
    import matplotlib.pyplot as plt

    p = Path(result_path)
    if not p.exists():
        raise FileNotFoundError(f"Result file not found: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))
    n_star = data["n_star"]
    eps = data["risk_tolerance"]
    curve = sorted(data["risk_curve"], key=lambda x: x[0])

    ns = [c[0] for c in curve]
    risks = [c[1] for c in curve]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(ns, risks, "o-", color="#E45756", linewidth=2, markersize=6, label="$\\hat{\\pi}(N)$")
    ax.axhline(y=eps, color="#4C78A8", linestyle="--", linewidth=1.5, label=f"$\\varepsilon$ = {eps}")
    if n_star > 0:
        ax.axvline(x=n_star, color="#54A24B", linestyle=":", linewidth=1.5, label=f"$N^*$ = {n_star}")
    ax.set_xlabel("Number of EVs (N)")
    ax.set_ylabel("Violation probability $\\hat{\\pi}(N)$")
    ax.set_title("Risk curve: EV hosting capacity")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    out = _ensure_outdir() / "04_risk_curve.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_bus_voltage_profile(*, n_vehicles: int = 1500, seed: int = 10) -> Path:
    """Plot 24h voltage trajectories for all buses at a given EV penetration level."""
    import matplotlib.pyplot as plt

    from ev_tripchain.config import load_config
    from ev_tripchain.grid.cases import load_case
    from ev_tripchain.grid.powerflow import run_powerflow
    from ev_tripchain.hosting_capacity.evaluate import _ensure_ev_load_elements
    from ev_tripchain.mobility.soc import SOCEvolutionParams
    from ev_tripchain.mobility.tripchain_profile import build_ev_profile_mw_tripchain
    from ev_tripchain.mobility.tripchain_sampling import TripChainSamplingParams

    rng = np.random.default_rng(seed)
    step_minutes = 15
    n_steps = 96

    cfg = load_config(Path("configs/tripchain_soc.yaml"))
    net = load_case(cfg.case.name, load_scale=cfg.case.load_scale)
    ev_idx = _ensure_ev_load_elements(net)
    buses = net.load.loc[ev_idx, "bus"].to_numpy()
    n_nodes = 50

    rng_map = np.random.default_rng(np.random.SeedSequence([42, 701]))
    from ev_tripchain.mobility.mapping import build_random_onehot_mapping

    mapping = build_random_onehot_mapping(n_nodes=n_nodes, bus_ids=buses, rng=rng_map)

    trip_params = TripChainSamplingParams(
        n_zones=n_nodes,
        other_stops_mean=0.8,
        first_departure_mean="07:30",
        first_departure_std_minutes=35,
        work_duration_mean_minutes=480,
        work_duration_std_minutes=45,
        other_dwell_mean_minutes=50,
        other_dwell_std_minutes=25,
        travel_minutes_per_km=2.0,
        distance_km_mean=20.0,
        distance_km_std=10.0,
    )
    soc_params = SOCEvolutionParams(
        battery_capacity_kwh=60.0,
        consumption_kwh_per_km=0.18,
        initial_soc_mean=0.75,
        initial_soc_std=0.12,
        charge_power_kw=7.2,
        charge_efficiency=0.92,
        charge_trigger_soc=0.50,
        charge_purposes=("home", "work"),
        allow_initial_stop_charging=False,
    )

    profile = build_ev_profile_mw_tripchain(
        n_vehicles=n_vehicles,
        step_minutes=step_minutes,
        n_steps=n_steps,
        mapping=mapping,
        trip_params=trip_params,
        soc_params=soc_params,
        rng=rng,
    )

    all_vm = np.zeros((n_steps, len(net.bus)), dtype=float)
    for t in range(n_steps):
        net.load.loc[ev_idx, "p_mw"] = profile[t, :]
        run_powerflow(net)
        all_vm[t, :] = net.res_bus.vm_pu.to_numpy()

    hours = np.arange(n_steps) * (step_minutes / 60.0)
    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)

    for b in range(all_vm.shape[1]):
        ax.plot(hours, all_vm[:, b], linewidth=0.8, alpha=0.7, label=f"Bus {b}")

    ax.axhline(y=0.95, color="red", linestyle="--", linewidth=1.2, label="$V_{min}$ = 0.95")
    ax.axhline(y=1.05, color="red", linestyle="--", linewidth=1.2, label="$V_{max}$ = 1.05")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Voltage (p.u.)")
    ax.set_title(f"Bus voltage profiles, N={n_vehicles} EVs")
    ax.grid(True, alpha=0.3)

    out = _ensure_outdir() / "05_bus_voltage_profile.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_load_model_comparison(*, seed: int = 5, n_vehicles: int = 1000) -> Path:
    """Compare total load curves from session-based and trip-chain models."""
    import matplotlib.pyplot as plt

    from pathlib import Path as P

    from ev_tripchain.config import load_config
    from ev_tripchain.grid.cases import load_case
    from ev_tripchain.hosting_capacity.evaluate import _ensure_ev_load_elements
    from ev_tripchain.mobility.profile import build_ev_profile_mw

    step_minutes = 15
    n_steps = 96
    hours = np.arange(n_steps) * (step_minutes / 60.0)

    # session-based model
    cfg_sess = load_config(P("configs/example.yaml"))
    net_sess = load_case(cfg_sess.case.name, load_scale=cfg_sess.case.load_scale)
    ev_idx_sess = _ensure_ev_load_elements(net_sess)
    buses_sess = net_sess.load.loc[ev_idx_sess, "bus"].to_numpy()
    rng1 = np.random.default_rng(seed)
    prof_sess = build_ev_profile_mw(
        cfg=cfg_sess, n_vehicles=n_vehicles, buses=buses_sess, n_buses=len(ev_idx_sess), rng=rng1
    )
    total_sess_kw = prof_sess.sum(axis=1) * 1000.0

    # trip-chain model
    cfg_tc = load_config(P("configs/tripchain_soc.yaml"))
    net_tc = load_case(cfg_tc.case.name, load_scale=cfg_tc.case.load_scale)
    ev_idx_tc = _ensure_ev_load_elements(net_tc)
    buses_tc = net_tc.load.loc[ev_idx_tc, "bus"].to_numpy()
    rng2 = np.random.default_rng(seed)
    prof_tc = build_ev_profile_mw(
        cfg=cfg_tc, n_vehicles=n_vehicles, buses=buses_tc, n_buses=len(ev_idx_tc), rng=rng2
    )
    total_tc_kw = prof_tc.sum(axis=1) * 1000.0

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.plot(hours, total_sess_kw, color="#4C78A8", linewidth=2.2, label="Session-based model")
    ax.plot(hours, total_tc_kw, color="#E45756", linewidth=2.2, label="Trip-chain + SOC model")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Total charging power (kW)")
    ax.set_title(f"Load model comparison, N={n_vehicles}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = _ensure_outdir() / "06_load_model_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    p1 = plot_input_histograms()
    print(str(p1))

    p2 = plot_single_vehicle_soc()
    print(str(p2))

    p3 = plot_total_charging_load()
    print(str(p3))

    try:
        p4 = plot_risk_curve()
        print(str(p4))
    except FileNotFoundError:
        print("Skipping risk curve (no result file)")

    p5 = plot_bus_voltage_profile()
    print(str(p5))

    p6 = plot_load_model_comparison()
    print(str(p6))


if __name__ == "__main__":
    main()
