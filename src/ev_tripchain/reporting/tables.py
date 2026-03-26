"""Data export to CSV for thesis tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def _write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path


def export_risk_curve(
    path: Path,
    risk_points: list[dict[str, Any]],
    n_star: int,
) -> Path:
    pts = sorted(risk_points, key=lambda x: x["n"])
    rows = [
        [
            p["n"],
            f"{p['p_hat']:.4f}",
            f"{p['ci95_low']:.4f}",
            f"{p['ci95_high']:.4f}",
            f"{p.get('hard_constraints', {}).get('voltage_limit_exceedance', {}).get('p_hat', 0.0):.4f}",
            f"{p.get('hard_constraints', {}).get('line_limit_exceedance', {}).get('p_hat', 0.0):.4f}",
            f"{p.get('hard_constraints', {}).get('trafo_limit_exceedance', {}).get('p_hat', 0.0):.4f}",
            f"{p.get('hard_constraints', {}).get('solver_failure', {}).get('p_hat', 0.0):.4f}",
            f"{p.get('soft_metrics', {}).get('mean_peak_voltage_deviation_pu', 0.0):.4f}",
            f"{p.get('soft_metrics', {}).get('mean_peak_network_loss_mw', 0.0):.4f}",
            f"{p.get('soft_metrics', {}).get('mean_peak_line_loading_percent', 0.0):.4f}",
            f"{p.get('soft_metrics', {}).get('mean_peak_trafo_loading_percent', 0.0):.4f}",
        ]
        for p in pts
    ]
    rows.append([])
    rows.append(["N*", n_star, "", "", "", "", "", "", "", "", "", ""])
    return _write_csv(
        path,
        [
            "N",
            "p_hat",
            "ci95_low",
            "ci95_high",
            "voltage_p_hat",
            "line_p_hat",
            "trafo_p_hat",
            "solver_failure_p_hat",
            "mean_peak_voltage_deviation_pu",
            "mean_peak_network_loss_mw",
            "mean_peak_line_loading_percent",
            "mean_peak_trafo_loading_percent",
        ],
        rows,
    )


def export_strategy_comparison(
    path: Path,
    tc_results: dict[str, int],
    sess_results: dict[str, int],
) -> Path:
    rows: list[list[Any]] = []
    for key, n_star in tc_results.items():
        rows.append(["tripchain_soc", key, n_star])
    for key, n_star in sess_results.items():
        rows.append(["session", key, n_star])
    return _write_csv(path, ["model", "strategy", "N*"], rows)


def export_method_comparison(
    path: Path,
    method_results: dict[str, int],
    method_times: dict[str, float] | None = None,
) -> Path:
    rows = []
    for key, n_star in method_results.items():
        t = method_times.get(key, 0) if method_times else 0
        rows.append([key, n_star, f"{t:.1f}"])
    return _write_csv(path, ["method", "N*", "time_s"], rows)


def export_ordered_delay(
    path: Path,
    hours: np.ndarray,
    p_uncontrolled: np.ndarray,
    p_no_delay: np.ndarray,
    p_with_delay: np.ndarray,
) -> Path:
    rows = [
        [
            f"{float(h):.2f}",
            f"{float(pu):.4f}",
            f"{float(pn):.4f}",
            f"{float(py):.4f}",
        ]
        for h, pu, pn, py in zip(hours, p_uncontrolled, p_no_delay, p_with_delay, strict=True)
    ]
    return _write_csv(
        path,
        ["hour", "uncontrolled_kw", "ordered_no_delay_kw", "ordered_with_delay_kw"],
        rows,
    )


def export_bottleneck(
    path: Path,
    buses: list[dict[str, Any]],
    lines: list[dict[str, Any]],
) -> Path:
    rows: list[list[Any]] = [["=== Bottleneck Buses ==="]]
    rows.append(["bus_id", "worst_vmin_pu", "avg_vmin_pu"])
    for b in buses:
        rows.append([b["bus"], f"{b['worst_vmin_pu']:.4f}", f"{b['avg_vmin_pu']:.4f}"])
    rows.append([])
    rows.append(["=== Bottleneck Lines ==="])
    rows.append(["line_id", "worst_loading_pct", "avg_loading_pct"])
    for ln in lines:
        rows.append([ln["line"], f"{ln['worst_loading_pct']:.2f}", f"{ln['avg_loading_pct']:.2f}"])
    return _write_csv(path, ["item", "value1", "value2"], rows)


def export_parameter_sweep(
    path: Path,
    load_scales: list[float],
    charge_powers: list[float],
    n_star_grid: np.ndarray,
) -> Path:
    headers = ["load_scale"] + [f"P={p:.1f}kW" for p in charge_powers]
    rows = []
    for i, ls in enumerate(load_scales):
        rows.append([f"{ls:.2f}"] + [str(int(n_star_grid[i, j])) for j in range(len(charge_powers))])
    return _write_csv(path, headers, rows)
