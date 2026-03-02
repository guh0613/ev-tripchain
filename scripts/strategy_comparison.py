"""Strategy comparison: run hosting capacity under different charging strategies."""
from __future__ import annotations

import json
from pathlib import Path

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.pipelines.run import run_hosting_capacity


STRATEGIES = [
    {"name": "uncontrolled"},
    {"name": "ordered", "window_start": "22:00", "window_end": "06:00"},
    {"name": "nearest"},
    {"name": "navigation", "candidate_k": 6},
]


def compare_strategies() -> list[dict]:
    base_cfg = load_config(Path("configs/example.yaml"))
    results: list[dict] = []

    for strat in STRATEGIES:
        cfg_dict = base_cfg.model_dump()
        cfg_dict["strategy"]["name"] = strat["name"]
        if "window_start" in strat:
            cfg_dict["strategy"]["ordered"]["window_start"] = strat["window_start"]
            cfg_dict["strategy"]["ordered"]["window_end"] = strat["window_end"]
        if "candidate_k" in strat:
            cfg_dict["strategy"]["navigation"]["candidate_k"] = strat["candidate_k"]
        # use fewer scenarios for speed
        cfg_dict["hosting_capacity"]["scenarios"] = 15
        cfg_dict["hosting_capacity"]["binary_search"]["max_iter"] = 12
        cfg_dict["hosting_capacity"]["binary_search"]["min_step"] = 20

        cfg = ProjectConfig.model_validate(cfg_dict)
        label = strat["name"]
        print(f"Running strategy: {label} ...")
        result = run_hosting_capacity(cfg)
        row = {
            "strategy": label,
            "n_star": result.n_star,
            "risk_curve": result.risk_curve,
        }
        results.append(row)
        print(f"  -> N* = {result.n_star}")

    return results


def main() -> None:
    results = compare_strategies()
    outdir = Path("docs/results")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "strategy_comparison.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")

    print(f"\n{'Strategy':>16} {'N*':>6}")
    print("-" * 26)
    for r in results:
        print(f"{r['strategy']:>16} {r['n_star']:>6}")


if __name__ == "__main__":
    main()
