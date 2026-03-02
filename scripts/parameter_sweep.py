"""Parameter sweep: evaluate N* under varying charge power and load scale."""
from __future__ import annotations

import json
from pathlib import Path

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.pipelines.run import run_hosting_capacity


def sweep() -> list[dict]:
    base_cfg = load_config(Path("configs/tripchain_soc.yaml"))
    charge_powers = [3.7, 7.2, 11.0, 22.0]
    load_scales = [0.5, 0.7, 0.9]
    results: list[dict] = []

    for ls in load_scales:
        for cp in charge_powers:
            cfg_dict = base_cfg.model_dump()
            cfg_dict["case"]["load_scale"] = ls
            cfg_dict["ev"]["charge_power_kw"] = cp
            cfg_dict["hosting_capacity"]["scenarios"] = 15
            cfg_dict["hosting_capacity"]["binary_search"]["max_iter"] = 12
            cfg_dict["hosting_capacity"]["binary_search"]["min_step"] = 20
            cfg = ProjectConfig.model_validate(cfg_dict)
            print(f"Running: load_scale={ls}, charge_power_kw={cp} ...")
            result = run_hosting_capacity(cfg)
            row = {
                "load_scale": ls,
                "charge_power_kw": cp,
                "n_star": result.n_star,
                "risk_curve": result.risk_curve,
            }
            results.append(row)
            print(f"  -> N* = {result.n_star}")

    return results


def main() -> None:
    results = sweep()
    outdir = Path("docs/results")
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "parameter_sweep.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")

    # summary table
    print(f"\n{'load_scale':>12} {'charge_kw':>10} {'N*':>6}")
    print("-" * 32)
    for r in results:
        print(f"{r['load_scale']:>12.1f} {r['charge_power_kw']:>10.1f} {r['n_star']:>6}")


if __name__ == "__main__":
    main()
