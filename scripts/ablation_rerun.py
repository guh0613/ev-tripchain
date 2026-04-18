"""Re-run voltage-related ablation experiments after fixing keep_mask gating."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ev_tripchain.config import ProjectConfig, load_config
from ev_tripchain.pipelines.run import run_hosting_capacity


def load_base_config() -> ProjectConfig:
    return load_config(Path("configs/tripchain_soc.yaml"))


def run_variant(label: str, cfg: ProjectConfig) -> dict:
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = run_hosting_capacity(cfg, progress=lambda msg: print(f"  {msg}"))
    elapsed = time.time() - t0
    curve = {
        rp.n: {"p_hat": rp.p_hat, "ci95_low": rp.ci95_low, "ci95_high": rp.ci95_high}
        for rp in result.risk_curve_detail
    }
    print(f"  => N* = {result.n_star}  ({elapsed:.1f}s)")
    return {"label": label, "n_star": result.n_star, "elapsed": round(elapsed, 1), "curve": curve}


def main() -> None:
    results = []

    # 1. Dynamic no-voltage (keep_mask + w_voltage both disabled)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    cfg.strategy.navigation.disable_voltage_factor = True
    results.append(run_variant("Nav dynamic no-voltage (fixed)", cfg))

    # 2. Dynamic temporal-only (no w_voltage, no w_path)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    cfg.strategy.navigation.path_congestion_weight = 0.0
    cfg.strategy.navigation.disable_voltage_factor = True
    results.append(run_variant("Nav dynamic temporal-only (fixed)", cfg))

    print("\n" + "=" * 60)
    print("RE-RUN RESULTS")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:45s}  N*={r['n_star']:4d}  ({r['elapsed']:.1f}s)")

    with open("output/ablation_rerun.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
