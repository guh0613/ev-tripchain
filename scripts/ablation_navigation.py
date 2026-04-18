"""Navigation factor ablation study for thesis Chapter 4.

Tests different factor combinations to quantify marginal contribution of each factor,
including w_voltage isolation and softmax temperature sensitivity.
"""
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
    nav = cfg.strategy.navigation
    print(f"  strategy={cfg.strategy.name}, dynamic={nav.dynamic_scoring}, "
          f"K={nav.candidate_k}, beta={nav.distance_beta}, "
          f"path_w={nav.path_congestion_weight}, "
          f"temp={nav.softmax_temperature}, "
          f"disable_v={nav.disable_voltage_factor}")
    print(f"{'='*60}")
    t0 = time.time()
    result = run_hosting_capacity(cfg, progress=lambda msg: print(f"  {msg}"))
    elapsed = time.time() - t0
    curve_detail = {
        rp.n: {"p_hat": rp.p_hat, "ci95_low": rp.ci95_low, "ci95_high": rp.ci95_high}
        for rp in result.risk_curve_detail
    }
    print(f"  => N* = {result.n_star}  ({elapsed:.1f}s)")
    return {
        "label": label,
        "n_star": result.n_star,
        "elapsed": round(elapsed, 1),
        "risk_curve_detail": curve_detail,
    }


def main() -> None:
    results = []

    # 1. Baseline: uncontrolled
    cfg = load_base_config()
    cfg.strategy.name = "uncontrolled"
    results.append(run_variant("Uncontrolled (baseline)", cfg))

    # 2. Nearest (distance only, no bus scoring)
    cfg = load_base_config()
    cfg.strategy.name = "nearest"
    results.append(run_variant("Nearest (distance only)", cfg))

    # 3. Navigation static (w_dist + w_score)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = False
    results.append(run_variant("Nav static (dist+score)", cfg))

    # 4. Navigation dynamic, w_path disabled
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    cfg.strategy.navigation.path_congestion_weight = 0.0
    results.append(run_variant("Nav dynamic no-path", cfg))

    # 5. Navigation dynamic, w_voltage disabled (isolate voltage factor)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    cfg.strategy.navigation.disable_voltage_factor = True
    results.append(run_variant("Nav dynamic no-voltage", cfg))

    # 6. Navigation dynamic, both w_path and w_voltage disabled (only temporal)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    cfg.strategy.navigation.path_congestion_weight = 0.0
    cfg.strategy.navigation.disable_voltage_factor = True
    results.append(run_variant("Nav dynamic temporal-only", cfg))

    # 7. Navigation full dynamic (all 5 factors)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    results.append(run_variant("Nav dynamic full", cfg))

    # 8. Static with larger distance penalty (beta=2)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = False
    cfg.strategy.navigation.distance_beta = 2.0
    results.append(run_variant("Nav static beta=2.0", cfg))

    # 9. Static with fewer candidates (K=3)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = False
    cfg.strategy.navigation.candidate_k = 3
    results.append(run_variant("Nav static K=3", cfg))

    # 10. Dynamic full with higher temperature (more random)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    cfg.strategy.navigation.softmax_temperature = 1.0
    results.append(run_variant("Nav dynamic temp=1.0", cfg))

    # 11. Dynamic full with lower temperature (more greedy)
    cfg = load_base_config()
    cfg.strategy.name = "navigation"
    cfg.strategy.navigation.dynamic_scoring = True
    cfg.strategy.navigation.softmax_temperature = 0.1
    results.append(run_variant("Nav dynamic temp=0.1", cfg))

    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['label']:45s}  N*={r['n_star']:4d}  ({r['elapsed']:.1f}s)")

    with open("output/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to output/ablation_results.json")


if __name__ == "__main__":
    main()
