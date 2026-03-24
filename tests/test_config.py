from pathlib import Path

from ev_tripchain.config import ProjectConfig
from ev_tripchain.config import load_config


def test_load_config_minimal(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 1\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.seed == 1
    assert cfg.time.n_steps > 0


def test_default_risk_metric_matches_report_definition() -> None:
    cfg = ProjectConfig()
    assert cfg.hosting_capacity.risk_metric == "p_hat"


def test_repo_configs_keep_p_hat_for_hosting_capacity_decision() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for rel in ("configs/example.yaml", "configs/tripchain_soc.yaml"):
        cfg = load_config(repo_root / rel)
        assert cfg.hosting_capacity.risk_metric == "p_hat"
