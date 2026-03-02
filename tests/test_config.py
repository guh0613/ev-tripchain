from pathlib import Path

from ev_tripchain.config import load_config


def test_load_config_minimal(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 1\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.seed == 1
    assert cfg.time.n_steps > 0

