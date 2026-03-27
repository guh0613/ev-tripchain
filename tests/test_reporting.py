import matplotlib

from ev_tripchain.config import ProjectConfig
from ev_tripchain.reporting.report import FIGURE_GENERATORS, FIGURE_REGISTRY, generate_report

matplotlib.use("Agg")


def test_figure_generator_registry_stays_in_sync() -> None:
    assert set(FIGURE_GENERATORS) == set(FIGURE_REGISTRY)


def test_generate_report_for_lightweight_figures(tmp_path) -> None:
    cfg = ProjectConfig()
    saved = generate_report(
        cfg_tc=cfg,
        cfg_sess=cfg,
        output_dir=tmp_path,
        figure_ids=[1, 2],
        fmt="png",
    )

    assert set(saved) == {1, 2}
    for path in saved.values():
        assert path.exists()
