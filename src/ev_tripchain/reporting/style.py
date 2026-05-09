"""Unified matplotlib style for thesis figures."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# Color palette (consistent across all figures)
COLORS = {
    "primary": "#4C78A8",
    "secondary": "#E45756",
    "success": "#54A24B",
    "warning": "#F58518",
    "purple": "#B279A2",
    "gray": "#64748b",
    # Strategy-specific
    "uncontrolled": "#64748b",
    "ordered_no_delay": "#ef4444",
    "ordered_delay": "#22c55e",
    "nearest": "#3b82f6",
    "navigation_static": "#8b5cf6",
    "navigation_dynamic": "#22c55e",
    # Method-specific
    "deterministic": "#dc2626",
    "sensitivity_representative": "#f59e0b",
    "sensitivity_weakest": "#f97316",
    "sensitivity_uniform": "#eab308",
    "monte_carlo": "#22c55e",
}

STRATEGY_LABELS = {
    "uncontrolled": "无序充电",
    "ordered_no_delay": "有序充电\n（无延迟）",
    "ordered_with_delay": "有序充电\n（延迟）",
    "nearest": "就近充电",
    "navigation_static": "导航引导\n（静态）",
    "navigation_dynamic": "导航引导\n（动态）",
}

METHOD_LABELS = {
    "deterministic": "确定性法\n（典型模板）",
    "sensitivity_representative": "灵敏度法\n（典型模板）",
    "sensitivity_weakest": "灵敏度法\n（最薄弱母线）",
    "sensitivity_uniform": "灵敏度法\n（均匀分配）",
    "monte_carlo": "蒙特卡洛法",
}


def apply_style() -> None:
    """Apply unified matplotlib style for all thesis figures."""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "font.family": ["Songti SC", "STHeiti", "SimSong", "serif"],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "figure.constrained_layout.use": True,
        }
    )
