"""Generate Figure 1-1: EV stock trends 2010-2024."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data" / "electric-car-stocks.csv"
OUT_PICS = ROOT / "pics"
OUT_LATEX = ROOT / "latex" / "pics"

ENTITIES = {
    "World": ("全球", "#8B4513", "o"),
    "China": ("中国", "#00008B", "s"),
    "European Union (27)": ("欧盟 (27 国)", "#CC0000", "^"),
    "United States": ("美国", "#228B22", "D"),
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Songti SC", "STHeiti", "SimSong", "serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "figure.constrained_layout.use": True,
        }
    )


def load_data() -> dict[str, tuple[list[int], list[float]]]:
    rows: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(DATA, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entity = row["Entity"]
            if entity in ENTITIES:
                year = int(row["Year"])
                stock_million = int(row["Electric car stocks"]) / 1e6
                rows[entity].append((year, stock_million))
    result = {}
    for entity, pts in rows.items():
        pts.sort()
        result[entity] = ([p[0] for p in pts], [p[1] for p in pts])
    return result


def main() -> None:
    _apply_style()
    data = load_data()

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for entity, (label, color, marker) in ENTITIES.items():
        years, stocks = data[entity]
        ax.plot(years, stocks, color=color, marker=marker, markersize=5,
                linewidth=1.8, label=label)

    ax.set_xlabel("年份")
    ax.set_ylabel("保有量（百万辆）")
    ax.set_xlim(2009.5, 2024.5)
    ax.set_xticks(range(2010, 2025, 2))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linewidth=0.3, alpha=0.5)

    ax.annotate(
        "注：保有量指在使用中的车辆数量（累计销售减去报废数量），\n"
        "包含纯电动及插电式混合动力汽车。",
        xy=(0.02, 0.58), xycoords="axes fraction",
        fontsize=8, color="gray", va="top",
    )

    OUT_PICS.mkdir(exist_ok=True)
    OUT_LATEX.mkdir(parents=True, exist_ok=True)

    fig.savefig(OUT_PICS / "1-1.png")
    fig.savefig(OUT_LATEX / "1-1.png")
    plt.close(fig)
    print(f"Saved to {OUT_PICS / '1-1.png'} and {OUT_LATEX / '1-1.png'}")


if __name__ == "__main__":
    main()
