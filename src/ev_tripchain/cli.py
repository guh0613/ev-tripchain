from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from ev_tripchain.config import load_config
from ev_tripchain.pipelines.run import run_hosting_capacity, run_method_comparison
from ev_tripchain.reporting.report import FIGURE_REGISTRY, generate_report

app = typer.Typer(no_args_is_help=True)

CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    "-c",
    exists=True,
    dir_okay=False,
    readable=True,
)
OUT_OPTION = typer.Option(None, "--out", "-o", help="Write results JSON to this file")


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _emit_payload(payload: dict, out: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered + "\n", encoding="utf-8")


def _run(config: Path, out: Path | None) -> None:
    cfg = load_config(config)
    result = run_hosting_capacity(cfg, progress=_progress, progress_label="mc")
    _emit_payload(result.model_dump(), out)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    out: Path | None = OUT_OPTION,
) -> None:
    """Run probabilistic hosting-capacity assessment."""
    if ctx.invoked_subcommand is not None:
        return
    if config is None:
        print(ctx.get_help())
        raise typer.Exit()
    _run(config, out)


@app.command()
def run(
    config: Path = CONFIG_OPTION,
    out: Path | None = OUT_OPTION,
) -> None:
    """Alias for the default command."""
    _run(config, out)


@app.command()
def compare(
    config: Path = CONFIG_OPTION,
    out: Path | None = OUT_OPTION,
) -> None:
    """Run all three HC methods (MC, deterministic, sensitivity) and compare."""
    cfg = load_config(config)
    result = run_method_comparison(cfg, progress=_progress)
    _emit_payload(result.model_dump(), out)


@app.command()
def report(
    config: Path = CONFIG_OPTION,
    out: Path = typer.Option(
        Path("output"),
        "--out",
        "-o",
        help="Output directory for figures/tables/data",
    ),
    session_config: Path | None = typer.Option(
        None,
        "--session-config",
        "-s",
        help="Session model config (default: configs/example.yaml)",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Comma-separated figure IDs to generate (e.g. '1,4,8')",
    ),
    fmt: str = typer.Option("png", "--fmt", "-f", help="Image format: png, pdf, svg"),
    list_figures: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List available figures and exit",
    ),
) -> None:
    """Generate thesis figures, tables, and data exports."""
    if list_figures:
        print(f"{'ID':>4}  {'Name':<28} {'Heavy':>5}  Description")
        print("-" * 70)
        for fid, (name, desc, heavy) in sorted(FIGURE_REGISTRY.items()):
            tag = "*" if heavy else " "
            print(f"{fid:>4}  {name:<28} {tag:>5}  {desc}")
        print("\n* = computationally heavy (runs MC / binary search)")
        return

    cfg_tc = load_config(config)
    cfg_sess = load_config(session_config) if session_config else None

    figure_ids = [int(x.strip()) for x in only.split(",")] if only else None

    generate_report(
        cfg_tc=cfg_tc,
        cfg_sess=cfg_sess,
        output_dir=out,
        figure_ids=figure_ids,
        fmt=fmt,
    )


if __name__ == "__main__":
    app()
