from __future__ import annotations

import json
from pathlib import Path

import typer

from ev_tripchain.config import load_config
from ev_tripchain.pipelines.run import run_hosting_capacity

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


@app.command()
def run(
    config: Path = CONFIG_OPTION,
    out: Path | None = OUT_OPTION,
) -> None:
    """Run probabilistic hosting-capacity assessment."""
    cfg = load_config(config)
    result = run_hosting_capacity(cfg)

    payload = result.model_dump()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
