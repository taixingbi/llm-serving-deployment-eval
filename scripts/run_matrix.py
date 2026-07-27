#!/usr/bin/env python3
"""Expand experiment YAML × backends and invoke run_one.sh for each cell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Allow `python scripts/run_matrix.py` without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from envutil import load_dotenv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "configs" / "experiments"


def load_experiment(name: str) -> dict[str, Any]:
    path = EXPERIMENTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment config: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def expand_cells(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(cfg.get("defaults") or {})
    matrix = dict(cfg.get("matrix") or {})
    cells: list[dict[str, Any]] = []

    if "concurrency" in matrix:
        for conc in matrix["concurrency"]:
            cell = {**defaults, "concurrency": int(conc)}
            cells.append(cell)
    elif "mean_input_tokens" in matrix:
        for n in matrix["mean_input_tokens"]:
            cell = {**defaults, "mean_input_tokens": int(n)}
            cells.append(cell)
    elif "mean_output_tokens" in matrix:
        for n in matrix["mean_output_tokens"]:
            cell = {**defaults, "mean_output_tokens": int(n)}
            cells.append(cell)
    else:
        raise ValueError(f"Unsupported matrix keys in experiment: {list(matrix)}")

    return cells


def cell_exp_id(experiment: str, cell: dict[str, Any]) -> str:
    conc = cell.get("concurrency", 1)
    mean_in = cell.get("mean_input_tokens", 550)
    mean_out = cell.get("mean_output_tokens", 150)
    return f"{experiment}_c{conc}_in{mean_in}_out{mean_out}"


def resolve_backend(backend: str) -> None:
    script = ROOT / "scripts" / "resolve_endpoints.sh"
    cmd = (
        f'export RESOLVE_QUIET=1; source "{script}" "{backend}" && '
        f'printf "%s\\0%s" "$OPENAI_API_BASE" "$OPENAI_API_KEY"'
    )
    out = subprocess.check_output(["bash", "-lc", cmd])
    base, key = out.split(b"\0", 1)
    os.environ["OPENAI_API_BASE"] = base.decode()
    os.environ["OPENAI_API_KEY"] = key.decode()


def run_cell(
    backend: str,
    experiment: str,
    cell: dict[str, Any],
    *,
    dry_run: bool,
    sample_resources: bool,
) -> dict[str, Any]:
    exp_id = cell_exp_id(experiment, cell)
    env = os.environ.copy()
    env.update(
        {
            "BACKEND": backend,
            "EXP_ID": exp_id,
            "MEAN_IN": str(cell.get("mean_input_tokens", 550)),
            "MEAN_OUT": str(cell.get("mean_output_tokens", 150)),
            "CONC": str(cell.get("concurrency", 1)),
            "N_REQ": str(cell.get("max_num_completed_requests", 50)),
            "TIMEOUT": str(cell.get("timeout_s", 600)),
            "SAMPLE_RESOURCES": "1" if sample_resources else "0",
        }
    )
    record = {
        "backend": backend,
        "experiment": experiment,
        "exp_id": exp_id,
        "cell": cell,
        "started_at": time.time(),
    }
    if dry_run:
        record["status"] = "dry_run"
        print(f"[dry-run] {backend} {exp_id}")
        return record

    resolve_backend(backend)
    env["OPENAI_API_BASE"] = os.environ["OPENAI_API_BASE"]
    env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]

    cmd = ["bash", str(ROOT / "scripts" / "run_one.sh")]
    print(f"=== {backend} {exp_id} ===")
    proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
    record["finished_at"] = time.time()
    record["returncode"] = proc.returncode
    record["status"] = "ok" if proc.returncode == 0 else "failed"
    return record


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["concurrency", "prompt_length", "output_length"],
    )
    parser.add_argument(
        "--backends",
        default="selfhost,ecs,bedrock",
        help="Comma-separated backends",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sample-resources",
        action="store_true",
        help="Start sample_resources.py during each run (self-host/ECS GPU host)",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Path for run manifest JSON (default results/manifest_<exp>_<ts>.json)",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    cfg = load_experiment(args.experiment)
    cells = expand_cells(cfg)

    records: list[dict[str, Any]] = []
    for backend in backends:
        for cell in cells:
            records.append(
                run_cell(
                    backend,
                    args.experiment,
                    cell,
                    dry_run=args.dry_run,
                    sample_resources=args.sample_resources,
                )
            )

    manifest_path = Path(
        args.manifest
        or (
            ROOT
            / "results"
            / f"manifest_{args.experiment}_{int(time.time())}.json"
        )
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": args.experiment,
        "backends": backends,
        "cells": cells,
        "runs": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote manifest {manifest_path}")

    failed = sum(1 for r in records if r.get("status") == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
