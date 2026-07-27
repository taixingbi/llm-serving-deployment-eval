#!/usr/bin/env python3
"""Aggregate LLMPerf summary JSON + run.json (+ optional resources) into CSV."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# LLMPerf metric keys
TTFT = "ttft_s"
E2E = "end_to_end_latency_s"
OUT_TPUT = "mean_output_throughput_token_per_s"
NUM_COMPLETED = "num_completed_requests"
COMPLETED_PER_MIN = "num_completed_requests_per_min"
ERROR_RATE = "error_rate"
NUM_ERRORS = "number_errors"
NUM_STARTED = "num_requests_started"


def find_summary(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("*_summary.json"))
    return matches[0] if matches else None


def quantiles(block: dict[str, Any] | None) -> dict[str, float | None]:
    if not block:
        return {"p50": None, "p95": None, "p99": None, "mean": None}
    q = block.get("quantiles") or {}
    return {
        "p50": q.get("p50"),
        "p95": q.get("p95"),
        "p99": q.get("p99"),
        "mean": block.get("mean"),
    }


def summarize_resources(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {
            "gpu_util_mean": None,
            "gpu_mem_used_mb_mean": None,
            "gpu_power_w_mean": None,
            "cpu_pct_mean": None,
            "ram_pct_mean": None,
        }
    gpu_utils: list[float] = []
    gpu_mem: list[float] = []
    gpu_power: list[float] = []
    cpu: list[float] = []
    ram: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cpu.append(float(row.get("cpu_pct") or 0))
            ram.append(float(row.get("ram_pct") or 0))
            for g in row.get("gpus") or []:
                if "util_gpu_pct" in g:
                    gpu_utils.append(float(g["util_gpu_pct"]))
                if "mem_used_mb" in g:
                    gpu_mem.append(float(g["mem_used_mb"]))
                if "power_w" in g:
                    gpu_power.append(float(g["power_w"]))

    def mean(xs: list[float]) -> float | None:
        return statistics.fmean(xs) if xs else None

    return {
        "gpu_util_mean": mean(gpu_utils),
        "gpu_mem_used_mb_mean": mean(gpu_mem),
        "gpu_power_w_mean": mean(gpu_power),
        "cpu_pct_mean": mean(cpu),
        "ram_pct_mean": mean(ram),
    }


def flatten_run(run_dir: Path) -> dict[str, Any] | None:
    run_path = run_dir / "run.json"
    summary_path = find_summary(run_dir)
    if not run_path.exists() or summary_path is None:
        return None

    run = json.loads(run_path.read_text())
    summary_doc = json.loads(summary_path.read_text())
    # LLMPerfResults wraps metadata; support both shapes
    meta = summary_doc.get("metadata") or summary_doc
    results = meta.get("results") or {}

    ttft = quantiles(results.get(TTFT))
    e2e = quantiles(results.get(E2E))
    resources = summarize_resources(run_dir / "resources.jsonl")

    completed = results.get(NUM_COMPLETED) or 0
    per_min = results.get(COMPLETED_PER_MIN) or 0
    req_per_s = (per_min / 60.0) if per_min else None

    return {
        "backend": run.get("backend"),
        "exp_id": run.get("exp_id"),
        "mean_input_tokens": run.get("mean_input_tokens"),
        "mean_output_tokens": run.get("mean_output_tokens"),
        "concurrency": run.get("concurrency"),
        "model": run.get("model") or meta.get("model"),
        "ttft_p50_s": ttft["p50"],
        "ttft_p95_s": ttft["p95"],
        "ttft_p99_s": ttft["p99"],
        "ttft_mean_s": ttft["mean"],
        "e2e_p50_s": e2e["p50"],
        "e2e_p95_s": e2e["p95"],
        "e2e_p99_s": e2e["p99"],
        "e2e_mean_s": e2e["mean"],
        "output_tokens_per_s": results.get(OUT_TPUT),
        "requests_per_s": req_per_s,
        "num_completed_requests": completed,
        "num_requests_started": results.get(NUM_STARTED),
        "error_rate": results.get(ERROR_RATE),
        "num_errors": results.get(NUM_ERRORS),
        "summary_path": str(summary_path),
        **resources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=str(ROOT / "results" / "raw"),
        help="Root containing <backend>/<exp_id>/",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "aggregated.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    rows: list[dict[str, Any]] = []
    if raw_root.exists():
        for run_json in sorted(raw_root.glob("*/*/run.json")):
            row = flatten_run(run_json.parent)
            if row:
                rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
