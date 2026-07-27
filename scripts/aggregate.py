#!/usr/bin/env python3
"""Aggregate LLMPerf summary JSON + run.json (+ optional resources) into CSV.

LLMPerf's LLMPerfResults.to_dict() flattens nested metrics, e.g.:
  results_ttft_s_quantiles_p50
  results_end_to_end_latency_s_mean
This script accepts that flat shape (and a nested shape if present).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Nested metric names (pre-flatten)
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


def experiment_from_exp_id(exp_id: str | None) -> str | None:
    if not exp_id:
        return None
    for prefix in ("concurrency", "prompt_length", "output_length", "smoke"):
        if exp_id.startswith(prefix):
            return prefix
    return "other"


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


def flat_get(doc: dict[str, Any], *parts: str) -> Any:
    """Read a flattened LLMPerf key: parts ('results','ttft_s','quantiles','p50')."""
    key = "_".join(parts)
    return doc.get(key)


def metrics_from_summary(summary_doc: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics from nested or flat LLMPerf summary JSON."""
    # Nested: {metadata: {results: {ttft_s: {quantiles: ..., mean: ...}}}}
    meta = summary_doc.get("metadata") or summary_doc
    results = meta.get("results") if isinstance(meta, dict) else None
    if isinstance(results, dict) and TTFT in results:
        ttft = quantiles(results.get(TTFT))
        e2e = quantiles(results.get(E2E))
        return {
            "ttft": ttft,
            "e2e": e2e,
            "output_tokens_per_s": results.get(OUT_TPUT),
            "num_completed_requests": results.get(NUM_COMPLETED),
            "num_requests_started": results.get(NUM_STARTED),
            "completed_per_min": results.get(COMPLETED_PER_MIN),
            "error_rate": results.get(ERROR_RATE),
            "num_errors": results.get(NUM_ERRORS),
        }

    # Flat (LLMPerfResults.to_dict / flatten_dict): results_ttft_s_quantiles_p50
    src = summary_doc
    ttft = {
        "p50": flat_get(src, "results", TTFT, "quantiles", "p50"),
        "p95": flat_get(src, "results", TTFT, "quantiles", "p95"),
        "p99": flat_get(src, "results", TTFT, "quantiles", "p99"),
        "mean": flat_get(src, "results", TTFT, "mean"),
    }
    e2e = {
        "p50": flat_get(src, "results", E2E, "quantiles", "p50"),
        "p95": flat_get(src, "results", E2E, "quantiles", "p95"),
        "p99": flat_get(src, "results", E2E, "quantiles", "p99"),
        "mean": flat_get(src, "results", E2E, "mean"),
    }
    return {
        "ttft": ttft,
        "e2e": e2e,
        "output_tokens_per_s": flat_get(src, "results", OUT_TPUT),
        "num_completed_requests": flat_get(src, "results", NUM_COMPLETED),
        "num_requests_started": flat_get(src, "results", NUM_STARTED),
        "completed_per_min": flat_get(src, "results", COMPLETED_PER_MIN),
        "error_rate": flat_get(src, "results", ERROR_RATE),
        "num_errors": flat_get(src, "results", NUM_ERRORS),
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
    m = metrics_from_summary(summary_doc)
    resources = summarize_resources(run_dir / "resources.jsonl")

    completed = m["num_completed_requests"] or 0
    per_min = m["completed_per_min"] or 0
    req_per_s = (float(per_min) / 60.0) if per_min else None
    exp_id = run.get("exp_id")

    # Optional CloudWatch / sidecar metrics (see fetch_bedrock_metrics.py)
    model_copies = run.get("model_copies")
    metrics_path = run_dir / "bedrock_metrics.json"
    if metrics_path.exists():
        try:
            bm = json.loads(metrics_path.read_text())
            if model_copies is None and bm.get("model_copies") is not None:
                model_copies = bm["model_copies"]
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "backend": run.get("backend"),
        "experiment": experiment_from_exp_id(exp_id),
        "exp_id": exp_id,
        "mean_input_tokens": run.get("mean_input_tokens"),
        "mean_output_tokens": run.get("mean_output_tokens"),
        "concurrency": run.get("concurrency"),
        "model": run.get("model") or summary_doc.get("model"),
        "finished_at": run.get("finished_at"),
        "model_copies": model_copies,
        "ttft_p50_s": m["ttft"]["p50"],
        "ttft_p95_s": m["ttft"]["p95"],
        "ttft_p99_s": m["ttft"]["p99"],
        "ttft_mean_s": m["ttft"]["mean"],
        "e2e_p50_s": m["e2e"]["p50"],
        "e2e_p95_s": m["e2e"]["p95"],
        "e2e_p99_s": m["e2e"]["p99"],
        "e2e_mean_s": m["e2e"]["mean"],
        "output_tokens_per_s": m["output_tokens_per_s"],
        "requests_per_s": req_per_s,
        "num_completed_requests": completed,
        "num_requests_started": m["num_requests_started"],
        "error_rate": m["error_rate"],
        "num_errors": m["num_errors"],
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
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke_* experiment dirs (excluded by default)",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    if raw_root.exists():
        for run_json in sorted(raw_root.glob("*/*/run.json")):
            run_dir = run_json.parent
            if not args.include_smoke and run_dir.name.startswith("smoke_"):
                continue
            row = flatten_run(run_dir)
            if row:
                if row.get("ttft_p50_s") is None and row.get("e2e_p50_s") is None:
                    skipped.append(f"{run_dir} (no latency metrics in summary)")
                rows.append(row)
            else:
                skipped.append(f"{run_dir} (missing run.json or *_summary.json)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["experiment", "backend", "exp_id"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows → {out_path}")
    if not df.empty and "experiment" in df.columns and "backend" in df.columns:
        print(df.groupby(["experiment", "backend"]).size().to_string())
    null_lat = (
        int(((df["ttft_p50_s"].isna()) & (df["e2e_p50_s"].isna())).sum())
        if not df.empty
        else 0
    )
    if null_lat:
        print(f"WARNING: {null_lat} rows have null TTFT/E2E (check summary JSON shape)")
    if skipped:
        print(f"Skipped/notes ({len(skipped)}):")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
