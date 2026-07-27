#!/usr/bin/env python3
"""Estimate cost metrics from aggregated CSV + configs/cost.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cost() -> dict:
    with (ROOT / "configs" / "cost.yaml").open() as f:
        return yaml.safe_load(f)


def cost_per_hour(backend: str, cfg: dict) -> float:
    section = cfg.get(backend) or {}
    hourly = float(section.get("cost_per_hour") or 0.0)
    if backend == "selfhost" and section.get("use_electricity"):
        watts = float(section.get("power_watts") or 0.0)
        rate = float(section.get("electricity_usd_per_kwh") or 0.0)
        hourly += (watts / 1000.0) * rate
    return hourly


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        backend = str(r["backend"])
        hourly = cost_per_hour(backend, cfg)
        section = cfg.get(backend) or {}

        completed = float(r.get("num_completed_requests") or 0)
        # Approximate duration from completed requests / throughput
        rps = r.get("requests_per_s")
        if rps and float(rps) > 0 and completed > 0:
            duration_s = completed / float(rps)
        else:
            duration_s = None

        mean_out = float(r.get("mean_output_tokens") or 0)
        mean_in = float(r.get("mean_input_tokens") or 0)
        total_out = mean_out * completed
        total_in = mean_in * completed

        cost_hour_component = 0.0
        if duration_s is not None:
            cost_hour_component = hourly * (duration_s / 3600.0)

        token_cost = 0.0
        if backend == "bedrock":
            in_rate = float(section.get("input_usd_per_1m_tokens") or 0.0)
            out_rate = float(section.get("output_usd_per_1m_tokens") or 0.0)
            token_cost = (total_in / 1e6) * in_rate + (total_out / 1e6) * out_rate

        total_cost = cost_hour_component + token_cost
        cost_per_req = (total_cost / completed) if completed else None
        cost_per_1m_out = (
            (total_cost / total_out * 1e6) if total_out > 0 else None
        )

        out = r.to_dict()
        out["cost_per_hour_usd"] = hourly
        out["approx_duration_s"] = duration_s
        out["approx_total_cost_usd"] = total_cost
        out["cost_per_request_usd"] = cost_per_req
        out["cost_per_1m_output_tokens_usd"] = cost_per_1m_out
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregated",
        default=str(ROOT / "results" / "aggregated.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "with_cost.csv"),
    )
    args = parser.parse_args()

    agg_path = Path(args.aggregated)
    if not agg_path.exists():
        raise SystemExit(f"Missing aggregated CSV: {agg_path} (run aggregate.py first)")

    df = pd.read_csv(agg_path)
    cfg = load_cost()
    out = enrich(df, cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
