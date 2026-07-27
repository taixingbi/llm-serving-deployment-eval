#!/usr/bin/env python3
"""Estimate cost metrics from aggregated CSV + configs/cost.yaml.

Dispatches on each backend's `billing` field:
  electricity — self-host wall power (+ optional GPU amortization)
  hourly      — cloud instance $/hour × duration
  token       — Bedrock-style on-demand FM: input/output $/1M tokens
  cmu         — Bedrock Custom Model Import: active model-copy CMU minutes
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

BILLING_LABELS = {
    "electricity": "Electricity",
    "hourly": "Hourly",
    "token": "Per-token",
    "cmu": "CMU (active copy)",
}


def load_cost() -> dict:
    with (ROOT / "configs" / "cost.yaml").open() as f:
        return yaml.safe_load(f)


def hardware_amortization_per_hour(section: dict) -> float:
    hw = section.get("hardware") or {}
    purchase = float(hw.get("gpu_purchase_usd") or 0.0)
    lifetime = float(hw.get("lifetime_hours") or 0.0)
    if purchase > 0 and lifetime > 0:
        return purchase / lifetime
    return 0.0


def electricity_per_hour(section: dict) -> float:
    if not section.get("use_electricity", True):
        return 0.0
    watts = float(section.get("power_watts") or 0.0)
    rate = float(section.get("electricity_usd_per_kwh") or 0.0)
    return (watts / 1000.0) * rate


def cmu_per_hour(section: dict) -> float:
    """Effective $/hour while one continuous active copy is running."""
    copies = float(section.get("model_copies") or 1.0)
    cmu = float(section.get("custom_model_units_per_copy") or 0.0)
    rate = float(section.get("usd_per_cmu_per_minute") or 0.0)
    return copies * cmu * rate * 60.0


def cost_per_hour(backend: str, cfg: dict) -> float:
    """Reporting $/hour for the backend's primary capacity unit (when busy)."""
    section = cfg.get(backend) or {}
    billing = str(section.get("billing") or "").lower()

    if billing == "electricity":
        return electricity_per_hour(section) + hardware_amortization_per_hour(section)
    if billing == "hourly":
        return float(section.get("cost_per_hour") or 0.0)
    if billing == "cmu":
        return cmu_per_hour(section)
    if billing == "token":
        # On-demand token APIs have no idle hourly charge unless provisioned.
        return float(section.get("cost_per_hour") or 0.0)
    # Legacy fallback
    hourly = float(section.get("cost_per_hour") or 0.0)
    if backend == "selfhost" and section.get("use_electricity"):
        hourly += electricity_per_hour(section)
    return hourly


def approx_duration_s(row: pd.Series) -> float | None:
    completed = float(row.get("num_completed_requests") or 0)
    rps = row.get("requests_per_s")
    if rps and float(rps) > 0 and completed > 0:
        return completed / float(rps)
    return None


def bill_electricity_or_hourly(
    hourly: float, duration_s: float | None
) -> float:
    if duration_s is None or duration_s <= 0:
        return 0.0
    return hourly * (duration_s / 3600.0)


def bill_tokens(section: dict, total_in: float, total_out: float) -> float:
    in_rate = float(section.get("input_usd_per_1m_tokens") or 0.0)
    out_rate = float(section.get("output_usd_per_1m_tokens") or 0.0)
    return (total_in / 1e6) * in_rate + (total_out / 1e6) * out_rate


def bill_cmu(section: dict, duration_s: float | None) -> float:
    """Custom Model Import: ceil duration into billing windows, charge per CMU-minute."""
    if duration_s is None or duration_s <= 0:
        return 0.0
    window_min = float(section.get("billing_window_minutes") or 5.0)
    window_s = window_min * 60.0
    windows = max(1, math.ceil(duration_s / window_s))
    billed_minutes = windows * window_min
    copies = float(section.get("model_copies") or 1.0)
    cmu = float(section.get("custom_model_units_per_copy") or 0.0)
    rate = float(section.get("usd_per_cmu_per_minute") or 0.0)
    return copies * cmu * rate * billed_minutes


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        backend = str(r["backend"])
        section = cfg.get(backend) or {}
        billing = str(section.get("billing") or "").lower()
        if not billing:
            # Infer legacy configs
            if backend == "selfhost":
                billing = "electricity"
            elif backend == "bedrock" and (
                section.get("input_usd_per_1m_tokens")
                or section.get("output_usd_per_1m_tokens")
            ):
                billing = "token"
            elif backend == "bedrock" and section.get("usd_per_cmu_per_minute"):
                billing = "cmu"
            else:
                billing = "hourly"

        hourly = cost_per_hour(backend, cfg)
        duration_s = approx_duration_s(r)

        completed = float(r.get("num_completed_requests") or 0)
        mean_out = float(r.get("mean_output_tokens") or 0)
        mean_in = float(r.get("mean_input_tokens") or 0)
        total_out = mean_out * completed
        total_in = mean_in * completed

        if billing in ("electricity", "hourly"):
            total_cost = bill_electricity_or_hourly(hourly, duration_s)
        elif billing == "token":
            total_cost = bill_tokens(section, total_in, total_out)
            # Optional provisioned / idle hourly add-on
            total_cost += bill_electricity_or_hourly(
                float(section.get("cost_per_hour") or 0.0), duration_s
            )
        elif billing == "cmu":
            total_cost = bill_cmu(section, duration_s)
        else:
            raise ValueError(
                f"Unknown billing={billing!r} for backend={backend!r} "
                f"(expected electricity|hourly|token|cmu)"
            )

        cost_per_req = (total_cost / completed) if completed else None
        cost_per_1m_out = (
            (total_cost / total_out * 1e6) if total_out > 0 else None
        )

        out = r.to_dict()
        out["billing"] = billing
        out["billing_label"] = BILLING_LABELS.get(billing, billing)
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
    # Paper-friendly summary of billing modes
    for backend in sorted(out["backend"].unique()):
        label = out.loc[out["backend"] == backend, "billing_label"].iloc[0]
        cph = out.loc[out["backend"] == backend, "cost_per_hour_usd"].iloc[0]
        print(f"  {backend}: billing={label}  cost_per_hour_usd={cph:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
