#!/usr/bin/env python3
"""Estimate dual cost metrics from aggregated CSV + configs/cost.yaml.

Reports for every row:
  normalized_compute_cost_usd     — cost_per_hour × duration/3600 (busy capacity)
  billed_session_cost_usd         — Bedrock CMU: one 5-min-window bill for the
                                    whole (backend, experiment) session
  allocated_session_cost_usd      — session bill allocated to this cell
  cost_per_request_normalized_usd
  cost_per_request_billed_usd     — allocated_session / completed requests
  cell_floor_billed_cost_usd      — if this cell ran alone (ceil to one window)

Also keeps approx_total_cost_usd / cost_per_request_usd as aliases of the
**normalized** figures (paper efficiency primary).

Self-host emits both electricity-only and amortized $/hour.
ECS is compute-only (g5.xlarge). Bedrock uses CMU, not tokens.
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
    "electricity": "Electricity / amortized",
    "hourly": "Hourly (compute-only)",
    "token": "Per-token",
    "cmu": "CMU (active copy)",
}


def load_cost() -> dict:
    with (ROOT / "configs" / "cost.yaml").open() as f:
        return yaml.safe_load(f)


def hardware_amortization_per_hour(section: dict) -> float:
    hw = section.get("hardware") or {}
    # Prefer whole-system purchase; fall back to legacy gpu_purchase_usd
    purchase = float(
        hw.get("system_purchase_usd")
        or hw.get("gpu_purchase_usd")
        or 0.0
    )
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


def cmu_per_hour(section: dict, copies: float) -> float:
    """Effective $/hour while `copies` continuous active copies are running."""
    cmu = float(section.get("custom_model_units_per_copy") or 0.0)
    rate = float(section.get("usd_per_cmu_per_minute") or 0.0)
    return copies * cmu * rate * 60.0


def resolve_billing(backend: str, section: dict) -> str:
    billing = str(section.get("billing") or "").lower()
    if billing:
        return billing
    if backend == "selfhost":
        return "electricity"
    if backend == "bedrock" and (
        section.get("input_usd_per_1m_tokens")
        or section.get("output_usd_per_1m_tokens")
    ):
        return "token"
    if backend == "bedrock" and section.get("usd_per_cmu_per_minute"):
        return "cmu"
    return "hourly"


def hourly_rates(backend: str, section: dict, copies: float) -> dict[str, float]:
    """Return primary + scenario $/hour values."""
    billing = resolve_billing(backend, section)
    out: dict[str, float] = {
        "cost_per_hour_electricity_usd": 0.0,
        "cost_per_hour_amortized_usd": 0.0,
        "cost_per_hour_usd": 0.0,
    }

    if billing == "electricity":
        elec = electricity_per_hour(section)
        amort = elec + hardware_amortization_per_hour(section)
        out["cost_per_hour_electricity_usd"] = elec
        out["cost_per_hour_amortized_usd"] = amort
        primary = str(section.get("primary_hourly") or "amortized").lower()
        out["cost_per_hour_usd"] = amort if primary == "amortized" else elec
        return out

    if billing == "hourly":
        h = float(section.get("cost_per_hour") or 0.0)
        out["cost_per_hour_usd"] = h
        return out

    if billing == "cmu":
        h = cmu_per_hour(section, copies)
        out["cost_per_hour_usd"] = h
        return out

    if billing == "token":
        out["cost_per_hour_usd"] = float(section.get("cost_per_hour") or 0.0)
        return out

    out["cost_per_hour_usd"] = float(section.get("cost_per_hour") or 0.0)
    return out


def approx_duration_s(row: pd.Series) -> float | None:
    if pd.notna(row.get("approx_duration_s")):
        try:
            v = float(row["approx_duration_s"])
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    completed = float(row.get("num_completed_requests") or 0)
    rps = row.get("requests_per_s")
    if rps and float(rps) > 0 and completed > 0:
        return completed / float(rps)
    return None


def row_copies(row: pd.Series, section: dict) -> float:
    if pd.notna(row.get("model_copies")):
        try:
            return max(1.0, float(row["model_copies"]))
        except (TypeError, ValueError):
            pass
    return max(1.0, float(section.get("model_copies") or 1.0))


def bill_cmu_minutes(section: dict, copies: float, duration_s: float) -> float:
    """Charge for ceil(duration / window) × window minutes."""
    window_min = float(section.get("billing_window_minutes") or 5.0)
    window_s = window_min * 60.0
    windows = max(1, math.ceil(duration_s / window_s))
    billed_minutes = windows * window_min
    cmu = float(section.get("custom_model_units_per_copy") or 0.0)
    rate = float(section.get("usd_per_cmu_per_minute") or 0.0)
    return copies * cmu * rate * billed_minutes


def bill_tokens(section: dict, total_in: float, total_out: float) -> float:
    in_rate = float(section.get("input_usd_per_1m_tokens") or 0.0)
    out_rate = float(section.get("output_usd_per_1m_tokens") or 0.0)
    return (total_in / 1e6) * in_rate + (total_out / 1e6) * out_rate


def session_wall_span_s(group: pd.DataFrame) -> float | None:
    """Wall-clock session length from finished_at − duration (if available)."""
    if "finished_at" not in group.columns or group["finished_at"].isna().all():
        # Fallback: sum of cell durations (assumes sequential, no idle gaps)
        durs = [approx_duration_s(r) for _, r in group.iterrows()]
        durs = [d for d in durs if d is not None]
        return sum(durs) if durs else None

    starts: list[float] = []
    ends: list[float] = []
    for _, r in group.iterrows():
        fin = r.get("finished_at")
        if pd.isna(fin):
            continue
        fin_f = float(fin)
        dur = approx_duration_s(r) or 0.0
        starts.append(fin_f - dur)
        ends.append(fin_f)
    if not starts:
        return None
    return max(ends) - min(starts)


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    allocation = str(cfg.get("session_allocation") or "duration").lower()
    work = df.copy()

    # Pass 1: per-row normalized + cell-floor CMU bill
    base_rows: list[dict[str, Any]] = []
    for _, r in work.iterrows():
        backend = str(r["backend"])
        section = cfg.get(backend) or {}
        billing = resolve_billing(backend, section)
        copies = row_copies(r, section)
        rates = hourly_rates(backend, section, copies)
        hourly = rates["cost_per_hour_usd"]
        duration_s = approx_duration_s(r)

        completed = float(r.get("num_completed_requests") or 0)
        mean_out = float(r.get("mean_output_tokens") or 0)
        mean_in = float(r.get("mean_input_tokens") or 0)
        total_out = mean_out * completed
        total_in = mean_in * completed

        # Normalized busy capacity cost (no billing-window floor)
        if duration_s and duration_s > 0:
            if billing == "token":
                normalized = bill_tokens(section, total_in, total_out)
                normalized += hourly * (duration_s / 3600.0)
            else:
                normalized = hourly * (duration_s / 3600.0)
        else:
            normalized = 0.0

        # If this cell ran in isolation (triggers ≥1 CMU window)
        cell_floor = 0.0
        if billing == "cmu" and duration_s and duration_s > 0:
            cell_floor = bill_cmu_minutes(section, copies, duration_s)
        elif billing in ("electricity", "hourly") and duration_s and duration_s > 0:
            cell_floor = normalized
        elif billing == "token":
            cell_floor = normalized

        out = r.to_dict()
        out["billing"] = billing
        out["billing_label"] = BILLING_LABELS.get(billing, billing)
        out["model_copies_used"] = copies
        out["cost_scope"] = section.get("cost_scope") or (
            "compute_only" if billing == "hourly" else None
        )
        out.update(rates)
        out["approx_duration_s"] = duration_s
        out["normalized_compute_cost_usd"] = normalized
        out["cell_floor_billed_cost_usd"] = cell_floor
        # Placeholders filled in pass 2 for CMU sessions
        out["billed_session_cost_usd"] = None
        out["session_wall_s"] = None
        out["session_billed_minutes"] = None
        out["allocated_session_cost_usd"] = None
        out["cost_per_request_normalized_usd"] = (
            (normalized / completed) if completed else None
        )
        out["cost_per_request_billed_usd"] = None
        out["cost_per_1m_output_tokens_normalized_usd"] = (
            (normalized / total_out * 1e6) if total_out > 0 else None
        )
        # Backward-compatible primary columns = normalized (efficiency)
        out["approx_total_cost_usd"] = normalized
        out["cost_per_request_usd"] = out["cost_per_request_normalized_usd"]
        out["cost_per_1m_output_tokens_usd"] = out[
            "cost_per_1m_output_tokens_normalized_usd"
        ]
        base_rows.append(out)

    result = pd.DataFrame(base_rows)

    # Pass 2: session-level CMU billing + allocation
    if result.empty:
        return result

    for (backend, experiment), idx in result.groupby(
        ["backend", "experiment"]
    ).groups.items():
        section = cfg.get(str(backend)) or {}
        billing = resolve_billing(str(backend), section)
        group = result.loc[idx]

        if billing != "cmu":
            # Non-CMU: session bill = sum of normalized; allocated = normalized
            for i in idx:
                norm = float(result.at[i, "normalized_compute_cost_usd"] or 0.0)
                completed = float(result.at[i, "num_completed_requests"] or 0)
                result.at[i, "billed_session_cost_usd"] = float(
                    group["normalized_compute_cost_usd"].sum()
                )
                result.at[i, "session_wall_s"] = float(
                    group["approx_duration_s"].fillna(0).sum()
                )
                result.at[i, "allocated_session_cost_usd"] = norm
                result.at[i, "cost_per_request_billed_usd"] = (
                    (norm / completed) if completed else None
                )
            continue

        span_s = session_wall_span_s(group)
        if span_s is None or span_s <= 0:
            # Fall back to sum of durations
            span_s = float(group["approx_duration_s"].fillna(0).sum()) or 0.0

        # Use max observed copies in the session (conservative for cost)
        copies = float(group["model_copies_used"].max())
        window_min = float(section.get("billing_window_minutes") or 5.0)
        session_cost = (
            bill_cmu_minutes(section, copies, span_s) if span_s > 0 else 0.0
        )
        billed_minutes = (
            max(1, math.ceil(span_s / (window_min * 60.0))) * window_min
            if span_s > 0
            else 0.0
        )

        if allocation == "requests":
            weights = group["num_completed_requests"].astype(float).fillna(0.0)
        else:
            weights = group["approx_duration_s"].astype(float).fillna(0.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            weights = pd.Series(1.0, index=group.index)
            weight_sum = float(len(group))

        for i in idx:
            w = float(weights.loc[i]) if i in weights.index else 0.0
            allocated = session_cost * (w / weight_sum) if weight_sum else 0.0
            completed = float(result.at[i, "num_completed_requests"] or 0)
            result.at[i, "billed_session_cost_usd"] = session_cost
            result.at[i, "session_wall_s"] = span_s
            result.at[i, "session_billed_minutes"] = billed_minutes
            result.at[i, "allocated_session_cost_usd"] = allocated
            result.at[i, "cost_per_request_billed_usd"] = (
                (allocated / completed) if completed else None
            )

    return result


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

    for backend in sorted(out["backend"].dropna().unique()):
        sub = out.loc[out["backend"] == backend]
        label = sub["billing_label"].iloc[0]
        cph = float(sub["cost_per_hour_usd"].iloc[0])
        print(f"  {backend}: billing={label}  cost_per_hour_usd={cph:.4f}")
        if backend == "selfhost":
            elec = float(sub["cost_per_hour_electricity_usd"].iloc[0])
            amort = float(sub["cost_per_hour_amortized_usd"].iloc[0])
            print(f"    electricity-only=${elec:.4f}/hr  amortized-TCO=${amort:.4f}/hr")
        if backend == "bedrock":
            for exp, g in sub.groupby("experiment"):
                sess = g["billed_session_cost_usd"].iloc[0]
                wall = g["session_wall_s"].iloc[0]
                mins = g["session_billed_minutes"].iloc[0]
                print(
                    f"    session[{exp}]: wall={wall:.1f}s  "
                    f"billed_minutes={mins}  billed_session=${sess:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
