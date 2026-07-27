#!/usr/bin/env python3
"""Estimate dual cost metrics from aggregated CSV + configs/cost.yaml.

Paper-facing columns (primary):
  normalized_compute_cost_usd   — cost_per_hour × duration/3600 (no billing floor)
  standalone_billed_cost_usd    — if this cell ran alone (Bedrock: ≥1×5-min window)
  session_allocated_cost_usd    — share of the matrix session CMU bill

Also writes results/session_costs.csv with one row per (backend, experiment).

Self-host reports electricity-only AND amortized TCO $/hour.
ECS is compute-only (g5.xlarge). Bedrock CMU is not per-token.
"""

from __future__ import annotations

import argparse
import json
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
        out["cost_per_hour_usd"] = float(section.get("cost_per_hour") or 0.0)
        return out

    if billing == "cmu":
        out["cost_per_hour_usd"] = cmu_per_hour(section, copies)
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


def resolve_copies(row: pd.Series, section: dict) -> tuple[float, str]:
    """Return (copies, source). Prefer CloudWatch / aggregated observation."""
    if pd.notna(row.get("model_copies")):
        try:
            copies = max(1.0, float(row["model_copies"]))
            src = str(row.get("model_copy_source") or "aggregated")
            if src in ("", "nan", "None"):
                src = "aggregated"
            return copies, src
        except (TypeError, ValueError):
            pass
    return max(1.0, float(section.get("model_copies") or 1.0)), "configured_assumption"


def bill_cmu_minutes(section: dict, copies: float, duration_s: float) -> tuple[float, float]:
    """Return (cost_usd, billed_minutes) for ceil(duration/window) windows."""
    window_min = float(section.get("billing_window_minutes") or 5.0)
    window_s = window_min * 60.0
    if duration_s <= 0:
        return 0.0, 0.0
    windows = max(1, math.ceil(duration_s / window_s))
    billed_minutes = windows * window_min
    cmu = float(section.get("custom_model_units_per_copy") or 0.0)
    rate = float(section.get("usd_per_cmu_per_minute") or 0.0)
    return copies * cmu * rate * billed_minutes, billed_minutes


def bill_tokens(section: dict, total_in: float, total_out: float) -> float:
    in_rate = float(section.get("input_usd_per_1m_tokens") or 0.0)
    out_rate = float(section.get("output_usd_per_1m_tokens") or 0.0)
    return (total_in / 1e6) * in_rate + (total_out / 1e6) * out_rate


def load_manifest_spans() -> dict[tuple[str, str], float]:
    """(backend, experiment) → wall seconds from earliest started_at to latest finished_at."""
    spans: dict[tuple[str, str], tuple[float, float]] = {}
    for path in sorted((ROOT / "results").glob("manifest_*.json")):
        try:
            doc = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for run in doc.get("runs") or []:
            if run.get("status") not in ("ok", "failed"):
                continue
            backend = run.get("backend")
            experiment = run.get("experiment")
            started = run.get("started_at")
            finished = run.get("finished_at")
            if backend is None or experiment is None:
                continue
            if started is None or finished is None:
                continue
            key = (str(backend), str(experiment))
            s, e = float(started), float(finished)
            if key not in spans:
                spans[key] = (s, e)
            else:
                lo, hi = spans[key]
                spans[key] = (min(lo, s), max(hi, e))
    return {k: hi - lo for k, (lo, hi) in spans.items() if hi > lo}


def session_wall_from_rows(group: pd.DataFrame) -> float | None:
    if "finished_at" in group.columns and group["finished_at"].notna().any():
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
        if starts:
            return max(ends) - min(starts)
    durs = [approx_duration_s(r) for _, r in group.iterrows()]
    durs = [d for d in durs if d is not None]
    return sum(durs) if durs else None


def enrich(
    df: pd.DataFrame, cfg: dict, manifest_spans: dict[tuple[str, str], float] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allocation = str(cfg.get("session_allocation") or "requests").lower()
    manifest_spans = manifest_spans or {}
    work = df.copy()
    session_rows: list[dict[str, Any]] = []

    base_rows: list[dict[str, Any]] = []
    for _, r in work.iterrows():
        backend = str(r["backend"])
        section = cfg.get(backend) or {}
        billing = resolve_billing(backend, section)
        copies, copy_source = resolve_copies(r, section)
        rates = hourly_rates(backend, section, copies)
        hourly = rates["cost_per_hour_usd"]
        duration_s = approx_duration_s(r)

        completed = float(r.get("num_completed_requests") or 0)
        mean_out = float(r.get("mean_output_tokens") or 0)
        mean_in = float(r.get("mean_input_tokens") or 0)
        total_out = mean_out * completed
        total_in = mean_in * completed

        # 1) Normalized busy capacity (NO 5-minute floor)
        if duration_s and duration_s > 0:
            if billing == "token":
                normalized = bill_tokens(section, total_in, total_out)
                normalized += hourly * (duration_s / 3600.0)
            else:
                normalized = hourly * (duration_s / 3600.0)
        else:
            normalized = 0.0

        # 2) Standalone billed: cell alone (Bedrock cold/bursty floor)
        if billing == "cmu" and duration_s and duration_s > 0:
            standalone, _ = bill_cmu_minutes(section, copies, duration_s)
        else:
            standalone = normalized

        out = r.to_dict()
        out["billing"] = billing
        out["billing_label"] = BILLING_LABELS.get(billing, billing)
        out["model_copies_observed"] = copies
        out["model_copy_source"] = copy_source
        out["model_copies_used"] = copies  # legacy alias
        out["cost_scope"] = section.get("cost_scope") or (
            "compute_only" if billing == "hourly" else None
        )
        out.update(rates)
        out["approx_duration_s"] = duration_s
        out["normalized_compute_cost_usd"] = normalized
        out["standalone_billed_cost_usd"] = standalone
        out["cell_floor_billed_cost_usd"] = standalone  # legacy alias
        out["billed_session_cost_usd"] = None
        out["session_wall_s"] = None
        out["session_billed_minutes"] = None
        out["session_allocated_cost_usd"] = None
        out["allocated_session_cost_usd"] = None  # legacy alias
        out["cost_per_request_normalized_usd"] = (
            (normalized / completed) if completed else None
        )
        out["cost_per_request_billed_usd"] = None
        out["cost_per_1m_output_tokens_normalized_usd"] = (
            (normalized / total_out * 1e6) if total_out > 0 else None
        )
        # Primary paper aliases → normalized (efficiency)
        out["approx_total_cost_usd"] = normalized
        out["cost_per_request_usd"] = out["cost_per_request_normalized_usd"]
        out["cost_per_1m_output_tokens_usd"] = out[
            "cost_per_1m_output_tokens_normalized_usd"
        ]
        base_rows.append(out)

    result = pd.DataFrame(base_rows)
    if result.empty:
        return result, pd.DataFrame()

    for (backend, experiment), idx in result.groupby(
        ["backend", "experiment"]
    ).groups.items():
        section = cfg.get(str(backend)) or {}
        billing = resolve_billing(str(backend), section)
        group = result.loc[idx]

        key = (str(backend), str(experiment))
        span_s = manifest_spans.get(key)
        if span_s is None:
            span_s = session_wall_from_rows(group)
        if span_s is None or span_s <= 0:
            span_s = float(group["approx_duration_s"].fillna(0).sum()) or 0.0

        copies = float(group["model_copies_observed"].max())
        copy_source = (
            "cloudwatch"
            if (group["model_copy_source"] == "cloudwatch").any()
            else str(group["model_copy_source"].iloc[0])
        )

        if billing == "cmu":
            session_cost, billed_minutes = (
                bill_cmu_minutes(section, copies, span_s) if span_s > 0 else (0.0, 0.0)
            )
        else:
            session_cost = float(group["normalized_compute_cost_usd"].sum())
            billed_minutes = span_s / 60.0 if span_s else 0.0

        total_requests = float(group["num_completed_requests"].fillna(0).sum())
        session_rows.append(
            {
                "backend": backend,
                "experiment": experiment,
                "billing": billing,
                "session_duration_s": span_s,
                "billed_minutes": billed_minutes,
                "model_copies": copies,
                "model_copy_source": copy_source,
                "total_requests": total_requests,
                "n_cells": int(len(group)),
                "session_cost_usd": session_cost,
                "span_source": "manifest" if key in manifest_spans else "row_finished_at",
            }
        )

        if allocation == "duration":
            weights = group["approx_duration_s"].astype(float).fillna(0.0)
        else:
            # Default / paper: allocate by completed requests
            weights = group["num_completed_requests"].astype(float).fillna(0.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            weights = pd.Series(1.0, index=group.index)
            weight_sum = float(len(group))

        for i in idx:
            if billing == "cmu":
                w = float(weights.loc[i]) if i in weights.index else 0.0
                allocated = session_cost * (w / weight_sum) if weight_sum else 0.0
            else:
                allocated = float(result.at[i, "normalized_compute_cost_usd"] or 0.0)
            completed = float(result.at[i, "num_completed_requests"] or 0)
            result.at[i, "billed_session_cost_usd"] = session_cost
            result.at[i, "session_wall_s"] = span_s
            result.at[i, "session_billed_minutes"] = billed_minutes
            result.at[i, "session_allocated_cost_usd"] = allocated
            result.at[i, "allocated_session_cost_usd"] = allocated
            result.at[i, "cost_per_request_billed_usd"] = (
                (allocated / completed) if completed else None
            )

    return result, pd.DataFrame(session_rows)


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
    parser.add_argument(
        "--session-out",
        default=str(ROOT / "results" / "session_costs.csv"),
    )
    args = parser.parse_args()

    agg_path = Path(args.aggregated)
    if not agg_path.exists():
        raise SystemExit(f"Missing aggregated CSV: {agg_path} (run aggregate.py first)")

    df = pd.read_csv(agg_path)
    cfg = load_cost()
    manifests = load_manifest_spans()
    out, sessions = enrich(df, cfg, manifests)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows → {out_path}")

    sess_path = Path(args.session_out)
    sessions.to_csv(sess_path, index=False)
    print(f"Wrote {len(sessions)} sessions → {sess_path}")

    for backend in sorted(out["backend"].dropna().unique()):
        sub = out.loc[out["backend"] == backend]
        label = sub["billing_label"].iloc[0]
        cph = float(sub["cost_per_hour_usd"].iloc[0])
        print(f"  {backend}: billing={label}  primary_$/hr={cph:.4f}")
        if backend == "selfhost":
            elec = float(sub["cost_per_hour_electricity_usd"].iloc[0])
            amort = float(sub["cost_per_hour_amortized_usd"].iloc[0])
            print(f"    electricity-only=${elec:.4f}/hr  amortized-TCO=${amort:.4f}/hr")
        if backend == "bedrock":
            src = sub["model_copy_source"].value_counts().to_dict()
            print(f"    model_copy_source={src}")

    if not sessions.empty:
        print("Sessions:")
        for _, s in sessions.iterrows():
            print(
                f"  {s['backend']}/{s['experiment']}: "
                f"wall={s['session_duration_s']:.1f}s  "
                f"billed_min={s['billed_minutes']}  "
                f"cost=${s['session_cost_usd']:.4f}  "
                f"({s['span_source']}, copies={s['model_copies']}/{s['model_copy_source']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
