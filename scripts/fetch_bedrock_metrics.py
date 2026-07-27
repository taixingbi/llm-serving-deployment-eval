#!/usr/bin/env python3
"""Fetch Bedrock Custom Model Import CloudWatch metrics for a run window.

Writes results/raw/bedrock/<exp_id>/bedrock_metrics.json with at least:
  model_copies  — max/avg ModelCopy during [start, end]
  plus optional InvocationLatency / Throttles / token counts when present.

Usage (after a cell finishes):
  python scripts/fetch_bedrock_metrics.py \\
    --exp-id concurrency_c64_in550_out150 \\
    --imported-model-arn arn:aws:bedrock:...:imported-model/...

Requires AWS credentials and CloudWatch access. Metric names/dimensions vary by
region and import job; adjust --namespace / --dimensions if empty.

See: https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-cloudwatch.html
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import boto3
except ImportError:  # pragma: no cover
    print("ERROR: pip install boto3", file=sys.stderr)
    raise SystemExit(1)


def load_run(exp_id: str) -> dict:
    path = ROOT / "results" / "raw" / "bedrock" / exp_id / "run.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    return json.loads(path.read_text())


def approx_duration_s(run: dict, summary_hint_s: float | None) -> float:
    if summary_hint_s and summary_hint_s > 0:
        return summary_hint_s
    return float(run.get("timeout_s") or 600)


def get_metric_stats(
    cw,
    *,
    namespace: str,
    metric_name: str,
    dimensions: list[dict],
    start: datetime,
    end: datetime,
    stat: str = "Maximum",
) -> float | None:
    try:
        resp = cw.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions,
            StartTime=start,
            EndTime=end,
            Period=60,
            Statistics=[stat],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: {metric_name}: {exc}", file=sys.stderr)
        return None
    points = resp.get("Datapoints") or []
    if not points:
        return None
    values = [float(p[stat]) for p in points if stat in p]
    if not values:
        return None
    return max(values) if stat == "Maximum" else sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument(
        "--imported-model-arn",
        default="",
        help="Imported model ARN (dimension). If empty, only writes a stub.",
    )
    parser.add_argument("--region", default="")
    parser.add_argument(
        "--namespace",
        default="AWS/Bedrock",
        help="CloudWatch namespace (confirm in console)",
    )
    parser.add_argument(
        "--pad-seconds",
        type=float,
        default=120.0,
        help="Pad before/after the run window for metric lookback",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Override run duration (else finished_at − pad estimate)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from envutil import load_dotenv

    load_dotenv()

    run = load_run(args.exp_id)
    finished = float(run.get("finished_at") or time.time())
    duration = args.duration_s or approx_duration_s(run, None)
    pad = float(args.pad_seconds)
    start_ts = finished - duration - pad
    end_ts = finished + pad
    start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end = datetime.fromtimestamp(end_ts, tz=timezone.utc)

    region = args.region or __import__("os").environ.get("AWS_REGION", "us-east-1")
    out_path = ROOT / "results" / "raw" / "bedrock" / args.exp_id / "bedrock_metrics.json"

    payload: dict = {
        "exp_id": args.exp_id,
        "fetched_at": time.time(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "model_copies": None,
        "invocation_throttles": None,
        "notes": [],
    }

    if not args.imported_model_arn:
        payload["notes"].append(
            "No --imported-model-arn; wrote stub. "
            "Re-run with ARN to populate ModelCopy from CloudWatch."
        )
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote stub {out_path}")
        return 0

    cw = boto3.client("cloudwatch", region_name=region)
    # Dimension keys differ across accounts; try common shapes.
    dim_candidates = [
        [{"Name": "ModelId", "Value": args.imported_model_arn}],
        [{"Name": "ImportedModelArn", "Value": args.imported_model_arn}],
        [{"Name": "ModelArn", "Value": args.imported_model_arn}],
    ]

    copies = None
    for dims in dim_candidates:
        copies = get_metric_stats(
            cw,
            namespace=args.namespace,
            metric_name="ModelCopy",
            dimensions=dims,
            start=start,
            end=end,
            stat="Maximum",
        )
        if copies is not None:
            payload["dimensions_used"] = dims
            break

    if copies is None:
        payload["notes"].append(
            "ModelCopy returned no datapoints; check namespace/dimensions in "
            "CloudWatch console and pass matching --namespace / ARN."
        )
    else:
        payload["model_copies"] = max(1, int(round(copies)))

    # Best-effort extras (may be empty)
    if payload.get("dimensions_used"):
        throttles = get_metric_stats(
            cw,
            namespace=args.namespace,
            metric_name="InvocationThrottles",
            dimensions=payload["dimensions_used"],
            start=start,
            end=end,
            stat="Sum",
        )
        payload["invocation_throttles"] = throttles

    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    # Also merge into run.json for aggregate convenience
    run_path = out_path.parent / "run.json"
    if run_path.exists() and payload.get("model_copies") is not None:
        run["model_copies"] = payload["model_copies"]
        run_path.write_text(json.dumps(run, indent=2) + "\n")

    print(f"Wrote {out_path}")
    print(f"  model_copies={payload.get('model_copies')}")
    for n in payload.get("notes") or []:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
