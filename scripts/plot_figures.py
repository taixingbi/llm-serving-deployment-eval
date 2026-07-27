#!/usr/bin/env python3
"""Generate Phase 1 paper figures from with_cost.csv (or aggregated.csv)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing data file: {path}")
    return pd.read_csv(path)


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_ttft_vs_concurrency(df: pd.DataFrame, out_dir: Path) -> None:
    # Concurrency experiment rows: varying concurrency, similar prompt sizes
    sub = df.copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in sub.groupby("backend"):
        g = g.sort_values("concurrency")
        y = g["ttft_p50_s"] if "ttft_p50_s" in g else g.get("ttft_mean_s")
        ax.plot(g["concurrency"], y, marker="o", label=backend)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("TTFT P50 (s)")
    ax.set_title("TTFT vs Concurrency")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / "ttft_vs_concurrency.png")


def plot_latency_vs_prompt(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[df["concurrency"] == 1].copy() if "concurrency" in df else df.copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in sub.groupby("backend"):
        g = g.sort_values("mean_input_tokens")
        ax.plot(g["mean_input_tokens"], g["e2e_p50_s"], marker="o", label=backend)
    ax.set_xlabel("Mean input tokens")
    ax.set_ylabel("E2E latency P50 (s)")
    ax.set_title("Latency vs Prompt Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / "latency_vs_prompt_length.png")


def plot_throughput_vs_concurrency(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in df.groupby("backend"):
        g = g.sort_values("concurrency")
        ax.plot(
            g["concurrency"],
            g["output_tokens_per_s"],
            marker="o",
            label=backend,
        )
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output tokens / s")
    ax.set_title("Throughput vs Concurrency")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / "throughput_vs_concurrency.png")


def plot_gpu_vs_concurrency(df: pd.DataFrame, out_dir: Path) -> None:
    if "gpu_util_mean" not in df.columns:
        return
    sub = df[df["backend"].isin(["selfhost", "ecs"])].dropna(subset=["gpu_util_mean"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in sub.groupby("backend"):
        g = g.sort_values("concurrency")
        ax.plot(g["concurrency"], g["gpu_util_mean"], marker="o", label=backend)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("GPU utilization mean (%)")
    ax.set_title("GPU Utilization vs Concurrency")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / "gpu_util_vs_concurrency.png")


def plot_cost_vs_throughput(df: pd.DataFrame, out_dir: Path) -> None:
    if "cost_per_request_usd" not in df.columns:
        return
    sub = df.dropna(subset=["cost_per_request_usd", "output_tokens_per_s"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in sub.groupby("backend"):
        ax.scatter(
            g["output_tokens_per_s"],
            g["cost_per_request_usd"],
            label=backend,
            alpha=0.8,
        )
    ax.set_xlabel("Output tokens / s")
    ax.set_ylabel("Cost / request (USD)")
    ax.set_title("Cost vs Throughput")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / "cost_vs_throughput.png")


def plot_cost_vs_latency(df: pd.DataFrame, out_dir: Path) -> None:
    if "cost_per_request_usd" not in df.columns:
        return
    sub = df.dropna(subset=["cost_per_request_usd", "e2e_p50_s"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in sub.groupby("backend"):
        ax.scatter(g["e2e_p50_s"], g["cost_per_request_usd"], label=backend, alpha=0.8)
    ax.set_xlabel("E2E latency P50 (s)")
    ax.set_ylabel("Cost / request (USD)")
    ax.set_title("Cost vs Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / "cost_vs_latency.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(ROOT / "results" / "with_cost.csv"),
        help="Prefer with_cost.csv; falls back guidance if missing",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "paper" / "figures"),
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        alt = ROOT / "results" / "aggregated.csv"
        if alt.exists():
            data_path = alt
            print(f"with_cost.csv missing; using {alt}")
        else:
            raise SystemExit(
                f"Missing {args.data}. Run aggregate.py and estimate_cost.py first."
            )

    df = load_df(data_path)
    out_dir = Path(args.out_dir)

    # Heuristic splits for nicer plots when mixed experiments are present
    conc = df.copy()
    if "exp_id" in df.columns:
        conc = df[df["exp_id"].astype(str).str.startswith("concurrency_")].copy()
        if conc.empty:
            conc = df.copy()
        prompt = df[df["exp_id"].astype(str).str.startswith("prompt_length_")].copy()
        if prompt.empty:
            prompt = df[df["concurrency"] == 1].copy() if "concurrency" in df else df
    else:
        prompt = df

    plot_ttft_vs_concurrency(conc, out_dir)
    plot_latency_vs_prompt(prompt, out_dir)
    plot_throughput_vs_concurrency(conc, out_dir)
    plot_gpu_vs_concurrency(conc, out_dir)
    plot_cost_vs_throughput(df, out_dir)
    plot_cost_vs_latency(df, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
