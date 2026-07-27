#!/usr/bin/env python3
"""Generate Phase 1 paper figures from with_cost.csv (or aggregated.csv).

Cost scatter plots are split by experiment so each point can be read as
backend + the varying parameter (concurrency / prompt / output length).
"""

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


def pick_cost_col(df: pd.DataFrame) -> str | None:
    for col in ("cost_per_request_normalized_usd", "cost_per_request_usd"):
        if col in df.columns:
            return col
    return None


def point_label(row: pd.Series, kind: str) -> str:
    if kind == "concurrency":
        return f"c{int(row['concurrency'])}"
    if kind == "prompt":
        return str(int(row["mean_input_tokens"]))
    if kind == "output":
        return str(int(row["mean_output_tokens"]))
    return ""


def annotate_points(
    ax: plt.Axes, g: pd.DataFrame, xcol: str, ycol: str, kind: str
) -> None:
    for _, row in g.iterrows():
        label = point_label(row, kind)
        if not label:
            continue
        ax.annotate(
            label,
            (row[xcol], row[ycol]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            alpha=0.85,
        )


def plot_ttft_vs_concurrency(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in df.groupby("backend"):
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
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in df.groupby("backend"):
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


def plot_cost_scatter(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    kind: str,
    xcol: str,
    xlabel: str,
    title: str,
    filename: str,
) -> None:
    cost_col = pick_cost_col(df)
    if cost_col is None or xcol not in df.columns or df.empty:
        return
    sub = df.dropna(subset=[cost_col, xcol])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in sub.groupby("backend"):
        ax.scatter(g[xcol], g[cost_col], label=backend, alpha=0.85)
        annotate_points(ax, g, xcol, cost_col, kind)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cost / request normalized (USD)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / filename)


def plot_cost_vs_throughput_for_kind(
    df: pd.DataFrame, out_dir: Path, kind: str, title: str, filename: str
) -> None:
    plot_cost_scatter(
        df,
        out_dir,
        kind=kind,
        xcol="output_tokens_per_s",
        xlabel="Output tokens / s",
        title=title,
        filename=filename,
    )


def plot_cost_vs_latency_for_kind(
    df: pd.DataFrame, out_dir: Path, kind: str, title: str, filename: str
) -> None:
    plot_cost_scatter(
        df,
        out_dir,
        kind=kind,
        xcol="e2e_p50_s",
        xlabel="E2E latency P50 (s)",
        title=title,
        filename=filename,
    )


def plot_cost_normalized_vs_billed(
    df: pd.DataFrame, out_dir: Path, *, kind: str, title: str, filename: str
) -> None:
    need = {"cost_per_request_normalized_usd", "cost_per_request_billed_usd", "backend"}
    if not need.issubset(df.columns) or df.empty:
        return
    sub = df.dropna(
        subset=["cost_per_request_normalized_usd", "cost_per_request_billed_usd"]
    )
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for backend, g in sub.groupby("backend"):
        ax.scatter(
            g["cost_per_request_normalized_usd"],
            g["cost_per_request_billed_usd"],
            label=backend,
            alpha=0.85,
        )
        annotate_points(
            ax,
            g,
            "cost_per_request_normalized_usd",
            "cost_per_request_billed_usd",
            kind,
        )
    ax.set_xlabel("Cost / request normalized (USD)")
    ax.set_ylabel("Cost / request session-allocated (USD)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, out_dir / filename)


def split_experiments(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "experiment" in df.columns:
        conc = df[df["experiment"] == "concurrency"].copy()
        prompt = df[df["experiment"] == "prompt_length"].copy()
        output = df[df["experiment"] == "output_length"].copy()
        return conc, prompt, output

    if "exp_id" not in df.columns:
        return df.copy(), df.copy(), df.copy()

    exp_id = df["exp_id"].astype(str)
    conc = df[exp_id.str.startswith("concurrency_")].copy()
    prompt = df[exp_id.str.startswith("prompt_length_")].copy()
    output = df[exp_id.str.startswith("output_length_")].copy()
    return conc, prompt, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(ROOT / "results" / "with_cost.csv"),
        help="Prefer with_cost.csv; falls back to aggregated.csv if missing",
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
    conc, prompt, output = split_experiments(df)

    plot_ttft_vs_concurrency(conc, out_dir)
    plot_latency_vs_prompt(prompt, out_dir)
    plot_throughput_vs_concurrency(conc, out_dir)
    plot_gpu_vs_concurrency(conc, out_dir)

    plot_cost_vs_throughput_for_kind(
        conc,
        out_dir,
        kind="concurrency",
        title="Normalized Cost vs Throughput (Concurrency)",
        filename="cost_vs_throughput_concurrency.png",
    )
    plot_cost_vs_latency_for_kind(
        conc,
        out_dir,
        kind="concurrency",
        title="Normalized Cost vs Latency (Concurrency)",
        filename="cost_vs_latency_concurrency.png",
    )
    plot_cost_normalized_vs_billed(
        conc,
        out_dir,
        kind="concurrency",
        title="Normalized vs Session-Allocated Cost (Concurrency)",
        filename="cost_normalized_vs_billed_concurrency.png",
    )

    plot_cost_vs_throughput_for_kind(
        prompt,
        out_dir,
        kind="prompt",
        title="Normalized Cost vs Throughput (Prompt Length)",
        filename="cost_vs_throughput_prompt.png",
    )
    plot_cost_vs_latency_for_kind(
        prompt,
        out_dir,
        kind="prompt",
        title="Normalized Cost vs Latency (Prompt Length)",
        filename="cost_vs_latency_prompt.png",
    )
    plot_cost_normalized_vs_billed(
        prompt,
        out_dir,
        kind="prompt",
        title="Normalized vs Session-Allocated Cost (Prompt Length)",
        filename="cost_normalized_vs_billed_prompt.png",
    )

    plot_cost_vs_throughput_for_kind(
        output,
        out_dir,
        kind="output",
        title="Normalized Cost vs Throughput (Output Length)",
        filename="cost_vs_throughput_output.png",
    )
    plot_cost_vs_latency_for_kind(
        output,
        out_dir,
        kind="output",
        title="Normalized Cost vs Latency (Output Length)",
        filename="cost_vs_latency_output.png",
    )
    plot_cost_normalized_vs_billed(
        output,
        out_dir,
        kind="output",
        title="Normalized vs Session-Allocated Cost (Output Length)",
        filename="cost_normalized_vs_billed_output.png",
    )

    # Keep legacy filenames as concurrency-only for older links
    plot_cost_vs_throughput_for_kind(
        conc,
        out_dir,
        kind="concurrency",
        title="Normalized Cost vs Throughput",
        filename="cost_vs_throughput.png",
    )
    plot_cost_vs_latency_for_kind(
        conc,
        out_dir,
        kind="concurrency",
        title="Normalized Cost vs Latency",
        filename="cost_vs_latency.png",
    )
    plot_cost_normalized_vs_billed(
        conc,
        out_dir,
        kind="concurrency",
        title="Normalized vs Session-Allocated Cost / Request",
        filename="cost_normalized_vs_billed.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
