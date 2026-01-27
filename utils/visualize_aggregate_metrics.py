#!/usr/bin/env python3
"""
Visualize aggregate metrics from a RealHiTBench results JSON.

This script:
1) Extracts `aggregate_metrics` into a tidy table.
2) Writes CSV and Markdown tables for easy reading.
3) Produces a heatmap and per-metric bar charts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


PREFERRED_METRIC_ORDER = [
    "EM",
    "F1",
    "ROUGE-L",
    "SacreBLEU",
    "Pass",
    "ECR",
]

# Metrics that are stored as fractions in the results file and are easier to read
# when shown as percentages.
FRACTION_METRICS = {"Pass", "ECR"}


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    return cleaned.lower() or "metric"


def _ordered_columns(columns: Iterable[str]) -> list[str]:
    cols = list(columns)
    ordered = [c for c in PREFERRED_METRIC_ORDER if c in cols]
    remainder = sorted(c for c in cols if c not in ordered)
    return ordered + remainder


def load_aggregate_metrics(results_path: Path) -> Tuple[pd.DataFrame, Dict]:
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    aggregate = data.get("aggregate_metrics")
    if not aggregate:
        raise ValueError(f"No aggregate_metrics found in {results_path}")

    df = pd.DataFrame(aggregate).T
    df = df.apply(pd.to_numeric, errors="coerce")

    # Convert fraction-based metrics into percentages for consistency.
    for metric in FRACTION_METRICS:
        if metric in df.columns:
            df[metric] = df[metric] * 100.0

    df = df[_ordered_columns(df.columns)]
    return df, data.get("config", {})


def write_tables(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "aggregate_metrics.csv"
    df.to_csv(csv_path, float_format="%.6f")

    md_path = out_dir / "aggregate_metrics.md"
    df_fmt = df.copy()
    for col in df_fmt.columns:
        df_fmt[col] = df_fmt[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    md_path.write_text(df_fmt.to_markdown(), encoding="utf-8")

    return csv_path, md_path


def plot_heatmap(df: pd.DataFrame, out_dir: Path, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    values = df.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(color="#f2f2f2")

    fig_w = max(8.0, 1.2 * len(df.columns) + 4.0)
    fig_h = max(4.0, 0.7 * len(df.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(masked, aspect="auto", vmin=0, vmax=100, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Question Type")

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_xticklabels(df.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_yticklabels(df.index)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Score (%)")

    fig.tight_layout()
    heatmap_path = out_dir / "aggregate_heatmap.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    return heatmap_path


def plot_metric_bars(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for metric in df.columns:
        series = df[metric].dropna()
        if series.empty:
            continue

        series = series.sort_values(ascending=True)

        fig_h = max(3.0, 0.6 * len(series) + 1.5)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        ax.barh(series.index, series.values, color="#4C78A8")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Score (%)")
        ax.set_title(metric)

        for i, (label, val) in enumerate(zip(series.index, series.values)):
            ax.text(min(val + 1.0, 99.0), i, f"{val:.1f}", va="center", fontsize=9)

        fig.tight_layout()
        path = out_dir / f"metric_{_sanitize_filename(metric)}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_json",
        type=Path,
        help="Path to results_*.json that contains aggregate_metrics",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for tables and plots (default: <results_dir>/plots)",
    )
    args = parser.parse_args()

    results_path: Path = args.results_json
    out_dir = args.out_dir or (results_path.parent / "plots")

    df, config = load_aggregate_metrics(results_path)

    csv_path, md_path = write_tables(df, out_dir)
    title_bits = ["Aggregate Metrics"]
    if config:
        modality = config.get("modality")
        fmt = config.get("format")
        if modality:
            title_bits.append(str(modality))
        if fmt:
            title_bits.append(str(fmt))
    heatmap_title = " | ".join(title_bits)
    heatmap_path = plot_heatmap(df, out_dir, heatmap_title)
    bar_paths = plot_metric_bars(df, out_dir)

    print("Wrote:")
    print(f"- {csv_path}")
    print(f"- {md_path}")
    print(f"- {heatmap_path}")
    for p in bar_paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()

