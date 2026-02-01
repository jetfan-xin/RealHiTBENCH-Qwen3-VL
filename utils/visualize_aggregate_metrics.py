#!/usr/bin/env python3
"""
Visualize aggregate metrics from RealHiTBench results JSON files.

This script:
1) Extracts `aggregate_metrics` into a tidy table.
2) Writes CSV and Markdown tables for easy reading.
3) Produces a heatmap and per-metric bar charts.

Usage:
    # Process single file with all outputs
    python visualize_aggregate_metrics.py path/to/results.json

    # Process all results.json in compiled directory, CSV only
    python visualize_aggregate_metrics.py --batch --csv-only

    # Process all results.json with full visualization
    python visualize_aggregate_metrics.py --batch
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional

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

# Default compiled results directory
DEFAULT_COMPILED_DIR = Path(__file__).parent.parent / "result" / "complied"


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")
    return cleaned.lower() or "metric"


def _ordered_columns(columns: Iterable[str]) -> list[str]:
    cols = list(columns)
    ordered = [c for c in PREFERRED_METRIC_ORDER if c in cols]
    remainder = sorted(c for c in cols if c not in ordered)
    return ordered + remainder


def load_aggregate_metrics(results_path: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load aggregate metrics from results.json.
    
    Supports both old format (flat dict) and new format (with by_question_type and overall).
    """
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    aggregate = data.get("aggregate_metrics")
    if not aggregate:
        raise ValueError(f"No aggregate_metrics found in {results_path}")

    # Handle new format with by_question_type and overall
    if "by_question_type" in aggregate:
        by_type = aggregate["by_question_type"]
        overall = aggregate.get("overall", {})
        # Add overall as a row
        metrics_dict = dict(by_type)
        if overall:
            metrics_dict["Overall"] = overall
        df = pd.DataFrame(metrics_dict).T
    else:
        # Old flat format
        df = pd.DataFrame(aggregate).T
    
    df = df.apply(pd.to_numeric, errors="coerce")

    # Note: Pass and ECR are already in percentage format from add_aggregate_to_each.py
    # Only convert if they appear to be fractions (< 1)
    for metric in FRACTION_METRICS:
        if metric in df.columns:
            # Check if values are fractions (all <= 1) or already percentages
            max_val = df[metric].max()
            if not pd.isna(max_val) and max_val <= 1.0:
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


def process_single_file(results_path: Path, out_dir: Optional[Path] = None, csv_only: bool = False) -> None:
    """Process a single results.json file."""
    out_dir = out_dir or results_path.parent
    
    try:
        df, config = load_aggregate_metrics(results_path)
    except ValueError as e:
        print(f"Skipping {results_path}: {e}")
        return
    
    csv_path, md_path = write_tables(df, out_dir)
    print(f"✓ {results_path.parent.name}:")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    
    if not csv_only:
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
        print(f"  - {heatmap_path}")
        for p in bar_paths:
            print(f"  - {p}")


def process_batch(compiled_dir: Path, csv_only: bool = False) -> None:
    """Process all results.json files in compiled directory."""
    results_files = list(compiled_dir.rglob("results.json"))
    
    if not results_files:
        print(f"No results.json files found in {compiled_dir}")
        return
    
    print(f"Found {len(results_files)} results.json files\n")
    
    for results_path in sorted(results_files):
        # Output to same directory as results.json
        process_single_file(results_path, results_path.parent, csv_only)
        print()
    
    print(f"Done! Processed {len(results_files)} files.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "results_json",
        type=Path,
        nargs="?",
        default=None,
        help="Path to results_*.json that contains aggregate_metrics (not needed with --batch)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for tables and plots (default: same as results.json)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all results.json files in compiled directory",
    )
    parser.add_argument(
        "--compiled-dir",
        type=Path,
        default=DEFAULT_COMPILED_DIR,
        help=f"Compiled results directory for batch mode (default: {DEFAULT_COMPILED_DIR})",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only generate CSV and Markdown tables, skip plots",
    )
    args = parser.parse_args()

    if args.batch:
        process_batch(args.compiled_dir, args.csv_only)
    elif args.results_json:
        process_single_file(args.results_json, args.out_dir, args.csv_only)
    else:
        parser.print_help()
        print("\nError: Either provide a results.json path or use --batch mode")
        exit(1)


if __name__ == "__main__":
    main()

