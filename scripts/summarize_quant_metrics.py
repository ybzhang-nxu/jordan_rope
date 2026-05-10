#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir, write_csv


GROUP_KEYS = [
    "stage",
    "method",
    "dataset",
    "cache_protocol",
    "split",
    "quant_method",
    "rotation",
    "kac_depth",
    "b_numeric",
    "b_storage",
    "distance_bucket",
    "metric",
    "aggregation",
]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def summarize(rows: list[dict[str, str]]) -> list[dict[str, float | str | int]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get("value", ""))
        if value is None:
            continue
        key = tuple(row.get(field, "") for field in GROUP_KEYS)
        groups[key].append(value)

    out: list[dict[str, float | str | int]] = []
    for key, values in sorted(groups.items()):
        base = {field: value for field, value in zip(GROUP_KEYS, key)}
        mean = sum(values) / len(values)
        values_sorted = sorted(values)
        mid = len(values_sorted) // 2
        median = values_sorted[mid] if len(values_sorted) % 2 else 0.5 * (values_sorted[mid - 1] + values_sorted[mid])
        out.append(
            {
                **base,
                "layer_count": len(values),
                "value_mean": mean,
                "value_median": median,
                "value_min": min(values),
                "value_max": max(values),
            }
        )
    return out


def row_value(rows, *, metric: str, method: str, quant_method: str, b_numeric: str, kac_depth: str = "0"):
    for row in rows:
        if (
            row.get("metric") == metric
            and row.get("method") == method
            and row.get("quant_method") == quant_method
            and row.get("b_numeric") == b_numeric
            and row.get("kac_depth") == kac_depth
            and row.get("aggregation") == "mean"
            and row.get("distance_bucket") == "all"
        ):
            return float(row["value_mean"])
    return None


def plot_stage1(summary: list[dict[str, str]], out_dir: Path, bit: str) -> None:
    import matplotlib.pyplot as plt

    methods = sorted({row["method"] for row in summary if row.get("distance_bucket") == "all"})
    depth_values = ["8", "16", "32"]
    for metric, filename, ylabel in [
        ("logit_mse", "stage1_logit_mse_b3.png", "P1 logit MSE"),
        ("coordinate_flatness", "stage1_flatness_b3.png", "coordinate flatness"),
    ]:
        plt.figure(figsize=(7.5, 4.5))
        for method in methods:
            xs = [int(depth) for depth in depth_values]
            ys = [
                row_value(
                    summary,
                    metric=metric,
                    method=method,
                    quant_method="kac_rot_uniform",
                    b_numeric=bit,
                    kac_depth=depth,
                )
                for depth in depth_values
            ]
            if any(value is not None for value in ys):
                plt.plot(xs, ys, marker="o", linewidth=1.6, label=method)
        plt.xlabel("Kac depth")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} at b={bit}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()

    plt.figure(figsize=(7.5, 4.5))
    xs = []
    ys = []
    labels = []
    for method in methods:
        value = row_value(
            summary,
            metric="norm_growth_k",
            method=method,
            quant_method="identity",
            b_numeric="0.0",
            kac_depth="0",
        )
        if value is not None:
            labels.append(method)
            xs.append(len(xs))
            ys.append(value)
    plt.bar(xs, ys, color="#4979a8")
    plt.xticks(xs, labels, rotation=25, ha="right")
    plt.ylabel("NormGrowth_K")
    plt.title("Key Position Norm Growth")
    plt.tight_layout()
    plt.savefig(out_dir / "stage1_norm_growth_k.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize layer-expanded phase-2 quantization metrics.")
    parser.add_argument("--metrics", default="runs/phase2/stage1_kac_frontier/metrics/metrics.csv")
    parser.add_argument("--out-dir", default="runs/phase2/stage1_kac_frontier/summary")
    parser.add_argument("--plot-bit", default="3.0")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rows = read_rows(args.metrics)
    summary = summarize(rows)
    write_csv(out_dir / "summary.csv", summary)
    plot_stage1(summary, out_dir, args.plot_bit)
    print(f"Wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
