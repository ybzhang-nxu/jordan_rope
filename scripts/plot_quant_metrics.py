#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return float("nan")


def label_for(row: dict[str, str]) -> str:
    method = row.get("quant_method", "")
    bits = as_float(row, "b_numeric")
    depth = row.get("kac_depth", "0")
    if method == "identity":
        return "identity"
    if method == "kac_rot_uniform":
        return f"kac L={depth}, b={bits:g}"
    return f"{method}, b={bits:g}"


def plot_bar(rows: list[dict[str, str]], metric: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row.get("metric") == metric
        and row.get("aggregation") == "mean"
        and row.get("distance_bucket") == "all"
    ]
    labels = [label_for(row) for row in selected]
    values = [as_float(row, "value") for row in selected]
    plt.figure(figsize=(max(7.0, 0.35 * len(labels)), 4.0))
    plt.bar(range(len(values)), values, color="#3b6ea8")
    plt.xticks(range(len(values)), labels, rotation=35, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_norm_profile(rows: list[dict[str, str]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.0, 4.0))
    for metric, color in [("norm_profile_k", "#3b6ea8"), ("norm_profile_q", "#b54a4a")]:
        selected = [
            row
            for row in rows
            if row.get("metric") == metric and row.get("distance_bucket", "").startswith("position_")
        ]
        selected.sort(key=lambda row: int(row["distance_bucket"].removeprefix("position_")))
        xs = [int(row["distance_bucket"].removeprefix("position_")) for row in selected]
        ys = [as_float(row, "value") for row in selected]
        plt.plot(xs, ys, linewidth=1.8, label=metric, color=color)
    plt.xlabel("position")
    plt.ylabel("mean L2 norm")
    plt.title("Position Norm Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot phase-2 smoke quantization metrics.")
    parser.add_argument("--metrics", default="runs/phase2/stage0_smoke/metrics/metrics.csv")
    parser.add_argument("--out-dir", default="runs/phase2/stage0_smoke/figures")
    args = parser.parse_args()

    rows = read_rows(args.metrics)
    out_dir = ensure_dir(args.out_dir)
    plot_bar(rows, "coordinate_flatness", out_dir / "flatness.png", "Coordinate Flatness")
    plot_bar(rows, "logit_mse", out_dir / "logit_mse.png", "P1 Logit MSE")
    plot_norm_profile(rows, out_dir / "position_norm_profile.png")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
