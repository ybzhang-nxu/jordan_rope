#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir


def bucket_key(name: str) -> int:
    try:
        return int(name.split("_", 1)[0])
    except (ValueError, IndexError):
        return 0


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def grouped_mean(rows, metric: str, quant_method: str, bit: str):
    groups = defaultdict(list)
    for row in rows:
        if row.get("metric") != metric:
            continue
        if metric in {"N_Q", "SNR_J"}:
            if row.get("quant_method") != quant_method or row.get("b_numeric") != bit:
                continue
        groups[(row.get("method", ""), row.get("distance_bucket", ""))].append(float(row["value"]))
    return {key: sum(values) / len(values) for key, values in groups.items() if values}


def plot_lines(rows, metric: str, out_path: Path, quant_method: str, bit: str) -> None:
    import matplotlib.pyplot as plt

    values = grouped_mean(rows, metric, quant_method, bit)
    methods = sorted({method for method, _ in values})
    buckets = sorted({bucket for _, bucket in values}, key=bucket_key)
    xs = list(range(len(buckets)))
    plt.figure(figsize=(8.0, 4.5))
    for method in methods:
        ys = [values.get((method, bucket)) for bucket in buckets]
        if any(value is not None for value in ys):
            plt.plot(xs, ys, marker="o", linewidth=1.6, label=method)
    plt.xticks(xs, buckets, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by distance bucket")
    if metric == "SNR_J":
        plt.axhline(1.0, color="#666666", linewidth=1.0, linestyle="--")
        plt.axhline(3.0, color="#999999", linewidth=1.0, linestyle=":")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot P0/P1 Jordan-mode survival diagnostics.")
    parser.add_argument("--metrics", default="runs/phase2/stage2_jordan_survival_smoke/metrics/jordan_mode_metrics.csv")
    parser.add_argument("--out-dir", default="runs/phase2/stage2_jordan_survival_smoke/figures")
    parser.add_argument("--quant-method", default="kac_rot_uniform")
    parser.add_argument("--bit", default="3.0")
    args = parser.parse_args()

    rows = read_rows(args.metrics)
    out_dir = ensure_dir(args.out_dir)
    plot_lines(rows, "S_J", out_dir / "jordan_signal.png", args.quant_method, args.bit)
    plot_lines(rows, "N_Q", out_dir / "quant_noise.png", args.quant_method, args.bit)
    plot_lines(rows, "SNR_J", out_dir / "snr_j.png", args.quant_method, args.bit)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
