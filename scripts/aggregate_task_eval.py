#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir, write_csv


GROUP_KEYS = ["method", "context", "quant_method", "kac_depth", "b_numeric"]
METRICS = ["loss", "accuracy", "delta_loss_same", "delta_acc_same", "drop_acc_same"]


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def seed_from_checkpoint(path: str) -> str:
    match = re.search(r"_seed(\d+)\.pt$", path)
    return match.group(1) if match else ""


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def mean_std(values: list[float]) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(var)
    return mean, std, std / math.sqrt(len(values))


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, str | float | int]]:
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        seed = seed_from_checkpoint(row.get("checkpoint", ""))
        key = tuple(row.get(field, "") for field in GROUP_KEYS)
        for metric in METRICS:
            value = as_float(row.get(metric, ""))
            if value is not None:
                groups[key][metric][seed] = value

    out = []
    for key, metric_values in sorted(groups.items()):
        base = {field: value for field, value in zip(GROUP_KEYS, key)}
        row = {**base}
        for metric in METRICS:
            values_by_seed = metric_values.get(metric, {})
            values = list(values_by_seed.values())
            if values:
                mean, std, sem = mean_std(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_sem"] = sem
                row[f"{metric}_n"] = len(values)
        out.append(row)
    return out


def plot_drop(agg_rows: list[dict[str, str | float | int]], out_dir: Path, context: str) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in agg_rows
        if str(row.get("context")) == context
        and row.get("quant_method") in {"scalar_uniform_no_rotation", "kac_rot_uniform"}
    ]
    methods = sorted({str(row["method"]) for row in selected})
    xs = list(range(len(methods)))
    width = 0.38
    for offset, quant_method, color, label in [
        (-width / 2, "scalar_uniform_no_rotation", "#b75d4a", "scalar uniform"),
        (width / 2, "kac_rot_uniform", "#4b79a8", "Kac uniform"),
    ]:
        means = []
        errors = []
        for method in methods:
            row = next((r for r in selected if r["method"] == method and r["quant_method"] == quant_method), None)
            means.append(float(row.get("drop_acc_same_mean", "nan")) if row else float("nan"))
            errors.append(float(row.get("drop_acc_same_sem", 0.0)) if row else 0.0)
        plt.bar([x + offset for x in xs], means, width=width, yerr=errors, capsize=3, color=color, label=label)
    plt.axhline(0.0, color="#666666", linewidth=1.0)
    plt.xticks(xs, methods, rotation=25, ha="right")
    plt.ylabel("DropAcc_same mean")
    plt.title(f"Multiseed K-only Task Drop @ {context}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"drop_compare_multiseed_{context}.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate task-level K-only quantization evaluation across checkpoint seeds.")
    parser.add_argument("--input", default="runs/phase2/stage2_task_eval_medium_multiseed_b2/summary/task_eval_with_preserve.csv")
    parser.add_argument("--out-dir", default="runs/phase2/stage2_task_eval_medium_multiseed_b2/aggregate")
    parser.add_argument("--plot-context", default="4096")
    args = parser.parse_args()

    rows = read_rows(args.input)
    out_dir = ensure_dir(args.out_dir)
    agg_rows = aggregate(rows)
    write_csv(out_dir / "task_eval_aggregate.csv", agg_rows)
    plot_drop(agg_rows, out_dir, args.plot_context)
    print(f"Wrote {out_dir / 'task_eval_aggregate.csv'}")


if __name__ == "__main__":
    main()
