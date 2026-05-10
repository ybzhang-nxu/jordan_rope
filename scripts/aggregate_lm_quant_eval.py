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


GROUP_KEYS = ["method", "cache_target", "context", "quant_method", "kac_depth", "b_numeric"]
METRICS = ["loss", "ppl", "delta_loss_same"]


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def seed_from_checkpoint(path: str) -> str:
    match = re.search(r"_seed(\d+)\.pt$", path)
    return match.group(1) if match else path


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
        row = {field: value for field, value in zip(GROUP_KEYS, key)}
        for metric in METRICS:
            values = list(metric_values.get(metric, {}).values())
            if not values:
                continue
            mean, std, sem = mean_std(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_sem"] = sem
            row[f"{metric}_n"] = len(values)
        out.append(row)
    return out


def plot_delta(rows: list[dict[str, str | float | int]], out_dir: Path, context: str, cache_target: str) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if str(row.get("context")) == context
        and row.get("cache_target") == cache_target
        and row.get("quant_method") != "identity"
    ]
    methods = sorted({str(row["method"]) for row in selected})
    xs = list(range(len(methods)))
    means = []
    errors = []
    for method in methods:
        row = next((r for r in selected if r["method"] == method), None)
        means.append(float(row.get("delta_loss_same_mean", "nan")) if row else float("nan"))
        errors.append(float(row.get("delta_loss_same_sem", 0.0)) if row else 0.0)
    plt.bar(xs, means, yerr=errors, capsize=3, color="#4b79a8")
    plt.axhline(0.0, color="#666666", linewidth=1.0)
    plt.xticks(xs, methods, rotation=25, ha="right")
    plt.ylabel("Delta loss vs fp16")
    plt.title(f"LM cache quantization extra loss @ {context}, {cache_target}")
    plt.tight_layout()
    plt.savefig(out_dir / f"delta_loss_{cache_target}_{context}.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate LM quantized evaluation across checkpoint seeds.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--plot-context", default="2048")
    parser.add_argument("--plot-cache-target", default="kv")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    rows = aggregate(read_rows(args.input))
    write_csv(out_dir / "lm_quant_eval_aggregate.csv", rows)
    plot_delta(rows, out_dir, args.plot_context, args.plot_cache_target)
    print(f"Wrote {out_dir / 'lm_quant_eval_aggregate.csv'}")


if __name__ == "__main__":
    main()
