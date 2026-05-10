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


HEAD_KEY_FIELDS = ["method", "checkpoint", "layer", "head"]
GROUP_FIELDS = ["method", "quant_method", "kac_depth", "b_numeric", "b_storage", "distance_bucket", "metric"]


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def finite_float(value: str) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def head_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row.get(field, "") for field in HEAD_KEY_FIELDS)


def mean_stats(values: list[float]) -> dict[str, float | int]:
    values_sorted = sorted(values)
    mean = sum(values) / len(values)
    mid = len(values_sorted) // 2
    median = values_sorted[mid] if len(values_sorted) % 2 else 0.5 * (values_sorted[mid - 1] + values_sorted[mid])
    return {
        "value_mean": mean,
        "value_median": median,
        "value_min": min(values),
        "value_max": max(values),
        "value_count": len(values),
    }


def select_top_heads(
    rows: list[dict[str, str]],
    *,
    bucket: str,
    top_fraction: float,
) -> tuple[set[tuple[str, str, str, str]], list[dict[str, str | float | int]]]:
    by_method: dict[str, list[tuple[tuple[str, str, str, str], float]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("aggregation") != "per_head_mean"
            or row.get("quant_method") != "identity"
            or row.get("metric") != "S_J"
            or row.get("distance_bucket") != bucket
            or row.get("head") in {"", "all"}
        ):
            continue
        value = finite_float(row.get("value", ""))
        if value is None:
            continue
        by_method[row.get("method", "")].append((head_key(row), value))

    selected: set[tuple[str, str, str, str]] = set()
    selected_rows: list[dict[str, str | float | int]] = []
    for method, values in sorted(by_method.items()):
        values.sort(key=lambda item: item[1], reverse=True)
        top_count = max(1, int(math.ceil(len(values) * float(top_fraction))))
        for rank, (key, value) in enumerate(values[:top_count], start=1):
            selected.add(key)
            selected_rows.append(
                {
                    "method": method,
                    "checkpoint": key[1],
                    "layer": key[2],
                    "head": key[3],
                    "rank": rank,
                    "S_J": value,
                    "top_fraction": float(top_fraction),
                    "selected_count": top_count,
                    "candidate_count": len(values),
                    "selection_bucket": bucket,
                }
            )
    return selected, selected_rows


def summarize_p1(
    rows: list[dict[str, str]],
    *,
    selected: set[tuple[str, str, str, str]],
    metrics: set[str],
    buckets: set[str],
    quant_methods: set[str] | None,
) -> list[dict[str, str | float | int]]:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    total_selected_by_method: dict[str, int] = defaultdict(int)
    total_heads_by_method: dict[str, set[tuple[str, str, str, str]]] = defaultdict(set)
    for key in selected:
        total_selected_by_method[key[0]] += 1

    for row in rows:
        if row.get("aggregation") != "per_head_mean" or row.get("head") in {"", "all"}:
            continue
        if row.get("metric") not in metrics or row.get("distance_bucket") not in buckets:
            continue
        if quant_methods is not None and row.get("quant_method") not in quant_methods:
            continue
        value = finite_float(row.get("value", ""))
        if value is None:
            continue
        key = head_key(row)
        total_heads_by_method[row.get("method", "")].add(key)
        subset_names = ["all_heads"]
        if key in selected:
            subset_names.append("top_jordan_heads")
        for subset in subset_names:
            group_key = tuple([subset] + [row.get(field, "") for field in GROUP_FIELDS])
            groups[group_key].append(value)

    out: list[dict[str, str | float | int]] = []
    for key, values in sorted(groups.items()):
        subset = key[0]
        base = {"subset": subset, **{field: value for field, value in zip(GROUP_FIELDS, key[1:])}}
        method = str(base["method"])
        out.append(
            {
                **base,
                **mean_stats(values),
                "selected_head_count": total_selected_by_method.get(method, 0),
                "candidate_head_count": len(total_heads_by_method.get(method, set())),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize held-out P1 metrics on calibration-selected top Jordan heads.")
    parser.add_argument("--p0-calib", nargs="+", required=True)
    parser.add_argument("--p1-eval", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selection-bucket", default="2049_4096")
    parser.add_argument("--p1-buckets", default="all,2049_4096")
    parser.add_argument("--metrics", default="logit_mse,relative_logit_mse,attention_kl,top1_agreement,top5_agreement")
    parser.add_argument("--quant-methods", default="")
    parser.add_argument("--top-fraction", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    p0_rows = read_rows(args.p0_calib)
    p1_rows = read_rows(args.p1_eval)
    selected, selected_rows = select_top_heads(
        p0_rows,
        bucket=args.selection_bucket,
        top_fraction=float(args.top_fraction),
    )
    quant_methods = {item for item in args.quant_methods.split(",") if item.strip()} or None
    summary = summarize_p1(
        p1_rows,
        selected=selected,
        metrics={item for item in args.metrics.split(",") if item.strip()},
        buckets={item for item in args.p1_buckets.split(",") if item.strip()},
        quant_methods=quant_methods,
    )
    write_csv(out_dir / "selected_heads.csv", selected_rows)
    write_csv(out_dir / "top_jordan_head_summary.csv", summary)
    print(f"Wrote {out_dir / 'top_jordan_head_summary.csv'}")


if __name__ == "__main__":
    main()
