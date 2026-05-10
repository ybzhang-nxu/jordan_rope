#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir, write_csv


KEYS = ["method", "context", "quant_method", "kac_depth", "b_numeric"]
METRICS = ["accuracy_mean", "loss_mean", "drop_acc_same_mean", "delta_loss_same_mean"]


def read_rows(path: str | Path) -> dict[tuple[str, ...], dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {tuple(row.get(key, "") for key in KEYS): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare task aggregates for two cache targets.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--left-label", default="k")
    parser.add_argument("--right", required=True)
    parser.add_argument("--right-label", default="kv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context", default=None)
    args = parser.parse_args()

    left = read_rows(args.left)
    right = read_rows(args.right)
    rows = []
    for key in sorted(set(left) & set(right)):
        base = dict(zip(KEYS, key))
        if args.context is not None and base["context"] != args.context:
            continue
        row = {**base}
        for metric in METRICS:
            lval = left[key].get(metric, "")
            rval = right[key].get(metric, "")
            row[f"{args.left_label}_{metric}"] = lval
            row[f"{args.right_label}_{metric}"] = rval
            try:
                row[f"{args.right_label}_minus_{args.left_label}_{metric}"] = float(rval) - float(lval)
            except ValueError:
                row[f"{args.right_label}_minus_{args.left_label}_{metric}"] = ""
        rows.append(row)

    out_dir = ensure_dir(args.out_dir)
    write_csv(out_dir / "task_target_comparison.csv", rows)
    print(f"Wrote {out_dir / 'task_target_comparison.csv'}")


if __name__ == "__main__":
    main()
