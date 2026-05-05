#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir


def as_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_file(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    metric_keys = [key for key in rows[0] if key in {"mse", "accuracy", "loss", "ppl", "delta_loss"}]
    group_keys = [key for key in rows[0] if key not in metric_keys and key != "seed"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_keys)].append(row)
    out = []
    for group, items in groups.items():
        base = {"source": path.name}
        base.update({key: value for key, value in zip(group_keys, group)})
        for metric in metric_keys:
            vals = [as_float(item.get(metric, "")) for item in items]
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            if not vals:
                continue
            base[f"{metric}_mean"] = mean(vals)
            base[f"{metric}_std"] = 0.0 if len(vals) == 1 else stdev(vals)
            base[f"{metric}_n"] = len(vals)
        out.append(base)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment CSV files into mean/std tables.")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--out", default="runs/summary.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows: list[dict] = []
    for path in run_dir.rglob("*.csv"):
        if path.name == Path(args.out).name:
            continue
        rows.extend(summarize_file(path))

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        print(f"No CSV rows found under {run_dir}")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
