#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retrieval accuracy as a function of distance.")
    parser.add_argument("--csv", default="runs/retrieval/retrieval_eval.csv")
    parser.add_argument("--out-dir", default="runs/retrieval")
    parser.add_argument("--context", type=int, default=None)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Matplotlib is required for plotting. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        ) from exc

    rows = []
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["context"] = int(row["context"])
            row["distance"] = int(row["distance"])
            row["accuracy"] = float(row["accuracy"])
            row["step"] = int(row.get("step", 0) or 0)
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    max_step = max(row["step"] for row in rows)
    rows = [row for row in rows if row["step"] == max_step]
    context = args.context or max(row["context"] for row in rows)
    rows = [row for row in rows if row["context"] == context]
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["distance"])].append(row["accuracy"])

    by_method: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (method, distance), values in grouped.items():
        by_method[method].append((distance, mean(values)))

    plt.figure(figsize=(9, 5))
    for method, points in sorted(by_method.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=method)
    plt.xscale("log", base=2)
    plt.ylim(0.0, 1.0)
    plt.xlabel("query-key distance")
    plt.ylabel("accuracy")
    plt.title(f"Retrieval accuracy-distance curve, T={context}, step={max_step}")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out_dir = ensure_dir(args.out_dir)
    out = out_dir / f"retrieval_accuracy_distance_T{context}.png"
    plt.savefig(out, dpi=160)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
