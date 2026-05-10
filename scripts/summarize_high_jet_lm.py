#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize_eval(path: Path) -> dict:
    rows = read_rows(path)
    if not rows:
        return {"path": str(path), "rows": 0, "summary": []}
    max_step = max(int(row["step"]) for row in rows)
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        if int(row["step"]) == max_step:
            grouped[(int(row["context"]), row["method"])].append(row)

    summary = []
    by_context: dict[int, list[dict]] = defaultdict(list)
    for (context, method), group in grouped.items():
        losses = [float(row["loss"]) for row in group]
        accs = [float(row["accuracy"]) for row in group]
        loss_mean, loss_std = mean_std(losses)
        acc_mean, acc_std = mean_std(accs)
        item = {
            "context": context,
            "method": method,
            "n": len(group),
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
        }
        summary.append(item)
        by_context[context].append(item)

    best = []
    for context, items in sorted(by_context.items()):
        best_acc = max(items, key=lambda item: (item["accuracy_mean"], -item["loss_mean"]))
        best_loss = min(items, key=lambda item: item["loss_mean"])
        best.append(
            {
                "context": context,
                "best_accuracy_method": best_acc["method"],
                "best_accuracy": best_acc["accuracy_mean"],
                "best_accuracy_std": best_acc["accuracy_std"],
                "best_loss_method": best_loss["method"],
                "best_loss": best_loss["loss_mean"],
                "best_loss_std": best_loss["loss_std"],
            }
        )
    summary.sort(key=lambda item: (item["context"], item["loss_mean"]))
    return {"path": str(path), "rows": len(rows), "step": max_step, "summary": summary, "best": best}


def write_markdown(results: dict[str, dict], out_path: Path) -> None:
    lines = ["# High-Jet Synthetic LM Medium Summary", ""]
    for name, result in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Final step: `{result['step']}`.")
        lines.append("")
        lines.append("| context | best acc method | acc mean | acc std | best loss method | loss mean | loss std |")
        lines.append("|---:|---|---:|---:|---|---:|---:|")
        for item in result["best"]:
            lines.append(
                f"| {item['context']} | {item['best_accuracy_method']} | "
                f"{item['best_accuracy']:.4f} | {item['best_accuracy_std']:.4f} | "
                f"{item['best_loss_method']} | {item['best_loss']:.4f} | {item['best_loss_std']:.4f} |"
            )
        lines.append("")
        lines.append("| context | method | acc mean | acc std | loss mean | loss std | n |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for item in result["summary"]:
            lines.append(
                f"| {item['context']} | {item['method']} | "
                f"{item['accuracy_mean']:.4f} | {item['accuracy_std']:.4f} | "
                f"{item['loss_mean']:.4f} | {item['loss_std']:.4f} | {item['n']} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=Path("runs/phase2/high_jet_kernel_lm_medium/summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("runs/phase2/high_jet_kernel_lm_medium/summary.md"))
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length")

    results = {name: summarize_eval(Path(path)) for name, path in zip(args.names, args.inputs)}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
