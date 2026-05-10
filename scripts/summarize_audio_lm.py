#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize_one(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"path": str(path), "rows": 0, "seeds": [], "summary": [], "best": []}
    max_step = max(int(row["step"]) for row in rows)
    seeds = sorted({row["seed"] for row in rows}, key=lambda value: int(value))
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        if int(row["step"]) == max_step:
            grouped[(int(row["context"]), row["method"])].append(row)

    summary = []
    by_context: dict[int, list[dict]] = defaultdict(list)
    for (context, method), group in grouped.items():
        losses = [float(row["loss"]) for row in group]
        ppls = [float(row["ppl"]) for row in group]
        deltas = [float(row["delta_loss"]) for row in group]
        loss_mean, loss_std = mean_std(losses)
        ppl_mean, ppl_std = mean_std(ppls)
        delta_mean, delta_std = mean_std(deltas)
        item = {
            "context": context,
            "method": method,
            "n": len(group),
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "ppl_mean": ppl_mean,
            "ppl_std": ppl_std,
            "delta_loss_mean": delta_mean,
            "delta_loss_std": delta_std,
        }
        summary.append(item)
        by_context[context].append(item)

    best = []
    for context, items in sorted(by_context.items()):
        best_loss = min(items, key=lambda item: item["loss_mean"])
        best.append(
            {
                "context": context,
                "best_loss_method": best_loss["method"],
                "best_loss": best_loss["loss_mean"],
                "best_loss_std": best_loss["loss_std"],
                "best_ppl": best_loss["ppl_mean"],
                "best_ppl_std": best_loss["ppl_std"],
            }
        )
    summary.sort(key=lambda item: (item["context"], item["loss_mean"]))
    return {"path": str(path), "rows": len(rows), "step": max_step, "seeds": seeds, "summary": summary, "best": best}


def write_markdown(results: dict[str, dict], out_path: Path) -> None:
    lines = ["# LM Summary", ""]
    lines.append("Run summary: final-step means/stds by dataset, method, and evaluation context.")
    lines.append("")
    for name, result in results.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Final step: `{result['step']}`. Seeds: `{', '.join(result['seeds'])}`.")
        lines.append("")
        lines.append("| context | best method | loss mean | loss std | ppl mean | ppl std |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for item in result["best"]:
            lines.append(
                f"| {item['context']} | {item['best_loss_method']} | "
                f"{item['best_loss']:.4f} | {item['best_loss_std']:.4f} | "
                f"{item['best_ppl']:.2f} | {item['best_ppl_std']:.2f} |"
            )
        lines.append("")
        lines.append("| context | method | loss mean | loss std | delta mean | ppl mean | n |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for item in result["summary"]:
            lines.append(
                f"| {item['context']} | {item['method']} | "
                f"{item['loss_mean']:.4f} | {item['loss_std']:.4f} | "
                f"{item['delta_loss_mean']:.4f} | {item['ppl_mean']:.2f} | {item['n']} |"
            )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=Path("runs/phase2/audio_lm_medium/summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("runs/phase2/audio_lm_medium/summary.md"))
    args = parser.parse_args()
    if len(args.inputs) != len(args.names):
        raise SystemExit("--inputs and --names must have the same length")
    results = {name: summarize_one(Path(path)) for name, path in zip(args.names, args.inputs)}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(results, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
