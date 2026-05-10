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


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def aggregate_snr(rows: list[dict[str, str]], quant_method: str, bit: str, bucket: str) -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("distance_bucket") != bucket:
            continue
        method = row.get("method", "")
        metric = row.get("metric", "")
        if metric in {"S_J", "S_base", "S_total"}:
            groups[method][metric].append(f(row, "value"))
        elif (
            row.get("quant_method") == quant_method
            and row.get("b_numeric") == bit
            and metric in {"N_Q", "SNR_J"}
        ):
            groups[method][metric].append(f(row, "value"))
    out = {}
    for method, metrics in groups.items():
        out[method] = {}
        for metric, values in metrics.items():
            finite = [value for value in values if math.isfinite(value)]
            if finite:
                out[method][metric] = sum(finite) / len(finite)
    return out


def build_joined(
    snr_rows: list[dict[str, str]],
    task_rows: list[dict[str, str]],
    *,
    quant_method: str,
    bit: str,
    bucket: str,
) -> list[dict[str, str | float]]:
    snr = aggregate_snr(snr_rows, quant_method, bit, bucket)
    out = []
    for row in task_rows:
        if row.get("quant_method") != quant_method or row.get("b_numeric") != bit:
            continue
        method = row.get("method", "")
        stats = snr.get(method, {})
        out.append(
            {
                "method": method,
                "context": row.get("context", ""),
                "quant_method": quant_method,
                "b_numeric": bit,
                "snr_bucket": bucket,
                "S_J": stats.get("S_J", ""),
                "N_Q": stats.get("N_Q", ""),
                "SNR_J": stats.get("SNR_J", ""),
                "task_accuracy": f(row, "accuracy"),
                "drop_acc_same": f(row, "drop_acc_same"),
                "delta_acc_same": f(row, "delta_acc_same"),
                "delta_loss_same": f(row, "delta_loss_same"),
                "preserve_j": row.get("preserve_j", ""),
            }
        )
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def plot(joined: list[dict[str, str | float]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    by_context = sorted({str(row["context"]) for row in joined}, key=lambda value: int(value))
    for context in by_context:
        selected = [row for row in joined if str(row["context"]) == context]
        xs = [float(row["SNR_J"]) for row in selected if row["SNR_J"] != ""]
        ys = [float(row["drop_acc_same"]) for row in selected if row["SNR_J"] != ""]
        labels = [str(row["method"]) for row in selected if row["SNR_J"] != ""]
        plt.figure(figsize=(6.2, 4.2))
        plt.scatter(xs, ys, color="#4b79a8")
        for x, y, label in zip(xs, ys, labels):
            plt.annotate(label, (x, y), fontsize=8, xytext=(4, 3), textcoords="offset points")
        plt.xscale("symlog", linthresh=1.0)
        plt.axvline(1.0, color="#777777", linestyle="--", linewidth=1.0)
        plt.xlabel("SNR_J")
        plt.ylabel("DropAcc_same")
        plt.title(f"SNR_J vs Task Drop @ {context}")
        plt.tight_layout()
        plt.savefig(out_dir / f"snr_vs_drop_{context}.png")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Join Jordan-mode SNR diagnostics with task-level quantized evaluation.")
    parser.add_argument("--snr", default="runs/phase2/stage2_jordan_survival_medium/metrics/jordan_mode_metrics.csv")
    parser.add_argument("--task", default="runs/phase2/stage2_task_eval_medium_b2_eval32/summary/task_eval_with_preserve.csv")
    parser.add_argument("--out-dir", default="runs/phase2/stage2_task_eval_medium_b2_eval32/correlation")
    parser.add_argument("--quant-method", default="kac_rot_uniform")
    parser.add_argument("--bit", default="2.0")
    parser.add_argument("--bucket", default="513_1024")
    args = parser.parse_args()

    snr_rows = read_rows(args.snr)
    task_rows = read_rows(args.task)
    out_dir = ensure_dir(args.out_dir)
    joined = build_joined(
        snr_rows,
        task_rows,
        quant_method=args.quant_method,
        bit=args.bit,
        bucket=args.bucket,
    )
    write_csv(out_dir / "snr_task_joined.csv", joined)

    correlations = []
    for context in sorted({str(row["context"]) for row in joined}, key=lambda value: int(value)):
        selected = [row for row in joined if str(row["context"]) == context]
        xs = [float(row["SNR_J"]) for row in selected if row["SNR_J"] != ""]
        drop = [float(row["drop_acc_same"]) for row in selected if row["SNR_J"] != ""]
        loss = [float(row["delta_loss_same"]) for row in selected if row["SNR_J"] != ""]
        correlations.append(
            {
                "context": context,
                "quant_method": args.quant_method,
                "b_numeric": args.bit,
                "snr_bucket": args.bucket,
                "pearson_snr_drop_acc": pearson(xs, drop),
                "pearson_snr_delta_loss": pearson(xs, loss),
                "n": len(xs),
            }
        )
    write_csv(out_dir / "correlations.csv", correlations)
    plot(joined, out_dir)
    print(f"Wrote {out_dir / 'snr_task_joined.csv'}")


if __name__ == "__main__":
    main()
