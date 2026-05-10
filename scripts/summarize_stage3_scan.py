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


def f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except ValueError:
        return default


def mean(values: list[float]) -> float:
    finite = [x for x in values if math.isfinite(x)]
    return sum(finite) / len(finite) if finite else float("nan")


def parse_scan_method(method: str) -> tuple[float, float]:
    c_value = 1.0
    eta_value = 0.0
    for token in method.split("_"):
        if token.startswith("c") and token[1:].isdigit():
            c_value = int(token[1:]) / 100.0
        if token.startswith("eta") and token[3:].isdigit() and len(token[3:]) >= 3:
            eta_value = int(token[3:]) / 1000.0
    return c_value, eta_value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def task_metrics(rows: list[dict[str, str]], context: str, bit: str, kac_depth: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("context") != context:
            continue
        method = row["method"]
        if row.get("quant_method") == "identity":
            grouped[method]["acc_fp16"].append(f(row, "accuracy"))
            grouped[method]["loss_fp16"].append(f(row, "loss"))
        elif (
            row.get("quant_method") == "kac_rot_uniform"
            and row.get("b_numeric") == bit
            and row.get("kac_depth") == kac_depth
        ):
            grouped[method]["acc_kac"].append(f(row, "accuracy"))
            grouped[method]["loss_kac"].append(f(row, "loss"))
            grouped[method]["drop_acc_kac"].append(f(row, "drop_acc_same"))
            grouped[method]["delta_loss_kac"].append(f(row, "delta_loss_same"))
        elif (
            row.get("quant_method") == "scalar_uniform_no_rotation"
            and row.get("b_numeric") == bit
            and row.get("kac_depth") == "0"
        ):
            grouped[method]["acc_scalar"].append(f(row, "accuracy"))
            grouped[method]["loss_scalar"].append(f(row, "loss"))
            grouped[method]["drop_acc_scalar"].append(f(row, "drop_acc_same"))
            grouped[method]["delta_loss_scalar"].append(f(row, "delta_loss_same"))
    return {method: {metric: mean(values) for metric, values in metrics.items()} for method, metrics in grouped.items()}


def quant_metrics(rows: list[dict[str, str]], bit: str, kac_depth: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("distance_bucket") != "all":
            continue
        method = row["method"]
        if (
            row.get("quant_method") == "kac_rot_uniform"
            and row.get("b_numeric") == bit
            and row.get("kac_depth") == kac_depth
            and row.get("metric") in {"logit_mse", "relative_logit_mse"}
        ):
            out[method][row["metric"]] = f(row, "value_mean")
        if row.get("quant_method") == "identity" and row.get("metric") == "norm_growth_k":
            out[method]["norm_growth_k"] = f(row, "value_mean")
    return out


def jordan_metrics(rows: list[dict[str, str]], bucket: str, bit: str, kac_depth: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("distance_bucket") != bucket:
            continue
        method = row["method"]
        metric = row.get("metric", "")
        if metric in {"S_J", "S_base", "S_total"} and row.get("quant_method") == "identity":
            grouped[method][metric].append(f(row, "value"))
        if (
            metric in {"N_Q", "SNR_J"}
            and row.get("quant_method") == "kac_rot_uniform"
            and row.get("b_numeric") == bit
            and row.get("kac_depth") == kac_depth
        ):
            grouped[method][metric].append(f(row, "value"))
    return {method: {metric: mean(values) for metric, values in metrics.items()} for method, metrics in grouped.items()}


def write_heatmap(rows: list[dict[str, float | str]], metric: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cs = sorted({float(row["c"]) for row in rows})
    etas = sorted({float(row["eta"]) for row in rows})
    matrix = np.full((len(etas), len(cs)), np.nan, dtype=float)
    for row in rows:
        i = etas.index(float(row["eta"]))
        j = cs.index(float(row["c"]))
        matrix[i, j] = float(row[metric])

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    image = ax.imshow(matrix, aspect="auto", origin="lower")
    ax.set_xticks(range(len(cs)), [f"{c:g}" for c in cs])
    ax.set_yticks(range(len(etas)), [f"{eta:g}" for eta in etas])
    ax.set_xlabel("c")
    ax.set_ylabel("eta init")
    ax.set_title(metric)
    for i in range(len(etas)):
        for j in range(len(cs)):
            value = matrix[i, j]
            if math.isfinite(float(value)):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage 3 c/eta scan outputs.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--quant-summary", required=True)
    parser.add_argument("--jordan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--context", default="8192")
    parser.add_argument("--snr-bucket", default="2049_4096")
    parser.add_argument("--bit", default="3.0")
    parser.add_argument("--kac-depth", default="16")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    task = task_metrics(load_csv(Path(args.task)), args.context, args.bit, args.kac_depth)
    quant = quant_metrics(load_csv(Path(args.quant_summary)), args.bit, args.kac_depth)
    jordan = jordan_metrics(load_csv(Path(args.jordan)), args.snr_bucket, args.bit, args.kac_depth)

    methods = sorted(set(task) | set(quant) | set(jordan))
    rows: list[dict[str, float | str]] = []
    for method in methods:
        c_value, eta_value = parse_scan_method(method)
        inv_decay_long = math.exp(c_value * 4096.0 / 1024.0)
        row: dict[str, float | str] = {
            "method": method,
            "c": c_value,
            "eta": eta_value,
            "context": args.context,
            "snr_bucket": args.snr_bucket,
            "inv_decay_4096": inv_decay_long,
        }
        row.update(task.get(method, {}))
        row.update(quant.get(method, {}))
        row.update(jordan.get(method, {}))
        rows.append(row)

    rows.sort(key=lambda row: (float(row["c"]), float(row["eta"])))
    write_csv(out_dir / "stage3_summary.csv", rows)

    for metric in ["acc_fp16", "acc_kac", "drop_acc_kac", "delta_loss_kac", "logit_mse", "S_J", "SNR_J", "norm_growth_k"]:
        if any(metric in row and math.isfinite(float(row[metric])) for row in rows):
            write_heatmap(rows, metric, out_dir / f"{metric}_heatmap.png")

    print(f"Wrote {out_dir / 'stage3_summary.csv'}")


if __name__ == "__main__":
    main()
