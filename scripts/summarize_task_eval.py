#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir, write_csv


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def add_preserve_j(rows: list[dict[str, str]]) -> list[dict[str, str | float]]:
    identity = {}
    by_quant = {}
    for row in rows:
        key = (row["method"], row["context"])
        qkey = (row["method"], row["context"], row["quant_method"], row["kac_depth"], row["b_numeric"])
        if row["quant_method"] == "identity":
            identity[key] = f(row, "accuracy")
        by_quant[qkey] = f(row, "accuracy")

    out = []
    for row in rows:
        context = row["context"]
        qkey_rope = ("rope", context, row["quant_method"], row["kac_depth"], row["b_numeric"])
        rope_quant = by_quant.get(qkey_rope)
        rope_fp16 = identity.get(("rope", context))
        method_fp16 = identity.get((row["method"], context))
        acc = f(row, "accuracy")
        preserve = ""
        if rope_quant is not None and rope_fp16 is not None and method_fp16 is not None:
            denom = method_fp16 - rope_fp16
            if abs(denom) >= 1e-6:
                preserve = (acc - rope_quant) / denom
        out.append({**row, "preserve_j": preserve})
    return out


def plot_accuracy(rows: list[dict[str, str]], out_dir: Path, context: str, quant_method: str) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["context"] == context and row["quant_method"] in {"identity", quant_method}
    ]
    methods = sorted({row["method"] for row in selected})
    xs = list(range(len(methods)))
    width = 0.38
    identity = [next((f(row, "accuracy") for row in selected if row["method"] == method and row["quant_method"] == "identity"), float("nan")) for method in methods]
    quant = [next((f(row, "accuracy") for row in selected if row["method"] == method and row["quant_method"] == quant_method), float("nan")) for method in methods]
    plt.figure(figsize=(8.0, 4.5))
    plt.bar([x - width / 2 for x in xs], identity, width=width, label="identity", color="#4b79a8")
    plt.bar([x + width / 2 for x in xs], quant, width=width, label=quant_method, color="#b75d4a")
    plt.xticks(xs, methods, rotation=25, ha="right")
    plt.ylabel("accuracy")
    plt.title(f"Task Accuracy @ {context}")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"accuracy_{context}_{quant_method}.png")
    plt.close()


def plot_drop_compare(rows: list[dict[str, str]], out_dir: Path, context: str) -> None:
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["context"] == context
        and row["quant_method"] in {"scalar_uniform_no_rotation", "kac_rot_uniform"}
    ]
    methods = sorted({row["method"] for row in selected})
    xs = list(range(len(methods)))
    width = 0.38
    scalar = [
        next((f(row, "drop_acc_same") for row in selected if row["method"] == method and row["quant_method"] == "scalar_uniform_no_rotation"), float("nan"))
        for method in methods
    ]
    kac = [
        next((f(row, "drop_acc_same") for row in selected if row["method"] == method and row["quant_method"] == "kac_rot_uniform"), float("nan"))
        for method in methods
    ]
    plt.figure(figsize=(8.0, 4.5))
    plt.bar([x - width / 2 for x in xs], scalar, width=width, label="scalar uniform", color="#b75d4a")
    plt.bar([x + width / 2 for x in xs], kac, width=width, label="Kac uniform", color="#4b79a8")
    plt.axhline(0.0, color="#666666", linewidth=1.0)
    plt.xticks(xs, methods, rotation=25, ha="right")
    plt.ylabel("DropAcc_same")
    plt.title(f"K-only Quantization Task Drop @ {context}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"drop_compare_{context}.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize K-only quantized kernel-LM task evaluation.")
    parser.add_argument("--input", default="runs/phase2/stage2_task_eval_medium/task_eval.csv")
    parser.add_argument("--out-dir", default="runs/phase2/stage2_task_eval_medium/summary")
    parser.add_argument("--plot-context", default="4096")
    parser.add_argument("--plot-quant-method", default="kac_rot_uniform")
    args = parser.parse_args()

    rows = add_preserve_j(read_rows(args.input))
    out_dir = ensure_dir(args.out_dir)
    write_csv(out_dir / "task_eval_with_preserve.csv", rows)
    plot_accuracy(rows, out_dir, args.plot_context, args.plot_quant_method)
    plot_drop_compare(rows, out_dir, args.plot_context)
    print(f"Wrote {out_dir / 'task_eval_with_preserve.csv'}")


if __name__ == "__main__":
    main()
