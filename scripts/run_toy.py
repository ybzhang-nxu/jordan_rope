#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, load_config, require_torch, save_config, set_seed, write_csv


TASKS = ("phase", "linear", "mixed")


def target_values(task: str, deltas, cfg):
    import torch

    theta = float(cfg["position"]["theta"])
    toy = cfg["toy"]
    head_dim = int(toy["head_dim"])
    freq_idx = int(toy["target_freq_index"])
    train_context = float(toy["train_context"])
    omega = theta ** (-2.0 * freq_idx / head_dim)
    d = deltas.to(torch.float32)
    if task == "phase":
        return torch.cos(omega * d)
    if task == "linear":
        return -(d / train_context)
    if task == "mixed":
        return (d / train_context) * torch.cos(omega * d)
    raise ValueError(f"Unknown task: {task}")


def fit_ridge(x, y, ridge: float):
    import torch

    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    lhs = x.T @ x + ridge * eye
    rhs = x.T @ y
    return torch.linalg.solve(lhs, rhs)


def mse(pred, target) -> float:
    import torch

    return float(torch.mean((pred - target) ** 2).detach().cpu())


def make_features(method: str, deltas, cfg):
    from jordan_rope.positional import causal_delta_features

    toy = cfg["toy"]
    pos = cfg["position"]
    return causal_delta_features(
        method,
        deltas,
        num_freqs=int(toy["num_freqs"]),
        head_dim=int(toy["head_dim"]),
        theta=float(pos["theta"]),
        gamma=float(toy.get("gamma", pos.get("gamma_min", 1e-4))),
        train_context=int(toy["train_context"]),
        bounded_tau=bool(pos.get("bounded_tau", True)),
    )


def maybe_plot_mixed(out_dir: Path, rows: list[dict], predictions: dict[str, tuple], context: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    plt.figure(figsize=(10, 5))
    for name, (deltas, pred, target) in predictions.items():
        if name == "target":
            continue
        plt.plot(deltas.cpu(), pred.cpu(), label=name, linewidth=1.2)
    any_item = next(iter(predictions.values()))
    plt.plot(any_item[0].cpu(), any_item[2].cpu(), label="target", linewidth=2.0, color="black")
    plt.title(f"Toy mixed target extrapolation, context={context}")
    plt.xlabel("Delta")
    plt.ylabel("score")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"toy_mixed_predictions_T{context}.png", dpi=160)
    plt.close()


def main() -> None:
    require_torch()
    import torch

    parser = argparse.ArgumentParser(description="Run synthetic basis-fitting diagnostics.")
    parser.add_argument("--config", default="configs/full_research.yaml")
    parser.add_argument("--out-dir", default="runs/toy")
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    methods = args.methods or (
        list(cfg["methods"]) + list(cfg.get("ablation_methods", [])) + list(cfg.get("toy_extra_methods", []))
    )
    out_dir = ensure_dir(args.out_dir)
    save_config(cfg, out_dir / "config.yaml")
    device = choose_device(cfg.get("device", "auto"))
    rows: list[dict] = []
    largest_context = max(int(x) for x in cfg["toy"]["eval_contexts"])
    mixed_predictions: dict[str, tuple] = {}

    for seed in cfg["seeds"]:
        set_seed(int(seed))
        train_context = int(cfg["toy"]["train_context"])
        train_points = int(cfg["toy"].get("train_points", train_context))
        all_train = torch.arange(0, train_context, device=device)
        if train_points < train_context:
            perm = torch.randperm(train_context, device=device)[:train_points]
            train_deltas = all_train[perm].sort().values
        else:
            train_deltas = all_train

        for task in TASKS:
            y_train = target_values(task, train_deltas, cfg)
            for method in methods:
                x_train = make_features(method, train_deltas, cfg)
                weights = fit_ridge(x_train, y_train, float(cfg["toy"]["ridge"]))
                train_pred = x_train @ weights
                rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "method": method,
                        "context": train_context,
                        "split": "train",
                        "mse": mse(train_pred, y_train),
                    }
                )
                for context in cfg["toy"]["eval_contexts"]:
                    context = int(context)
                    eval_deltas = torch.arange(0, context, device=device)
                    x_eval = make_features(method, eval_deltas, cfg)
                    y_eval = target_values(task, eval_deltas, cfg)
                    pred = x_eval @ weights
                    rows.append(
                        {
                            "seed": seed,
                            "task": task,
                            "method": method,
                            "context": context,
                            "split": "eval",
                            "mse": mse(pred, y_eval),
                        }
                    )
                    if seed == cfg["seeds"][0] and task == "mixed" and context == largest_context:
                        # Plot a readable subset for long contexts.
                        stride = max(1, math.ceil(context / 1000))
                        mixed_predictions[method] = (
                            eval_deltas[::stride].detach().cpu(),
                            pred[::stride].detach().cpu(),
                            y_eval[::stride].detach().cpu(),
                        )

    write_csv(out_dir / "toy_metrics.csv", rows)
    if mixed_predictions:
        maybe_plot_mixed(out_dir, rows, mixed_predictions, largest_context)
    print(f"Wrote {out_dir / 'toy_metrics.csv'}")


if __name__ == "__main__":
    main()
