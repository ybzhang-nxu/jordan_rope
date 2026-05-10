#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, load_config, require_torch, save_config, set_seed, write_csv


BENCHMARKS = {
    "phase_drift": "RoPE frequency slightly misspecified; Jordan tangent can apply a first-order correction.",
    "seasonal_trend": "A seasonal component whose amplitude grows with distance.",
    "damped_wave": "A damped oscillatory response with a polynomial wave term.",
    "rhythm_envelope": "Two rhythmic components with distance-dependent envelope.",
    "motif_spacing": "A repeated motif-spacing signal with distance modulation.",
    "high_jet_r1": "First frequency-jet target (d/L) cos(omega d).",
    "high_jet_r2": "Second frequency-jet target (d/L)^2 cos(omega d).",
    "high_jet_r3": "Third frequency-jet target (d/L)^3 cos(omega d).",
    "scaled_high_jet_r2": "Scaled second frequency-jet target (d/L)^2 exp(-c d/L) cos(omega d).",
    "scaled_high_jet_r3": "Scaled third frequency-jet target (d/L)^3 exp(-c d/L) cos(omega d).",
}


def frequencies(cfg):
    import torch

    bench = cfg["benchmarks"]
    theta = float(cfg["position"]["theta"])
    head_dim = int(bench["head_dim"])
    idx = int(bench["target_freq_index"])
    idx2 = int(bench["second_freq_index"])
    omega = theta ** (-2.0 * idx / head_dim)
    omega2 = theta ** (-2.0 * idx2 / head_dim)
    return torch.tensor(float(omega)), torch.tensor(float(omega2))


def target_values(name: str, deltas, cfg):
    import torch

    bench = cfg["benchmarks"]
    train_context = float(bench["train_context"])
    gamma = float(bench.get("gamma", cfg["position"].get("gamma_min", 1e-4)))
    omega, omega2 = frequencies(cfg)
    omega = omega.to(device=deltas.device, dtype=torch.float32)
    omega2 = omega2.to(device=deltas.device, dtype=torch.float32)
    d = deltas.to(torch.float32)
    x = d / train_context

    if name == "phase_drift":
        eps = float(bench.get("drift_relative_eps", 5e-5))
        return torch.cos((omega * (1.0 + eps)) * d)

    if name == "seasonal_trend":
        return (0.35 + 0.9 * x) * torch.cos(omega * d) + 0.25 * torch.sin(omega * d)

    if name == "damped_wave":
        decay = torch.exp(-gamma * d)
        return decay * (torch.cos(omega * d) + 0.75 * x * torch.sin(omega * d))

    if name == "rhythm_envelope":
        carrier = 0.65 * torch.cos(omega * d) + 0.35 * torch.sin(omega2 * d)
        return (0.25 + 0.85 * x) * carrier

    if name == "motif_spacing":
        # A motif/repeat proxy: periodic correlation plus distance-dependent confidence.
        motif = 0.7 * torch.cos(omega2 * d) - 0.3 * torch.sin(omega2 * d)
        return (x / (1.0 + 0.25 * x)) * motif

    if name.startswith("high_jet_r"):
        order = int(name.removeprefix("high_jet_r"))
        return x.pow(order) * torch.cos(omega * d)

    if name.startswith("scaled_high_jet_r"):
        order = int(name.removeprefix("scaled_high_jet_r"))
        c_value = float(bench.get("scaled_c", 0.1))
        return x.pow(order) * torch.exp(-c_value * x) * torch.cos(omega * d)

    raise ValueError(f"Unknown benchmark: {name}")


def make_features(method: str, deltas, cfg):
    from jordan_rope.positional import causal_delta_features

    bench = cfg["benchmarks"]
    pos = cfg["position"]
    return causal_delta_features(
        method,
        deltas,
        num_freqs=int(bench["num_freqs"]),
        head_dim=int(bench["head_dim"]),
        theta=float(pos["theta"]),
        gamma=float(bench.get("gamma", pos.get("gamma_min", 1e-4))),
        train_context=int(bench["train_context"]),
        bounded_tau=bool(pos.get("bounded_tau", True)),
    )


def fit_ridge(x, y, ridge: float):
    import torch

    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    return torch.linalg.solve(x.T @ x + ridge * eye, x.T @ y)


def metrics(pred, target):
    import torch

    err = pred - target
    mse = torch.mean(err.square())
    denom = torch.mean((target - target.mean()).square()).clamp_min(1e-12)
    r2 = 1.0 - mse / denom
    return float(mse.detach().cpu()), float(r2.detach().cpu())


def plot_predictions(out_dir: Path, benchmark: str, context: int, predictions: dict[str, tuple]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    order = [
        "target",
        "rope",
        "rope_alibi",
        "direct_sum",
        "jordan_rope",
        "jordan_raw_tau",
        "jordan_m3",
        "jordan_m4",
        "jordan_scaled_m3_c010",
        "jordan_scaled_m4_c010",
    ]
    plt.figure(figsize=(10, 5))
    for name in order:
        if name not in predictions:
            continue
        deltas, pred, target = predictions[name]
        if name == "target":
            plt.plot(deltas.cpu(), target.cpu(), label="target", color="black", linewidth=2.0)
        else:
            plt.plot(deltas.cpu(), pred.cpu(), label=name, linewidth=1.15)
    plt.title(f"{benchmark} extrapolation, T={context}")
    plt.xlabel("Delta")
    plt.ylabel("score")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"{benchmark}_T{context}.png", dpi=160)
    plt.close()


def main() -> None:
    require_torch()
    import torch

    parser = argparse.ArgumentParser(description="Run structured synthetic benchmarks for Jordan-RoPE.")
    parser.add_argument("--config", default="configs/structured_benchmarks.yaml")
    parser.add_argument("--out-dir", default="runs/structured_benchmarks")
    parser.add_argument("--benchmarks", nargs="*", default=None, choices=sorted(BENCHMARKS))
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    save_config(cfg, out_dir / "config.yaml")
    methods = args.methods or list(cfg["methods"])
    benchmarks = args.benchmarks or list(BENCHMARKS)
    device = choose_device(cfg.get("device", "auto"))
    rows: list[dict] = []
    largest_context = max(int(x) for x in cfg["benchmarks"]["eval_contexts"])

    for seed in cfg["seeds"]:
        set_seed(int(seed))
        train_context = int(cfg["benchmarks"]["train_context"])
        train_points = int(cfg["benchmarks"].get("train_points", train_context))
        all_train = torch.arange(0, train_context, device=device)
        if train_points < train_context:
            train_deltas = all_train[torch.randperm(train_context, device=device)[:train_points]].sort().values
        else:
            train_deltas = all_train

        for benchmark in benchmarks:
            y_train = target_values(benchmark, train_deltas, cfg)
            plot_cache: dict[str, tuple] = {}
            for method in methods:
                x_train = make_features(method, train_deltas, cfg)
                weights = fit_ridge(x_train, y_train, float(cfg["benchmarks"]["ridge"]))
                train_pred = x_train @ weights
                train_mse, train_r2 = metrics(train_pred, y_train)
                rows.append(
                    {
                        "seed": seed,
                        "benchmark": benchmark,
                        "method": method,
                        "context": train_context,
                        "split": "train",
                        "mse": train_mse,
                        "r2": train_r2,
                    }
                )
                for context in cfg["benchmarks"]["eval_contexts"]:
                    context = int(context)
                    eval_deltas = torch.arange(0, context, device=device)
                    y_eval = target_values(benchmark, eval_deltas, cfg)
                    pred = make_features(method, eval_deltas, cfg) @ weights
                    eval_mse, eval_r2 = metrics(pred, y_eval)
                    rows.append(
                        {
                            "seed": seed,
                            "benchmark": benchmark,
                            "method": method,
                            "context": context,
                            "split": "eval",
                            "mse": eval_mse,
                            "r2": eval_r2,
                        }
                    )
                    if seed == cfg["seeds"][0] and context == largest_context:
                        stride = max(1, math.ceil(context / 1200))
                        plot_cache[method] = (
                            eval_deltas[::stride].detach().cpu(),
                            pred[::stride].detach().cpu(),
                            y_eval[::stride].detach().cpu(),
                        )
                        plot_cache["target"] = (
                            eval_deltas[::stride].detach().cpu(),
                            pred[::stride].detach().cpu(),
                            y_eval[::stride].detach().cpu(),
                        )
            if plot_cache:
                plot_predictions(out_dir, benchmark, largest_context, plot_cache)

    write_csv(out_dir / "structured_benchmark_metrics.csv", rows)
    print(f"Wrote {out_dir / 'structured_benchmark_metrics.csv'}")


if __name__ == "__main__":
    main()
