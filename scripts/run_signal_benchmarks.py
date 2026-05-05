#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.model import RMSNorm, TransformerBlock, TransformerConfig
from jordan_rope.utils import append_jsonl, choose_device, ensure_dir, load_config, require_torch, save_config, set_seed, write_csv


def signal_frequencies(cfg: dict) -> tuple[float, float]:
    sig = cfg["signal"]
    theta = float(cfg["position"]["theta"])
    d_head = int(sig["model"]["d_model"]) // int(sig["model"]["n_heads"])
    idx = int(sig["target_freq_index"])
    idx2 = int(sig["second_freq_index"])
    return theta ** (-2.0 * idx / d_head), theta ** (-2.0 * idx2 / d_head)


def raw_kernel_values(name: str, deltas, cfg: dict):
    import torch

    sig = cfg["signal"]
    train_context = float(sig["seq_len_train"])
    gamma = float(sig.get("gamma", cfg["position"].get("gamma_min", 1e-4)))
    omega, omega2 = signal_frequencies(cfg)
    d = deltas.to(torch.float32)
    x = d / train_context
    omega = torch.tensor(omega, device=d.device, dtype=d.dtype)
    omega2 = torch.tensor(omega2, device=d.device, dtype=d.dtype)

    if name == "seasonal_trend":
        return (0.35 + 0.9 * x) * torch.cos(omega * d) + 0.25 * torch.sin(omega * d)
    if name == "damped_wave":
        decay = torch.exp(-gamma * d)
        return decay * (torch.cos(omega * d) + 0.75 * x * torch.sin(omega * d))
    if name == "rhythm_envelope":
        carrier = 0.65 * torch.cos(omega * d) + 0.35 * torch.sin(omega2 * d)
        return (0.25 + 0.85 * x) * carrier
    if name == "motif_spacing":
        motif = 0.7 * torch.cos(omega2 * d) - 0.3 * torch.sin(omega2 * d)
        return (x / (1.0 + 0.25 * x)) * motif
    raise ValueError(f"Unknown signal benchmark: {name}")


def kernel_vector(name: str, length: int, cfg: dict, device):
    import torch

    deltas = torch.arange(length, device=device)
    kernel = raw_kernel_values(name, deltas, cfg)
    kernel = kernel.clone()
    kernel[0] = 0.0
    train_len = int(cfg["signal"]["seq_len_train"])
    train_d = torch.arange(train_len, device=device)
    train_kernel = raw_kernel_values(name, train_d, cfg)
    train_kernel[0] = 0.0
    norm = torch.linalg.vector_norm(train_kernel).clamp_min(1e-6)
    return kernel / norm


def kernel_matrix(name: str, length: int, cfg: dict, device):
    import torch

    d = torch.arange(length, device=device)
    delta = d[:, None] - d[None, :]
    kernel = kernel_vector(name, length, cfg, device)
    mat = torch.zeros(length, length, device=device, dtype=torch.float32)
    mask = delta > 0
    mat[mask] = kernel[delta[mask]]
    return mat


def build_signal_model(cfg: dict, method: str):
    import torch
    from torch import nn

    sig = cfg["signal"]
    model_cfg = sig["model"]
    pos = cfg["position"]
    tf_cfg = TransformerConfig(
        vocab_size=1,
        d_model=int(model_cfg["d_model"]),
        n_heads=int(model_cfg["n_heads"]),
        n_layers=int(model_cfg["n_layers"]),
        mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        position_method=method,
        theta=float(pos["theta"]),
        gamma_min=float(pos["gamma_min"]),
        init_gamma=float(pos["init_gamma"]),
        eta_max=float(pos["eta_max"]),
        init_eta=float(pos.get("init_eta", 0.0)),
        train_context=int(pos["train_context"]),
        bounded_tau=bool(pos.get("bounded_tau", True)),
        max_exponent=float(pos.get("max_exponent", 30.0)),
    )

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(int(sig["input_dim"]), tf_cfg.d_model, bias=False)
            self.blocks = nn.ModuleList([TransformerBlock(tf_cfg) for _ in range(tf_cfg.n_layers)])
            self.norm = RMSNorm(tf_cfg.d_model)
            self.output = nn.Linear(tf_cfg.d_model, int(sig["output_dim"]), bias=False)

        def forward(self, x):
            positions = torch.arange(x.shape[1], device=x.device)
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h, positions)
            return self.output(self.norm(h))

    return Model()


def generate_batch(benchmark: str, length: int, batch_size: int, cfg: dict, device, kmat_cache: dict):
    import torch

    input_dim = int(cfg["signal"]["input_dim"])
    x = torch.randn(batch_size, length, input_dim, device=device)
    key = (benchmark, length, str(device))
    if key not in kmat_cache:
        kmat_cache[key] = kernel_matrix(benchmark, length, cfg, device)
    y = torch.einsum("ij,bjc->bic", kmat_cache[key], x)
    return x, y


def regression_metrics(pred, target) -> tuple[float, float]:
    import torch

    mse = torch.mean((pred - target).square())
    denom = torch.mean((target - target.mean()).square()).clamp_min(1e-12)
    r2 = 1.0 - mse / denom
    return float(mse.detach().cpu()), float(r2.detach().cpu())


def evaluate(model, benchmark: str, cfg: dict, device, method: str, seed: int, step: int, kmat_cache: dict) -> list[dict]:
    import torch

    sig = cfg["signal"]
    rows = []
    model.eval()
    with torch.no_grad():
        for length in sig["eval_lengths"]:
            mses = []
            r2s = []
            for _ in range(int(sig["eval_batches"])):
                x, y = generate_batch(
                    benchmark,
                    int(length),
                    int(sig["eval_batch_size"]),
                    cfg,
                    device,
                    kmat_cache,
                )
                pred = model(x)
                mse, r2 = regression_metrics(pred, y)
                mses.append(mse)
                r2s.append(r2)
            rows.append(
                {
                    "seed": seed,
                    "benchmark": benchmark,
                    "method": method,
                    "step": step,
                    "context": int(length),
                    "mse": sum(mses) / len(mses),
                    "r2": sum(r2s) / len(r2s),
                }
            )
    model.train()
    return rows


def main() -> None:
    require_torch()
    import torch
    from torch.nn import functional as F

    parser = argparse.ArgumentParser(description="Train causal Transformer signal-kernel benchmarks.")
    parser.add_argument("--config", default="configs/signal_benchmarks.yaml")
    parser.add_argument("--out-dir", default="runs/signal_benchmarks")
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    save_config(cfg, out_dir / "config.yaml")
    device = choose_device(cfg.get("device", "auto"))
    methods = args.methods or list(cfg["methods"])
    benchmarks = args.benchmarks or list(cfg["signal"]["benchmarks"])
    sig = cfg["signal"]
    rows: list[dict] = []

    for benchmark in benchmarks:
        for seed in cfg["seeds"]:
            seed = int(seed)
            for method in methods:
                set_seed(seed)
                model = build_signal_model(cfg, method).to(device)
                opt = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(sig["lr"]),
                    weight_decay=float(sig.get("weight_decay", 0.0)),
                )
                kmat_cache: dict = {}
                log_path = out_dir / f"{benchmark}_{method}_seed{seed}.jsonl"
                for step in range(1, int(sig["train_steps"]) + 1):
                    x, y = generate_batch(
                        benchmark,
                        int(sig["seq_len_train"]),
                        int(sig["batch_size"]),
                        cfg,
                        device,
                        kmat_cache,
                    )
                    pred = model(x)
                    loss = F.mse_loss(pred, y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                    if step % int(sig["log_every"]) == 0:
                        mse, r2 = regression_metrics(pred, y)
                        append_jsonl(
                            log_path,
                            {
                                "seed": seed,
                                "benchmark": benchmark,
                                "method": method,
                                "step": step,
                                "mse": mse,
                                "r2": r2,
                            },
                        )

                    if step % int(sig["eval_every"]) == 0 or step == int(sig["train_steps"]):
                        eval_rows = evaluate(model, benchmark, cfg, device, method, seed, step, kmat_cache)
                        rows.extend(eval_rows)
                        write_csv(out_dir / "signal_eval.csv", rows)

    write_csv(out_dir / "signal_eval.csv", rows)
    print(f"Wrote {out_dir / 'signal_eval.csv'}")


if __name__ == "__main__":
    main()
