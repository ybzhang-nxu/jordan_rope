#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.model import CausalTransformerLM, TransformerConfig
from jordan_rope.utils import append_jsonl, choose_device, ensure_dir, load_config, require_torch, save_config, set_seed, write_csv


BIT0 = 0
BIT1 = 1
QUERY = 2
VOCAB_SIZE = 3


def build_config(cfg: dict, method: str) -> TransformerConfig:
    model = cfg["kernel_lm"]["model"]
    pos = cfg["position"]
    return TransformerConfig(
        vocab_size=VOCAB_SIZE,
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        mlp_ratio=int(model.get("mlp_ratio", 4)),
        dropout=float(model.get("dropout", 0.0)),
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


def target_frequencies(cfg: dict) -> tuple[float, float]:
    klm = cfg["kernel_lm"]
    theta = float(cfg["position"]["theta"])
    d_head = int(klm["model"]["d_model"]) // int(klm["model"]["n_heads"])
    idx = int(klm["target_freq_index"])
    idx2 = int(klm["second_freq_index"])
    return theta ** (-2.0 * idx / d_head), theta ** (-2.0 * idx2 / d_head)


def kernel_values(length: int, cfg: dict, device):
    import torch

    klm = cfg["kernel_lm"]
    benchmark = klm["benchmark"]
    train_context = float(klm["seq_len_train"])
    gamma = float(klm.get("gamma", cfg["position"].get("gamma_min", 1e-4)))
    omega, omega2 = target_frequencies(cfg)
    d = torch.arange(1, length, device=device, dtype=torch.float32)
    x = d / train_context
    omega = torch.tensor(omega, device=device)
    omega2 = torch.tensor(omega2, device=device)

    if benchmark == "mixed_raw":
        k = x * torch.cos(omega * d)
    elif benchmark == "mixed_bounded":
        tau = d / (1.0 + d / train_context)
        k = (tau / train_context) * torch.cos(omega * d)
    elif benchmark == "damped_mixed":
        decay = torch.exp(-gamma * d)
        k = decay * (torch.cos(omega * d) + 0.75 * x * torch.sin(omega * d))
    elif benchmark == "rhythm_envelope":
        carrier = 0.65 * torch.cos(omega * d) + 0.35 * torch.sin(omega2 * d)
        k = (0.25 + 0.85 * x) * carrier
    elif benchmark.startswith("high_jet_r"):
        order = int(benchmark.removeprefix("high_jet_r"))
        k = x.pow(order) * torch.cos(omega * d)
    elif benchmark.startswith("scaled_high_jet_r"):
        order = int(benchmark.removeprefix("scaled_high_jet_r"))
        c_value = float(klm.get("scaled_c", 0.1))
        k = x.pow(order) * torch.exp(-c_value * x) * torch.cos(omega * d)
    else:
        raise ValueError(f"Unknown kernel_lm benchmark: {benchmark}")

    train_d = torch.arange(1, int(klm["seq_len_train"]), device=device, dtype=torch.float32)
    train_x = train_d / train_context
    if benchmark == "mixed_raw":
        train_k = train_x * torch.cos(omega * train_d)
    elif benchmark == "mixed_bounded":
        train_tau = train_d / (1.0 + train_d / train_context)
        train_k = (train_tau / train_context) * torch.cos(omega * train_d)
    elif benchmark == "damped_mixed":
        train_decay = torch.exp(-gamma * train_d)
        train_k = train_decay * (torch.cos(omega * train_d) + 0.75 * train_x * torch.sin(omega * train_d))
    elif benchmark == "rhythm_envelope":
        train_carrier = 0.65 * torch.cos(omega * train_d) + 0.35 * torch.sin(omega2 * train_d)
        train_k = (0.25 + 0.85 * train_x) * train_carrier
    elif benchmark.startswith("high_jet_r"):
        order = int(benchmark.removeprefix("high_jet_r"))
        train_k = train_x.pow(order) * torch.cos(omega * train_d)
    elif benchmark.startswith("scaled_high_jet_r"):
        order = int(benchmark.removeprefix("scaled_high_jet_r"))
        c_value = float(klm.get("scaled_c", 0.1))
        train_k = train_x.pow(order) * torch.exp(-c_value * train_x) * torch.cos(omega * train_d)
    else:
        raise ValueError(f"Unknown kernel_lm benchmark: {benchmark}")
    norm = torch.linalg.vector_norm(train_k).clamp_min(1e-6)
    return k / norm


def generate_batch(length: int, batch_size: int, cfg: dict, device, generator):
    import torch

    bits = torch.randint(0, 2, (batch_size, length - 1), generator=generator, device=device)
    signs = bits.float() * 2.0 - 1.0
    kernel = kernel_values(length, cfg, device)
    # Query is at the final input position, so deltas run length-1, ..., 1.
    score = torch.sum(signs * kernel.flip(0)[None, :], dim=1)
    if cfg["kernel_lm"].get("label_mode", "threshold") == "stochastic":
        prob = torch.sigmoid(float(cfg["kernel_lm"].get("logit_scale", 3.0)) * score)
        labels = torch.bernoulli(prob, generator=generator).long()
    else:
        labels = (score > 0).long()
    tokens = torch.empty(batch_size, length, dtype=torch.long, device=device)
    tokens[:, :-1] = bits
    tokens[:, -1] = QUERY
    return tokens, labels, score.detach()


def batch_metrics(logits, labels) -> tuple[float, float]:
    import torch
    from torch.nn import functional as F

    loss = F.cross_entropy(logits, labels)
    acc = (logits.argmax(dim=-1) == labels).float().mean()
    return float(loss.detach().cpu()), float(acc.detach().cpu())


def evaluate(model, cfg: dict, device, method: str, seed: int, step: int) -> list[dict]:
    import torch
    from torch.nn import functional as F

    klm = cfg["kernel_lm"]
    rows = []
    model.eval()
    with torch.no_grad():
        for length in klm["eval_lengths"]:
            losses = []
            accs = []
            score_abs = []
            gen = torch.Generator(device=device)
            gen.manual_seed(int(klm.get("eval_seed", 9109)) + seed * 1009 + int(length))
            for _ in range(int(klm["eval_batches"])):
                tokens, labels, score = generate_batch(
                    int(length), int(klm["eval_batch_size"]), cfg, device, gen
                )
                logits = model(tokens)[:, -1, :2]
                loss = F.cross_entropy(logits, labels)
                losses.append(float(loss.detach().cpu()))
                accs.append(float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu()))
                score_abs.append(float(score.abs().mean().detach().cpu()))
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "benchmark": klm["benchmark"],
                    "step": step,
                    "context": int(length),
                    "loss": sum(losses) / len(losses),
                    "accuracy": sum(accs) / len(accs),
                    "score_abs_mean": sum(score_abs) / len(score_abs),
                }
            )
    model.train()
    return rows


def main() -> None:
    require_torch()
    import torch
    from torch.nn import functional as F

    parser = argparse.ArgumentParser(description="Train query-style synthetic LM with Jordan-friendly kernels.")
    parser.add_argument("--config", default="configs/kernel_lm.yaml")
    parser.add_argument("--out-dir", default="runs/kernel_lm")
    parser.add_argument("--benchmark", default=None, help="Override cfg['kernel_lm']['benchmark'].")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = load_config(args.config)
    if args.benchmark is not None:
        cfg["kernel_lm"]["benchmark"] = args.benchmark
    if args.seeds is not None:
        cfg["seeds"] = args.seeds
    out_dir = ensure_dir(args.out_dir)
    save_config(cfg, out_dir / "config.yaml")
    device = choose_device(cfg.get("device", "auto"))
    methods = args.methods or list(cfg["methods"])
    klm = cfg["kernel_lm"]
    rows: list[dict] = []

    for seed in cfg["seeds"]:
        seed = int(seed)
        for method in methods:
            set_seed(seed)
            model = CausalTransformerLM(build_config(cfg, method)).to(device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(klm["lr"]),
                weight_decay=float(klm.get("weight_decay", 0.0)),
            )
            gen = torch.Generator(device=device)
            gen.manual_seed(int(klm.get("data_seed", 7107)) + seed * 1000003)
            log_path = out_dir / f"{method}_seed{seed}.jsonl"

            for step in range(1, int(klm["train_steps"]) + 1):
                tokens, labels, score = generate_batch(
                    int(klm["seq_len_train"]),
                    int(klm["batch_size"]),
                    cfg,
                    device,
                    gen,
                )
                logits = model(tokens)[:, -1, :2]
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % int(klm["log_every"]) == 0:
                    loss_value, acc = batch_metrics(logits.detach(), labels)
                    append_jsonl(
                        log_path,
                        {
                            "seed": seed,
                            "method": method,
                            "benchmark": klm["benchmark"],
                            "step": step,
                            "loss": loss_value,
                            "accuracy": acc,
                            "score_abs_mean": float(score.abs().mean().detach().cpu()),
                        },
                    )

                if step % int(klm["eval_every"]) == 0 or step == int(klm["train_steps"]):
                    eval_rows = evaluate(model, cfg, device, method, seed, step)
                    rows.extend(eval_rows)
                    write_csv(out_dir / "kernel_lm_eval.csv", rows)

            if args.save_checkpoints:
                ckpt_dir = ensure_dir(out_dir / "checkpoints")
                torch.save(
                    {"model": model.state_dict(), "config": build_config(cfg, method).__dict__},
                    ckpt_dir / f"{method}_seed{seed}.pt",
                )

    write_csv(out_dir / "kernel_lm_eval.csv", rows)
    print(f"Wrote {out_dir / 'kernel_lm_eval.csv'}")


if __name__ == "__main__":
    main()
