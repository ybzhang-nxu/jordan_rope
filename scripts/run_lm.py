#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import append_jsonl, choose_device, ensure_dir, load_config, require_torch, save_config, set_seed, write_csv


BYTE_VOCAB_SIZE = 257


def load_tokens(cfg: dict, split: str):
    from jordan_rope.data import load_hf_text_dataset, load_text_as_bytes

    data_cfg = cfg["lm"]["dataset"]
    if data_cfg.get("text_file"):
        return load_text_as_bytes(data_cfg["text_file"])
    split_name = data_cfg["train_split"] if split == "train" else data_cfg["eval_split"]
    return load_hf_text_dataset(data_cfg["name"], data_cfg["subset"], split_name)


def build_config(cfg: dict, method: str) -> TransformerConfig:
    from jordan_rope.model import TransformerConfig

    model = cfg["lm"]["model"]
    pos = cfg["position"]
    return TransformerConfig(
        vocab_size=BYTE_VOCAB_SIZE,
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


def evaluate_lm(model, tokens, cfg: dict, device, method: str, seed: int, step: int) -> list[dict]:
    import torch
    from torch.nn import functional as F
    from jordan_rope.data import sample_lm_batch_at_starts

    rows: list[dict] = []
    lm = cfg["lm"]
    eval_starts = make_eval_starts(tokens, cfg)
    stats = position_stats(model)
    model.eval()
    with torch.no_grad():
        base_loss = None
        for length in lm["eval_lengths"]:
            length = int(length)
            losses: list[float] = []
            for starts in eval_starts[length]:
                x, y = sample_lm_batch_at_starts(tokens, starts, length, device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(float(loss.item()))
            mean_loss = sum(losses) / max(len(losses), 1)
            if length == int(lm["seq_len_train"]):
                base_loss = mean_loss
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "step": step,
                    "context": length,
                    "loss": mean_loss,
                    "ppl": math.exp(min(mean_loss, 20.0)),
                    "delta_loss": 0.0 if base_loss is None else mean_loss - base_loss,
                    **stats,
                }
            )
    model.train()
    return rows


def make_train_starts(tokens, cfg: dict, seed: int):
    import torch

    lm = cfg["lm"]
    seq_len = int(lm["seq_len_train"])
    max_start = tokens.numel() - seq_len - 1
    if max_start <= 0:
        raise ValueError("Not enough train tokens for the requested seq_len.")
    generator = torch.Generator()
    generator.manual_seed(int(lm.get("train_sample_seed", 17)) + seed * 1000003)
    return torch.randint(
        0,
        max_start,
        (int(lm["train_steps"]), int(lm["batch_size"])),
        generator=generator,
    )


def make_eval_starts(tokens, cfg: dict):
    import torch

    lm = cfg["lm"]
    generator = torch.Generator()
    generator.manual_seed(int(lm.get("eval_sample_seed", 2027)))
    starts = {}
    for length in lm["eval_lengths"]:
        length = int(length)
        max_start = tokens.numel() - length - 1
        if max_start <= 0:
            raise ValueError("Not enough eval tokens for the requested eval length.")
        starts[length] = [
            torch.randint(0, max_start, (int(lm["eval_batch_size"]),), generator=generator)
            for _ in range(int(lm["eval_batches"]))
        ]
    return starts


def position_stats(model) -> dict[str, float | str]:
    import torch
    from jordan_rope.positional import JordanRoPE

    gammas = []
    etas = []
    for module in model.modules():
        if isinstance(module, JordanRoPE):
            gammas.append(module.gamma().detach().float().cpu().reshape(-1))
            etas.append(module.eta().detach().float().cpu().reshape(-1))
    if not gammas:
        return {
            "gamma_mean": "",
            "eta_mean": "",
            "eta_abs_mean": "",
            "eta_abs_max": "",
        }
    gamma = torch.cat(gammas)
    eta = torch.cat(etas)
    return {
        "gamma_mean": float(gamma.mean().item()),
        "eta_mean": float(eta.mean().item()),
        "eta_abs_mean": float(eta.abs().mean().item()),
        "eta_abs_max": float(eta.abs().max().item()),
    }


def main() -> None:
    require_torch()
    import torch
    from torch.nn import functional as F
    from jordan_rope.data import sample_lm_batch_at_starts
    from jordan_rope.model import CausalTransformerLM

    parser = argparse.ArgumentParser(description="Train/evaluate byte-level small causal LM.")
    parser.add_argument("--config", default="configs/full_research.yaml")
    parser.add_argument("--out-dir", default="runs/lm")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    save_config(cfg, out_dir / "config.yaml")
    device = choose_device(cfg.get("device", "auto"))
    methods = args.methods or (list(cfg["methods"]) + list(cfg.get("ablation_methods", [])))
    lm = cfg["lm"]
    train_tokens = load_tokens(cfg, "train")
    eval_tokens = load_tokens(cfg, "eval")
    all_eval_rows: list[dict] = []
    torch.set_float32_matmul_precision("high")

    for seed in cfg["seeds"]:
        seed = int(seed)
        train_starts = make_train_starts(train_tokens, cfg, seed)
        for method in methods:
            set_seed(seed)
            model = CausalTransformerLM(build_config(cfg, method)).to(device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(lm["lr"]),
                weight_decay=float(lm.get("weight_decay", 0.0)),
            )
            log_path = out_dir / f"{method}_seed{seed}.jsonl"
            for step in range(1, int(lm["train_steps"]) + 1):
                x, y = sample_lm_batch_at_starts(
                    train_tokens,
                    train_starts[step - 1],
                    int(lm["seq_len_train"]),
                    device,
                )
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % int(lm["log_every"]) == 0:
                    append_jsonl(
                        log_path,
                        {
                            "seed": seed,
                            "method": method,
                            "step": step,
                            "loss": float(loss.item()),
                            "ppl": math.exp(min(float(loss.item()), 20.0)),
                            **position_stats(model),
                        },
                    )

                if step % int(lm["eval_every"]) == 0 or step == int(lm["train_steps"]):
                    eval_rows = evaluate_lm(model, eval_tokens, cfg, device, method, seed, step)
                    all_eval_rows.extend(eval_rows)
                    write_csv(out_dir / "lm_eval.csv", all_eval_rows)

            if args.save_checkpoints:
                ckpt_dir = ensure_dir(out_dir / "checkpoints")
                torch.save({"model": model.state_dict(), "config": build_config(cfg, method).__dict__}, ckpt_dir / f"{method}_seed{seed}.pt")

    write_csv(out_dir / "lm_eval.csv", all_eval_rows)
    print(f"Wrote {out_dir / 'lm_eval.csv'}")


if __name__ == "__main__":
    main()
