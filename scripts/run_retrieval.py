#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import append_jsonl, choose_device, ensure_dir, load_config, require_torch, save_config, set_seed, write_csv


def build_config(cfg: dict, method: str, vocab_size: int) -> TransformerConfig:
    from jordan_rope.model import TransformerConfig

    model = cfg["retrieval"]["model"]
    pos = cfg["position"]
    return TransformerConfig(
        vocab_size=vocab_size,
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


def evaluate(model, cfg: dict, device, method: str, seed: int) -> list[dict]:
    import torch
    from jordan_rope.data import generate_retrieval_batch

    ret = cfg["retrieval"]
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for length in ret["eval_lengths"]:
            length = int(length)
            for distance in ret["distance_points"]:
                distance = int(distance)
                if distance >= length - 2:
                    continue
                correct = 0
                total = 0
                for _ in range(int(ret["eval_batches"])):
                    x, y, d = generate_retrieval_batch(
                        batch_size=int(ret["eval_batch_size"]),
                        seq_len=length,
                        num_pairs=int(ret["num_pairs"]),
                        num_keys=int(ret["num_keys"]),
                        num_values=int(ret["num_values"]),
                        device=device,
                        target_distance=distance,
                    )
                    logits = model(x)[:, -1, :]
                    pred = logits.argmax(dim=-1)
                    correct += int((pred == y).sum().item())
                    total += int(y.numel())
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "context": length,
                        "distance": distance,
                        "accuracy": correct / max(total, 1),
                    }
                )
    model.train()
    return rows


def main() -> None:
    require_torch()
    import torch
    from torch.nn import functional as F
    from jordan_rope.data import generate_retrieval_batch, retrieval_vocab_size
    from jordan_rope.model import CausalTransformerLM

    parser = argparse.ArgumentParser(description="Train/evaluate needle-style retrieval.")
    parser.add_argument("--config", default="configs/full_research.yaml")
    parser.add_argument("--out-dir", default="runs/retrieval")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    save_config(cfg, out_dir / "config.yaml")
    device = choose_device(cfg.get("device", "auto"))
    methods = args.methods or (list(cfg["methods"]) + list(cfg.get("ablation_methods", [])))
    ret = cfg["retrieval"]
    vocab_size = retrieval_vocab_size(int(ret["num_keys"]), int(ret["num_values"]))
    all_eval_rows: list[dict] = []

    for seed in cfg["seeds"]:
        seed = int(seed)
        for method in methods:
            set_seed(seed)
            model = CausalTransformerLM(build_config(cfg, method, vocab_size)).to(device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(ret["lr"]),
                weight_decay=float(ret.get("weight_decay", 0.0)),
            )
            log_path = out_dir / f"{method}_seed{seed}.jsonl"
            for step in range(1, int(ret["train_steps"]) + 1):
                x, y, distances = generate_retrieval_batch(
                    batch_size=int(ret["batch_size"]),
                    seq_len=int(ret["seq_len_train"]),
                    num_pairs=int(ret["num_pairs"]),
                    num_keys=int(ret["num_keys"]),
                    num_values=int(ret["num_values"]),
                    device=device,
                )
                logits = model(x)[:, -1, :]
                loss = F.cross_entropy(logits, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if step % int(ret["log_every"]) == 0:
                    acc = float((logits.argmax(dim=-1) == y).float().mean().item())
                    append_jsonl(
                        log_path,
                        {
                            "seed": seed,
                            "method": method,
                            "step": step,
                            "loss": float(loss.item()),
                            "accuracy": acc,
                            "mean_distance": float(distances.float().mean().item()),
                        },
                    )

                if step % int(ret["eval_every"]) == 0 or step == int(ret["train_steps"]):
                    eval_rows = evaluate(model, cfg, device, method, seed)
                    all_eval_rows.extend({**row, "step": step} for row in eval_rows)
                    write_csv(out_dir / "retrieval_eval.csv", all_eval_rows)

            if args.save_checkpoints:
                ckpt_dir = ensure_dir(out_dir / "checkpoints")
                torch.save({"model": model.state_dict(), "config": build_config(cfg, method, vocab_size).__dict__}, ckpt_dir / f"{method}_seed{seed}.pt")

    write_csv(out_dir / "retrieval_eval.csv", all_eval_rows)
    print(f"Wrote {out_dir / 'retrieval_eval.csv'}")


if __name__ == "__main__":
    main()
