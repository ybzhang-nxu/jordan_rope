#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, load_config, require_torch, write_csv
from scripts.evaluate_quantized_kernel_lm import KQuantizer, parse_ints, quant_method_bits
from scripts.run_lm import load_tokens, make_eval_starts


def load_checkpoint(path: str | Path, device):
    import torch
    from jordan_rope.model import CausalTransformerLM, TransformerConfig

    ckpt = torch.load(path, map_location=device)
    config = TransformerConfig(**ckpt["config"])
    model = CausalTransformerLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def quantized_logits(model, tokens, quantizer: KQuantizer, cache_target: str):
    if quantizer.method == "identity":
        return model(tokens)
    if cache_target == "k":
        return model.forward_with_cache_quantizer(
            tokens,
            quantize_k=lambda layer_id, tensor: quantizer(layer_id, tensor, role="k"),
        )
    if cache_target == "v":
        return model.forward_with_cache_quantizer(
            tokens,
            quantize_v=lambda layer_id, tensor: quantizer(layer_id, tensor, role="v"),
        )
    if cache_target == "kv":
        return model.forward_with_cache_quantizer(
            tokens,
            quantize_k=lambda layer_id, tensor: quantizer(layer_id, tensor, role="k"),
            quantize_v=lambda layer_id, tensor: quantizer(layer_id, tensor, role="v"),
        )
    raise ValueError(f"Unknown cache target: {cache_target}")


def evaluate_lm_checkpoint(model, eval_tokens, cfg: dict, device, quantizer: KQuantizer, cache_target: str) -> list[dict]:
    import torch
    from torch.nn import functional as F
    from jordan_rope.data import sample_lm_batch_at_starts

    rows = []
    starts_by_length = make_eval_starts(eval_tokens, cfg)
    with torch.no_grad():
        for length in cfg["lm"]["eval_lengths"]:
            length = int(length)
            losses = []
            for starts in starts_by_length[length]:
                x, y = sample_lm_batch_at_starts(eval_tokens, starts, length, device)
                logits = quantized_logits(model, x, quantizer, cache_target)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                losses.append(float(loss.detach().cpu()))
            mean_loss = sum(losses) / max(len(losses), 1)
            rows.append(
                {
                    "context": length,
                    "loss": mean_loss,
                    "ppl": math.exp(min(mean_loss, 20.0)),
                    "b_numeric": quantizer.b_numeric,
                    "b_storage": quantizer.b_storage,
                }
            )
    return rows


def method_from_checkpoint(path: Path) -> str:
    name = path.name
    for suffix in ("_seed0.pt", "_seed1.pt", "_seed2.pt", ".pt"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def main() -> None:
    require_torch()
    import torch

    parser = argparse.ArgumentParser(description="Evaluate byte-LM checkpoints with cache quantization.")
    parser.add_argument("--config", default="configs/phase2_wikitext_boundary.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/phase2/stage5_wikitext_boundary/quant_eval")
    parser.add_argument("--stage", default="stage5_wikitext_boundary")
    parser.add_argument("--quant-methods", default="identity,kac_rot_uniform")
    parser.add_argument("--cache-target", choices=["k", "v", "kv"], default="k")
    parser.add_argument("--bits", default="3")
    parser.add_argument("--kac-depths", default="16")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = load_config(args.config)
    if args.eval_batches is not None:
        cfg["lm"]["eval_batches"] = int(args.eval_batches)
    device = choose_device(args.device)
    eval_tokens = load_tokens(cfg, "eval")
    out_dir = ensure_dir(args.out_dir)
    quant_methods = [item for item in args.quant_methods.split(",") if item.strip()]
    bits_values = parse_ints(args.bits)
    kac_depths = parse_ints(args.kac_depths)
    rows = []

    for checkpoint_text in args.checkpoints:
        checkpoint = Path(checkpoint_text)
        model, model_config = load_checkpoint(checkpoint, device)
        method = model_config.position_method or method_from_checkpoint(checkpoint)
        identity_by_context = {}
        for quant_method in quant_methods:
            method_bits = quant_method_bits(quant_method, bits_values)
            method_depths = kac_depths if quant_method == "kac_rot_uniform" or quant_method.startswith("kac_mixed_") else [0]
            for bits in method_bits:
                for kac_depth in method_depths:
                    quantizer = KQuantizer(quant_method, bits=max(bits, 2), kac_depth=kac_depth, seed=int(args.seed))
                    eval_rows = evaluate_lm_checkpoint(model, eval_tokens, cfg, device, quantizer, args.cache_target)
                    for row in eval_rows:
                        if quant_method == "identity":
                            identity_by_context[int(row["context"])] = row
                        base = identity_by_context.get(int(row["context"]))
                        delta_loss = "" if base is None else row["loss"] - base["loss"]
                        rows.append(
                            {
                                "stage": args.stage,
                                "method": method,
                                "checkpoint": str(checkpoint),
                                "dataset": cfg["lm"]["dataset"].get("subset") or cfg["lm"]["dataset"].get("text_file", ""),
                                "seed": args.seed,
                                "cache_target": args.cache_target,
                                "split": "evaluation",
                                "quant_method": quant_method,
                                "rotation": "kac" if quant_method == "kac_rot_uniform" or quant_method.startswith("kac_mixed_") else ("dense" if quant_method == "dense_rot_uniform" else "none"),
                                "kac_depth": kac_depth,
                                "b_numeric": row["b_numeric"],
                                "b_storage": row["b_storage"],
                                "context": row["context"],
                                "loss": row["loss"],
                                "ppl": row["ppl"],
                                "delta_loss_same": delta_loss,
                            }
                        )
                    write_csv(out_dir / "lm_quant_eval.csv", rows)

    write_csv(out_dir / "lm_quant_eval.csv", rows)
    print(f"Wrote {out_dir / 'lm_quant_eval.csv'}")


if __name__ == "__main__":
    main()
