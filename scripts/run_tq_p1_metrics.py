#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, require_torch, write_csv
from scripts.run_quant_metrics import DEFAULT_BUCKETS, bucket_name, parse_buckets, parse_ints


def causal_masks(length: int, device, bucket: tuple[int, int] | None = None):
    import torch

    causal = torch.ones(length, length, device=device, dtype=torch.bool).tril()
    if bucket is None:
        return causal
    positions = torch.arange(length, device=device)
    distances = positions[:, None] - positions[None, :]
    lo, hi = bucket
    return causal & (distances >= int(lo)) & (distances <= int(hi))


def topk_agreement(true_masked, pred_masked, top_k: int):
    import torch

    top_k = max(1, min(int(top_k), int(true_masked.shape[-1])))
    true_top = torch.topk(true_masked, k=top_k, dim=-1).indices
    pred_top = torch.topk(pred_masked, k=top_k, dim=-1).indices
    matches = (true_top.unsqueeze(-1) == pred_top.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean()


def logit_metrics_from_logits(true, pred, mask):
    import torch

    err = (true - pred)[..., mask]
    bias = (pred - true)[..., mask]
    if err.numel() == 0:
        return {"logit_mse": 0.0, "relative_logit_mse": 0.0, "inner_product_bias": 0.0}
    denom = true[..., mask].square().mean().clamp_min(1e-12)
    return {
        "logit_mse": float(err.square().mean().detach().cpu()),
        "relative_logit_mse": float((err.square().mean() / denom).detach().cpu()),
        "inner_product_bias": float(bias.mean().detach().cpu()),
    }


def attention_metrics_from_logits(true, pred, causal, top_k: int):
    import torch
    from torch.nn import functional as F

    true_masked = true.masked_fill(~causal, float("-inf"))
    pred_masked = pred.masked_fill(~causal, float("-inf"))
    log_p = F.log_softmax(true_masked, dim=-1)
    log_q = F.log_softmax(pred_masked, dim=-1)
    p = log_p.exp()
    kl_terms = p * (log_p - log_q)
    kl_terms = kl_terms.masked_fill(~causal, 0.0)
    return {
        "attention_kl": float(kl_terms.sum(dim=-1).mean().detach().cpu()),
        "top1_agreement": float(topk_agreement(true_masked, pred_masked, 1).detach().cpu()),
        f"top{int(top_k)}_agreement": float(topk_agreement(true_masked, pred_masked, int(top_k)).detach().cpu()),
    }


def product_logits(q, encoded, scale: float):
    import torch

    mse_logits = torch.matmul(q, encoded.x_mse.transpose(-2, -1))
    projected_q = torch.matmul(q, encoded.qjl.projection.to(device=q.device, dtype=q.dtype).t())
    signs = encoded.qjl.signs.to(device=q.device, dtype=q.dtype)
    residual = encoded.qjl.residual_norm.to(device=q.device, dtype=q.dtype).squeeze(-1)
    correction = torch.einsum("bhqm,bhkm->bhqk", projected_q, signs)
    correction = correction / float(signs.shape[-1])
    correction = correction * residual[:, :, None, :] * math.sqrt(math.pi / 2.0)
    return (mse_logits + correction) * scale


def reconstruction_logits(q, k_hat, scale: float):
    import torch

    return torch.matmul(q, k_hat.transpose(-2, -1)) * scale


def quantized_logits_for_method(q, k, method: str, bits: int, kac_depth: int, seed: int, qjl_rows: int | None):
    from jordan_rope.quantization import (
        KacRotation,
        rotate_quantize_dequantize,
        turbo_mse_quantize_dequantize,
        turbo_product_quantize,
    )

    dim = int(k.shape[-1])
    scale = 1.0 / math.sqrt(dim)
    rotation = KacRotation.random(dim, int(kac_depth), seed=int(seed), device=k.device, dtype=k.dtype)
    if method == "kac_rot_uniform":
        k_hat, accounting = rotate_quantize_dequantize(k, bits=int(bits), rotation=rotation)
        return reconstruction_logits(q, k_hat, scale), accounting
    if method == "tq_mse_codebook":
        k_hat, accounting = turbo_mse_quantize_dequantize(k, bits=int(bits), rotation=rotation)
        return reconstruction_logits(q, k_hat, scale), accounting
    if method == "tq_prod_qjl":
        encoded = turbo_product_quantize(
            k,
            total_bits=int(bits),
            rotation=rotation,
            qjl_seed=int(seed) + 500009,
            qjl_rows=qjl_rows or dim,
        )
        return product_logits(q, encoded, scale), encoded.bit_accounting()
    if method == "identity":
        accounting = type(
            "IdentityAccounting",
            (),
            {
                "b_numeric": 0.0,
                "b_storage": 0.0,
                "num_scalars": int(k.numel()),
                "metadata_bits": 0,
            },
        )()
        return reconstruction_logits(q, k, scale), accounting
    raise ValueError(f"Unknown TQ P1 method: {method}")


def main() -> None:
    require_torch()
    import torch

    parser = argparse.ArgumentParser(description="Run small P1 attention-logit table for TQ-MSE and TQ-prod/QJL.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage", default="tq_p1_metrics")
    parser.add_argument("--dataset", default="exported")
    parser.add_argument("--methods", default="identity,kac_rot_uniform,tq_mse_codebook,tq_prod_qjl")
    parser.add_argument("--bits", default="3")
    parser.add_argument("--kac-depths", default="16")
    parser.add_argument("--qjl-trials", type=int, default=4)
    parser.add_argument("--qjl-rows", type=int, default=None)
    parser.add_argument("--buckets", default=",".join(f"{lo}:{hi}" for lo, hi in DEFAULT_BUCKETS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    methods = [item for item in args.methods.split(",") if item.strip()]
    bits_values = parse_ints(args.bits)
    kac_depths = parse_ints(args.kac_depths)
    buckets = parse_buckets(args.buckets)
    rows: list[dict] = []

    with torch.no_grad():
        for input_index, input_path in enumerate(args.input):
            payload = torch.load(input_path, map_location="cpu")
            metadata = payload.get("metadata", {})
            checkpoint = metadata.get("checkpoint", "")
            method_name = metadata.get("method", "")
            for record in payload["records"]:
                layer = int(record["layer"])
                q = record["q_pos"].float().to(device)
                k = record["k_pos"].float().to(device)
                dim = int(k.shape[-1])
                true = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dim)
                length = int(k.shape[-2])
                causal = causal_masks(length, device)
                for quant_method in methods:
                    method_bits = [0] if quant_method == "identity" else bits_values
                    method_depths = kac_depths if quant_method != "identity" else [0]
                    trial_count = int(args.qjl_trials) if quant_method == "tq_prod_qjl" else 1
                    for bits in method_bits:
                        for kac_depth in method_depths:
                            for trial in range(trial_count):
                                seed = int(args.seed) + input_index * 1000003 + layer * 1009 + int(kac_depth) * 9173 + trial * 104729
                                start = time.perf_counter()
                                pred, accounting = quantized_logits_for_method(
                                    q,
                                    k,
                                    quant_method,
                                    max(int(bits), 2),
                                    int(kac_depth),
                                    seed,
                                    args.qjl_rows,
                                )
                                wall_ms = (time.perf_counter() - start) * 1000.0
                                base = {
                                    "stage": args.stage,
                                    "method": method_name,
                                    "checkpoint": checkpoint,
                                    "dataset": args.dataset,
                                    "seed": metadata.get("seed", ""),
                                    "layer": layer,
                                    "head": "all",
                                    "cache_protocol": metadata.get("cache_protocol", "P1a_positioned_cache"),
                                    "split": "evaluation",
                                    "quant_method": quant_method,
                                    "rotation": "kac" if quant_method != "identity" else "none",
                                    "kac_depth": int(kac_depth),
                                    "qjl_trial": int(trial),
                                    "b_numeric": float(accounting.b_numeric),
                                    "b_storage": float(accounting.b_storage),
                                }
                                for metric, value in logit_metrics_from_logits(true, pred, causal).items():
                                    rows.append({**base, "distance_bucket": "all", "aggregation": "mean", "metric": metric, "value": value})
                                for metric, value in attention_metrics_from_logits(true, pred, causal, int(args.top_k)).items():
                                    rows.append({**base, "distance_bucket": "all", "aggregation": "mean", "metric": metric, "value": value})
                                rows.append({**base, "distance_bucket": "all", "aggregation": "mean", "metric": "quant_wall_ms", "value": wall_ms})
                                for bucket in buckets:
                                    mask = causal_masks(length, device, bucket=bucket)
                                    for metric, value in logit_metrics_from_logits(true, pred, mask).items():
                                        rows.append(
                                            {
                                                **base,
                                                "distance_bucket": bucket_name(bucket),
                                                "aggregation": "mean",
                                                "metric": metric,
                                                "value": value,
                                            }
                                        )

    write_csv(out_dir / "metrics.csv", rows)
    print(f"Wrote {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
