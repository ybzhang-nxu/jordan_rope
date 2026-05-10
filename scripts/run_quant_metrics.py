#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir, require_torch, write_csv


DEFAULT_BUCKETS = [(0, 128), (129, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192)]
MIXED_METHODS = {
    "kac_mixed_random_uniform",
    "kac_mixed_magnitude_uniform",
    "kac_mixed_sensitivity_uniform",
    "kac_mixed_head_random_uniform",
    "kac_mixed_head_magnitude_uniform",
    "kac_mixed_head_sensitivity_uniform",
}


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def parse_buckets(text: str) -> list[tuple[int, int]]:
    buckets = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split(":")
        buckets.append((int(lo), int(hi)))
    return buckets


def bucket_name(bucket: tuple[int, int]) -> str:
    return f"{bucket[0]}_{bucket[1]}"


def is_mixed_method(method: str) -> bool:
    return method in MIXED_METHODS


def mixed_strategy(method: str) -> str:
    if "random" in method:
        return "random"
    if "magnitude" in method:
        return "magnitude"
    if "sensitivity" in method:
        return "sensitivity"
    raise ValueError(f"Unknown mixed-bit strategy in method={method}")


def mixed_mask_scope(method: str) -> str:
    return "head" if "_head_" in method else "layer"


def random_high_mask(dim: int, high_fraction: float, seed: int, *, num_heads: int | None = None):
    import torch

    high_count = max(1, min(dim, int(round(dim * float(high_fraction)))))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    if num_heads is not None:
        mask = torch.zeros(int(num_heads), dim, dtype=torch.bool)
        for head in range(int(num_heads)):
            gen.manual_seed(int(seed) + head * 104729)
            indices = torch.randperm(dim, generator=gen)[:high_count]
            mask[head, indices] = True
        return mask
    indices = torch.randperm(dim, generator=gen)[:high_count]
    mask = torch.zeros(dim, dtype=torch.bool)
    mask[indices] = True
    return mask


def kac_rotation(dim: int, kac_depth: int, seed: int, device, dtype):
    from jordan_rope.quantization import KacRotation

    return KacRotation.random(dim, kac_depth, seed=seed, device=device, dtype=dtype)


def causal_metric_values(q, k, k_hat, bucket: tuple[int, int] | None = None):
    import torch

    d = q.shape[-1]
    true = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    pred = torch.matmul(q, k_hat.transpose(-2, -1)) / math.sqrt(d)
    length = q.shape[-2]
    causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    if bucket is not None:
        positions = torch.arange(length, device=q.device)
        distances = positions[:, None] - positions[None, :]
        lo, hi = bucket
        causal = causal & (distances >= lo) & (distances <= hi)
    err = (true - pred)[..., causal]
    bias = (pred - true)[..., causal]
    if err.numel() == 0:
        zero = torch.tensor(0.0, device=q.device)
        return {"logit_mse": 0.0, "relative_logit_mse": 0.0, "inner_product_bias": 0.0}
    denom = true[..., causal].square().mean().clamp_min(1e-12)
    return {
        "logit_mse": float(err.square().mean().detach().cpu()),
        "relative_logit_mse": float((err.square().mean() / denom).detach().cpu()),
        "inner_product_bias": float(bias.mean().detach().cpu()),
    }


def _topk_agreement(true_masked, pred_masked, top_k: int):
    import torch

    top_k = max(1, min(int(top_k), int(true_masked.shape[-1])))
    true_top = torch.topk(true_masked, k=top_k, dim=-1).indices
    pred_top = torch.topk(pred_masked, k=top_k, dim=-1).indices
    matches = (true_top.unsqueeze(-1) == pred_top.unsqueeze(-2)).any(dim=-1)
    return matches.float().mean(dim=-1)


def causal_attention_metric_values(q, k, k_hat, top_k: int = 5):
    import torch
    from torch.nn import functional as F

    d = q.shape[-1]
    true = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    pred = torch.matmul(q, k_hat.transpose(-2, -1)) / math.sqrt(d)
    length = q.shape[-2]
    causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    true_masked = true.masked_fill(~causal, float("-inf"))
    pred_masked = pred.masked_fill(~causal, float("-inf"))
    log_p = F.log_softmax(true_masked, dim=-1)
    log_q = F.log_softmax(pred_masked, dim=-1)
    p = log_p.exp()
    kl_terms = p * (log_p - log_q)
    kl_terms = kl_terms.masked_fill(~causal, 0.0)
    kl = kl_terms.sum(dim=-1)
    top1 = _topk_agreement(true_masked, pred_masked, 1)
    topk = _topk_agreement(true_masked, pred_masked, top_k)
    return {
        "attention_kl": float(kl.mean().detach().cpu()),
        "top1_agreement": float(top1.mean().detach().cpu()),
        f"top{int(top_k)}_agreement": float(topk.mean().detach().cpu()),
    }


def causal_metric_values_by_head(q, k, k_hat, bucket: tuple[int, int] | None = None, top_k: int = 5):
    import torch
    from torch.nn import functional as F

    d = q.shape[-1]
    true = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    pred = torch.matmul(q, k_hat.transpose(-2, -1)) / math.sqrt(d)
    length = q.shape[-2]
    causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    if bucket is not None:
        positions = torch.arange(length, device=q.device)
        distances = positions[:, None] - positions[None, :]
        lo, hi = bucket
        causal = causal & (distances >= lo) & (distances <= hi)

    mask = causal[None, None, :, :]
    count = float(max(int(causal.sum().item()) * int(q.shape[0]), 1))
    err = (true - pred).masked_fill(~mask, 0.0)
    bias = (pred - true).masked_fill(~mask, 0.0)
    true_masked_zero = true.masked_fill(~mask, 0.0)
    logit_mse = err.square().sum(dim=(0, 2, 3)) / count
    denom = true_masked_zero.square().sum(dim=(0, 2, 3)).clamp_min(1e-12) / count
    out = {
        "logit_mse": logit_mse.detach().cpu(),
        "relative_logit_mse": (logit_mse / denom).detach().cpu(),
        "inner_product_bias": (bias.sum(dim=(0, 2, 3)) / count).detach().cpu(),
    }
    if bucket is None:
        true_masked = true.masked_fill(~causal, float("-inf"))
        pred_masked = pred.masked_fill(~causal, float("-inf"))
        log_p = F.log_softmax(true_masked, dim=-1)
        log_q = F.log_softmax(pred_masked, dim=-1)
        p = log_p.exp()
        kl_terms = p * (log_p - log_q)
        kl_terms = kl_terms.masked_fill(~causal, 0.0)
        kl = kl_terms.sum(dim=-1)
        top1 = _topk_agreement(true_masked, pred_masked, 1)
        topk = _topk_agreement(true_masked, pred_masked, top_k)
        out.update(
            {
                "attention_kl": kl.mean(dim=(0, 2)).detach().cpu(),
                "top1_agreement": top1.mean(dim=(0, 2)).detach().cpu(),
                f"top{int(top_k)}_agreement": topk.mean(dim=(0, 2)).detach().cpu(),
            }
        )
    return out


def aggregate(name: str, values, base: dict, rows: list[dict]) -> None:
    import torch

    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        return
    rows.extend(
        [
            {**base, "metric": name, "aggregation": "mean", "value": float(flat.mean().cpu())},
            {**base, "metric": name, "aggregation": "median", "value": float(flat.median().cpu())},
            {**base, "metric": name, "aggregation": "p90", "value": float(torch.quantile(flat, 0.9).cpu())},
        ]
    )


def quantize_for_method(
    k,
    method: str,
    bits: int,
    kac_depth: int,
    seed: int,
    *,
    high_mask=None,
    mixed_low_bits: int = 2,
    mixed_high_bits: int = 4,
):
    import torch
    from jordan_rope.quantization import (
        KacRotation,
        dense_random_orthogonal,
        rotate_mixed_bit_quantize_dequantize,
        rotate_quantize_dequantize,
    )

    if method == "scalar_uniform_no_rotation":
        return rotate_quantize_dequantize(k, bits=bits, rotation=None)
    if method == "dense_rot_uniform":
        rotation = dense_random_orthogonal(k.shape[-1], seed=seed, device=k.device, dtype=k.dtype)
        return rotate_quantize_dequantize(k, bits=bits, rotation=rotation)
    if method == "kac_rot_uniform":
        rotation = KacRotation.random(k.shape[-1], kac_depth, seed=seed, device=k.device, dtype=k.dtype)
        return rotate_quantize_dequantize(k, bits=bits, rotation=rotation)
    if is_mixed_method(method):
        if high_mask is None:
            raise ValueError(f"Missing high_mask for mixed-bit method={method}")
        rotation = KacRotation.random(k.shape[-1], kac_depth, seed=seed, device=k.device, dtype=k.dtype)
        return rotate_mixed_bit_quantize_dequantize(
            k,
            low_bits=int(mixed_low_bits),
            high_bits=int(mixed_high_bits),
            high_mask=high_mask.to(device=k.device),
            rotation=rotation,
        )
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
        return k.clone(), accounting
    raise ValueError(f"Unknown quantization method: {method}")


def rotated_coordinates_for_stats(k, method: str, kac_depth: int, seed: int):
    from jordan_rope.quantization import KacRotation, apply_dense_rotation, dense_random_orthogonal

    if method in {"identity", "scalar_uniform_no_rotation"}:
        return k
    if method == "dense_rot_uniform":
        rotation = dense_random_orthogonal(k.shape[-1], seed=seed, device=k.device, dtype=k.dtype)
        return apply_dense_rotation(k, rotation)
    if method == "kac_rot_uniform" or method.startswith("kac_mixed_"):
        rotation = KacRotation.random(k.shape[-1], kac_depth, seed=seed, device=k.device, dtype=k.dtype)
        return rotation.apply(k)
    return k


def build_mixed_masks(
    calib_inputs: list[str],
    methods: list[str],
    kac_depths: list[int],
    *,
    seed: int,
    high_fraction: float,
) -> dict[tuple[str, int, str, int, int], object]:
    import torch

    stats = {}
    counts = {}
    masks = {}
    mixed_methods = [method for method in methods if is_mixed_method(method)]
    if not mixed_methods:
        return masks
    for input_path in calib_inputs:
        payload = torch.load(input_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        checkpoint = metadata.get("checkpoint", "")
        for record in payload["records"]:
            layer = int(record["layer"])
            q = record["q_pos"].float()
            k = record["k_pos"].float()
            dim = int(k.shape[-1])
            num_heads = int(k.shape[1])
            for method in mixed_methods:
                for kac_depth in kac_depths:
                    mask_key = (checkpoint, layer, method, int(kac_depth), dim)
                    scope = mixed_mask_scope(method)
                    if mixed_strategy(method) == "random":
                        if mask_key not in masks:
                            masks[mask_key] = random_high_mask(
                                dim,
                                high_fraction,
                                seed=int(seed) + layer * 1009 + int(kac_depth) * 9173,
                                num_heads=num_heads if scope == "head" else None,
                            )
                        continue
                    rotation_seed = int(seed) + layer * 1009 + int(kac_depth)
                    rotation = kac_rotation(dim, int(kac_depth), rotation_seed, k.device, k.dtype)
                    source = q if mixed_strategy(method) == "sensitivity" else k
                    rotated = rotation.apply(source)
                    if scope == "head":
                        value = rotated.square().mean(dim=(0, 2)).detach().cpu()
                    else:
                        value = rotated.square().mean(dim=(0, 1, 2)).detach().cpu()
                    stats[mask_key] = value if mask_key not in stats else stats[mask_key] + value
                    counts[mask_key] = counts.get(mask_key, 0) + 1
    for mask_key, value in stats.items():
        dim = mask_key[-1]
        high_count = max(1, min(dim, int(round(dim * float(high_fraction)))))
        score = value / float(max(counts[mask_key], 1))
        if score.ndim == 2:
            mask = torch.zeros_like(score, dtype=torch.bool)
            for head in range(int(score.shape[0])):
                top = torch.topk(score[head], high_count).indices
                mask[head, top] = True
        else:
            top = torch.topk(score, high_count).indices
            mask = torch.zeros(dim, dtype=torch.bool)
            mask[top] = True
        masks[mask_key] = mask
    return masks


def main() -> None:
    require_torch()
    import torch
    from jordan_rope.quantization import (
        coordinate_flatness,
        coordinate_kurtosis,
        norm_growth,
        outlier_ratio,
        position_norm_profile,
        vector_mse,
    )

    parser = argparse.ArgumentParser(description="Run phase-2 K-only quantization metrics on exported tensors.")
    parser.add_argument("--input", nargs="+", default=["runs/phase2/stage0_smoke/export/attention_tensors.pt"])
    parser.add_argument("--calib-input", nargs="*", default=None)
    parser.add_argument("--out-dir", default="runs/phase2/stage0_smoke/metrics")
    parser.add_argument("--stage", default="stage0_smoke")
    parser.add_argument("--dataset", default="exported")
    parser.add_argument("--methods", default="identity,scalar_uniform_no_rotation,dense_rot_uniform,kac_rot_uniform")
    parser.add_argument("--bits", default="2,3,4")
    parser.add_argument("--kac-depths", default="16")
    parser.add_argument("--mixed-low-bits", type=int, default=2)
    parser.add_argument("--mixed-high-bits", type=int, default=4)
    parser.add_argument("--mixed-high-fraction", type=float, default=0.25)
    parser.add_argument("--buckets", default=",".join(f"{lo}:{hi}" for lo, hi in DEFAULT_BUCKETS))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    methods = [item for item in args.methods.split(",") if item.strip()]
    bits_list = parse_ints(args.bits)
    kac_depths = parse_ints(args.kac_depths)
    buckets = parse_buckets(args.buckets)
    mixed_masks = build_mixed_masks(
        args.calib_input or args.input,
        methods,
        kac_depths,
        seed=int(args.seed),
        high_fraction=float(args.mixed_high_fraction),
    )
    rows: list[dict] = []

    for input_path in args.input:
        payload = torch.load(input_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        checkpoint = metadata.get("checkpoint", "")
        method_name = metadata.get("method", "")
        for record in payload["records"]:
            layer = int(record["layer"])
            q = record["q_pos"].float()
            k = record["k_pos"].float()
            q_profile = position_norm_profile(q)
            k_profile = position_norm_profile(k)
            profile_base = {
                "stage": args.stage,
                "method": method_name,
                "checkpoint": checkpoint,
                "dataset": args.dataset,
                "seed": metadata.get("seed", ""),
                "layer": layer,
                "head": "all",
                "cache_protocol": metadata.get("cache_protocol", "P1a_positioned_cache"),
                "split": "evaluation",
                "quant_method": "identity",
                "rotation": "none",
                "kac_depth": 0,
                "b_numeric": 0.0,
                "b_storage": 0.0,
                "aggregation": "mean",
            }
            for pos, value in enumerate(k_profile):
                rows.append(
                    {
                        **profile_base,
                        "distance_bucket": f"position_{pos}",
                        "metric": "norm_profile_k",
                        "value": float(value.cpu()),
                    }
                )
            for pos, value in enumerate(q_profile):
                rows.append(
                    {
                        **profile_base,
                        "distance_bucket": f"position_{pos}",
                        "metric": "norm_profile_q",
                        "value": float(value.cpu()),
                    }
                )
            rows.append(
                {
                    **profile_base,
                    "distance_bucket": "all",
                    "metric": "norm_growth_k",
                    "value": float(norm_growth(k_profile).cpu()),
                }
            )
            rows.append(
                {
                    **profile_base,
                    "distance_bucket": "all",
                    "metric": "norm_growth_q",
                    "value": float(norm_growth(q_profile).cpu()),
                }
            )

            for method in methods:
                method_bits = [0] if method == "identity" or is_mixed_method(method) else bits_list
                method_depths = kac_depths if method == "kac_rot_uniform" or method.startswith("kac_mixed_") else [0]
                for bits in method_bits:
                    for kac_depth in method_depths:
                        high_mask = mixed_masks.get((checkpoint, layer, method, int(kac_depth), int(k.shape[-1])))
                        start = time.perf_counter()
                        k_hat, accounting = quantize_for_method(
                            k,
                            method,
                            bits=max(bits, 2),
                            kac_depth=kac_depth,
                            seed=int(args.seed) + layer * 1009 + kac_depth,
                            high_mask=high_mask,
                            mixed_low_bits=int(args.mixed_low_bits),
                            mixed_high_bits=int(args.mixed_high_bits),
                        )
                        quant_wall_ms = (time.perf_counter() - start) * 1000.0
                        rotated = rotated_coordinates_for_stats(
                            k,
                            method,
                            kac_depth=kac_depth,
                            seed=int(args.seed) + layer * 1009 + kac_depth,
                        )
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
                            "quant_method": method,
                            "rotation": "kac" if method == "kac_rot_uniform" else ("dense" if method == "dense_rot_uniform" else "none"),
                            "kac_depth": kac_depth,
                            "b_numeric": float(accounting.b_numeric),
                            "b_storage": float(accounting.b_storage),
                            "distance_bucket": "all",
                        }
                        rows.append({**base, "aggregation": "mean", "metric": "vector_mse", "value": float(vector_mse(k, k_hat).cpu())})
                        rows.append({**base, "aggregation": "mean", "metric": "quant_wall_ms", "value": quant_wall_ms})
                        for metric, value in causal_metric_values(q, k, k_hat).items():
                            rows.append({**base, "aggregation": "mean", "metric": metric, "value": value})
                        for metric, value in causal_attention_metric_values(q, k, k_hat).items():
                            rows.append({**base, "aggregation": "mean", "metric": metric, "value": value})
                        for metric, values in causal_metric_values_by_head(q, k, k_hat).items():
                            for head, value in enumerate(values):
                                rows.append(
                                    {
                                        **base,
                                        "head": head,
                                        "aggregation": "per_head_mean",
                                        "metric": metric,
                                        "value": float(value),
                                    }
                                )
                        for bucket in buckets:
                            bucket_base = {**base, "distance_bucket": bucket_name(bucket)}
                            for metric, value in causal_metric_values(q, k, k_hat, bucket=bucket).items():
                                rows.append({**bucket_base, "aggregation": "mean", "metric": metric, "value": value})
                            for metric, values in causal_metric_values_by_head(q, k, k_hat, bucket=bucket).items():
                                for head, value in enumerate(values):
                                    rows.append(
                                        {
                                            **bucket_base,
                                            "head": head,
                                            "aggregation": "per_head_mean",
                                            "metric": metric,
                                            "value": float(value),
                                        }
                                    )
                        aggregate("coordinate_flatness", coordinate_flatness(rotated), base, rows)
                        aggregate("coordinate_kurtosis", coordinate_kurtosis(rotated), base, rows)
                        aggregate("outlier_ratio_4", outlier_ratio(rotated, 4.0), base, rows)

    write_csv(out_dir / "metrics.csv", rows)
    print(f"Wrote {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
