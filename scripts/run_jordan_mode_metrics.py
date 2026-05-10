#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir, require_torch, write_csv


DEFAULT_BUCKETS = [(0, 128), (129, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192)]


def parse_buckets(text: str) -> list[tuple[int, int]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split(":")
        out.append((int(lo), int(hi)))
    return out


def bucket_name(bucket: tuple[int, int]) -> str:
    return f"{bucket[0]}_{bucket[1]}"


def load_model_from_checkpoint(path: str, device):
    import torch
    from jordan_rope.model import CausalTransformerLM, TransformerConfig

    ckpt = torch.load(path, map_location=device)
    config = TransformerConfig(**ckpt["config"])
    model = CausalTransformerLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def p1_logit_noise_by_bucket(q, k, k_hat, query_indices, bucket):
    import torch

    d = q.shape[-1]
    q_sel = q.index_select(-2, query_indices)
    true = torch.matmul(q_sel, k.transpose(-2, -1)) / math.sqrt(d)
    pred = torch.matmul(q_sel, k_hat.transpose(-2, -1)) / math.sqrt(d)
    positions = torch.arange(k.shape[-2], device=q.device)
    distances = query_indices[:, None] - positions[None, :]
    lo, hi = bucket
    mask = (distances >= lo) & (distances <= hi)
    values = (pred - true).masked_fill(~mask[None, None, :, :], float("nan"))
    finite = torch.isfinite(values)
    return values.masked_fill(~finite, 0.0).square().sum() / finite.sum().clamp_min(1)


def p1_logit_noise_by_bucket_per_head(q, k, k_hat, query_indices, bucket):
    import torch

    d = q.shape[-1]
    q_sel = q.index_select(-2, query_indices)
    true = torch.matmul(q_sel, k.transpose(-2, -1)) / math.sqrt(d)
    pred = torch.matmul(q_sel, k_hat.transpose(-2, -1)) / math.sqrt(d)
    positions = torch.arange(k.shape[-2], device=q.device)
    distances = query_indices[:, None] - positions[None, :]
    lo, hi = bucket
    mask = (distances >= lo) & (distances <= hi)
    values = (pred - true).masked_fill(~mask[None, None, :, :], float("nan"))
    finite = torch.isfinite(values)
    count = finite.sum(dim=(0, 2, 3)).clamp_min(1)
    return values.masked_fill(~finite, 0.0).square().sum(dim=(0, 2, 3)) / count


def nanmean_square_per_head(values, mask):
    import torch

    values = values.masked_fill(~mask[None, None, :, :], float("nan"))
    finite = torch.isfinite(values)
    count = finite.sum(dim=(0, 2, 3)).clamp_min(1)
    return values.masked_fill(~finite, 0.0).square().sum(dim=(0, 2, 3)) / count


def stage2_quantized_k(k, method: str, bits: int, kac_depth: int, seed: int):
    from jordan_rope.quantization import KacRotation, dense_random_orthogonal, rotate_quantize_dequantize

    if method == "identity":
        return k.clone(), 0.0, 0.0
    if method == "scalar_uniform_no_rotation":
        k_hat, accounting = rotate_quantize_dequantize(k, bits=bits, rotation=None)
    elif method == "dense_rot_uniform":
        rotation = dense_random_orthogonal(k.shape[-1], seed=seed, device=k.device, dtype=k.dtype)
        k_hat, accounting = rotate_quantize_dequantize(k, bits=bits, rotation=rotation)
    elif method == "kac_rot_uniform":
        rotation = KacRotation.random(k.shape[-1], kac_depth, seed=seed, device=k.device, dtype=k.dtype)
        k_hat, accounting = rotate_quantize_dequantize(k, bits=bits, rotation=rotation)
    else:
        raise ValueError(f"Unknown quant method: {method}")
    return k_hat, accounting.b_numeric, accounting.b_storage


def main() -> None:
    require_torch()
    import torch
    from jordan_rope.diagnostics import distance_bucket_mask, nanmean_square, p0_jordan_logit_components
    from jordan_rope.positional import JordanRoPE

    parser = argparse.ArgumentParser(description="Compute P0 Jordan-mode signal and initial SNR diagnostics.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/phase2/stage2_jordan_survival_smoke/metrics")
    parser.add_argument("--stage", default="stage2_jordan_survival")
    parser.add_argument("--dataset", default="kernel_lm")
    parser.add_argument("--methods", default="identity,scalar_uniform_no_rotation,kac_rot_uniform")
    parser.add_argument("--bits", default="3")
    parser.add_argument("--kac-depths", default="16")
    parser.add_argument("--buckets", default=",".join(f"{lo}:{hi}" for lo, hi in DEFAULT_BUCKETS))
    parser.add_argument("--max-query-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    quant_methods = [item for item in args.methods.split(",") if item.strip()]
    bits_values = [int(item) for item in args.bits.split(",") if item.strip()]
    kac_depths = [int(item) for item in args.kac_depths.split(",") if item.strip()]
    buckets = parse_buckets(args.buckets)
    out_dir = ensure_dir(args.out_dir)
    rows: list[dict] = []
    models = {}

    for input_path in args.input:
        payload = torch.load(input_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        checkpoint = metadata.get("checkpoint", "")
        if not checkpoint:
            continue
        if checkpoint not in models:
            models[checkpoint] = load_model_from_checkpoint(checkpoint, torch.device("cpu"))
        model = models[checkpoint]
        method_name = metadata.get("method", model.config.position_method)

        for record in payload["records"]:
            layer = int(record["layer"])
            positioner = model.blocks[layer].attn.positioner
            if not isinstance(positioner, JordanRoPE) or positioner.order != 2:
                continue
            q0 = record["q0"].float()
            k0 = record["k0"].float()
            q_pos = record["q_pos"].float()
            k_pos = record["k_pos"].float()
            length = q0.shape[-2]
            if length <= int(args.max_query_rows):
                query_indices = torch.arange(length)
            else:
                query_indices = torch.linspace(0, length - 1, steps=int(args.max_query_rows)).round().long().unique()

            comps = p0_jordan_logit_components(positioner, q0, k0, query_indices=query_indices)
            for bucket in buckets:
                mask = distance_bucket_mask(query_indices, length, bucket)
                signal_j = nanmean_square(comps["jordan"], mask)
                signal_base = nanmean_square(comps["base"], mask)
                total_signal = nanmean_square(comps["total"], mask)
                signal_j_by_head = nanmean_square_per_head(comps["jordan"], mask)
                signal_base_by_head = nanmean_square_per_head(comps["base"], mask)
                total_signal_by_head = nanmean_square_per_head(comps["total"], mask)
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
                    "distance_bucket": bucket_name(bucket),
                    "aggregation": "mean",
                }
                rows.append({**base, "quant_method": "identity", "kac_depth": 0, "b_numeric": 0.0, "b_storage": 0.0, "metric": "S_J", "value": float(signal_j.cpu())})
                rows.append({**base, "quant_method": "identity", "kac_depth": 0, "b_numeric": 0.0, "b_storage": 0.0, "metric": "S_base", "value": float(signal_base.cpu())})
                rows.append({**base, "quant_method": "identity", "kac_depth": 0, "b_numeric": 0.0, "b_storage": 0.0, "metric": "S_total", "value": float(total_signal.cpu())})
                for head, (head_j, head_base, head_total) in enumerate(
                    zip(signal_j_by_head, signal_base_by_head, total_signal_by_head)
                ):
                    head_base_row = {**base, "head": head, "aggregation": "per_head_mean"}
                    rows.append({**head_base_row, "quant_method": "identity", "kac_depth": 0, "b_numeric": 0.0, "b_storage": 0.0, "metric": "S_J", "value": float(head_j.cpu())})
                    rows.append({**head_base_row, "quant_method": "identity", "kac_depth": 0, "b_numeric": 0.0, "b_storage": 0.0, "metric": "S_base", "value": float(head_base.cpu())})
                    rows.append({**head_base_row, "quant_method": "identity", "kac_depth": 0, "b_numeric": 0.0, "b_storage": 0.0, "metric": "S_total", "value": float(head_total.cpu())})

                for quant_method in quant_methods:
                    method_bits = [0] if quant_method == "identity" else bits_values
                    method_depths = kac_depths if quant_method == "kac_rot_uniform" else [0]
                    for bits in method_bits:
                        for kac_depth in method_depths:
                            k_hat, b_numeric, b_storage = stage2_quantized_k(
                                k_pos,
                                quant_method,
                                bits=max(bits, 2),
                                kac_depth=kac_depth,
                                seed=int(args.seed) + layer * 1009 + kac_depth,
                            )
                            noise = p1_logit_noise_by_bucket(q_pos, k_pos, k_hat, query_indices, bucket)
                            noise_by_head = p1_logit_noise_by_bucket_per_head(q_pos, k_pos, k_hat, query_indices, bucket)
                            snr = signal_j / noise.clamp_min(1e-12)
                            rows.append(
                                {
                                    **base,
                                    "quant_method": quant_method,
                                    "kac_depth": kac_depth,
                                    "b_numeric": b_numeric,
                                    "b_storage": b_storage,
                                    "metric": "N_Q",
                                    "value": float(noise.cpu()),
                                }
                            )
                            rows.append(
                                {
                                    **base,
                                    "quant_method": quant_method,
                                    "kac_depth": kac_depth,
                                    "b_numeric": b_numeric,
                                    "b_storage": b_storage,
                                    "metric": "SNR_J",
                                    "value": float(snr.cpu()),
                                }
                            )
                            snr_by_head = signal_j_by_head / noise_by_head.clamp_min(1e-12)
                            for head, (head_noise, head_snr) in enumerate(zip(noise_by_head, snr_by_head)):
                                rows.append(
                                    {
                                        **base,
                                        "head": head,
                                        "aggregation": "per_head_mean",
                                        "quant_method": quant_method,
                                        "kac_depth": kac_depth,
                                        "b_numeric": b_numeric,
                                        "b_storage": b_storage,
                                        "metric": "N_Q",
                                        "value": float(head_noise.cpu()),
                                    }
                                )
                                rows.append(
                                    {
                                        **base,
                                        "head": head,
                                        "aggregation": "per_head_mean",
                                        "quant_method": quant_method,
                                        "kac_depth": kac_depth,
                                        "b_numeric": b_numeric,
                                        "b_storage": b_storage,
                                        "metric": "SNR_J",
                                        "value": float(head_snr.cpu()),
                                    }
                                )

    write_csv(out_dir / "jordan_mode_metrics.csv", rows)
    print(f"Wrote {out_dir / 'jordan_mode_metrics.csv'}")


if __name__ == "__main__":
    main()
