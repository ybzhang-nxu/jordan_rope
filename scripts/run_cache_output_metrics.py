#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, require_torch, write_csv
from scripts.evaluate_quantized_kernel_lm import KQuantizer, is_mixed_method
from scripts.run_quant_metrics import bucket_name, parse_buckets, parse_ints


DEFAULT_BUCKETS = [(0, 128), (129, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192)]


def cache_protocol(cache_target: str) -> str:
    if cache_target == "k":
        return "P1a_positioned_key_cache_output"
    if cache_target == "v":
        return "P1_value_cache_output"
    if cache_target == "kv":
        return "P1a_positioned_k_plus_value_cache_output"
    raise ValueError(f"Unknown cache target: {cache_target}")


def select_query_rows(length: int, anchors: list[int], random_count: int, seed: int):
    import torch

    rows = {int(row) for row in anchors if 0 <= int(row) < int(length)}
    if random_count > 0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed) + int(length) * 7919)
        count = min(int(random_count), int(length))
        rows.update(int(row) for row in torch.randperm(int(length), generator=gen)[:count].tolist())
    if not rows:
        rows.add(int(length) - 1)
    return torch.tensor(sorted(rows), dtype=torch.long)


def output_metric_values(q, k, v, k_hat, v_hat, query_rows, *, bucket: tuple[int, int] | None = None):
    import torch
    from torch.nn import functional as F

    dim = int(q.shape[-1])
    length = int(k.shape[-2])
    rows = query_rows.to(device=q.device)
    q_rows = q.index_select(-2, rows)
    scores = torch.matmul(q_rows, k.transpose(-2, -1)) / math.sqrt(dim)
    scores_hat = torch.matmul(q_rows, k_hat.transpose(-2, -1)) / math.sqrt(dim)
    key_positions = torch.arange(length, device=q.device)
    causal = key_positions[None, :] <= rows[:, None]
    scores = scores.masked_fill(~causal, float("-inf"))
    scores_hat = scores_hat.masked_fill(~causal, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    attn_hat = F.softmax(scores_hat, dim=-1)
    if bucket is not None:
        distances = rows[:, None] - key_positions[None, :]
        lo, hi = bucket
        bucket_mask = causal & (distances >= int(lo)) & (distances <= int(hi))
        weight = bucket_mask.to(dtype=attn.dtype)
        attn = attn * weight
        attn_hat = attn_hat * weight
    true = torch.matmul(attn, v)
    pred = torch.matmul(attn_hat, v_hat)
    diff = pred - true
    mse_by_head = diff.square().mean(dim=(0, 2, 3)).detach().cpu()
    denom_by_head = true.square().mean(dim=(0, 2, 3)).clamp_min(1e-12).detach().cpu()
    bias_by_head = diff.mean(dim=(0, 2, 3)).detach().cpu()
    return {
        "output_mse": mse_by_head,
        "relative_output_mse": mse_by_head / denom_by_head,
        "output_bias": bias_by_head,
    }


def quantize_cache_tensors(quantizer: KQuantizer, layer_id: int, cache_target: str, k, v):
    k_hat = quantizer(layer_id, k, "k") if cache_target in {"k", "kv"} else k
    v_hat = quantizer(layer_id, v, "v") if cache_target in {"v", "kv"} else v
    return k_hat, v_hat


def add_metric_rows(rows: list[dict], metric_values: dict, base: dict) -> None:
    for metric, values in metric_values.items():
        values = values.float().reshape(-1)
        if values.numel() == 0:
            continue
        rows.append({**base, "head": "all", "aggregation": "mean", "metric": metric, "value": float(values.mean())})
        for head, value in enumerate(values.tolist()):
            rows.append(
                {
                    **base,
                    "head": int(head),
                    "aggregation": "per_head_mean",
                    "metric": metric,
                    "value": float(value),
                }
            )


def main() -> None:
    require_torch()
    import torch

    parser = argparse.ArgumentParser(description="Run sampled AttnV output-error diagnostics for cache quantization.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stage", default="stage6_output_metrics")
    parser.add_argument("--dataset", default="exported")
    parser.add_argument("--cache-targets", default="k,v,kv")
    parser.add_argument("--methods", default="identity,scalar_uniform_no_rotation,kac_rot_uniform")
    parser.add_argument("--bits", default="2,3")
    parser.add_argument("--kac-depths", default="16")
    parser.add_argument("--query-rows", default="1023,2047,4095,8191")
    parser.add_argument("--random-query-count", type=int, default=32)
    parser.add_argument("--buckets", default=",".join(f"{lo}:{hi}" for lo, hi in DEFAULT_BUCKETS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    cache_targets = [item for item in args.cache_targets.split(",") if item.strip()]
    methods = [item for item in args.methods.split(",") if item.strip()]
    bits_list = parse_ints(args.bits)
    kac_depths = parse_ints(args.kac_depths)
    anchors = parse_ints(args.query_rows)
    buckets = parse_buckets(args.buckets)
    rows: list[dict] = []

    with torch.no_grad():
        for input_path in args.input:
            payload = torch.load(input_path, map_location="cpu")
            metadata = payload.get("metadata", {})
            checkpoint = metadata.get("checkpoint", "")
            method_name = metadata.get("method", "")
            for record in payload["records"]:
                layer = int(record["layer"])
                q = record["q_pos"].float().to(device)
                k = record["k_pos"].float().to(device)
                v = record["v0"].float().to(device)
                query_rows = select_query_rows(
                    int(q.shape[-2]),
                    anchors,
                    int(args.random_query_count),
                    int(args.seed) + layer * 1009,
                ).to(device)
                for cache_target in cache_targets:
                    for method in methods:
                        method_bits = [0] if method == "identity" or is_mixed_method(method) else bits_list
                        method_depths = kac_depths if method == "kac_rot_uniform" or method.startswith("kac_mixed_") else [0]
                        for bits in method_bits:
                            for kac_depth in method_depths:
                                quantizer = KQuantizer(
                                    method,
                                    max(int(bits), 2),
                                    int(kac_depth),
                                    int(args.seed),
                                )
                                k_hat, v_hat = quantize_cache_tensors(quantizer, layer, cache_target, k, v)
                                if method == "identity":
                                    b_numeric = 0.0
                                    b_storage = 0.0
                                else:
                                    b_numeric = float(quantizer.b_numeric)
                                    b_storage = float(quantizer.b_storage)
                                base = {
                                    "stage": args.stage,
                                    "method": method_name,
                                    "checkpoint": checkpoint,
                                    "dataset": args.dataset,
                                    "seed": metadata.get("seed", ""),
                                    "layer": layer,
                                    "cache_protocol": cache_protocol(cache_target),
                                    "cache_target": cache_target,
                                    "split": "evaluation",
                                    "quant_method": method,
                                    "rotation": "kac"
                                    if method == "kac_rot_uniform"
                                    else ("dense" if method == "dense_rot_uniform" else "none"),
                                    "kac_depth": int(kac_depth),
                                    "b_numeric": b_numeric,
                                    "b_storage": b_storage,
                                    "query_count": int(query_rows.numel()),
                                }
                                all_values = output_metric_values(q, k, v, k_hat, v_hat, query_rows)
                                add_metric_rows(rows, all_values, {**base, "distance_bucket": "all"})
                                for bucket in buckets:
                                    values = output_metric_values(q, k, v, k_hat, v_hat, query_rows, bucket=bucket)
                                    add_metric_rows(rows, values, {**base, "distance_bucket": bucket_name(bucket)})

    write_csv(out_dir / "metrics.csv", rows)
    print(f"Wrote {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
