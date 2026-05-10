#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, require_torch, write_csv


def sample_random_pairs(num_pairs: int, dim: int, seed: int, device, *, mode: str, correlation: float):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    z = torch.randn(int(num_pairs), int(dim), generator=gen, device=device)
    if mode == "gaussian":
        y = torch.randn(int(num_pairs), int(dim), generator=gen, device=device)
    elif mode == "correlated":
        rho = max(-0.999, min(0.999, float(correlation)))
        noise = torch.randn(int(num_pairs), int(dim), generator=gen, device=device)
        y = rho * z + math.sqrt(max(1.0 - rho * rho, 0.0)) * noise
    else:
        raise ValueError(f"Unknown synthetic mode: {mode}")
    return y, z


def sample_attention_pairs(input_paths: list[str], max_pairs_per_record: int, seed: int, device):
    import torch

    ys = []
    zs = []
    for path_index, input_path in enumerate(input_paths):
        payload = torch.load(input_path, map_location="cpu")
        for record_index, record in enumerate(payload["records"]):
            q = record["q_pos"].float().to(device)
            k = record["k_pos"].float().to(device)
            batch, heads, length, dim = q.shape
            count = min(int(max_pairs_per_record), int(batch * heads * length))
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed) + path_index * 1000003 + record_index * 9176)
            b = torch.randint(0, batch, (count,), generator=gen, device=device)
            h = torch.randint(0, heads, (count,), generator=gen, device=device)
            i = torch.randint(0, length, (count,), generator=gen, device=device)
            u = torch.rand(count, generator=gen, device=device)
            j = torch.floor(u * (i + 1).float()).long()
            ys.append(q[b, h, i].reshape(count, dim))
            zs.append(k[b, h, j].reshape(count, dim))
    if not ys:
        raise ValueError("No attention records found in input paths.")
    return torch.cat(ys, dim=0), torch.cat(zs, dim=0)


def metric_row(stage, dataset, source, method, bits, b_storage, errors, threshold):
    import torch

    bias = float(errors.mean().detach().cpu())
    rmse = float(errors.square().mean().sqrt().detach().cpu())
    mae = float(errors.abs().mean().detach().cpu())
    ratio = abs(bias) / max(rmse, 1e-12)
    return {
        "stage": stage,
        "dataset": dataset,
        "source": source,
        "quant_method": method,
        "b_numeric": float(bits),
        "b_storage": float(b_storage),
        "num_errors": int(errors.numel()),
        "inner_product_bias": bias,
        "inner_product_rmse": rmse,
        "inner_product_mae": mae,
        "bias_over_rmse": ratio,
        "pass_threshold": float(threshold),
        "passed": ratio <= float(threshold),
    }


def main() -> None:
    require_torch()
    import torch
    from jordan_rope.quantization import (
        KacRotation,
        turbo_mse_quantize_dequantize,
        turbo_product_inner_product,
        turbo_product_quantize,
    )

    parser = argparse.ArgumentParser(description="Bias sanity check for TurboQuant-MSE and QJL residual product estimator.")
    parser.add_argument("--input", nargs="*", default=None, help="Optional exported attention_tensors.pt files.")
    parser.add_argument("--out-dir", default="runs/phase2/tq_bias_check")
    parser.add_argument("--stage", default="tq_bias_check")
    parser.add_argument("--dataset", default="synthetic_or_exported")
    parser.add_argument("--bits", type=int, default=3, help="Total bits for tq_prod_qjl; MSE-only also reports this bitwidth.")
    parser.add_argument("--num-pairs", type=int, default=4096)
    parser.add_argument("--synthetic-mode", choices=["gaussian", "correlated"], default="gaussian")
    parser.add_argument("--correlation", type=float, default=0.8)
    parser.add_argument("--max-pairs-per-record", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--kac-depth", type=int, default=16)
    parser.add_argument("--qjl-trials", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if int(args.bits) < 2:
        raise SystemExit("--bits must be >= 2 for tq_prod_qjl.")
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    if args.input:
        y, z = sample_attention_pairs(args.input, int(args.max_pairs_per_record), int(args.seed), device)
        source = "attention_export"
    else:
        y, z = sample_random_pairs(
            int(args.num_pairs),
            int(args.dim),
            int(args.seed),
            device,
            mode=args.synthetic_mode,
            correlation=float(args.correlation),
        )
        source = f"random_{args.synthetic_mode}"

    dim = int(z.shape[-1])
    rotation = KacRotation.random(dim, int(args.kac_depth), seed=int(args.seed) + 11, device=device, dtype=z.dtype)
    true = (y * z).sum(dim=-1)

    rows = []
    for mse_bits, label in ((int(args.bits), "tq_mse_codebook"), (int(args.bits) - 1, "tq_mse_codebook_prod_budget")):
        z_hat, accounting = turbo_mse_quantize_dequantize(z, bits=mse_bits, rotation=rotation)
        pred = (y * z_hat).sum(dim=-1)
        rows.append(
            metric_row(
                args.stage,
                args.dataset,
                source,
                label,
                float(accounting.b_numeric),
                float(accounting.b_storage),
                pred - true,
                args.threshold,
            )
        )

    prod_errors = []
    prod_storage = []
    for trial in range(int(args.qjl_trials)):
        encoded = turbo_product_quantize(
            z,
            total_bits=int(args.bits),
            rotation=rotation,
            qjl_seed=int(args.seed) + 1009 * trial,
            qjl_rows=dim,
        )
        pred = turbo_product_inner_product(y, encoded)
        prod_errors.append(pred - true)
        prod_storage.append(float(encoded.bit_accounting().b_storage))
    rows.append(
        metric_row(
            args.stage,
            args.dataset,
            source,
            "tq_prod_qjl",
            float(args.bits),
            sum(prod_storage) / max(len(prod_storage), 1),
            torch.cat(prod_errors, dim=0),
            args.threshold,
        )
    )

    write_csv(out_dir / "bias_check.csv", rows)
    print(f"Wrote {out_dir / 'bias_check.csv'}")


if __name__ == "__main__":
    main()
