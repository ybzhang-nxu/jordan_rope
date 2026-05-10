#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.model import CausalTransformerLM
from jordan_rope.positional import JordanRoPE
from jordan_rope.utils import choose_device, ensure_dir, load_config, require_torch
from scripts.run_lm import build_config, load_tokens


CKPT_RE = re.compile(r"(?P<method>.+)_seed(?P<seed>\d+)\.pt$")


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def checkpoint_identity(path: Path) -> tuple[str, int]:
    match = CKPT_RE.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse checkpoint name: {path}")
    return match.group("method"), int(match.group("seed"))


def sample_starts(tokens, length: int, batch_size: int, batches: int, seed: int):
    import torch

    max_start = tokens.numel() - length - 1
    if max_start <= 0:
        raise ValueError(f"Not enough tokens for length={length}")
    gen = torch.Generator()
    gen.manual_seed(seed + length * 1009)
    return [
        torch.randint(0, max_start, (batch_size,), generator=gen)
        for _ in range(batches)
    ]


def profile_one_model(model, tokens, lengths: list[int], batch_size: int, batches: int, seed: int, device):
    import torch
    from jordan_rope.data import sample_lm_batch_at_starts

    model.eval()
    out: dict[int, dict[str, list[float]]] = {}
    with torch.no_grad():
        for length in lengths:
            sums = {
                "q0_norm": torch.zeros(length, dtype=torch.float64),
                "k0_norm": torch.zeros(length, dtype=torch.float64),
                "q_pos_norm": torch.zeros(length, dtype=torch.float64),
                "k_pos_norm": torch.zeros(length, dtype=torch.float64),
            }
            record_count = 0
            for starts in sample_starts(tokens, length, batch_size, batches, seed):
                x, _ = sample_lm_batch_at_starts(tokens, starts, length, device)
                records = model.extract_attention_tensors(x, detach=True)
                for record in records:
                    for tensor_key, out_key in (
                        ("q0", "q0_norm"),
                        ("k0", "k0_norm"),
                        ("q_pos", "q_pos_norm"),
                        ("k_pos", "k_pos_norm"),
                    ):
                        value = record[tensor_key].float().norm(dim=-1).mean(dim=(0, 1)).cpu().to(torch.float64)
                        sums[out_key] += value
                    record_count += 1
            out[length] = {key: (value / max(record_count, 1)).tolist() for key, value in sums.items()}
    return out


def collect_jordan_params(model) -> tuple[list[dict], list[dict]]:
    detail = []
    summary = []
    for layer_id, block in enumerate(model.blocks):
        positioner = block.attn.positioner
        if not isinstance(positioner, JordanRoPE):
            continue
        gamma = positioner.gamma().detach().float().cpu()
        eta = positioner.eta().detach().float().cpu()
        summary.append(
            {
                "layer": layer_id,
                "order": positioner.order,
                "gamma_mean": float(gamma.mean()),
                "gamma_min": float(gamma.min()),
                "gamma_max": float(gamma.max()),
                "eta_mean": float(eta.mean()),
                "eta_abs_mean": float(eta.abs().mean()),
                "eta_abs_max": float(eta.abs().max()),
            }
        )
        for head in range(gamma.shape[0]):
            for block_id in range(gamma.shape[1]):
                detail.append(
                    {
                        "layer": layer_id,
                        "head": head,
                        "block": block_id,
                        "order": positioner.order,
                        "gamma": float(gamma[head, block_id]),
                        "eta": float(eta[head, block_id]),
                    }
                )
    return detail, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def write_markdown(out_path: Path, profile_rows: list[dict], param_rows: list[dict]) -> None:
    by_method_context: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in profile_rows:
        if int(row["position"]) in {0, int(row["context"]) - 1}:
            by_method_context[(row["method"], int(row["context"]))].append(row)

    lines = ["# Position Norm Profile Summary", ""]
    lines.append("Averages are over seeds, sampled eval batches, layers, heads, and batch items.")
    lines.append("")
    lines.append("## Endpoint Growth")
    lines.append("")
    lines.append("| method | context | k_pos_last/first | q_pos_last/first | k_transform_last/first | q_transform_last/first |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for (method, context), rows in sorted(by_method_context.items(), key=lambda item: (item[0][1], item[0][0])):
        first = [row for row in rows if int(row["position"]) == 0]
        last = [row for row in rows if int(row["position"]) == context - 1]
        if not first or not last:
            continue
        k_pos_first = mean([float(row["k_pos_norm"]) for row in first])
        q_pos_first = mean([float(row["q_pos_norm"]) for row in first])
        k0_first = mean([float(row["k0_norm"]) for row in first])
        q0_first = mean([float(row["q0_norm"]) for row in first])
        k_pos_last = mean([float(row["k_pos_norm"]) for row in last])
        q_pos_last = mean([float(row["q_pos_norm"]) for row in last])
        k0_last = mean([float(row["k0_norm"]) for row in last])
        q0_last = mean([float(row["q0_norm"]) for row in last])
        eps = 1e-12
        k_transform_first = k_pos_first / max(k0_first, eps)
        q_transform_first = q_pos_first / max(q0_first, eps)
        k_transform_last = k_pos_last / max(k0_last, eps)
        q_transform_last = q_pos_last / max(q0_last, eps)
        lines.append(
            f"| {method} | {context} | "
            f"{k_pos_last / max(k_pos_first, eps):.4f} | "
            f"{q_pos_last / max(q_pos_first, eps):.4f} | "
            f"{k_transform_last / max(k_transform_first, eps):.4f} | "
            f"{q_transform_last / max(q_transform_first, eps):.4f} |"
        )

    if param_rows:
        lines.append("")
        lines.append("## Learned Jordan Parameters")
        lines.append("")
        lines.append("| method | order | gamma mean | gamma max | eta mean | eta abs mean | eta abs max | n layers |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        by_method: dict[str, list[dict]] = defaultdict(list)
        for row in param_rows:
            by_method[row["method"]].append(row)
        for method, rows in sorted(by_method.items()):
            lines.append(
                f"| {method} | {int(float(rows[0]['order']))} | "
                f"{mean([float(row['gamma_mean']) for row in rows]):.6f} | "
                f"{mean([float(row['gamma_max']) for row in rows]):.6f} | "
                f"{mean([float(row['eta_mean']) for row in rows]):.6f} | "
                f"{mean([float(row['eta_abs_mean']) for row in rows]):.6f} | "
                f"{mean([float(row['eta_abs_max']) for row in rows]):.6f} | "
                f"{len(rows)} |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_torch()
    import torch

    parser = argparse.ArgumentParser(description="Analyze q/k positioned norm profiles and learned Jordan parameters.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lengths", default="256,512,1024,2048")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=424242)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = choose_device(args.device if args.device != "auto" else cfg.get("device", "auto"))
    eval_tokens = load_tokens(cfg, "eval")
    lengths = parse_ints(args.lengths)
    out_dir = ensure_dir(args.out_dir)

    checkpoint_paths = sorted(args.checkpoint_dir.glob("*.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"No checkpoints found in {args.checkpoint_dir}")

    profile_rows: list[dict] = []
    param_detail_rows: list[dict] = []
    param_summary_rows: list[dict] = []
    for path in checkpoint_paths:
        method, seed = checkpoint_identity(path)
        model = CausalTransformerLM(build_config(cfg, method)).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model"])

        profile = profile_one_model(
            model,
            eval_tokens,
            lengths,
            args.batch_size,
            args.batches,
            args.sample_seed + seed * 100003,
            device,
        )
        for context, values in profile.items():
            for position in range(context):
                profile_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "context": context,
                        "position": position,
                        "q0_norm": values["q0_norm"][position],
                        "k0_norm": values["k0_norm"][position],
                        "q_pos_norm": values["q_pos_norm"][position],
                        "k_pos_norm": values["k_pos_norm"][position],
                    }
                )

        detail, summary = collect_jordan_params(model)
        for row in detail:
            param_detail_rows.append({"method": method, "seed": seed, **row})
        for row in summary:
            param_summary_rows.append({"method": method, "seed": seed, **row})

    write_csv(out_dir / "norm_profile.csv", profile_rows)
    write_csv(out_dir / "jordan_param_detail.csv", param_detail_rows)
    write_csv(out_dir / "jordan_param_summary.csv", param_summary_rows)
    write_markdown(out_dir / "summary.md", profile_rows, param_summary_rows)
    print(f"wrote {out_dir / 'norm_profile.csv'}")
    print(f"wrote {out_dir / 'jordan_param_summary.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
