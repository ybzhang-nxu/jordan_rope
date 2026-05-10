#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, load_config, require_torch, write_csv
from scripts.run_kernel_lm import generate_batch


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


def load_checkpoint(path: str | Path, device):
    import torch
    from jordan_rope.model import CausalTransformerLM, TransformerConfig

    ckpt = torch.load(path, map_location=device)
    config = TransformerConfig(**ckpt["config"])
    model = CausalTransformerLM(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def is_mixed_method(method: str) -> bool:
    return method in MIXED_METHODS


def parse_strategy(method: str) -> str:
    if "random" in method:
        return "random"
    if "magnitude" in method:
        return "magnitude"
    if "sensitivity" in method:
        return "sensitivity"
    raise ValueError(f"Unknown mixed-bit strategy in method={method}")


def mixed_mask_scope(method: str) -> str:
    return "head" if "_head_" in method else "layer"


class KQuantizer:
    def __init__(
        self,
        method: str,
        bits: int,
        kac_depth: int,
        seed: int,
        *,
        mixed_low_bits: int = 2,
        mixed_high_bits: int = 4,
        mixed_high_fraction: float = 0.25,
    ) -> None:
        self.method = method
        self.bits = int(bits)
        self.kac_depth = int(kac_depth)
        self.seed = int(seed)
        self.mixed_low_bits = int(mixed_low_bits)
        self.mixed_high_bits = int(mixed_high_bits)
        self.mixed_high_fraction = float(mixed_high_fraction)
        self.mask_scope = mixed_mask_scope(method) if is_mixed_method(method) else "layer"
        self.rotations = {}
        self.masks = {}
        self.b_numeric = 0.0
        self.b_storage = 0.0

    def _rotation(self, layer_id: int, tensor, role: str):
        from jordan_rope.quantization import KacRotation, dense_random_orthogonal

        key = (role, layer_id, tensor.shape[-1], tensor.device, tensor.dtype, self.method, self.kac_depth)
        if key in self.rotations:
            return self.rotations[key]
        role_offset = 0 if role == "k" else 500009
        layer_seed = self.seed + role_offset + layer_id * 1000003 + self.kac_depth * 1009
        if self.method == "dense_rot_uniform":
            rotation = dense_random_orthogonal(tensor.shape[-1], seed=layer_seed, device=tensor.device, dtype=tensor.dtype)
        elif self.method == "kac_rot_uniform" or self.method.startswith("kac_mixed_"):
            rotation = KacRotation.random(tensor.shape[-1], self.kac_depth, seed=layer_seed, device=tensor.device, dtype=tensor.dtype)
        else:
            rotation = None
        self.rotations[key] = rotation
        return rotation

    def rotated(self, layer_id: int, tensor, role: str):
        from jordan_rope.quantization import apply_dense_rotation

        rotation = self._rotation(layer_id, tensor, role)
        if rotation is None:
            return tensor
        if hasattr(rotation, "apply"):
            return rotation.apply(tensor)
        return apply_dense_rotation(tensor, rotation)

    def _mask_key(self, layer_id: int, tensor, role: str):
        num_heads = int(tensor.shape[1]) if self.mask_scope == "head" else 0
        return (self.mask_scope, role, int(layer_id), num_heads, int(tensor.shape[-1]))

    def _random_mask(self, layer_id: int, tensor, role: str, device):
        import torch

        dim = int(tensor.shape[-1])
        key = self._mask_key(layer_id, tensor, role)
        if key in self.masks:
            return self.masks[key].to(device=device)
        high_count = max(1, min(dim, int(round(dim * self.mixed_high_fraction))))
        gen = torch.Generator(device="cpu")
        role_offset = 0 if role == "k" else 700001
        base_seed = self.seed + role_offset + layer_id * 1000003 + high_count * 997
        gen.manual_seed(base_seed)
        if self.mask_scope == "head":
            num_heads = int(tensor.shape[1])
            mask = torch.zeros(num_heads, dim, dtype=torch.bool)
            for head in range(num_heads):
                gen.manual_seed(base_seed + head * 104729)
                indices = torch.randperm(dim, generator=gen)[:high_count]
                mask[head, indices] = True
            self.masks[key] = mask
            return mask.to(device=device)
        indices = torch.randperm(dim, generator=gen)[:high_count]
        mask = torch.zeros(dim, dtype=torch.bool)
        mask[indices] = True
        self.masks[key] = mask
        return mask.to(device=device)

    def high_mask(self, layer_id: int, tensor, role: str, device):
        key = self._mask_key(layer_id, tensor, role)
        if key in self.masks:
            return self.masks[key].to(device=device)
        if parse_strategy(self.method) == "random":
            return self._random_mask(layer_id, tensor, role, device)
        raise ValueError(f"Missing calibrated mixed-bit mask for {self.method}, layer={layer_id}, role={role}")

    def __call__(self, layer_id: int, tensor, role: str = "k"):
        from jordan_rope.quantization import rotate_mixed_bit_quantize_dequantize, rotate_quantize_dequantize

        if self.method == "identity":
            return tensor
        if is_mixed_method(self.method):
            rotation = self._rotation(layer_id, tensor, role)
            high_mask = self.high_mask(layer_id, tensor, role, tensor.device)
            tensor_hat, accounting = rotate_mixed_bit_quantize_dequantize(
                tensor,
                low_bits=self.mixed_low_bits,
                high_bits=self.mixed_high_bits,
                high_mask=high_mask,
                rotation=rotation,
            )
            self.b_numeric = float(accounting.b_numeric)
            self.b_storage = float(accounting.b_storage)
            return tensor_hat
        if self.method == "scalar_uniform_no_rotation":
            rotation = None
        elif self.method in {"dense_rot_uniform", "kac_rot_uniform"}:
            rotation = self._rotation(layer_id, tensor, role)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")
        k_hat, accounting = rotate_quantize_dequantize(tensor, bits=self.bits, rotation=rotation)
        self.b_numeric = float(accounting.b_numeric)
        self.b_storage = float(accounting.b_storage)
        return k_hat


def parse_lengths(text: str, cfg: dict) -> list[int]:
    if text:
        return parse_ints(text)
    return [int(cfg["kernel_lm"]["seq_len_train"])]


def mixed_roles(cache_target: str) -> list[str]:
    if cache_target == "k":
        return ["k"]
    if cache_target == "v":
        return ["v"]
    if cache_target == "kv":
        return ["k", "v"]
    raise ValueError(f"Unknown cache target: {cache_target}")


def calibrate_mixed_masks(
    model,
    cfg: dict,
    device,
    quantizer: KQuantizer,
    *,
    cache_target: str,
    calib_batches: int,
    calib_seed: int,
    calib_lengths: list[int],
) -> None:
    import torch

    strategy = parse_strategy(quantizer.method)
    if strategy == "random":
        return
    stats = {}
    counts = {}
    roles = mixed_roles(cache_target)
    with torch.no_grad():
        for length in calib_lengths:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(calib_seed) + int(length) * 17)
            for _ in range(int(calib_batches)):
                tokens, _labels, _score = generate_batch(
                    int(length),
                    int(cfg["kernel_lm"]["eval_batch_size"]),
                    cfg,
                    device,
                    gen,
                )
                records = model.extract_attention_tensors(tokens, detach=True)
                for record in records:
                    layer_id = int(record["layer"])
                    for role in roles:
                        if role == "k" and strategy == "sensitivity":
                            tensor = record["q_pos"].float()
                        elif role == "k":
                            tensor = record["k_pos"].float()
                        elif role == "v":
                            tensor = record["v0"].float()
                        else:
                            raise ValueError(f"Unknown role: {role}")
                        rotated = quantizer.rotated(layer_id, tensor, role)
                        if quantizer.mask_scope == "head":
                            value = rotated.square().mean(dim=(0, 2)).detach().cpu()
                        else:
                            value = rotated.square().mean(dim=(0, 1, 2)).detach().cpu()
                        key = quantizer._mask_key(layer_id, tensor, role)
                        stats[key] = value if key not in stats else stats[key] + value
                        counts[key] = counts.get(key, 0) + 1

    for key, value in stats.items():
        dim = key[-1]
        high_count = max(1, min(dim, int(round(dim * quantizer.mixed_high_fraction))))
        score = value / float(max(counts[key], 1))
        if score.ndim == 2:
            mask = torch.zeros_like(score, dtype=torch.bool)
            for head in range(int(score.shape[0])):
                top = torch.topk(score[head], high_count).indices
                mask[head, top] = True
        else:
            top = torch.topk(score, high_count).indices
            mask = torch.zeros(dim, dtype=torch.bool)
            mask[top] = True
        quantizer.masks[key] = mask


def quant_method_bits(quant_method: str, bits_values: list[int]) -> list[int]:
    if quant_method == "identity" or is_mixed_method(quant_method):
        return [0]
    return bits_values


def evaluate_model(
    model,
    cfg: dict,
    device,
    quantizer: KQuantizer,
    seed: int,
    cache_target: str,
    eval_batches: int | None = None,
) -> list[dict]:
    import torch
    from torch.nn import functional as F

    klm = cfg["kernel_lm"]
    rows = []
    batches = int(eval_batches or klm["eval_batches"])
    with torch.no_grad():
        for length in klm["eval_lengths"]:
            length = int(length)
            losses = []
            accs = []
            score_abs = []
            gen = torch.Generator(device=device)
            gen.manual_seed(int(klm.get("eval_seed", 9109)) + seed * 1009 + length)
            for _ in range(batches):
                tokens, labels, score = generate_batch(length, int(klm["eval_batch_size"]), cfg, device, gen)
                if quantizer.method == "identity":
                    logits = model(tokens)[:, -1, :2]
                elif cache_target == "k":
                    logits = model.forward_with_cache_quantizer(
                        tokens,
                        quantize_k=lambda layer_id, tensor: quantizer(layer_id, tensor, role="k"),
                    )[:, -1, :2]
                elif cache_target == "v":
                    logits = model.forward_with_cache_quantizer(
                        tokens,
                        quantize_v=lambda layer_id, tensor: quantizer(layer_id, tensor, role="v"),
                    )[:, -1, :2]
                elif cache_target == "kv":
                    logits = model.forward_with_cache_quantizer(
                        tokens,
                        quantize_k=lambda layer_id, tensor: quantizer(layer_id, tensor, role="k"),
                        quantize_v=lambda layer_id, tensor: quantizer(layer_id, tensor, role="v"),
                    )[:, -1, :2]
                else:
                    raise ValueError(f"Unknown cache target: {cache_target}")
                loss = F.cross_entropy(logits, labels)
                losses.append(float(loss.detach().cpu()))
                accs.append(float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu()))
                score_abs.append(float(score.abs().mean().detach().cpu()))
            rows.append(
                {
                    "context": length,
                    "loss": sum(losses) / len(losses),
                    "accuracy": sum(accs) / len(accs),
                    "score_abs_mean": sum(score_abs) / len(score_abs),
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

    parser = argparse.ArgumentParser(description="Evaluate kernel-LM checkpoints with cache quantization.")
    parser.add_argument("--config", default="configs/phase2_kernel_lm_medium.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out-dir", default="runs/phase2/stage2_task_eval_medium")
    parser.add_argument("--stage", default="stage2_task_eval")
    parser.add_argument("--quant-methods", default="identity,scalar_uniform_no_rotation,dense_rot_uniform,kac_rot_uniform")
    parser.add_argument("--cache-target", choices=["k", "v", "kv"], default="k")
    parser.add_argument("--bits", default="3")
    parser.add_argument("--kac-depths", default="16")
    parser.add_argument("--mixed-low-bits", type=int, default=2)
    parser.add_argument("--mixed-high-bits", type=int, default=4)
    parser.add_argument("--mixed-high-fraction", type=float, default=0.25)
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--calib-seed", type=int, default=17001)
    parser.add_argument("--calib-lengths", default="")
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=9090)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = load_config(args.config)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    quant_methods = [item for item in args.quant_methods.split(",") if item.strip()]
    bits_values = parse_ints(args.bits)
    kac_depths = parse_ints(args.kac_depths)
    calib_lengths = parse_lengths(args.calib_lengths, cfg)
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
                    quantizer = KQuantizer(
                        quant_method,
                        bits=max(bits, 2),
                        kac_depth=kac_depth,
                        seed=int(args.seed),
                        mixed_low_bits=int(args.mixed_low_bits),
                        mixed_high_bits=int(args.mixed_high_bits),
                        mixed_high_fraction=float(args.mixed_high_fraction),
                    )
                    if is_mixed_method(quant_method):
                        calibrate_mixed_masks(
                            model,
                            cfg,
                            device,
                            quantizer,
                            cache_target=args.cache_target,
                            calib_batches=int(args.calib_batches),
                            calib_seed=int(args.calib_seed),
                            calib_lengths=calib_lengths,
                        )
                    eval_rows = evaluate_model(
                        model,
                        cfg,
                        device,
                        quantizer,
                        seed=int(args.seed),
                        cache_target=args.cache_target,
                        eval_batches=args.eval_batches,
                    )
                    for row in eval_rows:
                        if quant_method == "identity":
                            identity_by_context[int(row["context"])] = row
                        base = identity_by_context.get(int(row["context"]))
                        delta_loss = "" if base is None else row["loss"] - base["loss"]
                        delta_acc = "" if base is None else row["accuracy"] - base["accuracy"]
                        rows.append(
                            {
                                "stage": args.stage,
                                "method": method,
                                "checkpoint": str(checkpoint),
                                "dataset": "kernel_lm",
                                "seed": args.seed,
                                "calib_seed": args.calib_seed if is_mixed_method(quant_method) else "",
                                "calib_batches": args.calib_batches if is_mixed_method(quant_method) else "",
                                "calib_lengths": ",".join(str(length) for length in calib_lengths) if is_mixed_method(quant_method) else "",
                                "cache_protocol": (
                                    "P1a_positioned_k_cache"
                                    if args.cache_target == "k"
                                    else ("P1_value_cache" if args.cache_target == "v" else "P1a_positioned_k_plus_value_cache")
                                ),
                                "cache_target": args.cache_target,
                                "split": "evaluation",
                                "quant_method": quant_method,
                                "rotation": "kac" if quant_method == "kac_rot_uniform" or quant_method.startswith("kac_mixed_") else ("dense" if quant_method == "dense_rot_uniform" else "none"),
                                "kac_depth": kac_depth,
                                "mixed_low_bits": args.mixed_low_bits if is_mixed_method(quant_method) else "",
                                "mixed_high_bits": args.mixed_high_bits if is_mixed_method(quant_method) else "",
                                "mixed_high_fraction": args.mixed_high_fraction if is_mixed_method(quant_method) else "",
                                "b_numeric": row["b_numeric"],
                                "b_storage": row["b_storage"],
                                "context": row["context"],
                                "loss": row["loss"],
                                "accuracy": row["accuracy"],
                                "delta_loss_same": delta_loss,
                                "delta_acc_same": delta_acc,
                                "drop_acc_same": "" if delta_acc == "" else -delta_acc,
                                "score_abs_mean": row["score_abs_mean"],
                            }
                        )
                    write_csv(out_dir / "task_eval.csv", rows)

    write_csv(out_dir / "task_eval.csv", rows)
    print(f"Wrote {out_dir / 'task_eval.csv'}")


if __name__ == "__main__":
    main()
