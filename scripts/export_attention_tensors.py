#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import choose_device, ensure_dir, load_config, require_torch, save_config, set_seed


BYTE_VOCAB_SIZE = 257
KERNEL_LM_VOCAB_SIZE = 3


def build_config_from_file(cfg: dict, method: str):
    from jordan_rope.model import TransformerConfig

    pos = cfg.get("position", {})
    if "kernel_lm" in cfg:
        model = cfg["kernel_lm"]["model"]
        vocab_size = KERNEL_LM_VOCAB_SIZE
        train_context = int(cfg["kernel_lm"].get("seq_len_train", pos.get("train_context", 1024)))
    elif "lm" in cfg:
        model = cfg["lm"]["model"]
        vocab_size = BYTE_VOCAB_SIZE
        train_context = int(cfg["lm"].get("seq_len_train", pos.get("train_context", 1024)))
    else:
        model = cfg.get("model", {})
        vocab_size = int(cfg.get("vocab_size", BYTE_VOCAB_SIZE))
        train_context = int(cfg.get("train_context", pos.get("train_context", 1024)))

    return TransformerConfig(
        vocab_size=vocab_size,
        d_model=int(model.get("d_model", 64)),
        n_heads=int(model.get("n_heads", 4)),
        n_layers=int(model.get("n_layers", 2)),
        mlp_ratio=int(model.get("mlp_ratio", 4)),
        dropout=float(model.get("dropout", 0.0)),
        position_method=method,
        theta=float(pos.get("theta", 10000.0)),
        gamma_min=float(pos.get("gamma_min", 1e-4)),
        init_gamma=float(pos.get("init_gamma", 1e-4)),
        eta_max=float(pos.get("eta_max", 0.1)),
        init_eta=float(pos.get("init_eta", 0.0)),
        train_context=train_context,
        bounded_tau=bool(pos.get("bounded_tau", True)),
        max_exponent=float(pos.get("max_exponent", 30.0)),
    )


def load_model(args, device):
    import torch
    from jordan_rope.model import CausalTransformerLM, TransformerConfig

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = TransformerConfig(**ckpt["config"])
        model = CausalTransformerLM(config).to(device)
        model.load_state_dict(ckpt["model"])
        return model, config, {"checkpoint": str(args.checkpoint), "config_source": "checkpoint"}

    cfg = load_config(args.config) if args.config else {}
    method = args.method or cfg.get("method", "jordan_rope")
    config = build_config_from_file(cfg, method)
    model = CausalTransformerLM(config).to(device)
    return model, config, {"checkpoint": "", "config_source": str(args.config or "defaults")}


def parse_layers(text: str | None, n_layers: int) -> list[int] | None:
    if not text:
        return None
    if text == "all":
        return list(range(n_layers))
    return [int(item) for item in text.split(",") if item.strip()]


def main() -> None:
    require_torch()
    import torch

    parser = argparse.ArgumentParser(description="Export q/k/v and positioned q/k tensors for phase-2 diagnostics.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--method", default=None)
    parser.add_argument("--out-dir", default="runs/phase2/stage0_smoke/export")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer ids")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--token-mode", choices=["random", "binary_query"], default="random")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = ensure_dir(args.out_dir)
    model, config, source = load_model(args, device)
    model.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    if args.token_mode == "binary_query":
        if int(config.vocab_size) < 3:
            raise ValueError("binary_query token mode expects vocab_size >= 3.")
        tokens = torch.empty(int(args.batch_size), int(args.seq_len), dtype=torch.long, device=device)
        tokens[:, :-1] = torch.randint(0, 2, (int(args.batch_size), int(args.seq_len) - 1), generator=gen, device=device)
        tokens[:, -1] = 2
    else:
        tokens = torch.randint(
            0,
            int(config.vocab_size),
            (int(args.batch_size), int(args.seq_len)),
            generator=gen,
            device=device,
        )
    layers = parse_layers(args.layers, int(config.n_layers))

    with torch.no_grad():
        records = model.extract_attention_tensors(tokens, layers=layers)
        logits = model(tokens).detach().cpu()

    cpu_records = []
    for record in records:
        cpu_record = {"layer": int(record["layer"])}
        for key in ("q0", "k0", "v0", "q_pos", "k_pos", "positions"):
            cpu_record[key] = record[key].detach().cpu()
        cpu_records.append(cpu_record)

    payload = {
        "tokens": tokens.detach().cpu(),
        "logits": logits,
        "records": cpu_records,
        "metadata": {
            **source,
            "method": config.position_method,
            "cache_protocol": "P1a_positioned_cache",
            "seq_len": int(args.seq_len),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "device": str(device),
            "layers": "all" if layers is None else layers,
            "token_mode": args.token_mode,
        },
        "config": config.__dict__,
    }
    torch.save(payload, out_dir / "attention_tensors.pt")
    save_config(payload["metadata"], out_dir / "metadata.yaml")
    (out_dir / "config.json").write_text(json.dumps(payload["config"], indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'attention_tensors.pt'}")


if __name__ == "__main__":
    main()
