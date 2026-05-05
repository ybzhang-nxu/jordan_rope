#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jordan_rope.utils import ensure_dir, load_config, require_torch


def main() -> None:
    require_torch()
    import matplotlib.pyplot as plt
    import torch
    from jordan_rope.positional import causal_delta_features

    parser = argparse.ArgumentParser(description="Plot diagnostic basis functions.")
    parser.add_argument("--config", default="configs/full_research.yaml")
    parser.add_argument("--out-dir", default="runs/basis")
    parser.add_argument("--context", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    context = args.context or max(int(x) for x in cfg["toy"]["eval_contexts"])
    deltas = torch.arange(0, context)
    pos = cfg["position"]
    toy = cfg["toy"]
    features = causal_delta_features(
        "jordan_rope",
        deltas,
        num_freqs=1,
        head_dim=int(toy["head_dim"]),
        theta=float(pos["theta"]),
        gamma=float(toy.get("gamma", pos.get("gamma_min", 1e-4))),
        train_context=int(toy["train_context"]),
        bounded_tau=bool(pos.get("bounded_tau", True)),
    )
    labels = ["1", "decay*cos", "decay*sin", "tau*decay*cos", "tau*decay*sin"]
    plt.figure(figsize=(10, 5))
    for idx, label in enumerate(labels):
        plt.plot(deltas, features[:, idx], label=label)
    plt.xlabel("Delta")
    plt.ylabel("basis value")
    plt.title("Jordan-RoPE causal delta basis")
    plt.legend()
    plt.tight_layout()
    path = out_dir / "jordan_rope_basis.png"
    plt.savefig(path, dpi=160)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
