# Jordan-RoPE

Minimal reproduction code for **Jordan-RoPE**, a non-semisimple extension of
rotary relative positional encoding. The core object couples a rotary complex
phase with a nilpotent/Jordan response, giving primitive relative-position
features such as

```text
d * exp(i omega d),  d * exp(-gamma d) * cos(omega d)
```

This repository contains code, configs, and tests only. Paper sources, local
experiment outputs, and training logs are intentionally excluded from the
minimal GitHub version.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

The optional WikiText loader uses `datasets`, which is already listed in
`requirements.txt`. The small smoke LM can also run from `data/smoke_lm.txt`.

## Core Methods

The main implementation is in `jordan_rope/positional.py`.

- `rope`: standard RoPE.
- `alibi`: additive linear distance bias.
- `rope_alibi`: RoPE plus ALiBi-style distance channel.
- `damped_rope`: RoPE with learned damping.
- `direct_sum`: separate RoPE and real unipotent/Jordan subspaces.
- `jordan_rope`: **Stabilized Jordan-RoPE**, the bounded-shear default.
- `jordan_raw_tau`: exact/raw Jordan shear without bounded `tau`.
- `jordan_exact_scaled`: **Scaled-exact Jordan-RoPE** with `c=1`.
- `jordan_exact_c010`: scaled-exact variant with `c=0.1`.
- `jordan_no_gamma`, `jordan_m3`, `jordan_eta*`: ablations.

`JordanRoPE.apply(q, k, positions)` expects tensors shaped
`B x H x T x d_head`. The order-2 realified block requires `d_head % 4 == 0`.

## Quick Checks

Run unit tests:

```bash
pytest -q
```

Run a tiny toy diagnostic:

```bash
python3 scripts/run_toy.py \
  --config configs/smoke.yaml \
  --out-dir runs/smoke_toy
```

Run the Jordan-friendly synthetic LM smoke benchmark:

```bash
python3 scripts/run_kernel_lm.py \
  --config configs/kernel_lm_smoke.yaml \
  --out-dir runs/kernel_lm_smoke
```

Run a small local byte-LM smoke benchmark:

```bash
python3 scripts/run_lm.py \
  --config configs/smoke.yaml \
  --out-dir runs/smoke_lm \
  --methods jordan_rope rope
```

Generated outputs go under `runs/`, which is ignored by git.

## Longer Reproduction Runs

The configs used for the larger local experiments are included:

```bash
python3 scripts/run_toy.py --config configs/full_research.yaml --out-dir runs/toy
python3 scripts/run_structured_benchmarks.py --config configs/structured_benchmarks.yaml --out-dir runs/structured_benchmarks
python3 scripts/run_kernel_lm.py --config configs/kernel_lm.yaml --out-dir runs/kernel_lm
python3 scripts/run_lm.py --config configs/lm_focused.yaml --out-dir runs/lm_focused
```

These runs can be slow on CPU. GPU is recommended for the Transformer training
experiments.

## Repository Layout

```text
jordan_rope/     model and positional encoding implementations
scripts/         experiment runners
configs/         smoke and research configs
tests/           unit tests
data/            tiny local text file for smoke LM runs
```

## Notes

The non-orthogonal Jordan block uses a contragredient query action internally:
queries use the inverse-transpose positional action and keys use the primal
action. This is necessary for exact/raw Jordan blocks to induce a pure relative
score. The default stabilized variant bounds only the nilpotent shear time via
`tau(d) = d / (1 + d / L)`, which improves numerical behavior but is not an
exact one-parameter representation.
