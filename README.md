# Jordan-RoPE

Minimal reproduction code for **Jordan-RoPE**, a non-semisimple extension of
rotary relative positional encoding. The core object couples a rotary complex
phase with a nilpotent/Jordan response, giving primitive relative-position
features such as

```text
d * exp(i omega d),  d * exp(-gamma d) * cos(omega d)
```

This repository contains reproduction code, configs, and tests only. Paper
sources, local experiment plans, generated plots, checkpoints, downloaded
datasets, and training logs are intentionally excluded from the minimal GitHub
version.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

The optional WikiText loader uses `datasets`, which is already listed in
`requirements.txt`. The small smoke LM can also run from `data/smoke_lm.txt`.

For editable installs and optional extras:

```bash
python3 -m pip install -e ".[lm]"
python3 -m pip install -e ".[audio]"   # only needed for music/audio tokenization
python3 -m pip install -e ".[dev]"     # only needed for tests
```

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
- `jordan_exact_c150_eta010`: scaled-exact variant with parsed damping and
  initial shear suffixes.
- `jordan_m3`, `jordan_m4`, `jordan_exact_m{m}_...`: higher-order complex
  Jordan chains. An order-`m` realified block requires `d_head % (2*m) == 0`.
- `jordan_jetmix_m4_c150_g030_g010_g005`: independent high-order jet
  coefficients with implicit base coefficient `alpha0=1`.
- `jordan_jetmixfull_m4_c150_g100_g050_g030_g020`: full jet spectrum including
  `alpha0`.
- Any Jordan or jetmix method may be suffixed with `_alibi`, for example
  `jordan_jetmix_m3_c150_g010_g005_alibi`, to compose the Jordan/jet bilinear
  positional form with the same additive ALiBi bias used in `rope_alibi`.
- `jordan_no_gamma`, `jordan_eta*`: ablations.

`JordanRoPE.apply(q, k, positions)` expects tensors shaped
`B x H x T x d_head`. The implementation uses the contragredient query action
and the primal key action so that exact/raw Jordan blocks induce a relative
score.

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

The configs used for the larger local experiments are included. These runs can
be slow on CPU; GPU is recommended for the Transformer training experiments.

```bash
python3 scripts/run_toy.py --config configs/full_research.yaml --out-dir runs/toy
python3 scripts/run_structured_benchmarks.py --config configs/structured_benchmarks.yaml --out-dir runs/structured_benchmarks
python3 scripts/run_kernel_lm.py --config configs/kernel_lm.yaml --out-dir runs/kernel_lm
python3 scripts/run_lm.py --config configs/lm_focused.yaml --out-dir runs/lm_focused
```

### High-order jet probes

Run the high-order synthetic kernel-LM smoke check:

```bash
python3 scripts/run_kernel_lm.py \
  --config configs/high_jet_kernel_lm_smoke.yaml \
  --out-dir runs/high_jet_kernel_lm_smoke
```

Run the focused WikiText high-jet + ALiBI reproduction matrix:

```bash
python3 scripts/run_lm.py \
  --config configs/high_jet_wikitext103_jetmix_alibi_1000.yaml \
  --out-dir runs/high_jet_wikitext103_jetmix_alibi_1000

python3 scripts/summarize_lm.py \
  --inputs runs/high_jet_wikitext103_jetmix_alibi_1000/lm_eval.csv \
  --names wikitext_jetmix_alibi_1000 \
  --out-json runs/high_jet_wikitext103_jetmix_alibi_1000/summary.json \
  --out-md runs/high_jet_wikitext103_jetmix_alibi_1000/summary.md
```

This run trains a small byte-level WikiText-103 model at length 512 and
evaluates through length 8192. It compares `rope_alibi` against weak
Jordan/jet-spectrum corrections composed with the same ALiBi bias.

### Music/audio token transfer

Download the small default music/audio sources:

```bash
scripts/download_music_audio_datasets.sh start
scripts/download_music_audio_datasets.sh status
scripts/download_music_audio_datasets.sh verify
```

Prepare byte-token streams:

```bash
python3 scripts/prepare_audio_lm_tokens.py \
  --download-dir data/downloads \
  --out-dir data/processed/audio_lm_smoke \
  --max-maestro-files 32 \
  --max-fma-tracks 32 \
  --max-musicnet-wavs 16
```

Then run a smoke music-token LM:

```bash
python3 scripts/run_lm.py \
  --config configs/audio_lm_smoke_musicnet.yaml \
  --out-dir runs/audio_lm_smoke_musicnet
```

Downloaded archives and processed tokens live under `data/downloads/` and
`data/processed/`, both ignored by git.

### Cache quantization diagnostics

The repository also includes the code path used for K-cache/logit-side
diagnostics:

```bash
python3 scripts/export_attention_tensors.py --help
python3 scripts/run_quant_metrics.py --help
python3 scripts/run_jordan_mode_metrics.py --help
python3 scripts/run_tq_bias_check.py --help
```

Quantization method names are intentionally explicit: rotated-uniform baselines
use names such as `scalar_uniform_no_rotation`, `dense_rot_uniform`, and
`kac_rot_uniform`; TurboQuant-style names are reserved for the implemented
Lloyd-Max/codebook and QJL residual checks.

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
