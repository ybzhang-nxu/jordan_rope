from __future__ import annotations

import random
from pathlib import Path

import torch


PAD = 0
BOS = 1
QUERY = 2
NOISE = 3


def retrieval_vocab_size(num_keys: int, num_values: int) -> int:
    return 4 + num_keys + num_values


def key_token(key: int) -> int:
    return 4 + key


def value_token(num_keys: int, value: int) -> int:
    return 4 + num_keys + value


def generate_retrieval_batch(
    *,
    batch_size: int,
    seq_len: int,
    num_pairs: int,
    num_keys: int,
    num_values: int,
    device: torch.device,
    target_distance: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate key-value retrieval examples.

    The input sequence ends with [QUERY, target_key]. The label is the value
    associated with that key earlier in the context, predicted from the final
    position logits.
    """
    if seq_len < 8:
        raise ValueError("seq_len must be at least 8.")
    if num_pairs > num_keys:
        raise ValueError("num_pairs cannot exceed num_keys when unique keys are used.")

    tokens = torch.full((batch_size, seq_len), PAD, dtype=torch.long, device=device)
    labels = torch.empty((batch_size,), dtype=torch.long, device=device)
    distances = torch.empty((batch_size,), dtype=torch.long, device=device)
    max_pair_start = seq_len - 4

    for b in range(batch_size):
        tokens[b, 0] = BOS
        keys = random.sample(range(num_keys), k=num_pairs)
        values = [random.randrange(num_values) for _ in range(num_pairs)]
        if target_distance is None:
            target_start = random.randrange(1, max_pair_start)
        else:
            target_start = max(1, min(max_pair_start - 1, seq_len - 1 - target_distance))

        occupied = {0, seq_len - 2, seq_len - 1, target_start, target_start + 1}
        starts = [target_start]
        candidates = list(range(1, max_pair_start))
        random.shuffle(candidates)
        for cand in candidates:
            if len(starts) == num_pairs:
                break
            if cand in occupied or cand + 1 in occupied:
                continue
            occupied.add(cand)
            occupied.add(cand + 1)
            starts.append(cand)
        if len(starts) < num_pairs:
            raise RuntimeError("Could not place all retrieval pairs; reduce num_pairs or increase seq_len.")

        for idx, start in enumerate(starts):
            k = keys[idx]
            v = values[idx]
            tokens[b, start] = key_token(k)
            tokens[b, start + 1] = value_token(num_keys, v)

        target_key = keys[0]
        target_value = values[0]
        tokens[b, seq_len - 2] = QUERY
        tokens[b, seq_len - 1] = key_token(target_key)
        labels[b] = value_token(num_keys, target_value)
        distances[b] = seq_len - 1 - target_start

        noise_mask = tokens[b] == PAD
        noise = torch.randint(0, 4, size=(int(noise_mask.sum()),), device=device)
        tokens[b, noise_mask] = noise

    return tokens, labels, distances


def load_text_as_bytes(path: str | Path) -> torch.Tensor:
    data = Path(path).read_bytes()
    # Reserve token 0 for padding if needed; byte values become 1..256.
    values = [b + 1 for b in data]
    return torch.tensor(values, dtype=torch.long)


def load_hf_text_dataset(dataset_name: str, subset: str, split: str) -> torch.Tensor:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The `datasets` package is required for Hugging Face datasets. Install with "
            "`python3 -m pip install -r requirements.txt`, or pass a local text file."
        ) from exc
    ds = load_dataset(dataset_name, subset, split=split)
    text = "\n".join(str(row.get("text", "")) for row in ds)
    return torch.tensor([b + 1 for b in text.encode("utf-8")], dtype=torch.long)


def sample_lm_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= seq_len + 1:
        raise ValueError("Not enough tokens for the requested seq_len.")
    starts = torch.randint(0, tokens.numel() - seq_len - 1, (batch_size,))
    return sample_lm_batch_at_starts(tokens, starts, seq_len, device)


def sample_lm_batch_at_starts(
    tokens: torch.Tensor, starts: torch.Tensor, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.numel() <= seq_len + 1:
        raise ValueError("Not enough tokens for the requested seq_len.")
    starts = starts.to(torch.long).cpu()
    xs = [tokens[s : s + seq_len] for s in starts]
    ys = [tokens[s + 1 : s + seq_len + 1] for s in starts]
    return torch.stack(xs).to(device), torch.stack(ys).to(device)
