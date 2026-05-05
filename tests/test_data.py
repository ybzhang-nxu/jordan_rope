from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from jordan_rope.data import generate_retrieval_batch, retrieval_vocab_size


def test_generate_retrieval_batch_shapes():
    device = torch.device("cpu")
    x, y, distances = generate_retrieval_batch(
        batch_size=3,
        seq_len=32,
        num_pairs=4,
        num_keys=16,
        num_values=16,
        device=device,
        target_distance=12,
    )
    assert x.shape == (3, 32)
    assert y.shape == (3,)
    assert distances.shape == (3,)
    assert int(x.max()) < retrieval_vocab_size(16, 16)
