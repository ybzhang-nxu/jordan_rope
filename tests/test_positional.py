from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from jordan_rope.model import CausalTransformerLM, TransformerConfig
from jordan_rope.positional import JordanConfig, JordanRoPE, causal_delta_features


def test_jordan_rope_apply_shape_and_backward():
    module = JordanRoPE(8, 2, JordanConfig(train_context=16))
    q = torch.randn(2, 2, 5, 8, requires_grad=True)
    k = torch.randn(2, 2, 5, 8, requires_grad=True)
    q2, k2 = module.apply(q, k, torch.arange(5))
    assert q2.shape == q.shape
    assert k2.shape == k.shape
    loss = (q2.square().mean() + k2.square().mean())
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None


@pytest.mark.parametrize(
    "method",
    [
        "nope",
        "rope",
        "alibi",
        "rope_alibi",
        "damped_rope",
        "real_jordan",
        "direct_sum",
        "jordan_rope",
        "jordan_exact_scaled",
        "jordan_exact_c010",
        "jordan_exact_c010_eta1",
        "jordan_no_gamma",
        "jordan_raw_tau",
    ],
)
def test_transformer_forward_for_all_position_methods(method: str):
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method=method,
        train_context=16,
    )
    model = CausalTransformerLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, cfg.vocab_size)


def test_transformer_forward_for_order3_ablation():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=24,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method="jordan_m3",
        train_context=16,
    )
    model = CausalTransformerLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, cfg.vocab_size)


def test_toy_features_include_jordan_tangent_modes():
    deltas = torch.arange(0, 16)
    features = causal_delta_features(
        "jordan_rope",
        deltas,
        num_freqs=3,
        head_dim=12,
        train_context=16,
    )
    # 1 intercept + cos/sin + tau*cos/tau*sin for each frequency.
    assert features.shape == (16, 1 + 4 * 3)
    assert torch.isfinite(features).all()


def test_exact_scaled_features_use_normalized_exact_decay():
    deltas = torch.tensor([0, 16, 32])
    train_context = 16
    features = causal_delta_features(
        "jordan_exact_scaled",
        deltas,
        num_freqs=1,
        head_dim=4,
        train_context=train_context,
    )
    expected_decay = torch.exp(-deltas.float() / train_context)
    assert torch.allclose(features[:, 1], expected_decay * torch.cos(deltas.float()), atol=1e-6)
    assert torch.allclose(features[:, 3], (deltas.float() / train_context) * expected_decay * torch.cos(deltas.float()), atol=1e-6)
