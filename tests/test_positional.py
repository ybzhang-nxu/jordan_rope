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
        "jordan_exact_c050_eta005",
        "jordan_exact_m4_c010_eta010",
        "jordan_exact_c150_eta010_alibi",
        "jordan_jetmix_m4_c150_g050_g030_g020",
        "jordan_jetmix_m4_c150_g050_g030_g020_alibi",
        "jordan_jetmixfull_m4_c150_g100_g050_g030_g020",
        "jordan_jetmixfull_m4_c150_g100_g050_g030_g020_alibi",
        "jordan_no_gamma",
        "jordan_raw_tau",
        "jordan_m4",
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


def test_exact_method_suffix_sets_c_and_eta_init():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method="jordan_exact_c050_eta005",
        train_context=20,
    )
    model = CausalTransformerLM(cfg)
    positioner = model.blocks[0].attn.positioner
    assert isinstance(positioner, JordanRoPE)
    assert torch.allclose(positioner.gamma(), torch.full_like(positioner.gamma(), 0.5 / 20), atol=1e-6)
    assert torch.allclose(positioner.eta(), torch.full_like(positioner.eta(), 0.005), atol=1e-6)


def test_jetmix_method_suffix_sets_independent_gates():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method="jordan_jetmix_m4_c150_g050_g030_g020",
        train_context=20,
    )
    model = CausalTransformerLM(cfg)
    positioner = model.blocks[0].attn.positioner
    assert isinstance(positioner, JordanRoPE)
    assert torch.allclose(positioner.gamma(), torch.full_like(positioner.gamma(), 1.5 / 20), atol=1e-6)
    coeffs = positioner.jet_coefficients()
    assert coeffs is not None
    expected = torch.tensor([0.5, 0.3, 0.2], dtype=coeffs.dtype).view(3, 1, 1)
    assert torch.allclose(coeffs, expected.expand_as(coeffs), atol=1e-6)


def test_jetmixfull_method_suffix_sets_full_spectrum():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method="jordan_jetmixfull_m4_c150_g100_g050_g030_g020",
        train_context=20,
    )
    model = CausalTransformerLM(cfg)
    positioner = model.blocks[0].attn.positioner
    assert isinstance(positioner, JordanRoPE)
    spectrum = positioner.jet_spectrum()
    assert spectrum is not None
    expected = torch.tensor([1.0, 0.5, 0.3, 0.2], dtype=spectrum.dtype).view(4, 1, 1)
    assert torch.allclose(spectrum, expected.expand_as(spectrum), atol=1e-5)


def test_jordan_alibi_suffix_adds_bias_without_changing_positioner():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method="jordan_jetmixfull_m4_c150_g100_g050_g030_g020_alibi",
        train_context=20,
    )
    model = CausalTransformerLM(cfg)
    attn = model.blocks[0].attn
    assert isinstance(attn.positioner, JordanRoPE)
    assert attn.alibi is not None
    spectrum = attn.positioner.jet_spectrum()
    assert spectrum is not None
    expected = torch.tensor([1.0, 0.5, 0.3, 0.2], dtype=spectrum.dtype).view(4, 1, 1)
    assert torch.allclose(spectrum, expected.expand_as(spectrum), atol=1e-5)


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


def test_transformer_forward_for_order4_ablation():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method="jordan_m4",
        train_context=16,
    )
    model = CausalTransformerLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    logits = model(tokens)
    assert logits.shape == (2, 12, cfg.vocab_size)


def test_transformer_forward_for_exact_order3_ablation():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=24,
        n_heads=4,
        n_layers=1,
        mlp_ratio=2,
        position_method="jordan_exact_m3_c010_eta010",
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


def test_high_order_jet_features_are_finite():
    deltas = torch.arange(0, 16)
    features = causal_delta_features(
        "jordan_m4",
        deltas,
        num_freqs=3,
        head_dim=24,
        train_context=16,
    )
    # 1 intercept + order-4 cos/sin jet pairs for each frequency.
    assert features.shape == (16, 1 + 8 * 3)
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
