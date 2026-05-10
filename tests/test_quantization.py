import pytest

torch = pytest.importorskip("torch")

from jordan_rope.model import CausalTransformerLM, TransformerConfig
from jordan_rope.diagnostics import p0_jordan_logit_components
from jordan_rope.quantization import (
    KacRotation,
    dense_random_orthogonal,
    norm_growth,
    position_norm_profile,
    rotate_quantize_dequantize,
    rotate_mixed_bit_quantize_dequantize,
    turbo_mse_codebook,
    turbo_mse_quantize_dequantize,
    turbo_product_inner_product,
    turbo_product_quantize,
    uniform_quantize_dequantize,
)
from scripts.run_quant_metrics import causal_attention_metric_values, causal_metric_values_by_head
from scripts.run_cache_output_metrics import output_metric_values, select_query_rows


def test_uniform_quantizer_shape_and_bit_accounting():
    x = torch.randn(2, 3, 5, 8)
    x_hat, bits = uniform_quantize_dequantize(x, bits=3, scale_axis="vector")
    assert x_hat.shape == x.shape
    assert bits.b_numeric == 3.0
    assert bits.b_storage > bits.b_numeric
    assert bits.num_scalars == x.numel()


def test_dense_random_orthogonal_preserves_norm():
    x = torch.randn(4, 16)
    rot = dense_random_orthogonal(16, seed=7, dtype=x.dtype)
    y = torch.matmul(x, rot.t())
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_kac_rotation_inverse_and_norm_preservation():
    x = torch.randn(2, 3, 11, 16)
    rotation = KacRotation.random(16, depth=8, seed=19, dtype=x.dtype)
    y = rotation.apply(x)
    x_roundtrip = rotation.apply(y, inverse=True)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    assert torch.allclose(x, x_roundtrip, atol=1e-5)


def test_rotated_quantize_dequantize_shape():
    x = torch.randn(2, 4, 9, 8)
    rotation = KacRotation.random(8, depth=4, seed=23, dtype=x.dtype)
    x_hat, bits = rotate_quantize_dequantize(x, bits=2, rotation=rotation)
    assert x_hat.shape == x.shape
    assert bits.b_numeric == 2.0


def test_rotated_mixed_bit_quantize_dequantize_accounting():
    x = torch.randn(2, 4, 9, 8)
    rotation = KacRotation.random(8, depth=4, seed=29, dtype=x.dtype)
    high_mask = torch.tensor([True, True, False, False, False, False, False, False])
    x_hat, bits = rotate_mixed_bit_quantize_dequantize(
        x,
        low_bits=2,
        high_bits=4,
        high_mask=high_mask,
        rotation=rotation,
    )
    assert x_hat.shape == x.shape
    assert bits.b_numeric == 2.5
    assert bits.b_storage > bits.b_numeric


def test_turbo_mse_codebook_and_quantizer_accounting():
    codebook = turbo_mse_codebook(2, 16)
    assert codebook.shape == (4,)
    assert torch.all(codebook[:-1] < codebook[1:])
    assert torch.allclose(codebook, -codebook.flip(0), atol=1e-5)

    x = torch.randn(2, 3, 5, 16)
    rotation = KacRotation.random(16, depth=4, seed=37, dtype=x.dtype)
    x_hat, bits = turbo_mse_quantize_dequantize(x, bits=2, rotation=rotation)
    assert x_hat.shape == x.shape
    assert bits.b_numeric == 2.0
    assert bits.b_storage > bits.b_numeric


def test_turbo_product_qjl_bias_is_small_on_random_pairs():
    dim = 16
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(256, dim, generator=generator)
    y = torch.randn(256, dim, generator=generator)
    rotation = KacRotation.random(dim, depth=4, seed=41, dtype=x.dtype)
    estimates = []
    for seed in range(17, 25):
        encoded = turbo_product_quantize(
            x,
            total_bits=3,
            rotation=rotation,
            qjl_seed=seed,
            qjl_rows=dim,
        )
        estimates.append(turbo_product_inner_product(y, encoded))
        accounting = encoded.bit_accounting()
        assert accounting.b_numeric == 3.0
        assert accounting.b_storage > accounting.b_numeric
    true = (x * y).sum(dim=-1)
    error = torch.stack(estimates).mean(dim=0) - true
    rmse = error.square().mean().sqrt()
    assert error.mean().abs() <= 0.10 * rmse


def test_head_specific_mixed_bit_mask_accounting():
    x = torch.randn(2, 4, 9, 8)
    rotation = KacRotation.random(8, depth=4, seed=31, dtype=x.dtype)
    high_mask = torch.zeros(4, 8, dtype=torch.bool)
    high_mask[0, [0, 1]] = True
    high_mask[1, [2, 3]] = True
    high_mask[2, [4, 5]] = True
    high_mask[3, [6, 7]] = True
    x_hat, bits = rotate_mixed_bit_quantize_dequantize(
        x,
        low_bits=2,
        high_bits=4,
        high_mask=high_mask,
        rotation=rotation,
    )
    assert x_hat.shape == x.shape
    assert bits.b_numeric == 2.5
    assert bits.metadata_bits > high_mask.numel()
    assert bits.b_storage > bits.b_numeric


def test_p1_attention_and_per_head_metrics_identity():
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    attn_metrics = causal_attention_metric_values(q, k, k, top_k=3)
    head_metrics = causal_metric_values_by_head(q, k, k, top_k=3)
    assert attn_metrics["attention_kl"] == pytest.approx(0.0, abs=1e-6)
    assert attn_metrics["top1_agreement"] == pytest.approx(1.0, abs=1e-6)
    assert attn_metrics["top3_agreement"] == pytest.approx(1.0, abs=1e-6)
    assert head_metrics["attention_kl"].shape == (3,)
    assert torch.allclose(head_metrics["logit_mse"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(head_metrics["top1_agreement"], torch.ones(3), atol=1e-6)


def test_cache_output_metrics_identity_and_value_perturbation():
    q = torch.randn(2, 3, 6, 4)
    k = torch.randn(2, 3, 6, 4)
    v = torch.randn(2, 3, 6, 4)
    query_rows = select_query_rows(length=6, anchors=[2, 5], random_count=0, seed=17)
    identity = output_metric_values(q, k, v, k, v, query_rows)
    assert identity["output_mse"].shape == (3,)
    assert torch.allclose(identity["output_mse"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(identity["relative_output_mse"], torch.zeros(3), atol=1e-6)

    perturbed = output_metric_values(q, k, v, k, v + 0.1, query_rows)
    assert torch.all(perturbed["output_mse"] > 0)


def test_position_norm_profile_and_growth():
    x = torch.ones(2, 3, 4, 5)
    profile = position_norm_profile(x)
    assert profile.shape == (4,)
    assert torch.allclose(profile, torch.full((4,), 5**0.5))
    assert torch.allclose(norm_growth(profile), torch.tensor(1.0))


def test_model_extract_attention_tensors_shapes_and_forward_still_runs():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        position_method="jordan_rope",
        train_context=16,
    )
    model = CausalTransformerLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        logits = model(tokens)
        records = model.extract_attention_tensors(tokens, layers=[0])
    assert logits.shape == (2, 12, cfg.vocab_size)
    assert len(records) == 1
    record = records[0]
    assert record["q0"].shape == (2, cfg.n_heads, 12, cfg.head_dim)
    assert record["k0"].shape == (2, cfg.n_heads, 12, cfg.head_dim)
    assert record["v0"].shape == (2, cfg.n_heads, 12, cfg.head_dim)
    assert record["q_pos"].shape == (2, cfg.n_heads, 12, cfg.head_dim)
    assert record["k_pos"].shape == (2, cfg.n_heads, 12, cfg.head_dim)


def test_model_forward_with_cache_quantizer_identity_matches_forward():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=2,
        position_method="jordan_rope",
        train_context=16,
    )
    model = CausalTransformerLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 12))
    with torch.no_grad():
        base = model(tokens)
        quantized = model.forward_with_cache_quantizer(
            tokens,
            quantize_k=lambda _layer_id, x: x,
            quantize_v=lambda _layer_id, x: x,
        )
    assert torch.allclose(base, quantized, atol=1e-6)


def test_p0_jordan_components_reconstruct_positioned_logits_for_jordan_rope():
    cfg = TransformerConfig(
        vocab_size=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        position_method="jordan_rope",
        train_context=16,
    )
    model = CausalTransformerLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (2, 10))
    with torch.no_grad():
        record = model.extract_attention_tensors(tokens, layers=[0])[0]
    positioner = model.blocks[0].attn.positioner
    comps = p0_jordan_logit_components(positioner, record["q0"], record["k0"])
    positioned = torch.matmul(record["q_pos"], record["k_pos"].transpose(-2, -1))
    causal = torch.ones(10, 10, dtype=torch.bool).tril()
    assert torch.allclose(comps["total"][..., causal], positioned[..., causal], atol=1e-4)
    assert torch.allclose(comps["base"][..., causal] + comps["jordan"][..., causal], comps["total"][..., causal], atol=1e-5)
