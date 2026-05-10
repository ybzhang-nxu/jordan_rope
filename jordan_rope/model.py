from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .positional import (
    ALiBiBias,
    DirectSumRoPEUnipotent,
    JordanConfig,
    JordanRoPE,
    RotaryEmbedding,
    _parse_exact_scaled_c,
)


def _parse_method_eta_init(method: str) -> float | None:
    for token in method.split("_"):
        if token.startswith("eta") and token[3:].isdigit() and len(token[3:]) >= 3:
            return int(token[3:]) / 1000.0
    return None


def _parse_method_order(method: str, *, default: int = 2) -> int:
    for token in method.split("_"):
        if token.startswith("m") and token[1:].isdigit():
            return int(token[1:])
    return default


def _parse_method_jet_gates(method: str, *, count: int) -> tuple[float, ...]:
    gates = []
    for token in method.split("_"):
        if token.startswith("g") and token[1:].isdigit():
            gates.append(int(token[1:]) / 100.0)
    if len(gates) != count:
        raise ValueError(
            f"Expected {count} jet gate tokens like g050 in position_method={method}, got {len(gates)}."
        )
    return tuple(gates)


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    position_method: str = "jordan_rope"
    theta: float = 10000.0
    gamma_min: float = 1e-4
    init_gamma: float = 1e-4
    eta_max: float = 0.1
    init_eta: float = 0.0
    train_context: int = 1024
    bounded_tau: bool = True
    shear_time_scale: float = 1.0
    max_exponent: float = 30.0

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        return self.d_model // self.n_heads

    def jordan_config(self) -> JordanConfig:
        return JordanConfig(
            theta=self.theta,
            gamma_min=self.gamma_min,
            init_gamma=self.init_gamma,
            eta_max=self.eta_max,
            init_eta=self.init_eta,
            train_context=self.train_context,
            bounded_tau=self.bounded_tau,
            shear_time_scale=self.shear_time_scale,
            max_exponent=self.max_exponent,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        method = config.position_method
        self.positioner: nn.Module | None = None
        self.alibi: ALiBiBias | None = None
        if method == "nope":
            pass
        elif method == "rope":
            self.positioner = RotaryEmbedding(self.head_dim, theta=config.theta)
        elif method == "alibi":
            self.alibi = ALiBiBias(config.n_heads)
        elif method == "rope_alibi":
            self.positioner = RotaryEmbedding(self.head_dim, theta=config.theta)
            self.alibi = ALiBiBias(config.n_heads)
        elif method == "damped_rope":
            self.positioner = JordanRoPE(
                self.head_dim,
                config.n_heads,
                config=config.jordan_config(),
                learn_gamma=True,
                learn_eta=False,
                force_zero_eta=True,
            )
        elif method == "real_jordan":
            self.positioner = JordanRoPE(
                self.head_dim,
                config.n_heads,
                config=config.jordan_config(),
                learn_gamma=True,
                learn_eta=True,
                force_zero_omega=True,
            )
        elif method == "direct_sum":
            self.positioner = DirectSumRoPEUnipotent(
                self.head_dim, config.n_heads, config=config.jordan_config()
            )
        elif method == "jordan_rope":
            self.positioner = JordanRoPE(
                self.head_dim, config.n_heads, config=config.jordan_config(), learn_gamma=True, learn_eta=True
            )
        elif method.startswith("jordan_eta"):
            suffix = method.removeprefix("jordan_eta")
            try:
                eta = float(suffix) / 1000.0
            except ValueError as exc:
                raise ValueError(f"Could not parse eta init from position_method={method}") from exc
            base = config.jordan_config()
            eta_cfg = JordanConfig(**{**base.__dict__, "init_eta": eta})
            self.positioner = JordanRoPE(
                self.head_dim, config.n_heads, config=eta_cfg, learn_gamma=True, learn_eta=True
            )
        elif method.startswith("jordan_exact"):
            c_value = _parse_exact_scaled_c(method)
            base = config.jordan_config()
            eta_init = _parse_method_eta_init(method)
            order = _parse_method_order(method, default=2)
            eta_max = 1.0 if method.endswith("_eta1") or "_eta1_" in method else base.eta_max
            exact_cfg = JordanConfig(
                **{
                    **base.__dict__,
                    "bounded_tau": False,
                    "shear_time_scale": 1.0 / float(max(base.train_context, 1)),
                    "gamma_min": 0.0,
                    "init_gamma": c_value / float(max(base.train_context, 1)),
                    "init_eta": eta_init if eta_init is not None else base.init_eta,
                    "eta_max": eta_max,
                }
            )
            self.positioner = JordanRoPE(
                self.head_dim,
                config.n_heads,
                config=exact_cfg,
                learn_gamma=True,
                learn_eta=True,
                order=order,
            )
        elif method.startswith("jordan_jetmix"):
            c_value = _parse_exact_scaled_c(method)
            base = config.jordan_config()
            order = _parse_method_order(method, default=4)
            include_base = method.startswith("jordan_jetmixfull")
            jet_gates = _parse_method_jet_gates(method, count=order if include_base else order - 1)
            jet_base = jet_gates[0] if include_base else None
            jet_coefficients = jet_gates[1:] if include_base else jet_gates
            jet_cfg = JordanConfig(
                **{
                    **base.__dict__,
                    "shear_time_scale": 1.0 / float(max(base.train_context, 1)),
                    "gamma_min": 0.0,
                    "init_gamma": c_value / float(max(base.train_context, 1)),
                }
            )
            self.positioner = JordanRoPE(
                self.head_dim,
                config.n_heads,
                config=jet_cfg,
                learn_gamma=True,
                learn_eta=False,
                force_zero_eta=True,
                order=order,
                jet_coefficients=jet_coefficients,
                jet_base_coefficient=jet_base,
                jet_coefficient_max=2.0 if include_base else 1.0,
            )
        elif method in {"jordan_m3", "jordan_m4"}:
            order = int(method.removeprefix("jordan_m"))
            self.positioner = JordanRoPE(
                self.head_dim,
                config.n_heads,
                config=config.jordan_config(),
                learn_gamma=True,
                learn_eta=True,
                order=order,
            )
        elif method == "jordan_no_gamma":
            self.positioner = JordanRoPE(
                self.head_dim,
                config.n_heads,
                config=config.jordan_config(),
                learn_gamma=False,
                learn_eta=True,
                force_zero_gamma=True,
            )
        elif method == "jordan_raw_tau":
            raw_cfg = config.jordan_config()
            raw_cfg = JordanConfig(**{**raw_cfg.__dict__, "bounded_tau": False})
            self.positioner = JordanRoPE(self.head_dim, config.n_heads, config=raw_cfg)
        else:
            raise ValueError(f"Unknown position_method: {method}")

        if method.endswith("_alibi") and method not in {"alibi", "rope_alibi"}:
            self.alibi = ALiBiBias(config.n_heads)

    def project_qkv(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        bsz, length, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q0 = q.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        k0 = k.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if positions is None:
            positions = torch.arange(length, device=x.device)
        q_pos, k_pos = q0, k0
        if self.positioner is not None:
            q_pos, k_pos = self.positioner.apply(q0, k0, positions)

        return {
            "q0": q0,
            "k0": k0,
            "v0": v,
            "q_pos": q_pos,
            "k_pos": k_pos,
            "positions": positions,
        }

    def attention_scores(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        length = q.shape[-2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.alibi is not None:
            scores = scores + self.alibi(positions).to(device=q.device, dtype=scores.dtype)

        causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
        return scores

    def attend_projected(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        q = tensors["q_pos"]
        k = tensors["k_pos"]
        v = tensors["v0"]
        positions = tensors["positions"]
        bsz, _, length, _ = q.shape
        dim = self.config.d_model
        scores = self.attention_scores(q, k, positions)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(bsz, length, dim)
        return self.proj(y)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        tensors = self.project_qkv(x, positions)
        return self.attend_projected(tensors)

    def forward_with_k_quantizer(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None,
        quantize_k,
    ) -> torch.Tensor:
        return self.forward_with_cache_quantizer(x, positions, quantize_k=quantize_k)

    def forward_with_cache_quantizer(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None,
        *,
        quantize_k=None,
        quantize_v=None,
    ) -> torch.Tensor:
        tensors = self.project_qkv(x, positions)
        if quantize_k is not None:
            tensors = {**tensors, "k_pos": quantize_k(tensors["k_pos"])}
        if quantize_v is not None:
            tensors = {**tensors, "v0": quantize_v(tensors["v0"])}
        return self.attend_projected(tensors)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.d_model
        self.net = nn.Sequential(
            nn.Linear(config.d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        x = x + self.ff(self.norm2(x))
        return x

    def forward_with_k_quantizer(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None,
        quantize_k,
    ) -> torch.Tensor:
        return self.forward_with_cache_quantizer(x, positions, quantize_k=quantize_k)

    def forward_with_cache_quantizer(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None,
        *,
        quantize_k=None,
        quantize_v=None,
    ) -> torch.Tensor:
        x = x + self.attn.forward_with_cache_quantizer(
            self.norm1(x),
            positions,
            quantize_k=quantize_k,
            quantize_v=quantize_v,
        )
        x = x + self.ff(self.norm2(x))
        return x


class CausalTransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.drop(self.embed(tokens))
        for block in self.blocks:
            x = block(x, positions)
        x = self.norm(x)
        return self.lm_head(x)

    def forward_with_k_quantizer(self, tokens: torch.Tensor, quantize_k) -> torch.Tensor:
        return self.forward_with_cache_quantizer(tokens, quantize_k=quantize_k)

    def forward_with_cache_quantizer(
        self,
        tokens: torch.Tensor,
        *,
        quantize_k=None,
        quantize_v=None,
    ) -> torch.Tensor:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        x = self.drop(self.embed(tokens))
        for layer_id, block in enumerate(self.blocks):
            x = block.forward_with_cache_quantizer(
                x,
                positions,
                quantize_k=None if quantize_k is None else lambda k_pos, layer_id=layer_id: quantize_k(layer_id, k_pos),
                quantize_v=None if quantize_v is None else lambda v0, layer_id=layer_id: quantize_v(layer_id, v0),
            )
        x = self.norm(x)
        return self.lm_head(x)

    def extract_attention_tensors(
        self,
        tokens: torch.Tensor,
        *,
        layers: set[int] | list[int] | None = None,
        detach: bool = True,
    ) -> list[dict[str, torch.Tensor | int]]:
        _, length = tokens.shape
        positions = torch.arange(length, device=tokens.device)
        selected = set(range(len(self.blocks))) if layers is None else set(int(layer) for layer in layers)
        x = self.drop(self.embed(tokens))
        records: list[dict[str, torch.Tensor | int]] = []
        for layer_id, block in enumerate(self.blocks):
            h = block.norm1(x)
            tensors = block.attn.project_qkv(h, positions)
            if layer_id in selected:
                record: dict[str, torch.Tensor | int] = {"layer": layer_id}
                for key in ("q0", "k0", "v0", "q_pos", "k_pos", "positions"):
                    value = tensors[key]
                    record[key] = value.detach() if detach else value
                records.append(record)
            x = x + block.attn.attend_projected(tensors)
            x = x + block.ff(block.norm2(x))
        return records
