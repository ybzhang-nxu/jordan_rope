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
            eta_max = 1.0 if method.endswith("_eta1") or "_eta1_" in method else base.eta_max
            exact_cfg = JordanConfig(
                **{
                    **base.__dict__,
                    "bounded_tau": False,
                    "shear_time_scale": 1.0 / float(max(base.train_context, 1)),
                    "gamma_min": 0.0,
                    "init_gamma": c_value / float(max(base.train_context, 1)),
                    "eta_max": eta_max,
                }
            )
            self.positioner = JordanRoPE(
                self.head_dim, config.n_heads, config=exact_cfg, learn_gamma=True, learn_eta=True
            )
        elif method == "jordan_m3":
            self.positioner = JordanRoPE(
                self.head_dim,
                config.n_heads,
                config=config.jordan_config(),
                learn_gamma=True,
                learn_eta=True,
                order=3,
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

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        bsz, length, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, length, self.n_heads, self.head_dim).transpose(1, 2)

        if positions is None:
            positions = torch.arange(length, device=x.device)
        if self.positioner is not None:
            q, k = self.positioner.apply(q, k, positions)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.alibi is not None:
            scores = scores + self.alibi(positions).to(device=x.device, dtype=scores.dtype)

        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal[None, None, :, :], torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(bsz, length, dim)
        return self.proj(y)


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
