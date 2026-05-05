from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _as_1d_positions(positions: torch.Tensor | None, length: int, device: torch.device) -> torch.Tensor:
    if positions is None:
        return torch.arange(length, device=device)
    positions = positions.to(device)
    if positions.ndim != 1:
        raise ValueError("This implementation expects shared 1D positions of shape [T].")
    if positions.numel() != length:
        raise ValueError(f"Expected {length} positions, got {positions.numel()}.")
    return positions


def _rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x0, x1 = x.unbind(dim=-1)
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack((y0, y1), dim=-1)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=1e-8)
    return value + torch.log(-torch.expm1(-value))


def _atanh_clamped(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, min=-0.999999, max=0.999999)
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def default_alibi_slopes(num_heads: int) -> torch.Tensor:
    def slopes_power_of_2(n: int) -> list[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3.0)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = slopes_power_of_2(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        slopes = slopes_power_of_2(closest)
        extra = default_alibi_slopes(2 * closest)[0::2].tolist()
        slopes.extend(extra[: num_heads - closest])
    return torch.tensor(slopes, dtype=torch.float32)


class RotaryEmbedding(nn.Module):
    """Standard RoPE applied to q/k tensors shaped [B, H, T, D]."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        bsz, heads, length, dim = x.shape
        if dim != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {dim}.")
        pos = _as_1d_positions(positions, length, x.device).to(torch.float32)
        angles = pos[:, None] * self.inv_freq[None, :]
        cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
        pairs = x.reshape(bsz, heads, length, dim // 2, 2)
        return _rotate_pairs(pairs, cos, sin).reshape_as(x)

    def apply(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotate(q, positions), self.rotate(k, positions)


class ALiBiBias(nn.Module):
    """Causal ALiBi bias with one positive slope per head."""

    def __init__(self, num_heads: int, learnable: bool = False) -> None:
        super().__init__()
        slopes = default_alibi_slopes(num_heads)
        if learnable:
            self.raw_slopes = nn.Parameter(_inverse_softplus(slopes))
        else:
            self.register_buffer("slopes", slopes, persistent=False)
            self.raw_slopes = None

    def current_slopes(self) -> torch.Tensor:
        if self.raw_slopes is not None:
            return torch.nn.functional.softplus(self.raw_slopes)
        return self.slopes

    def forward(
        self, query_positions: torch.Tensor, key_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if key_positions is None:
            key_positions = query_positions
        qpos = query_positions.to(torch.float32)
        kpos = key_positions.to(torch.float32)
        delta = (qpos[:, None] - kpos[None, :]).clamp_min(0.0)
        slopes = self.current_slopes().to(device=qpos.device, dtype=qpos.dtype)
        bias = -slopes[:, None, None] * delta[None, :, :]
        return bias[None, :, :, :]


@dataclass(frozen=True)
class JordanConfig:
    theta: float = 10000.0
    gamma_min: float = 1e-4
    init_gamma: float = 1e-4
    eta_max: float = 0.1
    init_eta: float = 0.0
    train_context: int = 1024
    bounded_tau: bool = True
    shear_time_scale: float = 1.0
    max_exponent: float = 30.0


class JordanRoPE(nn.Module):
    """Complex Jordan-RoPE realification for small Jordan orders.

    The compatible `apply(q, k, positions)` path uses paired absolute transforms
    F(p)=A(-p)^(-T) for queries and G(p)=A(-p) for keys. With raw positions this
    gives q_i^T A(i-j) k_j exactly. If `bounded_tau=True`, only the nilpotent
    shear coordinate is saturated; this is the planned stable variant.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        config: JordanConfig | None = None,
        learn_gamma: bool = True,
        learn_eta: bool = True,
        force_zero_gamma: bool = False,
        force_zero_eta: bool = False,
        force_zero_omega: bool = False,
        order: int = 2,
    ) -> None:
        super().__init__()
        if order < 2:
            raise ValueError("JordanRoPE order must be at least 2.")
        real_block = 2 * order
        if head_dim % real_block != 0:
            raise ValueError(f"JordanRoPE order={order} requires head_dim divisible by {real_block}.")
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.order = order
        self.num_blocks = head_dim // real_block
        self.config = config or JordanConfig()
        self.learn_gamma = learn_gamma
        self.learn_eta = learn_eta
        self.force_zero_gamma = force_zero_gamma
        self.force_zero_eta = force_zero_eta
        self.force_zero_omega = force_zero_omega

        inv_freq = self.config.theta ** (
            -2.0 * torch.arange(self.num_blocks, dtype=torch.float32) / head_dim
        )
        self.register_buffer("omega", inv_freq, persistent=False)

        gamma_target = torch.full(
            (num_heads, self.num_blocks),
            max(self.config.init_gamma - self.config.gamma_min, 1e-8),
            dtype=torch.float32,
        )
        eta_target = torch.full(
            (num_heads, self.num_blocks),
            self.config.init_eta / max(self.config.eta_max, 1e-8),
            dtype=torch.float32,
        )
        if learn_gamma:
            self.raw_gamma = nn.Parameter(_inverse_softplus(gamma_target))
        else:
            self.register_buffer("fixed_gamma", gamma_target + self.config.gamma_min, persistent=False)
            self.raw_gamma = None
        if learn_eta:
            self.raw_eta = nn.Parameter(_atanh_clamped(eta_target))
        else:
            self.register_buffer("fixed_eta", eta_target * self.config.eta_max, persistent=False)
            self.raw_eta = None

    def gamma(self) -> torch.Tensor:
        if self.force_zero_gamma:
            return torch.zeros(
                self.num_heads, self.num_blocks, device=self.omega.device, dtype=self.omega.dtype
            )
        if self.raw_gamma is None:
            return self.fixed_gamma
        return torch.nn.functional.softplus(self.raw_gamma) + self.config.gamma_min

    def eta(self) -> torch.Tensor:
        if self.force_zero_eta:
            return torch.zeros(
                self.num_heads, self.num_blocks, device=self.omega.device, dtype=self.omega.dtype
            )
        if self.raw_eta is None:
            return self.fixed_eta
        return self.config.eta_max * torch.tanh(self.raw_eta)

    def effective_shear_time(self, t: torch.Tensor) -> torch.Tensor:
        if not self.config.bounded_tau:
            return t * float(self.config.shear_time_scale)
        scale = float(max(self.config.train_context, 1))
        return (t / (1.0 + t.abs() / scale)) * float(self.config.shear_time_scale)

    def _angles(self, t: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        omega = torch.zeros_like(self.omega) if self.force_zero_omega else self.omega
        angles = t[:, None].to(torch.float32) * omega[None, :]
        cos = angles.cos().to(dtype=dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin

    def _parameters_for_time(self, t: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = self.gamma().to(device=t.device, dtype=torch.float32)
        eta = self.eta().to(device=t.device, dtype=torch.float32)
        t_eff = self.effective_shear_time(t.to(torch.float32))
        shear = eta[None, :, None, :, None] * t_eff[None, None, :, None, None]
        decay_exp = (-gamma[None, :, None, :, None] * t[None, None, :, None, None]).clamp(
            -self.config.max_exponent, self.config.max_exponent
        )
        inv_decay_exp = (gamma[None, :, None, :, None] * t[None, None, :, None, None]).clamp(
            -self.config.max_exponent, self.config.max_exponent
        )
        return shear.to(dtype=dtype), (decay_exp, inv_decay_exp)

    def _apply_a(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, heads, length, dim = x.shape
        blocks = x.reshape(bsz, heads, length, self.num_blocks, self.order, 2)
        cos, sin = self._angles(t, x.dtype)
        shear, (decay_exp, _) = self._parameters_for_time(t, x.dtype)
        rotated = [_rotate_pairs(blocks[..., idx, :], cos, sin) for idx in range(self.order)]
        decay = decay_exp.exp().to(dtype=x.dtype)
        outputs = []
        for row in range(self.order):
            y = torch.zeros_like(rotated[row])
            for col in range(row, self.order):
                power = col - row
                if power == 0:
                    coef = 1.0
                else:
                    coef = shear.pow(power) / math.factorial(power)
                y = y + coef * rotated[col]
            outputs.append(decay * y)
        return torch.cat(outputs, dim=-1).reshape(bsz, heads, length, dim)

    def _apply_a_inverse_transpose(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        bsz, heads, length, dim = x.shape
        blocks = x.reshape(bsz, heads, length, self.num_blocks, self.order, 2)
        cos, sin = self._angles(t, x.dtype)
        shear, (_, inv_decay_exp) = self._parameters_for_time(t, x.dtype)
        rotated = [_rotate_pairs(blocks[..., idx, :], cos, sin) for idx in range(self.order)]
        inv_decay = inv_decay_exp.exp().to(dtype=x.dtype)
        outputs = []
        for row in range(self.order):
            y = torch.zeros_like(rotated[row])
            for col in range(0, row + 1):
                power = row - col
                if power == 0:
                    coef = 1.0
                else:
                    coef = ((-1.0) ** power) * shear.pow(power) / math.factorial(power)
                y = y + coef * rotated[col]
            outputs.append(inv_decay * y)
        return torch.cat(outputs, dim=-1).reshape(bsz, heads, length, dim)

    def apply(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape != k.shape:
            raise ValueError("JordanRoPE.apply expects q and k to have the same shape.")
        bsz, heads, length, dim = q.shape
        if heads != self.num_heads or dim != self.head_dim:
            raise ValueError(f"Expected [B,{self.num_heads},T,{self.head_dim}], got {tuple(q.shape)}.")
        pos = _as_1d_positions(positions, length, q.device).to(torch.float32)
        t = -pos
        return self._apply_a_inverse_transpose(q, t), self._apply_a(k, t)


class DirectSumRoPEUnipotent(nn.Module):
    """Direct-sum baseline: RoPE on one subspace, real unipotent/Jordan on another."""

    def __init__(self, head_dim: int, num_heads: int, config: JordanConfig | None = None) -> None:
        super().__init__()
        if head_dim % 8 != 0:
            raise ValueError("DirectSumRoPEUnipotent requires head_dim divisible by 8.")
        half = head_dim // 2
        self.head_dim = head_dim
        self.rope = RotaryEmbedding(half, theta=(config or JordanConfig()).theta)
        self.real_jordan = JordanRoPE(
            half,
            num_heads,
            config=config,
            learn_gamma=True,
            learn_eta=True,
            force_zero_omega=True,
        )

    def apply(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_rope, q_jordan = q.split(self.head_dim // 2, dim=-1)
        k_rope, k_jordan = k.split(self.head_dim // 2, dim=-1)
        q_rope, k_rope = self.rope.apply(q_rope, k_rope, positions)
        q_jordan, k_jordan = self.real_jordan.apply(q_jordan, k_jordan, positions)
        return torch.cat((q_rope, q_jordan), dim=-1), torch.cat((k_rope, k_jordan), dim=-1)


def causal_delta_features(
    method: str,
    deltas: torch.Tensor,
    *,
    num_freqs: int = 16,
    head_dim: int = 64,
    theta: float = 10000.0,
    gamma: float = 1e-4,
    train_context: int = 1024,
    bounded_tau: bool = True,
) -> torch.Tensor:
    """Basis features used by the synthetic diagnostic tasks."""
    if method == "jordan_no_gamma":
        method = "jordan_rope"
        gamma = 0.0
    elif method == "jordan_raw_tau":
        method = "jordan_rope"
        bounded_tau = False
    elif method.startswith("jordan_exact"):
        c_value = _parse_exact_scaled_c(method)
        method = "jordan_rope"
        bounded_tau = False
        gamma = c_value / float(max(train_context, 1))

    d = deltas.to(torch.float32)
    d_norm = d / float(max(train_context, 1))
    if bounded_tau:
        tau = d / (1.0 + d / float(max(train_context, 1)))
    else:
        tau = d
    tau_norm = tau / float(max(train_context, 1))
    freqs = theta ** (-2.0 * torch.arange(num_freqs, device=d.device, dtype=torch.float32) / head_dim)
    angles = d[:, None] * freqs[None, :]
    cos = angles.cos()
    sin = angles.sin()
    decay = torch.exp(-gamma * d)[:, None]
    ones = torch.ones_like(d[:, None])

    if method == "nope":
        return ones
    if method == "rope":
        return torch.cat((ones, cos, sin), dim=1)
    if method == "alibi":
        return torch.cat((ones, d_norm[:, None]), dim=1)
    if method == "rope_alibi":
        return torch.cat((ones, cos, sin, d_norm[:, None]), dim=1)
    if method == "damped_rope":
        return torch.cat((ones, decay * cos, decay * sin), dim=1)
    if method == "real_jordan":
        return torch.cat((ones, decay, tau_norm[:, None] * decay), dim=1)
    if method == "direct_sum":
        return torch.cat((ones, cos, sin, d_norm[:, None], decay, tau_norm[:, None] * decay), dim=1)
    if method == "jordan_rope":
        return torch.cat((ones, decay * cos, decay * sin, tau_norm[:, None] * decay * cos, tau_norm[:, None] * decay * sin), dim=1)
    if method == "jordan_m3":
        second = 0.5 * tau_norm[:, None].pow(2)
        return torch.cat(
            (
                ones,
                decay * cos,
                decay * sin,
                tau_norm[:, None] * decay * cos,
                tau_norm[:, None] * decay * sin,
                second * decay * cos,
                second * decay * sin,
            ),
            dim=1,
        )
    raise ValueError(f"Unknown method: {method}")


def _parse_exact_scaled_c(method: str) -> float:
    for token in method.split("_"):
        if token.startswith("c") and token[1:].isdigit():
            return int(token[1:]) / 100.0
    return 1.0
