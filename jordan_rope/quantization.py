from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


ScaleAxis = Literal["vector", "head", "tensor"]


@dataclass(frozen=True)
class BitAccounting:
    b_numeric: float
    b_storage: float
    num_scalars: int
    metadata_bits: int


@dataclass(frozen=True)
class UniformQuantized:
    codes: torch.Tensor
    scale: torch.Tensor
    bits: int
    scale_axis: ScaleAxis
    scale_bits: int = 16
    extra_metadata_bits: int = 0

    def dequantize(self) -> torch.Tensor:
        return self.codes.to(dtype=self.scale.dtype) * self.scale

    def bit_accounting(self) -> BitAccounting:
        num_scalars = int(self.codes.numel())
        metadata_bits = int(self.scale.numel()) * int(self.scale_bits) + int(self.extra_metadata_bits)
        b_storage = float(self.bits) + (metadata_bits / max(num_scalars, 1))
        return BitAccounting(
            b_numeric=float(self.bits),
            b_storage=b_storage,
            num_scalars=num_scalars,
            metadata_bits=metadata_bits,
        )


@dataclass(frozen=True)
class TurboMSEQuantized:
    codes: torch.Tensor
    norm: torch.Tensor
    codebook: torch.Tensor
    bits: int
    norm_bits: int = 16

    def bit_accounting(self) -> BitAccounting:
        num_scalars = int(self.codes.numel())
        metadata_bits = int(self.norm.numel()) * int(self.norm_bits)
        b_storage = float(self.bits) + (metadata_bits / max(num_scalars, 1))
        return BitAccounting(
            b_numeric=float(self.bits),
            b_storage=b_storage,
            num_scalars=num_scalars,
            metadata_bits=metadata_bits,
        )


@dataclass(frozen=True)
class QJLResidualQuantized:
    signs: torch.Tensor
    residual_norm: torch.Tensor
    projection: torch.Tensor
    norm_bits: int = 16

    def bit_accounting(self, num_scalars: int) -> BitAccounting:
        metadata_bits = int(self.residual_norm.numel()) * int(self.norm_bits)
        qjl_bits = int(self.signs.numel())
        b_numeric = qjl_bits / float(max(num_scalars, 1))
        b_storage = b_numeric + (metadata_bits / float(max(num_scalars, 1)))
        return BitAccounting(
            b_numeric=float(b_numeric),
            b_storage=float(b_storage),
            num_scalars=int(num_scalars),
            metadata_bits=metadata_bits,
        )


@dataclass(frozen=True)
class TurboProductQuantized:
    mse: TurboMSEQuantized
    qjl: QJLResidualQuantized
    x_mse: torch.Tensor
    rotation: torch.Tensor | KacRotation | None

    def bit_accounting(self) -> BitAccounting:
        mse_bits = self.mse.bit_accounting()
        qjl_bits = self.qjl.bit_accounting(mse_bits.num_scalars)
        return BitAccounting(
            b_numeric=float(mse_bits.b_numeric + qjl_bits.b_numeric),
            b_storage=float(mse_bits.b_storage + qjl_bits.b_storage),
            num_scalars=mse_bits.num_scalars,
            metadata_bits=int(mse_bits.metadata_bits + qjl_bits.metadata_bits),
        )


def _signed_range(bits: int) -> tuple[int, int]:
    if bits < 2:
        raise ValueError("Uniform signed quantization expects bits >= 2.")
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def _scale_for_axis(x: torch.Tensor, bits: int, scale_axis: ScaleAxis, eps: float) -> torch.Tensor:
    qmin, qmax = _signed_range(bits)
    max_code = float(max(abs(qmin), abs(qmax)))
    if scale_axis == "vector":
        amax = x.detach().abs().amax(dim=-1, keepdim=True)
    elif scale_axis == "head":
        if x.ndim < 4:
            raise ValueError("scale_axis='head' expects tensors shaped at least [B,H,T,D].")
        reduce_dims = tuple(dim for dim in range(x.ndim) if dim != 1)
        amax = x.detach().abs().amax(dim=reduce_dims, keepdim=True)
    elif scale_axis == "tensor":
        amax = x.detach().abs().amax()
    else:
        raise ValueError(f"Unknown scale_axis: {scale_axis}")
    return (amax / max_code).clamp_min(eps).to(dtype=x.dtype)


def uniform_quantize(
    x: torch.Tensor,
    bits: int,
    scale_axis: ScaleAxis = "vector",
    *,
    scale_bits: int = 16,
    extra_metadata_bits: int = 0,
    eps: float = 1e-8,
) -> UniformQuantized:
    """Signed uniform scalar quantization baseline.

    This is intentionally named as a uniform baseline, not TurboQuant. It uses
    a signed integer code range and floating scale values; packing is left to
    later storage experiments.
    """

    qmin, qmax = _signed_range(bits)
    scale = _scale_for_axis(x, bits, scale_axis, eps)
    codes = torch.round(x / scale).clamp(qmin, qmax).to(torch.int16)
    return UniformQuantized(
        codes=codes,
        scale=scale,
        bits=int(bits),
        scale_axis=scale_axis,
        scale_bits=int(scale_bits),
        extra_metadata_bits=int(extra_metadata_bits),
    )


def uniform_quantize_dequantize(
    x: torch.Tensor,
    bits: int,
    scale_axis: ScaleAxis = "vector",
    *,
    scale_bits: int = 16,
    extra_metadata_bits: int = 0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, BitAccounting]:
    encoded = uniform_quantize(
        x,
        bits,
        scale_axis,
        scale_bits=scale_bits,
        extra_metadata_bits=extra_metadata_bits,
        eps=eps,
    )
    return encoded.dequantize(), encoded.bit_accounting()


def lloyd_max_gaussian_codebook(
    bits: int,
    *,
    iterations: int = 80,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Lloyd-Max codebook for a standard normal scalar.

    TurboQuant uses the coordinate law of a random point on the sphere. For the
    head dimensions used in these experiments, this implementation uses the
    paper's Gaussian approximation to that law and later rescales by sqrt(dim).
    """

    if bits < 1:
        raise ValueError("Lloyd-Max codebook expects bits >= 1.")
    levels = 2 ** int(bits)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    probs = (torch.arange(levels, dtype=torch.float64) + 0.5) / float(levels)
    centroids = normal.icdf(probs).clamp(-8.0, 8.0)
    sqrt_two_pi = math.sqrt(2.0 * math.pi)
    for _ in range(int(iterations)):
        boundaries = torch.empty(levels + 1, dtype=torch.float64)
        boundaries[0] = -float("inf")
        boundaries[-1] = float("inf")
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        cdf_lo = normal.cdf(boundaries[:-1])
        cdf_hi = normal.cdf(boundaries[1:])
        mass = (cdf_hi - cdf_lo).clamp_min(1e-18)
        phi_lo = torch.exp(-0.5 * boundaries[:-1].clamp(-12.0, 12.0).square()) / sqrt_two_pi
        phi_hi = torch.exp(-0.5 * boundaries[1:].clamp(-12.0, 12.0).square()) / sqrt_two_pi
        phi_lo = torch.where(torch.isinf(boundaries[:-1]), torch.zeros_like(phi_lo), phi_lo)
        phi_hi = torch.where(torch.isinf(boundaries[1:]), torch.zeros_like(phi_hi), phi_hi)
        centroids = (phi_lo - phi_hi) / mass
    return centroids.to(device=device, dtype=dtype)


def turbo_mse_codebook(bits: int, dim: int, *, device=None, dtype=torch.float32) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive.")
    return lloyd_max_gaussian_codebook(bits, device=device, dtype=dtype) / math.sqrt(float(dim))


def _apply_rotation(x: torch.Tensor, rotation: torch.Tensor | KacRotation | None, *, inverse: bool = False) -> torch.Tensor:
    if rotation is None:
        return x.clone()
    if isinstance(rotation, KacRotation):
        return rotation.apply(x, inverse=inverse)
    return apply_dense_rotation(x, rotation, inverse=inverse)


def turbo_mse_quantize(
    x: torch.Tensor,
    *,
    bits: int,
    rotation: torch.Tensor | KacRotation | None,
    norm_bits: int = 16,
    eps: float = 1e-12,
) -> tuple[TurboMSEQuantized, torch.Tensor, BitAccounting]:
    if bits < 1:
        raise ValueError("TurboQuant-MSE codebook expects bits >= 1.")
    dim = int(x.shape[-1])
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    unit = x / norm
    rotated = _apply_rotation(unit, rotation)
    codebook = turbo_mse_codebook(bits, dim, device=x.device, dtype=x.dtype)
    boundaries = 0.5 * (codebook[:-1] + codebook[1:])
    codes = torch.bucketize(rotated.contiguous(), boundaries.contiguous()).to(torch.int16)
    quantized_unit_rot = codebook.index_select(0, codes.reshape(-1).long()).reshape_as(rotated)
    quantized_unit = _apply_rotation(quantized_unit_rot, rotation, inverse=True)
    x_hat = quantized_unit * norm
    encoded = TurboMSEQuantized(
        codes=codes,
        norm=norm.detach(),
        codebook=codebook.detach(),
        bits=int(bits),
        norm_bits=int(norm_bits),
    )
    return encoded, x_hat, encoded.bit_accounting()


def turbo_mse_quantize_dequantize(
    x: torch.Tensor,
    *,
    bits: int,
    rotation: torch.Tensor | KacRotation | None,
    norm_bits: int = 16,
) -> tuple[torch.Tensor, BitAccounting]:
    _encoded, x_hat, accounting = turbo_mse_quantize(
        x,
        bits=bits,
        rotation=rotation,
        norm_bits=norm_bits,
    )
    return x_hat, accounting


def gaussian_projection(rows: int, dim: int, *, seed: int, device=None, dtype=torch.float32) -> torch.Tensor:
    if rows <= 0 or dim <= 0:
        raise ValueError("projection rows and dim must be positive.")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    projection = torch.randn(int(rows), int(dim), generator=gen, dtype=torch.float64)
    return projection.to(device=device, dtype=dtype)


def qjl_residual_quantize(
    residual: torch.Tensor,
    *,
    projection: torch.Tensor,
    norm_bits: int = 16,
    eps: float = 1e-12,
) -> QJLResidualQuantized:
    if residual.shape[-1] != projection.shape[-1]:
        raise ValueError("projection dim must match residual.shape[-1].")
    projected = torch.matmul(residual, projection.t())
    signs = torch.where(projected >= 0, torch.ones_like(projected, dtype=torch.int8), -torch.ones_like(projected, dtype=torch.int8))
    residual_norm = residual.norm(dim=-1, keepdim=True).clamp_min(eps).detach()
    return QJLResidualQuantized(
        signs=signs,
        residual_norm=residual_norm,
        projection=projection.detach(),
        norm_bits=int(norm_bits),
    )


def qjl_residual_inner_product(y: torch.Tensor, qjl: QJLResidualQuantized) -> torch.Tensor:
    if y.shape[-1] != qjl.projection.shape[-1]:
        raise ValueError("projection dim must match y.shape[-1].")
    projected_y = torch.matmul(y, qjl.projection.to(device=y.device, dtype=y.dtype).t())
    signs = qjl.signs.to(device=y.device, dtype=y.dtype)
    correction = (projected_y * signs).mean(dim=-1)
    return qjl.residual_norm.to(device=y.device, dtype=y.dtype).squeeze(-1) * math.sqrt(math.pi / 2.0) * correction


def turbo_product_quantize(
    x: torch.Tensor,
    *,
    total_bits: int,
    rotation: torch.Tensor | KacRotation | None,
    qjl_seed: int,
    qjl_rows: int | None = None,
    norm_bits: int = 16,
) -> TurboProductQuantized:
    if total_bits < 2:
        raise ValueError("TurboQuant-prod expects total_bits >= 2.")
    mse_encoded, x_mse, _mse_bits = turbo_mse_quantize(
        x,
        bits=int(total_bits) - 1,
        rotation=rotation,
        norm_bits=norm_bits,
    )
    residual = x - x_mse
    dim = int(x.shape[-1])
    rows = int(qjl_rows or dim)
    projection = gaussian_projection(rows, dim, seed=int(qjl_seed), device=x.device, dtype=x.dtype)
    qjl = qjl_residual_quantize(residual, projection=projection, norm_bits=norm_bits)
    return TurboProductQuantized(mse=mse_encoded, qjl=qjl, x_mse=x_mse, rotation=rotation)


def turbo_product_inner_product(y: torch.Tensor, encoded: TurboProductQuantized) -> torch.Tensor:
    mse_part = (y * encoded.x_mse.to(device=y.device, dtype=y.dtype)).sum(dim=-1)
    residual_part = qjl_residual_inner_product(y, encoded.qjl)
    return mse_part + residual_part


def dense_random_orthogonal(
    dim: int,
    *,
    seed: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive.")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    mat = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs[None, :]
    return q.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class KacRotation:
    dim: int
    depth: int
    pairs: torch.Tensor
    angles: torch.Tensor

    @classmethod
    def random(
        cls,
        dim: int,
        depth: int,
        *,
        seed: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "KacRotation":
        if dim < 2:
            raise ValueError("KacRotation expects dim >= 2.")
        if depth < 0:
            raise ValueError("depth must be non-negative.")
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        num_pairs = dim // 2
        pairs = []
        angles = []
        for _ in range(depth):
            perm = torch.randperm(dim, generator=gen)
            pairs.append(perm[: 2 * num_pairs].reshape(num_pairs, 2))
            angles.append(2.0 * math.pi * torch.rand(num_pairs, generator=gen, dtype=torch.float64))
        if depth == 0:
            pair_tensor = torch.empty(0, num_pairs, 2, dtype=torch.long)
            angle_tensor = torch.empty(0, num_pairs, dtype=torch.float64)
        else:
            pair_tensor = torch.stack(pairs, dim=0)
            angle_tensor = torch.stack(angles, dim=0)
        return cls(
            dim=int(dim),
            depth=int(depth),
            pairs=pair_tensor.to(device=device),
            angles=angle_tensor.to(device=device, dtype=dtype),
        )

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> "KacRotation":
        return KacRotation(
            dim=self.dim,
            depth=self.depth,
            pairs=self.pairs.to(device=device),
            angles=self.angles.to(device=device, dtype=dtype or self.angles.dtype),
        )

    def apply(self, x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.shape[-1]}.")
        if self.depth == 0:
            return x.clone()
        y = x.clone()
        layer_iter = range(self.depth - 1, -1, -1) if inverse else range(self.depth)
        for layer in layer_iter:
            pairs = self.pairs[layer]
            theta = -self.angles[layer] if inverse else self.angles[layer]
            i = pairs[:, 0]
            j = pairs[:, 1]
            yi = y.index_select(-1, i)
            yj = y.index_select(-1, j)
            view_shape = (1,) * (y.ndim - 1) + (theta.numel(),)
            cos = theta.cos().reshape(view_shape).to(dtype=y.dtype)
            sin = theta.sin().reshape(view_shape).to(dtype=y.dtype)
            y_rot_i = yi * cos - yj * sin
            y_rot_j = yi * sin + yj * cos
            y = y.scatter(-1, i.expand_as(y_rot_i), y_rot_i)
            y = y.scatter(-1, j.expand_as(y_rot_j), y_rot_j)
        return y


def apply_dense_rotation(x: torch.Tensor, rotation: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    if x.shape[-1] != rotation.shape[0] or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("Rotation matrix must be square and match x.shape[-1].")
    mat = rotation.t() if inverse else rotation
    return torch.matmul(x, mat.t())


def rotate_quantize_dequantize(
    x: torch.Tensor,
    *,
    bits: int,
    rotation: torch.Tensor | KacRotation | None,
    scale_axis: ScaleAxis = "vector",
    scale_bits: int = 16,
) -> tuple[torch.Tensor, BitAccounting]:
    if rotation is None:
        rotated = x
    elif isinstance(rotation, KacRotation):
        rotated = rotation.apply(x)
    else:
        rotated = apply_dense_rotation(x, rotation)
    q_rotated, accounting = uniform_quantize_dequantize(
        rotated,
        bits,
        scale_axis=scale_axis,
        scale_bits=scale_bits,
    )
    if rotation is None:
        return q_rotated, accounting
    if isinstance(rotation, KacRotation):
        return rotation.apply(q_rotated, inverse=True), accounting
    return apply_dense_rotation(q_rotated, rotation, inverse=True), accounting


def mixed_bit_quantize_dequantize(
    x: torch.Tensor,
    *,
    low_bits: int,
    high_bits: int,
    high_mask: torch.Tensor,
    scale_axis: ScaleAxis = "vector",
    scale_bits: int = 16,
) -> tuple[torch.Tensor, BitAccounting]:
    """Two-group mixed-bit uniform quantization over the last dimension.

    ``high_mask`` may be a 1D ``[D]`` mask shared by all heads, or a 2D
    ``[H,D]`` mask for tensors shaped ``[B,H,T,D]``.
    """

    if high_mask.ndim == 2:
        if x.ndim < 4 or high_mask.shape[0] != x.shape[1] or high_mask.shape[1] != x.shape[-1]:
            raise ValueError("2D high_mask must be shaped [H,D] for an input shaped [B,H,T,D].")
        out = torch.empty_like(x)
        metadata_bits = int(high_mask.numel())
        numeric_bits = 0
        for head in range(int(x.shape[1])):
            head_mask = high_mask[head].to(device=x.device, dtype=torch.bool)
            head_hat, head_accounting = mixed_bit_quantize_dequantize(
                x.select(1, head),
                low_bits=low_bits,
                high_bits=high_bits,
                high_mask=head_mask,
                scale_axis=scale_axis,
                scale_bits=scale_bits,
            )
            out[:, head] = head_hat
            metadata_bits += int(head_accounting.metadata_bits) - int(head_mask.numel())
            numeric_bits += int(head_accounting.b_numeric * x.shape[-1])
        b_numeric = float(numeric_bits) / float(max(x.shape[1] * x.shape[-1], 1))
        b_storage = b_numeric + (float(metadata_bits) / float(max(x.numel(), 1)))
        return out, BitAccounting(
            b_numeric=b_numeric,
            b_storage=b_storage,
            num_scalars=int(x.numel()),
            metadata_bits=metadata_bits,
        )

    if high_mask.ndim != 1 or high_mask.numel() != x.shape[-1]:
        raise ValueError("high_mask must be a 1D boolean tensor matching x.shape[-1].")
    high_mask = high_mask.to(device=x.device, dtype=torch.bool)
    low_mask = ~high_mask
    out = torch.empty_like(x)
    metadata_bits = int(high_mask.numel())
    numeric_bits = 0

    for mask, bits in ((low_mask, low_bits), (high_mask, high_bits)):
        count = int(mask.sum().item())
        if count == 0:
            continue
        part = x.index_select(-1, mask.nonzero(as_tuple=False).flatten())
        q_part, accounting = uniform_quantize_dequantize(
            part,
            bits=int(bits),
            scale_axis=scale_axis,
            scale_bits=scale_bits,
        )
        out[..., mask] = q_part
        metadata_bits += int(accounting.metadata_bits)
        numeric_bits += int(bits) * count

    b_numeric = float(numeric_bits) / float(max(x.shape[-1], 1))
    b_storage = b_numeric + (float(metadata_bits) / float(max(x.numel(), 1)))
    return out, BitAccounting(
        b_numeric=b_numeric,
        b_storage=b_storage,
        num_scalars=int(x.numel()),
        metadata_bits=metadata_bits,
    )


def rotate_mixed_bit_quantize_dequantize(
    x: torch.Tensor,
    *,
    low_bits: int,
    high_bits: int,
    high_mask: torch.Tensor,
    rotation: torch.Tensor | KacRotation | None,
    scale_axis: ScaleAxis = "vector",
    scale_bits: int = 16,
) -> tuple[torch.Tensor, BitAccounting]:
    if rotation is None:
        rotated = x
    elif isinstance(rotation, KacRotation):
        rotated = rotation.apply(x)
    else:
        rotated = apply_dense_rotation(x, rotation)
    q_rotated, accounting = mixed_bit_quantize_dequantize(
        rotated,
        low_bits=low_bits,
        high_bits=high_bits,
        high_mask=high_mask,
        scale_axis=scale_axis,
        scale_bits=scale_bits,
    )
    if rotation is None:
        return q_rotated, accounting
    if isinstance(rotation, KacRotation):
        return rotation.apply(q_rotated, inverse=True), accounting
    return apply_dense_rotation(q_rotated, rotation, inverse=True), accounting


def vector_mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).square().sum(dim=-1).mean()


def inner_product_mse(y: torch.Tensor, z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    true = (y * z).sum(dim=-1)
    pred = (y * z_hat).sum(dim=-1)
    return (true - pred).square().mean()


def inner_product_bias(y: torch.Tensor, z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    true = (y * z).sum(dim=-1)
    pred = (y * z_hat).sum(dim=-1)
    return (pred - true).mean()


def coordinate_flatness(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.abs().amax(dim=-1) / x.norm(dim=-1).clamp_min(eps)


def coordinate_kurtosis(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1).clamp_min(eps)
    return centered.pow(4).mean(dim=-1) / var.square()


def outlier_ratio(x: torch.Tensor, lambda_value: float = 4.0, eps: float = 1e-12) -> torch.Tensor:
    threshold = float(lambda_value) * x.norm(dim=-1, keepdim=True).clamp_min(eps) / math.sqrt(x.shape[-1])
    return (x.abs() > threshold).to(torch.float32).mean(dim=-1)


def position_norm_profile(x: torch.Tensor) -> torch.Tensor:
    """Mean L2 norm by position for tensors shaped [B,H,T,D]."""

    if x.ndim != 4:
        raise ValueError("position_norm_profile expects [B,H,T,D].")
    return x.norm(dim=-1).mean(dim=(0, 1))


def norm_growth(profile: torch.Tensor, position: int = -1, eps: float = 1e-12) -> torch.Tensor:
    if profile.ndim != 1:
        raise ValueError("norm_growth expects a 1D position profile.")
    return profile[position] / profile[0].clamp_min(eps)
