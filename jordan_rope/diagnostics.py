from __future__ import annotations

import torch

from .positional import JordanRoPE, _rotate_pairs


def jordan_apply_a_order2_components(
    positioner: JordanRoPE,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply order-2 Jordan A(t) and split into base and nilpotent components.

    Returns `(base, jordan, total)` with the same shape as `x`. This mirrors the
    implementation of `JordanRoPE._apply_a` for `order=2`.
    """

    if positioner.order != 2:
        raise ValueError("P0 Jordan-mode decomposition currently supports order=2 only.")
    bsz, heads, length, dim = x.shape
    if heads != positioner.num_heads or dim != positioner.head_dim:
        raise ValueError(f"Expected [B,{positioner.num_heads},T,{positioner.head_dim}], got {tuple(x.shape)}.")
    if t.numel() != length:
        raise ValueError(f"Expected {length} relative times, got {t.numel()}.")

    blocks = x.reshape(bsz, heads, length, positioner.num_blocks, positioner.order, 2)
    cos, sin = positioner._angles(t.to(device=x.device), x.dtype)
    shear, (decay_exp, _) = positioner._parameters_for_time(t.to(device=x.device), x.dtype)
    rot0 = _rotate_pairs(blocks[..., 0, :], cos, sin)
    rot1 = _rotate_pairs(blocks[..., 1, :], cos, sin)
    decay = decay_exp.exp().to(dtype=x.dtype)

    base0 = decay * rot0
    base1 = decay * rot1
    jordan0 = decay * shear * rot1
    jordan1 = torch.zeros_like(base1)
    base = torch.cat((base0, base1), dim=-1).reshape(bsz, heads, length, dim)
    jordan = torch.cat((jordan0, jordan1), dim=-1).reshape(bsz, heads, length, dim)
    return base, jordan, base + jordan


def p0_jordan_logit_components(
    positioner: JordanRoPE,
    q0: torch.Tensor,
    k0: torch.Tensor,
    *,
    query_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute relative P0 base/Jordan logits from raw projected q/k.

    Output tensors are shaped `[B,H,Q,T]`, where `Q=len(query_indices)`.
    Future key positions are filled with NaN so reductions can ignore them.
    """

    if q0.shape != k0.shape:
        raise ValueError("q0 and k0 must have the same shape.")
    _, _, length, _ = q0.shape
    device = q0.device
    if query_indices is None:
        query_indices = torch.arange(length, device=device)
    else:
        query_indices = query_indices.to(device=device, dtype=torch.long)
    positions = torch.arange(length, device=device)
    base_rows = []
    jordan_rows = []
    total_rows = []
    for query_pos in query_indices:
        d = (query_pos - positions).to(torch.float32)
        base_k, jordan_k, total_k = jordan_apply_a_order2_components(positioner, k0, d)
        q = q0[:, :, int(query_pos), :].unsqueeze(-2)
        base_logit = (q * base_k).sum(dim=-1)
        jordan_logit = (q * jordan_k).sum(dim=-1)
        total_logit = (q * total_k).sum(dim=-1)
        causal = positions <= query_pos
        base_logit = base_logit.masked_fill(~causal[None, None, :], float("nan"))
        jordan_logit = jordan_logit.masked_fill(~causal[None, None, :], float("nan"))
        total_logit = total_logit.masked_fill(~causal[None, None, :], float("nan"))
        base_rows.append(base_logit)
        jordan_rows.append(jordan_logit)
        total_rows.append(total_logit)
    return {
        "base": torch.stack(base_rows, dim=2),
        "jordan": torch.stack(jordan_rows, dim=2),
        "total": torch.stack(total_rows, dim=2),
        "query_indices": query_indices,
    }


def distance_bucket_mask(
    query_indices: torch.Tensor,
    length: int,
    bucket: tuple[int, int],
    *,
    right_inclusive: bool = True,
) -> torch.Tensor:
    lo, hi = bucket
    positions = torch.arange(length, device=query_indices.device)
    distances = query_indices[:, None] - positions[None, :]
    if right_inclusive:
        return (distances >= lo) & (distances <= hi)
    return (distances >= lo) & (distances < hi)


def nanmean_square(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        values = values.masked_fill(~mask[None, None, :, :], float("nan"))
    finite = torch.isfinite(values)
    count = finite.sum().clamp_min(1)
    return values.masked_fill(~finite, 0.0).square().sum() / count
