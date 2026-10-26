from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor

def _pool_groups(x: Tensor, count: int = 4) -> Tensor:
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("spectral descriptor must be [N,B], B>0")
    return torch.stack(
        [
            chunk.mean(dim=1)
            for chunk in torch.tensor_split(x, min(count, x.shape[1]), dim=1)
        ],
        dim=1,
    )


def _spectral_descriptor(p: Tensor) -> Tensor:
    """Fixed band-index order; no physical wavelength-spacing assumption."""
    p = torch.as_tensor(p)
    if p.ndim == 3 and p.shape[-1] == 1:
        p = p[..., 0]
    if p.ndim != 2 or p.shape[1] < 2 or not bool(torch.isfinite(p).all()):
        raise ValueError("physical spectral evidence must be finite [N,B,1] or [N,B]")
    return torch.cat((_pool_groups(p, 4), _pool_groups(p[:, 1:] - p[:, :-1], 4)), dim=1)


@torch.no_grad()
def _spectral_conditions(
    p: Tensor, *, min_per_group: int
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Split by *native spectral evidence only*; no learned-Q leakage into groups.

    Returns membership, descriptor center, descriptor scale, principal spectral
    axis, and split threshold. For undersampled/uninformative classes, one group.
    The rule is estimated only when that class is first learned and is committed.
    """
    desc = _spectral_descriptor(p)
    n, dim = desc.shape
    membership = torch.zeros(n, dtype=torch.long, device=desc.device)
    center = desc.mean(dim=0)
    scale = desc.std(dim=0, unbiased=False).clamp_min(1e-4)
    axis = torch.zeros(dim, dtype=desc.dtype, device=desc.device)
    threshold = desc.new_zeros(())
    if n < 2 * min_per_group:
        return membership, center, scale, axis, threshold
    normalized = (desc - center) / scale
    cov = normalized.T @ normalized / max(n - 1, 1)
    eigval, eigvec = torch.linalg.eigh(cov)
    if not bool(torch.isfinite(eigval).all()) or float(eigval[-1]) < 1e-5:
        return membership, center, scale, axis, threshold
    axis = eigvec[:, -1].clone()
    # Eigenvectors are sign-ambiguous: choose a reproducible orientation.
    pivot = axis.abs().argmax()
    if axis[pivot] < 0:
        axis = -axis
    projection = normalized @ axis
    threshold = projection.median()
    candidate = (projection > threshold).long()
    if min(int((candidate == 0).sum()), int((candidate == 1).sum())) >= min_per_group:
        membership = candidate
    else:
        axis = torch.zeros_like(axis)
        threshold = threshold * 0
    return membership, center, scale, axis, threshold


@torch.no_grad()
def _compact_covariance(
    data: Tensor, *, rank: int, class_seed: int, shrinkage: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Deterministic low-rank + isotropic residual sample covariance.

    C = U diag(lambda) U^T + sigma2 (I-UU^T), shrunk toward trace(C)/F * I.
    No Gaussian model is fitted and no sampled features are generated.
    """
    if data.ndim != 2:
        raise ValueError("features must be [N,F]")
    n, f = data.shape
    axes = data.new_zeros(f, rank)
    lam = data.new_zeros(rank)
    eps = 1e-7
    if n < 2:
        return axes, lam, data.new_tensor(eps)
    x = data - data.mean(dim=0, keepdim=True)
    trace = x.square().sum() / (n - 1)
    r = min(rank, n - 1, f)
    if r < 1 or float(trace) <= eps:
        return axes, lam, (trace / f).clamp_min(eps)
    k = min(f, n - 1, r + 4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(class_seed) % (2**63 - 1))
    omega = torch.randn(f, k, generator=generator, dtype=torch.float64).to(
        device=x.device, dtype=x.dtype
    )
    # Matrix-free sample-covariance power iteration, O(N*F*rank).
    for _ in range(3):
        y = x.T @ (x @ omega)
        omega = torch.linalg.qr(y, mode="reduced").Q
    small = x @ omega
    _u, singular, vt = torch.linalg.svd(small, full_matrices=False)
    basis = omega @ vt[:r].T
    eig = singular[:r].square() / (n - 1)
    remaining = (trace - eig.sum()).clamp_min(0)
    isotropic = remaining / max(f - r, 1)
    mean_variance = trace / f
    axes[:, :r] = basis
    lam[:r] = (1.0 - shrinkage) * eig + shrinkage * mean_variance
    residual = ((1.0 - shrinkage) * isotropic + shrinkage * mean_variance).clamp_min(
        eps
    )
    if not bool(torch.isfinite(lam).all()) or not bool(torch.isfinite(residual)):
        raise RuntimeError("invalid spectral-condition covariance estimate")
    return axes, lam, residual




def _integer(value: object, name: str, *, minimum: int) -> int:
    if torch.is_tensor(value):
        if (
            value.ndim != 0
            or value.dtype == torch.bool
            or value.is_complex()
            or torch.is_floating_point(value)
        ):
            raise TypeError(f"{name} must be an integer scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _class_id_tuple(
    values: Sequence[int] | Tensor | Iterable[int],
    *,
    name: str,
) -> tuple[int, ...]:
    if torch.is_tensor(values):
        if values.ndim != 1 or values.numel() == 0 or values.dtype != torch.long:
            raise ValueError(f"{name} tensor must be non-empty rank-one torch.long")
        source = values.detach().cpu().tolist()
    else:
        source = list(values)
    result = tuple(_integer(value, name, minimum=0) for value in source)
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique class IDs")
    return result


def _band_positions(
    values: Sequence[float] | Tensor | None,
    *,
    count: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> Tensor:
    positions = (
        torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
        if values is None
        else torch.as_tensor(values, device=device, dtype=dtype).flatten()
    )
    if positions.shape != (count,):
        raise ValueError(f"spectral_band_positions must contain {count} values")
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("spectral_band_positions contain NaN or Inf")
    if count > 1 and bool((positions[1:] <= positions[:-1]).any()):
        raise ValueError("spectral_band_positions must be strictly increasing")
    return positions


def _deterministic_quantile_dim0(values: Tensor, q: float) -> Tensor:
    value = torch.as_tensor(values)
    if value.ndim < 1 or value.size(0) <= 0:
        raise ValueError("quantile input must have a non-empty leading dimension")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError("quantile input must be real floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quantile input contains NaN or Inf")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must be in [0,1]")

    original_device = value.device
    ordered = torch.sort(value.detach().to(device="cpu"), dim=0, stable=True).values
    sample_count = int(ordered.size(0))
    if sample_count == 1:
        result = ordered[0]
    else:
        position = float(q) * float(sample_count - 1)
        lower = int(position)
        upper = min(lower + 1, sample_count - 1)
        weight = position - float(lower)
        result = (
            ordered[lower]
            if upper == lower or weight == 0.0
            else ordered[lower] * (1.0 - weight) + ordered[upper] * weight
        )
    return result.to(device=original_device, dtype=value.dtype)


def _deterministic_median_dim0(values: Tensor) -> Tensor:
    return _deterministic_quantile_dim0(values, 0.5)


def _validate_ordered_samples(values: Tensor) -> Tensor:
    value = torch.as_tensor(values)
    if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) <= 0:
        raise ValueError("values must be non-empty [N,L,F], L>=2")
    if (
        not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("values must be finite real floating point")
    return value


def _validate_physical_evidence(values: Tensor) -> Tensor:
    value = torch.as_tensor(values)
    if value.ndim == 2:
        value = value.unsqueeze(-1)
    if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) != 1:
        raise ValueError("physical evidence must be direct full-band [N,B,1], B>=2")
    if (
        not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("physical evidence must be finite real floating point")
    return value


def normalize_ordered_decision_features(values: Tensor) -> Tensor:
    """Map Q to the canonical learned CDS coordinate."""
    value = _validate_ordered_samples(values)
    flat = value.reshape(value.size(0), -1)
    eps = torch.finfo(value.dtype).eps
    unit = flat / flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return unit.reshape_as(value)


def sanitize_ordered_variation(variation: Tensor) -> Tensor:
    """Validate signed, compact physical bounds [C,2,2*B-1,1].

    The first B slots describe residual per-band variation; the next B-1
    describe adjacent-band residual differences. No IQR or B-by-B tensor.
    """
    value = torch.as_tensor(variation)
    if (
        value.ndim != 4
        or value.size(1) != 2
        or value.size(-1) != 1
        or value.size(2) < 3
        or value.size(2) % 2 != 1
    ):
        raise ValueError("physical variation must be [C,2,2*B-1,1]")
    if (
        not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("physical variation must be finite real floating point")
    if bool((value[:, 0] > value[:, 1]).any()):
        raise ValueError("physical variation lower bounds exceed upper bounds")
    return value.contiguous()


def ordered_variation_operator(variation: Tensor) -> Tensor:
    return sanitize_ordered_variation(variation)


def estimate_relational_variation(learned_samples: Tensor, identity: Tensor) -> Tensor:
    """Class-conditional signed adjacent residual bounds [2,K-1,D].

    Both learned_samples and identity are in the canonical normalized Q
    coordinate. This is a nonparametric summary, not a containment guarantee.
    """
    q = _validate_ordered_samples(learned_samples)
    if identity.shape != q.shape[1:]:
        raise ValueError("identity must be [K,D] matching learned samples")
    residual = q - identity.unsqueeze(0)
    adjacent = residual[:, 1:, :] - residual[:, :-1, :]
    lo = _deterministic_quantile_dim0(adjacent, 0.10)
    hi = _deterministic_quantile_dim0(adjacent, 0.90)
    return torch.stack((lo, hi), dim=0).contiguous()


def validate_relational_variation(
    value: Tensor, *, length: int, features: int
) -> Tensor:
    """Validate [C,2,K-1,D] signed lower/upper interval summaries."""
    bounds = torch.as_tensor(value)
    if bounds.ndim != 4 or bounds.shape[1:] != (2, length - 1, features):
        raise ValueError("learned V_Q must be [C,2,K-1,D]")
    if not torch.is_floating_point(bounds) or not bool(torch.isfinite(bounds).all()):
        raise ValueError("learned V_Q must be finite floating-point")
    if bool((bounds[:, 0] > bounds[:, 1]).any()):
        raise ValueError("learned V_Q lower bound exceeds upper bound")
    return bounds


def learned_relational_support(
    sample: Tensor, identity: Tensor, variation: Tensor,
    absolute_deviation: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Identity, relational and combined compatibility [N,C] for learned Q.

    The relational penalty is zero for a residual difference within its
    class-conditional quantile interval. Soft hinges are differentiable with
    respect to the sample. The fixed sqrt(K*D) scale follows the unit-norm
    Q coordinate and introduces no learned calibration parameter.
    """
    q = normalize_ordered_decision_features(sample)
    center = torch.as_tensor(identity, device=q.device, dtype=q.dtype)
    if center.ndim != 3 or center.shape[1:] != q.shape[1:]:
        raise ValueError("learned identity must be [C,K,D] matching Q")
    c, k, d = center.shape
    bounds = validate_relational_variation(
        torch.as_tensor(variation, device=q.device, dtype=q.dtype), length=k, features=d
    )
    if bounds.shape[0] != c:
        raise ValueError("learned V_Q class count does not match I_Q")
    delta = q[:, None, :, :] - center[None, :, :, :]
    identity_distance = torch.linalg.vector_norm(
        delta.reshape(delta.size(0), delta.size(1), -1), dim=2
    )
    absolute = (1.0 / (1.0 + identity_distance)).clamp(0, 1)
    if absolute_deviation is not None:
        # A_Q measures the observed absolute deviation of each learned token
        # coordinate; it is NOT another covariance model. Distances beyond
        # this empirical within-class scale reduce absolute CDS support.
        a = torch.as_tensor(absolute_deviation,device=q.device,dtype=q.dtype).detach()
        if a.shape != center.shape or not bool(torch.isfinite(a).all()) or bool((a < 0).any()):
            raise ValueError("learned A_Q must be finite, nonnegative [C,K,D]")
        tolerance = math.sqrt(torch.finfo(q.dtype).eps) / math.sqrt(k*d)
        absolute_spread = 1.0 / (1.0 + (delta.abs() / (a[None] + tolerance)).mean((2,3)))
        absolute = torch.sqrt(absolute * absolute_spread.clamp(0,1))
    relation = delta[:, :, 1:, :] - delta[:, :, :-1, :]
    lower = bounds[None, :, 0, :, :]
    upper = bounds[None, :, 1, :, :]
    violation = (lower - relation).relu() + (relation - upper).relu()
    penalty = violation.mean(dim=(2, 3)) * math.sqrt(k * d)
    relational = (1.0 / (1.0 + penalty)).clamp(0, 1)
    return absolute, relational, torch.sqrt(absolute * relational)


def _interval_compatibility(
    delta: Tensor, lo: Tensor, hi: Tensor, reference: Tensor
) -> Tensor:
    """Continuous interval compatibility, nonzero even for sparse-class evidence."""
    violation = (lo - delta).relu() + (delta - hi).relu()
    # ponytail: this is a numerical floor, not a measured sensor-noise model.
    # Calibrate a shared physical tolerance if held-out rare-class evidence needs it.
    tolerance = torch.finfo(delta.dtype).eps ** 0.5 * (1.0 + reference.abs())
    width = (hi - lo).clamp_min(0.0)
    return (1.0 + violation / (width + tolerance)).reciprocal().clamp(0.0, 1.0)


def physical_support_components(
    sample: Tensor, identity: Tensor, variation: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-band, adjacent-band, joint support [N,C] from native ordered bands."""
    x = _validate_physical_evidence(sample)
    center = torch.as_tensor(identity, device=x.device, dtype=x.dtype)
    bounds = sanitize_ordered_variation(
        torch.as_tensor(variation, device=x.device, dtype=x.dtype)
    )
    if center.ndim != 3 or center.shape[1:] != x.shape[1:]:
        raise ValueError("physical identity must be [C,B,1] matching sample")
    c, b, _ = center.shape
    if bounds.shape != (c, 2, 2 * b - 1, 1):
        raise ValueError("physical variation must be [C,2,2*B-1,1]")
    residual = x[:, None] - center[None]
    absolute = _interval_compatibility(
        residual, bounds[None, :, 0, :b], bounds[None, :, 1, :b], center[None]
    ).mean(dim=(2, 3))
    adjacent = residual[:, :, 1:] - residual[:, :, :-1]
    center_change = (center[:, 1:] - center[:, :-1]).unsqueeze(0)
    ordered = _interval_compatibility(
        adjacent, bounds[None, :, 0, b:], bounds[None, :, 1, b:], center_change
    ).mean(dim=(2, 3))
    return absolute, ordered, torch.sqrt(absolute * ordered)


def ordered_cds_support_components(
    sample: Tensor, identity: Tensor, variation: Tensor,
    *, absolute_deviation: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Dispatch learned signed V_Q or compact signed physical V_P evidence."""
    x = torch.as_tensor(sample)
    center = torch.as_tensor(identity)
    v = torch.as_tensor(variation)
    if center.ndim != 3 or x.ndim != 3 or x.shape[1:] != center.shape[1:]:
        raise ValueError("sample and identity must be [N,L,F] and [C,L,F]")
    if v.ndim == 4 and v.shape[1:] == (2, center.size(1) - 1, center.size(2)):
        return learned_relational_support(x, center, v, absolute_deviation=absolute_deviation)
    if (
        center.size(2) == 1
        and v.ndim == 4
        and v.shape[1:] == (2, 2 * center.size(1) - 1, 1)
    ):
        return physical_support_components(x, center, v)
    raise ValueError("variation does not match learned V_Q or compact physical V_P")


def spectral_conditioned_learned_support(ordered_features, moments, physical_spectral_evidence=None):
    """Compatibility name: single unconditional low-rank covariance support.

    P does NOT partition feature distributions. Spectral variation enters the
    pair-risk and relational losses, never a second learned covariance bank.
    """
    q = normalize_ordered_decision_features(ordered_features)
    z = q.flatten(1)
    mu=torch.as_tensor(moments["mean"],device=z.device,dtype=z.dtype).detach()
    basis=torch.as_tensor(moments["basis"],device=z.device,dtype=z.dtype).detach()
    eig=torch.as_tensor(moments["eigenvalues"],device=z.device,dtype=z.dtype).detach()
    residual=torch.as_tensor(moments["residual_variance"],device=z.device,dtype=z.dtype).detach()
    n,f=z.shape
    if mu.ndim != 2 or mu.size(1)!=f or basis.shape[:2]!=mu.shape or eig.shape!=(len(mu),basis.size(-1)) or residual.shape!=(len(mu),):
        raise ValueError("unified Q moments must be mean[C,F], basis[C,F,R], eigen[C,R], residual[C]")
    if bool((eig<0).any()) or bool((residual<0).any()):
        raise ValueError("negative covariance")
    proj_z=torch.einsum('nf,cfr->ncr',z,basis)
    proj_mu=torch.einsum('cf,cfr->cr',mu,basis)
    projected=(proj_z-proj_mu.unsqueeze(0)).square()
    residual_sq=(z.square().sum(-1)[:,None]+mu.square().sum(-1)[None]-2*z@mu.T-projected.sum(-1)).clamp_min(0)
    used=(basis.square().sum(1)>1e-10).sum(-1)
    trace=eig.sum(-1)+(f-used)*residual
    ridge=(trace/f).clamp_min(1e-4)
    dist=((projected/(eig[None]+ridge[None,:,None])).sum(-1)+
          residual_sq/(residual[None]+ridge[None]))/f
    return (-0.5*dist.clamp_min(0)).exp().clamp(0,1)


def native_spectral_variation_support(physical_evidence, class_mean,
                                      class_variance, delta_variance):
    """Native-band mean/variance and adjacent-band variation (no V_P interval loss).

    Spectral coordinates come from the same fitted native-band normalizer across
    phases; physical evidence is detached and never acts as a second classifier.
    """
    p = _validate_physical_evidence(physical_evidence)[...,0]
    mu = torch.as_tensor(class_mean,device=p.device,dtype=p.dtype)[...,0].detach()
    v = torch.as_tensor(class_variance,device=p.device,dtype=p.dtype)[...,0].detach()
    vd = torch.as_tensor(delta_variance,device=p.device,dtype=p.dtype)[...,0].detach()
    if mu.ndim != 2 or mu.size(1) != p.size(1) or v.shape != mu.shape or vd.shape != (len(mu),p.size(1)-1):
        raise ValueError("native spectral statistics require [C,B] and [C,B-1]")
    if bool((v<0).any()) or bool((vd<0).any()):
        raise ValueError("negative native spectral variance")
    # Native P is a bounded, robustly normalized spectrum. Ridge prevents low-
    # count classes from acquiring infinite confidence along zero-variance bands.
    ridge=1e-3
    gap=(p[:,None,:]-mu[None])
    abs_score=torch.exp(-0.5*(gap.square()/(v[None]+ridge)).mean(-1)).clamp(0,1)
    dp=p[:,1:]-p[:,:-1]
    dmu=mu[:,1:]-mu[:,:-1]
    dg=dp[:,None,:]-dmu[None]
    rel_score=torch.exp(-0.5*(dg.square()/(vd[None]+ridge)).mean(-1)).clamp(0,1)
    return abs_score, rel_score, torch.sqrt(abs_score*rel_score).clamp(0,1)


def combine_learned_physical_relevance(learned: Tensor, physical: Tensor) -> Tensor:
    """Fuse learned and physical relevance with learned semantics as the gate.

    Physical similarity can strengthen an existing learned-space relation but
    cannot create semantic relevance by itself. This same rule is used by feature
    separation, historical decision protection, and Whole-CDS diagnostics.
    """
    learned_value = torch.as_tensor(learned)
    physical_value = torch.as_tensor(
        physical, device=learned_value.device, dtype=learned_value.dtype
    )
    if learned_value.shape != physical_value.shape:
        raise ValueError("learned and physical relevance must have identical shape")
    if not bool(torch.isfinite(learned_value).all()) or not bool(
        torch.isfinite(physical_value).all()
    ):
        raise ValueError("learned/physical relevance contains NaN or Inf")
    learned_value = learned_value.clamp(0.0, 1.0)
    physical_value = physical_value.clamp(0.0, 1.0)
    return (
        learned_value + (1.0 - learned_value) * learned_value * physical_value
    ).clamp(0.0, 1.0)


def combine_class_pair_rivalry(learned: Tensor, physical: Tensor) -> Tensor:
    """Joint R_Q/R_P *class-pair rivalry* (not sample-to-class support).

    Either independent source of class ambiguity can flag a difficult pair.
    A max is bounded, symmetric for symmetric inputs, parameter-free, and avoids
    counting the same ambiguity twice. Do not use this for sample support:
    combine_learned_physical_relevance has a different semantic contract.
    """
    r_q = torch.as_tensor(learned)
    r_p = torch.as_tensor(physical, device=r_q.device, dtype=r_q.dtype)
    if r_q.shape != r_p.shape or r_q.ndim != 2 or r_q.shape[0] != r_q.shape[1]:
        raise ValueError("R_Q/R_P class-pair rivalry must be aligned square [C,C]")
    if not all(bool(torch.isfinite(t).all()) for t in (r_q, r_p)):
        raise ValueError("R_Q/R_P contain NaN or Inf")
    return torch.maximum(r_q.clamp(0.0, 1.0), r_p.clamp(0.0, 1.0))


def _ordered_identity_and_variation(
    values: Tensor, *, row_chunk: int = 8
) -> tuple[Tensor, Tensor]:
    """Native spectrum arithmetic mean and signed 10th–90th percentile residual ranges.

    Returns I_P [B,1], compact V_P [2,2B-1,1]. row_chunk remains for
    source API compatibility, but no quadratic band-pair operation occurs.
    """
    sample = _validate_physical_evidence(values)
    identity = sample.mean(dim=0)
    residual = sample - identity.unsqueeze(0)
    adjacent = residual[:, 1:] - residual[:, :-1]
    evidence = torch.cat((residual, adjacent), dim=1)
    lo = _deterministic_quantile_dim0(evidence, 0.10)
    hi = _deterministic_quantile_dim0(evidence, 0.90)
    return identity.contiguous(), torch.stack((lo, hi)).contiguous()


def _learned_absolute_class_proximity(identity: Tensor, absolute_deviation: Tensor | None = None) -> Tensor:
    """Inter-class I_Q proximity relative to observed absolute within-class A_Q."""
    center = torch.as_tensor(identity)
    c, k, d = center.shape
    delta = center[:, None, :, :] - center[None, :, :, :]
    base = (1.0 / (1.0 + delta.norm(dim=-1) * math.sqrt(k))).clamp(0, 1)
    if absolute_deviation is None:
        return base
    a = torch.as_tensor(absolute_deviation,device=center.device,dtype=center.dtype)
    if a.shape != center.shape or not bool(torch.isfinite(a).all()) or bool((a < 0).any()):
        raise ValueError("A_Q must be finite, nonnegative [C,K,D]")
    tolerance = math.sqrt(torch.finfo(center.dtype).eps) / math.sqrt(k*d)
    scale = a[:, None] + a[None, :] + tolerance
    overlap = 1.0 / (1.0 + (delta.abs() / scale).mean(dim=-1))
    return torch.sqrt(base * overlap.clamp(0,1))


def _learned_relational_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
    """Symmetric cross-class quantile-interval intrusion; high means rivals."""
    center = torch.as_tensor(identity)
    c, k, d = center.shape
    v = validate_relational_variation(variation, length=k, features=d)
    if v.shape[0] != c:
        raise ValueError("learned identity/V_Q class counts disagree")
    relation = center[:, 1:, :] - center[:, :-1, :]
    delta = relation[None, :, :, :] - relation[:, None, :, :]
    lo_c, hi_c = v[:, 0, :, :][:, None], v[:, 1, :, :][:, None]
    lo_j, hi_j = v[:, 0, :, :][None, :], v[:, 1, :, :][None, :]
    violation_c = (lo_c - delta).relu() + (delta - hi_c).relu()
    violation_j = (lo_j + delta).relu() + (-delta - hi_j).relu()
    scale = math.sqrt(k * d)
    support_c = 1.0 / (1.0 + scale * violation_c.mean(dim=(-1, -2)))
    support_j = 1.0 / (1.0 + scale * violation_j.mean(dim=(-1, -2)))
    result = 0.5 * (support_c + support_j)
    result.diagonal().fill_(1.0)
    return result.clamp(0, 1)


def _physical_cross_class_support(
    identity: Tensor, variation: Tensor, *, adjacent: bool
) -> Tensor:
    """Symmetric rival compatibility of physical identities in stored intervals."""
    center = torch.as_tensor(identity)
    bounds = sanitize_ordered_variation(variation)
    if (
        center.ndim != 3
        or center.size(-1) != 1
        or bounds.shape != (center.size(0), 2, 2 * center.size(1) - 1, 1)
    ):
        raise ValueError("physical identity/variation shape mismatch")
    b = center.size(1)
    if adjacent:
        center = center[:, 1:] - center[:, :-1]
        lo, hi = bounds[:, 0, b:], bounds[:, 1, b:]
    else:
        lo, hi = bounds[:, 0, :b], bounds[:, 1, :b]
    # delta[c,j] is the observation from class c, evaluated by class j.
    delta = center[:, None] - center[None, :]
    as_j = _interval_compatibility(delta, lo[None], hi[None], center[None])
    as_c = _interval_compatibility(-delta, lo[:, None], hi[:, None], center[:, None])
    result = 0.5 * (as_j + as_c)
    diagonal = torch.arange(center.size(0), device=center.device)
    result[diagonal, diagonal] = 1.0
    return result.mean(dim=-1).clamp(0, 1)


def _absolute_class_proximity_by_slot(identity: Tensor, variation: Tensor) -> Tensor:
    return _physical_cross_class_support(identity, variation, adjacent=False)


def _relation_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
    return _physical_cross_class_support(identity, variation, adjacent=True).mean(dim=2)


@dataclass(frozen=True)
class PhysicalSpectralStructureBatch:
    class_ids: Tensor
    sample_counts: Tensor
    physical_identity: Tensor
    physical_variation: Tensor
    physical_variance: Tensor  # [C,B,1], exact native-band population variance
    physical_delta_variance: Tensor  # [C,B-1,1], adjacent-band difference variance

    @property
    def class_count(self) -> int:
        return int(self.class_ids.numel())


@dataclass(frozen=True)
class ClassStructureBatch:
    class_ids: Tensor
    sample_counts: Tensor
    learned_identity: Tensor  # [C,K,D], arithmetic mean in normalized Q
    learned_variation: Tensor  # [C,2,K-1,D], signed V_Q
    learned_absolute_deviation: Tensor  # [C,K,D], mean |Q - I_Q|; NOT covariance
    physical_identity: Tensor  # [C,B,1], mean native spectrum
    physical_variation: Tensor  # [C,2,2B-1,1], retained native-band diagnostics
    physical_variance: Tensor  # [C,B,1]
    physical_delta_variance: Tensor  # [C,B-1,1]
    learned_covariance_basis: Tensor  # [C,F,R], ONE learned distribution per class
    learned_covariance_eigenvalues: Tensor  # [C,R]
    learned_covariance_residual: Tensor  # [C], isotropic residual variance

    @property
    def class_count(self) -> int:
        return int(self.class_ids.numel())


@dataclass(frozen=True)
class ClassRelationState:
    class_ids: Tensor
    learned_identity_proximity_by_region: Tensor
    learned_relation_proximity: Tensor
    physical_identity_proximity_by_band: Tensor
    physical_relation_proximity: Tensor
    learned_prototype_cosine: Tensor  # PCDS-style angular inter-class structure

    @property
    def learned_identity_proximity(self):
        return self.learned_identity_proximity_by_region.mean(dim=2)

    @property
    def learned_combined_proximity(self):
        return torch.sqrt(
            self.learned_identity_proximity.clamp(0, 1)
            * self.learned_relation_proximity.clamp(0, 1)
        )

    @property
    def physical_identity_proximity(self):
        discrimination = (1 - self.physical_identity_proximity_by_band).clamp(0, 1)
        mass = discrimination.sum(dim=2)
        strength = discrimination.square().sum(dim=2) / mass.clamp_min(
            torch.finfo(discrimination.dtype).eps
        )
        return (1 - strength).clamp(0, 1)

    @property
    def physical_combined_proximity(self):
        return torch.sqrt(
            self.physical_identity_proximity.clamp(0, 1)
            * self.physical_relation_proximity.clamp(0, 1)
        )

    @property
    def physical_band_discriminability(self):
        return (1 - self.physical_identity_proximity).clamp(0, 1)


@dataclass(frozen=True)
class ClassDiscriminativeStructureState:
    classes: ClassStructureBatch
    relations: ClassRelationState
    spectral_band_positions: Tensor
    learned_moments: dict[str, Tensor] | None = None

    @property
    def conditional_moments(self):
        # Read-only compatibility for existing diagnostic callers. No conditions.
        return self.learned_moments


@dataclass(frozen=True)
class ClassStructureView:
    class_id: int
    sample_count: int
    learned_identity: Tensor
    learned_variation: Tensor
    learned_absolute_deviation: Tensor
    physical_identity: Tensor
    physical_variation: Tensor
    physical_variance: Tensor
    physical_delta_variance: Tensor
    learned_covariance_basis: Tensor
    learned_covariance_eigenvalues: Tensor
    learned_covariance_residual: Tensor


def estimate_physical_structure_batch(
    physical_evidence: Tensor, labels: Tensor, *, class_ids
) -> PhysicalSpectralStructureBatch:
    p = _validate_physical_evidence(physical_evidence)
    y = torch.as_tensor(labels, device=p.device, dtype=torch.long).flatten()
    ids = _class_id_tuple(class_ids, name="class_ids")
    if y.shape != (p.size(0),) or set(map(int, y.unique().tolist())) != set(ids):
        raise ValueError("physical training labels must match class IDs")
    identity, variation, count, variance, delta_variance = [], [], [], [], []
    for cid in ids:
        x = p[y == cid]
        i, v = _ordered_identity_and_variation(x)
        identity.append(i)
        variation.append(v)
        count.append(x.size(0))
        variance.append(x.var(dim=0, unbiased=False))
        delta_variance.append((x[:, 1:] - x[:, :-1]).var(dim=0, unbiased=False))
    return PhysicalSpectralStructureBatch(
        torch.tensor(ids, device=p.device, dtype=torch.long),
        torch.tensor(count, device=p.device, dtype=torch.long),
        torch.stack(identity), torch.stack(variation),
        torch.stack(variance), torch.stack(delta_variance),
    )


def combine_learned_and_physical_structure(
    ordered_features: Tensor,
    labels: Tensor,
    physical_structure: PhysicalSpectralStructureBatch,
    *,
    coupled_rank=None,
    coupled_quantile=1.0,
    moment_rank=8,
    moment_covariance_shrinkage=0.10,
) -> ClassStructureBatch:
    q = normalize_ordered_decision_features(ordered_features)
    y = torch.as_tensor(labels, device=q.device, dtype=torch.long).flatten()
    ids = physical_structure.class_ids.tolist()
    if set(y.unique().tolist()) != set(ids) or y.numel() != q.size(0):
        raise ValueError("Q class IDs mismatch")
    identity = []
    variation = []
    dispersion, basis, eigenvalues, residuals = [], [], [], []
    for cid in ids:
        x = q[y == cid]
        mean = x.mean(dim=0)
        identity.append(mean)
        # Old V_Q/A_Q remain read-only diagnostics for existing reports.
        variation.append(estimate_relational_variation(x, mean))
        dispersion.append((x - mean).abs().mean(dim=0))
        u, lam, res = _compact_covariance(
            x.flatten(1), rank=moment_rank, class_seed=73 + 997 * int(cid),
            shrinkage=moment_covariance_shrinkage,
        )
        basis.append(u); eigenvalues.append(lam); residuals.append(res)
    return ClassStructureBatch(
        physical_structure.class_ids,
        physical_structure.sample_counts,
        torch.stack(identity), torch.stack(variation), torch.stack(dispersion),
        physical_structure.physical_identity, physical_structure.physical_variation,
        physical_structure.physical_variance, physical_structure.physical_delta_variance,
        torch.stack(basis), torch.stack(eigenvalues), torch.stack(residuals),
    )


def estimate_structure_batch(
    ordered_features,
    physical_evidence,
    labels,
    *,
    class_ids,
    coupled_rank=None,
    coupled_quantile=1.0,
    moment_rank=8,
    moment_covariance_shrinkage=0.10,
) -> ClassStructureBatch:
    return combine_learned_and_physical_structure(
        ordered_features,
        labels,
        estimate_physical_structure_batch(
            physical_evidence, labels, class_ids=class_ids
        ),
        coupled_rank=coupled_rank,
        coupled_quantile=coupled_quantile,
        moment_rank=moment_rank,
        moment_covariance_shrinkage=moment_covariance_shrinkage,
    )


def merge_class_structure_batches(structures) -> ClassStructureBatch:
    xs = list(structures)
    if not xs:
        raise ValueError("missing class structures")
    ids = [i for x in xs for i in x.class_ids.tolist()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate class IDs")
    values = {
        name: torch.cat([getattr(x, name) for x in xs], dim=0)
        for name in ClassStructureBatch.__dataclass_fields__
    }
    return ClassStructureBatch(**values)


def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
    return ClassRelationState(
        structure.class_ids,
        _learned_absolute_class_proximity(structure.learned_identity,
                                          structure.learned_absolute_deviation),
        _learned_relational_class_proximity(
            structure.learned_identity, structure.learned_variation
        ),
        _absolute_class_proximity_by_slot(
            structure.physical_identity, structure.physical_variation
        ),
        _relation_class_proximity(
            structure.physical_identity, structure.physical_variation
        ),
        torch.nn.functional.normalize(structure.learned_identity.flatten(1), dim=1)
        @ torch.nn.functional.normalize(structure.learned_identity.flatten(1), dim=1).T,
    )


def normalized_moment_translation_feasible(moments, class_translation):
    """Per-class gate for a pure translation of unit-normalized Q moments.

    Historical within-class covariance is unchanged by a common Euclidean
    translation. A change in the squared mean norm must therefore not worsen
    the existing unit-sphere second-moment identity. We compare against the
    original compact covariance approximation error, not an exact dense model.
    """
    mu = moments["mean"]
    basis, eig, residual = (moments[k] for k in
        ("basis", "eigenvalues", "residual_variance"))
    counts = moments["counts"]
    if mu.ndim != 2:
        raise ValueError("unified historical means must be [C,F]")
    proposed = torch.as_tensor(class_translation, device=mu.device, dtype=mu.dtype)
    if proposed.numel() != mu.numel() or proposed.shape[0] != mu.size(0):
        raise ValueError("historical class translation and moments disagree")
    shift = proposed.reshape_as(mu)
    if not bool(torch.isfinite(shift).all()):
        raise ValueError("non-finite proposed historical translation")
    active_rank = (basis.square().sum(-2) > 1e-10).sum(-1)
    trace = eig.sum(-1) + (mu.size(-1) - active_rank) * residual
    factor = (counts - 1).clamp_min(0).to(mu) / counts.clamp_min(1).to(mu)
    old_energy = mu.square().sum(-1) + factor * trace
    new_energy = (mu + shift).square().sum(-1) + factor * trace
    valid = counts > 0
    # Numerical slack is only for floating-point accumulation; the original
    # compact approximation's own deviation is not silently treated as zero.
    slack = 128 * torch.finfo(mu.dtype).eps * (1.0 + old_energy.abs())
    acceptable = ((new_energy - 1.0).abs() <= (old_energy - 1.0).abs() + slack)
    acceptable &= (mu + shift).square().sum(-1) <= 1.0 + slack
    return acceptable | ~valid


def whole_cds_evidence_masks(classes: ClassStructureBatch):
    n = classes.sample_counts
    v = (n[:, None] > 1) & (n[None, :] > 1)
    return v, v


def whole_cds_pair_risk(cds: ClassDiscriminativeStructureState):
    """One inter-class risk from within-class learned variance and native spectra.

    Computation is descriptive (not calibrated overlap probability). All class
    pair scores are symmetric and detached when used as weighting evidence.
    V_Q and absolute-deviation diagnostics do not enter this training risk.
    """
    st = cds.classes
    mu = st.learned_identity.flatten(1)
    # Use covariance ORIENTATION, not just a trace or scalar radius.
    # Project each historical class covariance along the actual pairwise
    # mean-separation direction; that is the ambiguous direction of the pair.
    delta = mu[:, None, :] - mu[None, :, :]
    distance2 = delta.square().sum(-1)
    separation = delta.norm(dim=-1)
    direction = delta / separation.clamp_min(torch.finfo(mu.dtype).eps)[..., None]
    basis = st.learned_covariance_basis
    eigen = st.learned_covariance_eigenvalues
    residual = st.learned_covariance_residual
    source_proj = torch.einsum("ijf,ifr->ijr",direction,basis)
    source_var = (source_proj.square()*eigen[:,None,:]).sum(-1)
    source_var += residual[:,None]*(direction.square().sum(-1)-source_proj.square().sum(-1)).clamp_min(0)
    # Same directional quadratic form, with the *rival's* own covariance.
    rival_proj = torch.einsum("ijf,jfr->ijr",direction,basis)
    rival_var = (rival_proj.square()*eigen[None,:,:]).sum(-1)
    rival_var += residual[None,:]*(direction.square().sum(-1)-rival_proj.square().sum(-1)).clamp_min(0)
    counts = st.sample_counts.to(mu).clamp_min(1)
    variance = (source_var * ((counts-1)/counts)[:,None] +
                rival_var * ((counts-1)/counts)[None,:]).clamp_min(0)
    learned = variance / (variance + distance2).clamp_min(torch.finfo(mu.dtype).tiny)
    # With coincident means there is no separating direction to project onto.
    # Zero projected variance must not be interpreted as zero ambiguity.
    # This is a descriptive rivalry score, not an overlap probability.
    learned = torch.where(separation == 0, torch.ones_like(learned), learned)
    p = st.physical_identity[..., 0]
    p_var = st.physical_variance[..., 0]
    slope = p[:, 1:] - p[:, :-1]
    slope_var = st.physical_delta_variance[..., 0]
    def similarity(value, variance):
        delta = value[:, None] - value[None]
        spread_pair = variance[:, None] + variance[None] + 1e-4
        return torch.exp(-0.5 * (delta.square() / spread_pair).mean(-1))
    physical = (similarity(p, p_var) * similarity(slope, slope_var)).sqrt()
    # A spectral-only similarity cannot override an already well-separated
    # learned pair. Its purpose is to prioritize supported learned rivalry.
    effective = combine_learned_physical_relevance(learned, physical)
    eye = torch.eye(len(mu), device=mu.device, dtype=torch.bool)
    return tuple(v.masked_fill(eye, 0.0).clamp(0, 1).contiguous()
                 for v in (effective, learned, physical))


class HSIClassDiscriminativeStructureBank(nn.Module):
    """Canonical exemplar-free inter/intra/spectral class statistics.

    The bank alone owns one low-rank learned covariance per class and native spectral variation.
    No raw HSI samples, feature exemplars or dense covariance matrices persist.
    """

    def __init__(
        self,
        *,
        spectral_token_count,
        spectral_token_dim,
        spectral_bands,
        spectral_band_positions=None,
        coupled_rank=None,
        coupled_quantile=1.0,
        moment_rank=8,
        moment_min_per_group=24,
        moment_covariance_shrinkage=0.10,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.spectral_token_count = _integer(spectral_token_count, "spectral_token_count", minimum=2)
        self.spectral_token_dim = _integer(spectral_token_dim, "spectral_token_dim", minimum=1)
        self.spectral_bands = _integer(spectral_bands, "spectral_bands", minimum=2)
        self.moment_rank = _integer(moment_rank, "moment_rank", minimum=1)
        self.moment_min_per_group = _integer(moment_min_per_group, "moment_min_per_group", minimum=2)
        self.moment_covariance_shrinkage = float(moment_covariance_shrinkage)
        if not 0 < self.moment_covariance_shrinkage < 1:
            raise ValueError("moment covariance shrinkage must be in (0,1)")
        k = self.spectral_token_count
        d = self.spectral_token_dim
        b = self.spectral_bands
        self.register_buffer(
            "spectral_band_positions",
            _band_positions(
                spectral_band_positions, count=b, device=device, dtype=dtype
            ),
        )
        for key, x in {
            "class_ids": torch.empty(0, device=device, dtype=torch.long),
            "sample_count_bank": torch.empty(0, device=device, dtype=torch.long),
            "learned_identity_bank": torch.empty(0, k, d, device=device, dtype=dtype),
            "learned_variation_bank": torch.empty(
                0, 2, k - 1, d, device=device, dtype=dtype
            ),
            "learned_absolute_deviation_bank": torch.empty(
                0, k, d, device=device, dtype=dtype
            ),
            "physical_identity_bank": torch.empty(0, b, 1, device=device, dtype=dtype),
            "physical_variation_bank": torch.empty(
                0, 2, 2 * b - 1, 1, device=device, dtype=dtype
            ),
            "physical_variance_bank": torch.empty(0, b, 1, device=device, dtype=dtype),
            "physical_delta_variance_bank": torch.empty(0, b - 1, 1, device=device, dtype=dtype),
        }.items():
            self.register_buffer(key, x, persistent=True)
        f = k * d
        r = self.moment_rank
        # Canonical per-class low-rank moments. No spectral-condition groups.
        for key, x in {
            "moment_mean_bank": torch.empty(0, f, device=device, dtype=dtype),
            "moment_basis_bank": torch.empty(0, f, r, device=device, dtype=dtype),
            "moment_eigenvalues_bank": torch.empty(0, r, device=device, dtype=dtype),
            "moment_residual_variance_bank": torch.empty(0, device=device, dtype=dtype),
            "moment_counts_bank": torch.empty(0, device=device, dtype=torch.long),
        }.items():
            self.register_buffer(key, x, persistent=True)

    def __len__(self):
        return int(self.class_ids.numel())

    @property
    def device(self):
        return self.learned_identity_bank.device

    @property
    def dtype(self):
        return self.learned_identity_bank.dtype

    @property
    def seen_class_ids(self):
        return tuple(map(int, self.class_ids.tolist()))

    def validate_state(self):
        c = len(self)
        k = self.spectral_token_count
        d = self.spectral_token_dim
        b = self.spectral_bands
        for name, shape in [
            ("sample_count_bank", (c,)),
            ("learned_identity_bank", (c, k, d)),
            ("learned_variation_bank", (c, 2, k - 1, d)),
            ("learned_absolute_deviation_bank", (c, k, d)),
            ("physical_identity_bank", (c, b, 1)),
            ("physical_variation_bank", (c, 2, 2 * b - 1, 1)),
            ("physical_variance_bank", (c,b,1)),
            ("physical_delta_variance_bank", (c,b-1,1)),
        ]:
            t = getattr(self, name)
            if tuple(t.shape) != shape:
                raise RuntimeError(f"Whole-CDS {name} shape mismatch")
            if torch.is_floating_point(t) and not bool(torch.isfinite(t).all()):
                raise RuntimeError(f"{name} contains nonfinite statistics")
        if c and (
            (self.class_ids.unique().numel() != c)
            or bool((self.sample_count_bank <= 0).any())
        ):
            raise RuntimeError("invalid CDS classes")
        f = k * d
        r = self.moment_rank
        moments = {
            "moment_mean_bank": (c, f),
            "moment_basis_bank": (c, f, r),
            "moment_eigenvalues_bank": (c, r),
            "moment_residual_variance_bank": (c,),
            "moment_counts_bank": (c,),
            "physical_variance_bank": (c,b,1),
            "physical_delta_variance_bank": (c,b-1,1),
        }
        for name, shape in moments.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise RuntimeError(f"Unified CDS {name} shape mismatch: {value.shape} != {shape}")
            if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Unified CDS {name} contains non-finite statistics")
        if c:
            if not torch.equal(self.moment_counts_bank, self.sample_count_bank):
                raise RuntimeError("one CDS covariance requires one count per class")
            if not torch.allclose(self.moment_mean_bank, self.learned_identity_bank.flatten(1), rtol=2e-5,atol=2e-5):
                raise RuntimeError("unified class means disagree")
            for name in ("moment_eigenvalues_bank","moment_residual_variance_bank","physical_variance_bank","physical_delta_variance_bank"):
                if bool((getattr(self,name) < 0).any()):
                    raise RuntimeError(f"negative unified variance in {name}")
            # The complement covariance sigma2*(I-UU^T) is valid only for
            # orthonormal active columns; padding columns must remain zero.
            basis = self.moment_basis_bank
            active = basis.square().sum(dim=1) > 1e-10
            gram = basis.transpose(1, 2) @ basis
            target = torch.diag_embed(active.to(basis.dtype))
            tolerance = max(2e-5, 32 * torch.finfo(basis.dtype).eps)
            if not torch.allclose(gram, target, rtol=tolerance, atol=tolerance):
                raise RuntimeError("learned covariance basis must have orthonormal active columns")
            if bool((self.moment_eigenvalues_bank.masked_select(~active) != 0).any()):
                raise RuntimeError("inactive covariance columns must have zero eigenvalues")
        if bool((self.learned_absolute_deviation_bank < 0).any()):
            raise RuntimeError("negative dispersion")
        if bool(
            (
                self.learned_variation_bank[:, 0] > self.learned_variation_bank[:, 1]
            ).any()
        ):
            raise RuntimeError("V_Q interval inverted")
        if bool(
            (
                self.physical_variation_bank[:, 0] > self.physical_variation_bank[:, 1]
            ).any()
        ):
            raise RuntimeError("V_P interval inverted")
        return True

    def _validate_batch(self, structure):
        if not isinstance(structure, ClassStructureBatch):
            raise TypeError("expected ClassStructureBatch")
        if structure.learned_identity.shape[1:] != (
            self.spectral_token_count,
            self.spectral_token_dim,
        ):
            raise ValueError("learned feature size mismatch")
        if structure.physical_identity.shape[1:] != (self.spectral_bands, 1):
            raise ValueError("physical band count mismatch")
        if not bool((structure.learned_absolute_deviation >= 0).all()):
            raise ValueError("negative dispersion")
        for name in ClassStructureBatch.__dataclass_fields__:
            x = getattr(structure, name)
            if x.device != self.device:
                raise ValueError(f"{name} wrong device")
            if torch.is_floating_point(x) and x.dtype != self.dtype:
                raise ValueError(f"{name} wrong dtype")

    def estimate_physical(self, physical_evidence, labels, *, class_ids):
        return estimate_physical_structure_batch(
            physical_evidence, labels, class_ids=class_ids
        )

    def estimate_classes(
        self,
        ordered_features,
        physical_evidence,
        labels,
        *,
        class_ids,
        physical_structure=None,
    ):
        requested_ids = _class_id_tuple(class_ids, name="class_ids")
        if physical_structure is None:
            if physical_evidence is None:
                raise ValueError("missing physical evidence")
            physical_structure = self.estimate_physical(
                physical_evidence, labels, class_ids=requested_ids
            )
        if tuple(physical_structure.class_ids.tolist()) != requested_ids:
            raise ValueError("physical structure class IDs/order differ from requested class IDs")
        return combine_learned_and_physical_structure(
            ordered_features, labels, physical_structure,
            moment_rank=self.moment_rank,
            moment_covariance_shrinkage=self.moment_covariance_shrinkage,
        )

    @torch.no_grad()
    def estimate_compact_moments(self, ordered_features, physical_evidence, labels, *, class_ids):
        """One unconditional compact distribution per observed class; no synthesis."""
        q = normalize_ordered_decision_features(ordered_features).detach()
        p = _validate_physical_evidence(torch.as_tensor(
            physical_evidence, device=q.device, dtype=q.dtype)).detach()
        y = torch.as_tensor(labels, device=q.device, dtype=torch.long).flatten()
        ids = _class_id_tuple(class_ids, name="class_ids")
        if q.shape[1:] != (self.spectral_token_count,self.spectral_token_dim):
            raise ValueError("Q coordinate mismatch")
        if p.shape != (len(q),self.spectral_bands,1) or y.shape != (len(q),):
            raise ValueError("paired Q/P and labels required")
        if set(y.unique().tolist()) != set(ids):
            raise ValueError("moment estimation must use exactly current training classes")
        means, axes, eig, residual, counts = [],[],[],[],[]
        for cid in ids:
            x=q[y==cid].flatten(1)
            mu=x.mean(0)
            u,l,r=_compact_covariance(x,rank=self.moment_rank,
                class_seed=73+997*cid,shrinkage=self.moment_covariance_shrinkage)
            means.append(mu); axes.append(u); eig.append(l); residual.append(r)
            counts.append(len(x))
        return dict(moment_mean_bank=torch.stack(means),
                    moment_basis_bank=torch.stack(axes),
                    moment_eigenvalues_bank=torch.stack(eig),
                    moment_residual_variance_bank=torch.stack(residual),
                    moment_counts_bank=torch.tensor(counts,device=q.device,dtype=torch.long))

    @torch.no_grad()
    def replace_all_with_moments(self, structure, ordered_features, physical_evidence, labels):
        """Atomically replace CDS and moments, retaining prior old-class rows.

        No old examples are accessed. An old CDS coordinate change is rejected
        until a scientifically valid companion moment transport is implemented.
        """
        self._validate_batch(structure)
        all_ids = tuple(map(int, structure.class_ids.tolist()))
        current = set(self.seen_class_ids)
        incoming = tuple(cid for cid in all_ids if cid not in current)
        if not incoming:
            raise ValueError("commit needs at least one genuinely new class")
        y = torch.as_tensor(labels, device=self.device, dtype=torch.long).flatten()
        if set(y.tolist()) != set(incoming):
            raise ValueError("moment fitting may only read newly introduced classes")
        if set(self.seen_class_ids) - set(all_ids):
            raise ValueError("cannot drop committed historical classes")
        for cid in self.seen_class_ids:
            row = self._row_for_class(cid)
            next_row = all_ids.index(cid)
            for source, target in (
                ("learned_identity", "learned_identity_bank"),
                ("learned_variation", "learned_variation_bank"),
                ("learned_absolute_deviation", "learned_absolute_deviation_bank"),
                ("physical_identity", "physical_identity_bank"),
                ("physical_variation", "physical_variation_bank"),
                ("physical_variance", "physical_variance_bank"),
                ("physical_delta_variance", "physical_delta_variance_bank"),
                ("learned_covariance_basis", "moment_basis_bank"),
                ("learned_covariance_eigenvalues", "moment_eigenvalues_bank"),
                ("learned_covariance_residual", "moment_residual_variance_bank"),
            ):
                if not torch.equal(getattr(structure, source)[next_row], getattr(self, target)[row]):
                    raise RuntimeError("historical CDS has shifted without joint moment transport")
        estimate = self.estimate_compact_moments(
            ordered_features, physical_evidence, labels, class_ids=incoming
        )
        # A caller cannot silently commit contradictory class covariance state.
        # The same observed training features must produce the stored learned
        # covariance used in inter-class risk, historical risk and Q support.
        incoming_indices = torch.tensor([all_ids.index(cid) for cid in incoming],
                                        device=self.device,dtype=torch.long)
        for structure_name, moment_name in (
            ("learned_identity", "moment_mean_bank"),
            ("learned_covariance_basis", "moment_basis_bank"),
            ("learned_covariance_eigenvalues", "moment_eigenvalues_bank"),
            ("learned_covariance_residual", "moment_residual_variance_bank"),
        ):
            stored = getattr(structure,structure_name).index_select(0,incoming_indices)
            if structure_name == "learned_identity":
                stored = stored.flatten(1)
            derived = estimate[moment_name]
            if not torch.allclose(stored,derived,rtol=2e-5,atol=2e-5):
                raise RuntimeError(f"unified covariance mismatch for {structure_name}")
        pp = _validate_physical_evidence(torch.as_tensor(
            physical_evidence,device=self.device,dtype=self.dtype))
        for i,cid in enumerate(incoming):
            obs=pp[y==cid]
            expected_pv=obs.var(dim=0,unbiased=False)
            expected_dv=(obs[:,1:]-obs[:,:-1]).var(dim=0,unbiased=False)
            j=all_ids.index(cid)
            if not torch.allclose(structure.physical_identity[j], obs.mean(dim=0),
                                  rtol=2e-5, atol=2e-5):
                raise RuntimeError("unified physical spectral mean mismatch")
            if not torch.allclose(structure.physical_variance[j],expected_pv,rtol=2e-5,atol=2e-5) or not torch.allclose(
                structure.physical_delta_variance[j],expected_dv,rtol=2e-5,atol=2e-5):
                raise RuntimeError("unified physical spectral variance mismatch")
        expected = torch.tensor([int((y == cid).sum()) for cid in incoming], device=y.device)
        incoming_rows = torch.tensor([all_ids.index(cid) for cid in incoming], device=y.device)
        if not torch.equal(structure.sample_counts[incoming_rows], expected):
            raise RuntimeError("class counts differ from moment training samples")
        keys = tuple(estimate)
        combined = {}
        for key in keys:
            row_values = []
            for cid in all_ids:
                if cid in current:
                    row_values.append(getattr(self, key)[self._row_for_class(cid)])
                else:
                    row_values.append(estimate[key][incoming.index(cid)])
            combined[key] = torch.stack(row_values, dim=0)
        # Validate candidate in a temporary bank before changing this instance.
        candidate = HSIClassDiscriminativeStructureBank(
            spectral_token_count=self.spectral_token_count,
            spectral_token_dim=self.spectral_token_dim,
            spectral_bands=self.spectral_bands,
            spectral_band_positions=self.spectral_band_positions,
            moment_rank=self.moment_rank,
            moment_min_per_group=self.moment_min_per_group,
            moment_covariance_shrinkage=self.moment_covariance_shrinkage,
            device=self.device, dtype=self.dtype,
        )
        candidate._replace_base_unchecked(structure)
        for key, value in combined.items():
            setattr(candidate, key, value.detach().clone())
        candidate.validate_state()
        self._replace_base_unchecked(structure)
        for key, value in combined.items():
            setattr(self, key, value.detach().clone())
        self.validate_state()

    @torch.no_grad()
    def apply_historical_orthogonal_transform(self,class_ids,transform):
        """Atomic consistent historical mean/U update; spectral state untouched.

        The historical class-mean and covariance are BOTH rotated using the
        same unit-sphere isometry. No training features are stored or generated.
        """
        from models.phase_coordinate_alignment import rotate_historical_flat_rows
        if transform.rotation_basis is None:
            return
        basis = torch.as_tensor(transform.rotation_basis, device=self.device, dtype=self.dtype)
        rotation = torch.as_tensor(transform.rotation_matrix, device=self.device, dtype=self.dtype)
        if (basis.ndim != 2 or basis.size(0) != self.moment_mean_bank.size(1)
                or basis.size(1) < 1 or rotation.shape != (basis.size(1), basis.size(1))):
            raise ValueError("orthogonal transport basis/rotation shape mismatch")
        identity = torch.eye(basis.size(1), device=self.device, dtype=self.dtype)
        tolerance = max(2e-5, 32 * torch.finfo(self.dtype).eps)
        if (not bool(torch.isfinite(basis).all()) or not bool(torch.isfinite(rotation).all())
                or not torch.allclose(basis.T @ basis, identity, rtol=tolerance, atol=tolerance)
                or not torch.allclose(rotation.T @ rotation, identity, rtol=tolerance, atol=tolerance)):
            raise ValueError("historical transport must have an orthonormal basis and orthogonal rotation")
        ids=_class_id_tuple(class_ids,name="class_ids")
        rows=torch.tensor([self._row_for_class(i) for i in ids],device=self.device)
        previous_mean=self.moment_mean_bank.clone()
        previous_basis=self.moment_basis_bank.clone()
        proposed_mean=rotate_historical_flat_rows(previous_mean[rows],transform)
        proposed_basis=rotate_historical_flat_rows(previous_basis[rows].transpose(1,2),transform).transpose(1,2)
        before_norm=previous_mean[rows].norm(dim=1)
        if not bool(torch.allclose(proposed_mean.norm(dim=1),before_norm,rtol=1e-5,atol=1e-6)):
            raise RuntimeError("historical orthogonal transport did not preserve mean norm")
        self.moment_mean_bank[rows]=proposed_mean
        self.moment_basis_bank[rows]=proposed_basis
        self.learned_identity_bank[rows]=proposed_mean.view(-1,self.spectral_token_count,self.spectral_token_dim)
        try:
            self.validate_state()
        except Exception:
            self.moment_mean_bank.copy_(previous_mean)
            self.moment_basis_bank.copy_(previous_basis)
            self.learned_identity_bank[rows]=previous_mean[rows].view(-1,self.spectral_token_count,self.spectral_token_dim)
            raise

    @torch.no_grad()
    def apply_historical_translation(self,class_ids,class_translation):
        """Legacy unsafe migration stays prohibited; use orthogonal transport."""
        delta=torch.as_tensor(class_translation,device=self.device,dtype=self.dtype)
        if bool((delta != 0).any()):
            raise RuntimeError("Nonzero mean-only transport is prohibited; use apply_historical_orthogonal_transform")

    def class_moments(self, class_ids, *, device=None, dtype=None):
        """Detached, canonical, one-class-one-covariance historical description."""
        ids = _class_id_tuple(class_ids, name="class_ids")
        rows = torch.tensor([self._row_for_class(cid) for cid in ids], device=self.device)
        def take(key):
            v=getattr(self,key).index_select(0,rows).detach().clone()
            return v.to(device=device or self.device,dtype=(dtype or self.dtype) if v.is_floating_point() else torch.long)
        return dict(mean=take("moment_mean_bank"),basis=take("moment_basis_bank"),
                    eigenvalues=take("moment_eigenvalues_bank"),
                    residual_variance=take("moment_residual_variance_bank"),
                    counts=take("moment_counts_bank"),
                    physical_variance=take("physical_variance_bank")[...,0],
                    physical_delta_variance=take("physical_delta_variance_bank")[...,0])

    @staticmethod
    def _moment_keys():
        return ("moment_mean_bank","moment_basis_bank","moment_eigenvalues_bank",
                "moment_residual_variance_bank","moment_counts_bank")

    @torch.no_grad()
    def append_estimate(self, structure, *, ordered_features=None, physical_evidence=None, labels=None):
        """Commit additional observed classes; paired Q/P evidence is mandatory."""
        if ordered_features is None or physical_evidence is None or labels is None:
            raise ValueError("CDS inter/intra/spectral commit requires observed Q, P, labels")
        if set(self.seen_class_ids) & set(structure.class_ids.tolist()):
            raise ValueError("class already present")
        all_classes = structure if not len(self) else merge_class_structure_batches(
            [self.class_structures(), structure]
        )
        self.replace_all_with_moments(all_classes, ordered_features, physical_evidence, labels)

    @torch.no_grad()
    def replace_all(self, structure, *, ordered_features=None, physical_evidence=None, labels=None):
        if ordered_features is None or physical_evidence is None or labels is None:
            raise ValueError("CDS replacement requires observed Q, P, labels for new classes")
        self.replace_all_with_moments(structure, ordered_features, physical_evidence, labels)

    @torch.no_grad()
    def _replace_base_unchecked(self, structure):
        self._validate_batch(structure)
        mapping = {
            "class_ids": "class_ids",
            "sample_counts": "sample_count_bank",
            "learned_identity": "learned_identity_bank",
            "learned_variation": "learned_variation_bank",
            "learned_absolute_deviation": "learned_absolute_deviation_bank",
            "physical_identity": "physical_identity_bank",
            "physical_variation": "physical_variation_bank",
            "physical_variance": "physical_variance_bank",
            "physical_delta_variance": "physical_delta_variance_bank",
        }
        for source, target in mapping.items():
            setattr(self, target, getattr(structure, source).detach().clone())
        # Caller must attach the matching moment buffers and validate atomically.

    def _row_for_class(self, class_id):
        a = (self.class_ids == int(class_id)).nonzero().flatten()
        if a.numel() != 1:
            raise KeyError(class_id)
        return int(a.item())

    def class_structure(self, class_id):
        row = self._row_for_class(class_id)
        return ClassStructureView(
            class_id=int(class_id),
            sample_count=int(self.sample_count_bank[row]),
            learned_identity=self.learned_identity_bank[row].clone(),
            learned_variation=self.learned_variation_bank[row].clone(),
            learned_absolute_deviation=self.learned_absolute_deviation_bank[
                row
            ].clone(),
            physical_identity=self.physical_identity_bank[row].clone(),
            physical_variation=self.physical_variation_bank[row].clone(),
            physical_variance=self.physical_variance_bank[row].clone(),
            physical_delta_variance=self.physical_delta_variance_bank[row].clone(),
            learned_covariance_basis=self.moment_basis_bank[row].clone(),
            learned_covariance_eigenvalues=self.moment_eigenvalues_bank[row].clone(),
            learned_covariance_residual=self.moment_residual_variance_bank[row].clone(),
        )

    def class_structures(self, class_ids=None):
        if not len(self):
            raise RuntimeError("empty CDS bank")
        ids = (
            self.seen_class_ids
            if class_ids is None
            else _class_id_tuple(class_ids, name="class_ids")
        )
        rows = torch.tensor([self._row_for_class(x) for x in ids], device=self.device)
        return ClassStructureBatch(
            class_ids=self.class_ids[rows].clone(),
            sample_counts=self.sample_count_bank[rows].clone(),
            learned_identity=self.learned_identity_bank[rows].clone(),
            learned_variation=self.learned_variation_bank[rows].clone(),
            learned_absolute_deviation=self.learned_absolute_deviation_bank[
                rows
            ].clone(),
            physical_identity=self.physical_identity_bank[rows].clone(),
            physical_variation=self.physical_variation_bank[rows].clone(),
            physical_variance=self.physical_variance_bank[rows].clone(),
            physical_delta_variance=self.physical_delta_variance_bank[rows].clone(),
            learned_covariance_basis=self.moment_basis_bank[rows].clone(),
            learned_covariance_eigenvalues=self.moment_eigenvalues_bank[rows].clone(),
            learned_covariance_residual=self.moment_residual_variance_bank[rows].clone(),
        )

    def prototype_cosine(self, class_ids=None):
        """PCDS-style inter-class cosine of normalized learned class prototypes."""
        cls = self.class_structures(class_ids)
        mu = torch.nn.functional.normalize(cls.learned_identity.flatten(1), dim=1)
        return mu @ mu.T

    def build_cds(self, structure):
        self._validate_batch(structure)
        ids = tuple(map(int, structure.class_ids.detach().cpu().tolist()))
        committed = all(class_id in self.seen_class_ids for class_id in ids)
        if committed:
            for index, class_id in enumerate(ids):
                bank_row = self._row_for_class(class_id)
                for name, buffer_name in (
                    ("learned_identity", "learned_identity_bank"),
                    ("learned_variation", "learned_variation_bank"),
                    ("physical_identity", "physical_identity_bank"),
                    ("physical_variation", "physical_variation_bank"),
                ("physical_variance", "physical_variance_bank"),
                ("physical_delta_variance", "physical_delta_variance_bank"),
                ("learned_covariance_basis", "moment_basis_bank"),
                ("learned_covariance_eigenvalues", "moment_eigenvalues_bank"),
                ("learned_covariance_residual", "moment_residual_variance_bank"),
                ):
                    if not torch.equal(
                        getattr(structure, name)[index], getattr(self, buffer_name)[bank_row]
                    ):
                        committed = False
                        break
                if not committed:
                    break
        # The auxiliary classifier supports use moments only when the supplied
        # structure exactly matches one committed coordinate system. Raw cosine
        # logits never depend on these nonparametric statistics.
        moments = self.class_moments(ids) if committed else None
        return ClassDiscriminativeStructureState(
            classes=structure,
            relations=derive_class_relations(structure),
            spectral_band_positions=self.spectral_band_positions.detach().clone(),
            learned_moments=moments,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        required = (*self._moment_keys(), "physical_variance_bank", "physical_delta_variance_bank")
        mean_state = state_dict.get(prefix + "moment_mean_bank")
        if (any(prefix + key not in state_dict for key in required)
                or mean_state is not None and mean_state.ndim != 2):
            error_msgs.append("incompatible conditional or incomplete CDS checkpoint; rebuild Phase-0 from its training data under the unified CDS contract")
            return
        if any(prefix + key not in state_dict for key in self._moment_keys()):
            error_msgs.append("legacy checkpoint has no bank-owned moments; rebuild Phase-0 CDS from original training data")
            return
        if any(
            k.startswith(prefix + "learned_coupled_")
            or k == prefix + "learned_residual_scale_bank"
            for k in state_dict
        ):
            error_msgs.append(
                "G_Q checkpoint incompatible: rebuild class statistics from original phase TRAIN data"
            )
            return
        for key in (
            "class_ids",
            "sample_count_bank",
            "learned_identity_bank",
            "learned_variation_bank",
            "learned_absolute_deviation_bank",
            "physical_identity_bank",
            "physical_variation_bank",
            "physical_variance_bank", "physical_delta_variance_bank",
            *self._moment_keys(),
        ):
            k = prefix + key
            if k in state_dict and getattr(self, key).shape != state_dict[k].shape:
                x = state_dict[k]
                setattr(
                    self,
                    key,
                    torch.empty(
                        x.shape, device=self.device, dtype=getattr(self, key).dtype
                    ),
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if not error_msgs:
            self.validate_state()


__all__ = [
    "PhysicalSpectralStructureBatch",
    "ClassStructureBatch",
    "ClassRelationState",
    "ClassDiscriminativeStructureState",
    "ClassStructureView",
    "HSIClassDiscriminativeStructureBank",
    "estimate_physical_structure_batch",
    "combine_learned_and_physical_structure",
    "estimate_structure_batch",
    "merge_class_structure_batches",
    "normalize_ordered_decision_features",
    "sanitize_ordered_variation",
    "ordered_variation_operator",
    "ordered_cds_support_components",
    "estimate_relational_variation",
    "validate_relational_variation",
    "learned_relational_support",
    "combine_learned_physical_relevance",
    "combine_class_pair_rivalry",
    "derive_class_relations",
    "whole_cds_pair_risk",
    "whole_cds_evidence_masks",
    "physical_support_components",
    "spectral_conditioned_learned_support",
    "native_spectral_variation_support",
]

############## MEAN ENDS############

# from __future__ import annotations
# from dataclasses import dataclass
# from numbers import Integral
# import math
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor

# def _pool_groups(x: Tensor, count: int = 4) -> Tensor:
#     if x.ndim != 2 or x.shape[1] < 1:
#         raise ValueError("spectral descriptor must be [N,B], B>0")
#     return torch.stack(
#         [
#             chunk.mean(dim=1)
#             for chunk in torch.tensor_split(x, min(count, x.shape[1]), dim=1)
#         ],
#         dim=1,
#     )


# def _spectral_descriptor(p: Tensor) -> Tensor:
#     """Fixed band-index order; no physical wavelength-spacing assumption."""
#     p = torch.as_tensor(p)
#     if p.ndim == 3 and p.shape[-1] == 1:
#         p = p[..., 0]
#     if p.ndim != 2 or p.shape[1] < 2 or not bool(torch.isfinite(p).all()):
#         raise ValueError("physical spectral evidence must be finite [N,B,1] or [N,B]")
#     return torch.cat((_pool_groups(p, 4), _pool_groups(p[:, 1:] - p[:, :-1], 4)), dim=1)


# @torch.no_grad()
# def _spectral_conditions(
#     p: Tensor, *, min_per_group: int
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
#     """Split by *native spectral evidence only*; no learned-Q leakage into groups.

#     Returns membership, descriptor center, descriptor scale, principal spectral
#     axis, and split threshold. For undersampled/uninformative classes, one group.
#     The rule is estimated only when that class is first learned and is committed.
#     """
#     desc = _spectral_descriptor(p)
#     n, dim = desc.shape
#     membership = torch.zeros(n, dtype=torch.long, device=desc.device)
#     center = desc.mean(dim=0)
#     scale = desc.std(dim=0, unbiased=False).clamp_min(1e-4)
#     axis = torch.zeros(dim, dtype=desc.dtype, device=desc.device)
#     threshold = desc.new_zeros(())
#     if n < 2 * min_per_group:
#         return membership, center, scale, axis, threshold
#     normalized = (desc - center) / scale
#     cov = normalized.T @ normalized / max(n - 1, 1)
#     eigval, eigvec = torch.linalg.eigh(cov)
#     if not bool(torch.isfinite(eigval).all()) or float(eigval[-1]) < 1e-5:
#         return membership, center, scale, axis, threshold
#     axis = eigvec[:, -1].clone()
#     # Eigenvectors are sign-ambiguous: choose a reproducible orientation.
#     pivot = axis.abs().argmax()
#     if axis[pivot] < 0:
#         axis = -axis
#     projection = normalized @ axis
#     threshold = projection.median()
#     candidate = (projection > threshold).long()
#     if min(int((candidate == 0).sum()), int((candidate == 1).sum())) >= min_per_group:
#         membership = candidate
#     else:
#         axis = torch.zeros_like(axis)
#         threshold = threshold * 0
#     return membership, center, scale, axis, threshold


# @torch.no_grad()
# def _compact_covariance(
#     data: Tensor, *, rank: int, class_seed: int, shrinkage: float
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Deterministic low-rank + isotropic residual sample covariance.

#     C = U diag(lambda) U^T + sigma2 (I-UU^T), shrunk toward trace(C)/F * I.
#     No Gaussian model is fitted and no sampled features are generated.
#     """
#     if data.ndim != 2:
#         raise ValueError("features must be [N,F]")
#     n, f = data.shape
#     axes = data.new_zeros(f, rank)
#     lam = data.new_zeros(rank)
#     eps = 1e-7
#     if n < 2:
#         return axes, lam, data.new_tensor(eps)
#     x = data - data.mean(dim=0, keepdim=True)
#     trace = x.square().sum() / (n - 1)
#     r = min(rank, n - 1, f)
#     if r < 1 or float(trace) <= eps:
#         return axes, lam, (trace / f).clamp_min(eps)
#     k = min(f, n - 1, r + 4)
#     generator = torch.Generator(device="cpu")
#     generator.manual_seed(int(class_seed) % (2**63 - 1))
#     omega = torch.randn(f, k, generator=generator, dtype=torch.float64).to(
#         device=x.device, dtype=x.dtype
#     )
#     # Matrix-free sample-covariance power iteration, O(N*F*rank).
#     for _ in range(3):
#         y = x.T @ (x @ omega)
#         omega = torch.linalg.qr(y, mode="reduced").Q
#     small = x @ omega
#     _u, singular, vt = torch.linalg.svd(small, full_matrices=False)
#     basis = omega @ vt[:r].T
#     eig = singular[:r].square() / (n - 1)
#     remaining = (trace - eig.sum()).clamp_min(0)
#     isotropic = remaining / max(f - r, 1)
#     mean_variance = trace / f
#     axes[:, :r] = basis
#     lam[:r] = (1.0 - shrinkage) * eig + shrinkage * mean_variance
#     residual = ((1.0 - shrinkage) * isotropic + shrinkage * mean_variance).clamp_min(
#         eps
#     )
#     if not bool(torch.isfinite(lam).all()) or not bool(torch.isfinite(residual)):
#         raise RuntimeError("invalid spectral-condition covariance estimate")
#     return axes, lam, residual


# def _integer(value: object, name: str, *, minimum: int) -> int:
#     if torch.is_tensor(value):
#         if (
#             value.ndim != 0
#             or value.dtype == torch.bool
#             or value.is_complex()
#             or torch.is_floating_point(value)
#         ):
#             raise TypeError(f"{name} must be an integer scalar")
#         value = value.item()
#     if isinstance(value, bool) or not isinstance(value, Integral):
#         raise TypeError(f"{name} must be an integer")
#     result = int(value)
#     if result < minimum:
#         raise ValueError(f"{name} must be >= {minimum}")
#     return result


# def _class_id_tuple(
#     values: Sequence[int] | Tensor | Iterable[int],
#     *,
#     name: str,
# ) -> tuple[int, ...]:
#     if torch.is_tensor(values):
#         if values.ndim != 1 or values.numel() == 0 or values.dtype != torch.long:
#             raise ValueError(f"{name} tensor must be non-empty rank-one torch.long")
#         source = values.detach().cpu().tolist()
#     else:
#         source = list(values)
#     result = tuple(_integer(value, name, minimum=0) for value in source)
#     if not result or len(result) != len(set(result)):
#         raise ValueError(f"{name} must contain unique class IDs")
#     return result


# def _band_positions(
#     values: Sequence[float] | Tensor | None,
#     *,
#     count: int,
#     device: torch.device | str | None,
#     dtype: torch.dtype,
# ) -> Tensor:
#     positions = (
#         torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
#         if values is None
#         else torch.as_tensor(values, device=device, dtype=dtype).flatten()
#     )
#     if positions.shape != (count,):
#         raise ValueError(f"spectral_band_positions must contain {count} values")
#     if not bool(torch.isfinite(positions).all()):
#         raise ValueError("spectral_band_positions contain NaN or Inf")
#     if count > 1 and bool((positions[1:] <= positions[:-1]).any()):
#         raise ValueError("spectral_band_positions must be strictly increasing")
#     return positions


# def _deterministic_quantile_dim0(values: Tensor, q: float) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("quantile input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("quantile input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("quantile input contains NaN or Inf")
#     if not 0.0 <= float(q) <= 1.0:
#         raise ValueError("q must be in [0,1]")

#     original_device = value.device
#     ordered = torch.sort(value.detach().to(device="cpu"), dim=0, stable=True).values
#     sample_count = int(ordered.size(0))
#     if sample_count == 1:
#         result = ordered[0]
#     else:
#         position = float(q) * float(sample_count - 1)
#         lower = int(position)
#         upper = min(lower + 1, sample_count - 1)
#         weight = position - float(lower)
#         result = (
#             ordered[lower]
#             if upper == lower or weight == 0.0
#             else ordered[lower] * (1.0 - weight) + ordered[upper] * weight
#         )
#     return result.to(device=original_device, dtype=value.dtype)


# def _deterministic_median_dim0(values: Tensor) -> Tensor:
#     return _deterministic_quantile_dim0(values, 0.5)


# def _validate_ordered_samples(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) <= 0:
#         raise ValueError("values must be non-empty [N,L,F], L>=2")
#     if (
#         not torch.is_floating_point(value)
#         or value.is_complex()
#         or not bool(torch.isfinite(value).all())
#     ):
#         raise ValueError("values must be finite real floating point")
#     return value


# def _validate_physical_evidence(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim == 2:
#         value = value.unsqueeze(-1)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) != 1:
#         raise ValueError("physical evidence must be direct full-band [N,B,1], B>=2")
#     if (
#         not torch.is_floating_point(value)
#         or value.is_complex()
#         or not bool(torch.isfinite(value).all())
#     ):
#         raise ValueError("physical evidence must be finite real floating point")
#     return value


# def normalize_ordered_decision_features(values: Tensor) -> Tensor:
#     """Map Q to the canonical learned CDS coordinate."""
#     value = _validate_ordered_samples(values)
#     flat = value.reshape(value.size(0), -1)
#     eps = torch.finfo(value.dtype).eps
#     unit = flat / flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
#     return unit.reshape_as(value)


# def sanitize_ordered_variation(variation: Tensor) -> Tensor:
#     """Validate signed, compact physical bounds [C,2,2*B-1,1].

#     The first B slots describe residual per-band variation; the next B-1
#     describe adjacent-band residual differences. No IQR or B-by-B tensor.
#     """
#     value = torch.as_tensor(variation)
#     if (
#         value.ndim != 4
#         or value.size(1) != 2
#         or value.size(-1) != 1
#         or value.size(2) < 3
#         or value.size(2) % 2 != 1
#     ):
#         raise ValueError("physical variation must be [C,2,2*B-1,1]")
#     if (
#         not torch.is_floating_point(value)
#         or value.is_complex()
#         or not bool(torch.isfinite(value).all())
#     ):
#         raise ValueError("physical variation must be finite real floating point")
#     if bool((value[:, 0] > value[:, 1]).any()):
#         raise ValueError("physical variation lower bounds exceed upper bounds")
#     return value.contiguous()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     return sanitize_ordered_variation(variation)


# def estimate_relational_variation(learned_samples: Tensor, identity: Tensor) -> Tensor:
#     """Class-conditional signed adjacent residual bounds [2,K-1,D].

#     Both learned_samples and identity are in the canonical normalized Q
#     coordinate. This is a nonparametric summary, not a containment guarantee.
#     """
#     q = _validate_ordered_samples(learned_samples)
#     if identity.shape != q.shape[1:]:
#         raise ValueError("identity must be [K,D] matching learned samples")
#     residual = q - identity.unsqueeze(0)
#     adjacent = residual[:, 1:, :] - residual[:, :-1, :]
#     lo = _deterministic_quantile_dim0(adjacent, 0.10)
#     hi = _deterministic_quantile_dim0(adjacent, 0.90)
#     return torch.stack((lo, hi), dim=0).contiguous()


# def validate_relational_variation(
#     value: Tensor, *, length: int, features: int
# ) -> Tensor:
#     """Validate [C,2,K-1,D] signed lower/upper interval summaries."""
#     bounds = torch.as_tensor(value)
#     if bounds.ndim != 4 or bounds.shape[1:] != (2, length - 1, features):
#         raise ValueError("learned V_Q must be [C,2,K-1,D]")
#     if not torch.is_floating_point(bounds) or not bool(torch.isfinite(bounds).all()):
#         raise ValueError("learned V_Q must be finite floating-point")
#     if bool((bounds[:, 0] > bounds[:, 1]).any()):
#         raise ValueError("learned V_Q lower bound exceeds upper bound")
#     return bounds


# def learned_relational_support(
#     sample: Tensor, identity: Tensor, variation: Tensor,
#     absolute_deviation: Tensor | None = None,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Identity, relational and combined compatibility [N,C] for learned Q.

#     The relational penalty is zero for a residual difference within its
#     class-conditional quantile interval. Soft hinges are differentiable with
#     respect to the sample. The fixed sqrt(K*D) scale follows the unit-norm
#     Q coordinate and introduces no learned calibration parameter.
#     """
#     q = normalize_ordered_decision_features(sample)
#     center = torch.as_tensor(identity, device=q.device, dtype=q.dtype)
#     if center.ndim != 3 or center.shape[1:] != q.shape[1:]:
#         raise ValueError("learned identity must be [C,K,D] matching Q")
#     c, k, d = center.shape
#     bounds = validate_relational_variation(
#         torch.as_tensor(variation, device=q.device, dtype=q.dtype), length=k, features=d
#     )
#     if bounds.shape[0] != c:
#         raise ValueError("learned V_Q class count does not match I_Q")
#     delta = q[:, None, :, :] - center[None, :, :, :]
#     identity_distance = torch.linalg.vector_norm(
#         delta.reshape(delta.size(0), delta.size(1), -1), dim=2
#     )
#     absolute = (1.0 / (1.0 + identity_distance)).clamp(0, 1)
#     if absolute_deviation is not None:
#         # A_Q measures the observed absolute deviation of each learned token
#         # coordinate; it is NOT another covariance model. Distances beyond
#         # this empirical within-class scale reduce absolute CDS support.
#         a = torch.as_tensor(absolute_deviation,device=q.device,dtype=q.dtype).detach()
#         if a.shape != center.shape or not bool(torch.isfinite(a).all()) or bool((a < 0).any()):
#             raise ValueError("learned A_Q must be finite, nonnegative [C,K,D]")
#         tolerance = math.sqrt(torch.finfo(q.dtype).eps) / math.sqrt(k*d)
#         absolute_spread = 1.0 / (1.0 + (delta.abs() / (a[None] + tolerance)).mean((2,3)))
#         absolute = torch.sqrt(absolute * absolute_spread.clamp(0,1))
#     relation = delta[:, :, 1:, :] - delta[:, :, :-1, :]
#     lower = bounds[None, :, 0, :, :]
#     upper = bounds[None, :, 1, :, :]
#     violation = (lower - relation).relu() + (relation - upper).relu()
#     penalty = violation.mean(dim=(2, 3)) * math.sqrt(k * d)
#     relational = (1.0 / (1.0 + penalty)).clamp(0, 1)
#     return absolute, relational, torch.sqrt(absolute * relational)


# def _interval_compatibility(
#     delta: Tensor, lo: Tensor, hi: Tensor, reference: Tensor
# ) -> Tensor:
#     """Continuous interval compatibility, nonzero even for sparse-class evidence."""
#     violation = (lo - delta).relu() + (delta - hi).relu()
#     # ponytail: this is a numerical floor, not a measured sensor-noise model.
#     # Calibrate a shared physical tolerance if held-out rare-class evidence needs it.
#     tolerance = torch.finfo(delta.dtype).eps ** 0.5 * (1.0 + reference.abs())
#     width = (hi - lo).clamp_min(0.0)
#     return (1.0 + violation / (width + tolerance)).reciprocal().clamp(0.0, 1.0)


# def physical_support_components(
#     sample: Tensor, identity: Tensor, variation: Tensor
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Per-band, adjacent-band, joint support [N,C] from native ordered bands."""
#     x = _validate_physical_evidence(sample)
#     center = torch.as_tensor(identity, device=x.device, dtype=x.dtype)
#     bounds = sanitize_ordered_variation(
#         torch.as_tensor(variation, device=x.device, dtype=x.dtype)
#     )
#     if center.ndim != 3 or center.shape[1:] != x.shape[1:]:
#         raise ValueError("physical identity must be [C,B,1] matching sample")
#     c, b, _ = center.shape
#     if bounds.shape != (c, 2, 2 * b - 1, 1):
#         raise ValueError("physical variation must be [C,2,2*B-1,1]")
#     residual = x[:, None] - center[None]
#     absolute = _interval_compatibility(
#         residual, bounds[None, :, 0, :b], bounds[None, :, 1, :b], center[None]
#     ).mean(dim=(2, 3))
#     adjacent = residual[:, :, 1:] - residual[:, :, :-1]
#     center_change = (center[:, 1:] - center[:, :-1]).unsqueeze(0)
#     ordered = _interval_compatibility(
#         adjacent, bounds[None, :, 0, b:], bounds[None, :, 1, b:], center_change
#     ).mean(dim=(2, 3))
#     return absolute, ordered, torch.sqrt(absolute * ordered)


# def ordered_cds_support_components(
#     sample: Tensor, identity: Tensor, variation: Tensor,
#     *, absolute_deviation: Tensor | None = None,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Dispatch learned signed V_Q or compact signed physical V_P evidence."""
#     x = torch.as_tensor(sample)
#     center = torch.as_tensor(identity)
#     v = torch.as_tensor(variation)
#     if center.ndim != 3 or x.ndim != 3 or x.shape[1:] != center.shape[1:]:
#         raise ValueError("sample and identity must be [N,L,F] and [C,L,F]")
#     if v.ndim == 4 and v.shape[1:] == (2, center.size(1) - 1, center.size(2)):
#         return learned_relational_support(x, center, v, absolute_deviation=absolute_deviation)
#     if (
#         center.size(2) == 1
#         and v.ndim == 4
#         and v.shape[1:] == (2, 2 * center.size(1) - 1, 1)
#     ):
#         return physical_support_components(x, center, v)
#     raise ValueError("variation does not match learned V_Q or compact physical V_P")


# def spectral_conditioned_learned_support(ordered_features, moments, physical_spectral_evidence=None):
#     """Compatibility name: single unconditional low-rank covariance support.

#     P does NOT partition feature distributions. Spectral variation enters the
#     pair-risk and relational losses, never a second learned covariance bank.
#     """
#     q = normalize_ordered_decision_features(ordered_features)
#     z = q.flatten(1)
#     mu=torch.as_tensor(moments["mean"],device=z.device,dtype=z.dtype).detach()
#     basis=torch.as_tensor(moments["basis"],device=z.device,dtype=z.dtype).detach()
#     eig=torch.as_tensor(moments["eigenvalues"],device=z.device,dtype=z.dtype).detach()
#     residual=torch.as_tensor(moments["residual_variance"],device=z.device,dtype=z.dtype).detach()
#     n,f=z.shape
#     if mu.ndim != 2 or mu.size(1)!=f or basis.shape[:2]!=mu.shape or eig.shape!=(len(mu),basis.size(-1)) or residual.shape!=(len(mu),):
#         raise ValueError("unified Q moments must be mean[C,F], basis[C,F,R], eigen[C,R], residual[C]")
#     if bool((eig<0).any()) or bool((residual<0).any()):
#         raise ValueError("negative covariance")
#     proj_z=torch.einsum('nf,cfr->ncr',z,basis)
#     proj_mu=torch.einsum('cf,cfr->cr',mu,basis)
#     projected=(proj_z-proj_mu.unsqueeze(0)).square()
#     residual_sq=(z.square().sum(-1)[:,None]+mu.square().sum(-1)[None]-2*z@mu.T-projected.sum(-1)).clamp_min(0)
#     used=(basis.square().sum(1)>1e-10).sum(-1)
#     trace=eig.sum(-1)+(f-used)*residual
#     ridge=(trace/f).clamp_min(1e-4)
#     dist=((projected/(eig[None]+ridge[None,:,None])).sum(-1)+
#           residual_sq/(residual[None]+ridge[None]))/f
#     return (-0.5*dist.clamp_min(0)).exp().clamp(0,1)


# def native_spectral_variation_support(physical_evidence, class_mean,
#                                       class_variance, delta_variance):
#     """Native-band mean/variance and adjacent-band variation (no V_P interval loss).

#     Spectral coordinates come from the same fitted native-band normalizer across
#     phases; physical evidence is detached and never acts as a second classifier.
#     """
#     p = _validate_physical_evidence(physical_evidence)[...,0]
#     mu = torch.as_tensor(class_mean,device=p.device,dtype=p.dtype)[...,0].detach()
#     v = torch.as_tensor(class_variance,device=p.device,dtype=p.dtype)[...,0].detach()
#     vd = torch.as_tensor(delta_variance,device=p.device,dtype=p.dtype)[...,0].detach()
#     if mu.ndim != 2 or mu.size(1) != p.size(1) or v.shape != mu.shape or vd.shape != (len(mu),p.size(1)-1):
#         raise ValueError("native spectral statistics require [C,B] and [C,B-1]")
#     if bool((v<0).any()) or bool((vd<0).any()):
#         raise ValueError("negative native spectral variance")
#     # Native P is a bounded, robustly normalized spectrum. Ridge prevents low-
#     # count classes from acquiring infinite confidence along zero-variance bands.
#     ridge=1e-3
#     gap=(p[:,None,:]-mu[None])
#     abs_score=torch.exp(-0.5*(gap.square()/(v[None]+ridge)).mean(-1)).clamp(0,1)
#     dp=p[:,1:]-p[:,:-1]
#     dmu=mu[:,1:]-mu[:,:-1]
#     dg=dp[:,None,:]-dmu[None]
#     rel_score=torch.exp(-0.5*(dg.square()/(vd[None]+ridge)).mean(-1)).clamp(0,1)
#     return abs_score, rel_score, torch.sqrt(abs_score*rel_score).clamp(0,1)


# def combine_learned_physical_relevance(learned: Tensor, physical: Tensor) -> Tensor:
#     """Fuse learned and physical relevance with learned semantics as the gate.

#     Physical similarity can strengthen an existing learned-space relation but
#     cannot create semantic relevance by itself. This same rule is used by feature
#     separation, historical decision protection, and Whole-CDS diagnostics.
#     """
#     learned_value = torch.as_tensor(learned)
#     physical_value = torch.as_tensor(
#         physical, device=learned_value.device, dtype=learned_value.dtype
#     )
#     if learned_value.shape != physical_value.shape:
#         raise ValueError("learned and physical relevance must have identical shape")
#     if not bool(torch.isfinite(learned_value).all()) or not bool(
#         torch.isfinite(physical_value).all()
#     ):
#         raise ValueError("learned/physical relevance contains NaN or Inf")
#     learned_value = learned_value.clamp(0.0, 1.0)
#     physical_value = physical_value.clamp(0.0, 1.0)
#     return (
#         learned_value + (1.0 - learned_value) * learned_value * physical_value
#     ).clamp(0.0, 1.0)


# def combine_class_pair_rivalry(learned: Tensor, physical: Tensor) -> Tensor:
#     """Joint R_Q/R_P *class-pair rivalry* (not sample-to-class support).

#     Either independent source of class ambiguity can flag a difficult pair.
#     A max is bounded, symmetric for symmetric inputs, parameter-free, and avoids
#     counting the same ambiguity twice. Do not use this for sample support:
#     combine_learned_physical_relevance has a different semantic contract.
#     """
#     r_q = torch.as_tensor(learned)
#     r_p = torch.as_tensor(physical, device=r_q.device, dtype=r_q.dtype)
#     if r_q.shape != r_p.shape or r_q.ndim != 2 or r_q.shape[0] != r_q.shape[1]:
#         raise ValueError("R_Q/R_P class-pair rivalry must be aligned square [C,C]")
#     if not all(bool(torch.isfinite(t).all()) for t in (r_q, r_p)):
#         raise ValueError("R_Q/R_P contain NaN or Inf")
#     return torch.maximum(r_q.clamp(0.0, 1.0), r_p.clamp(0.0, 1.0))


# def _ordered_identity_and_variation(
#     values: Tensor, *, row_chunk: int = 8
# ) -> tuple[Tensor, Tensor]:
#     """Native spectrum arithmetic mean and signed 10th–90th percentile residual ranges.

#     Returns I_P [B,1], compact V_P [2,2B-1,1]. row_chunk remains for
#     source API compatibility, but no quadratic band-pair operation occurs.
#     """
#     sample = _validate_physical_evidence(values)
#     identity = sample.mean(dim=0)
#     residual = sample - identity.unsqueeze(0)
#     adjacent = residual[:, 1:] - residual[:, :-1]
#     evidence = torch.cat((residual, adjacent), dim=1)
#     lo = _deterministic_quantile_dim0(evidence, 0.10)
#     hi = _deterministic_quantile_dim0(evidence, 0.90)
#     return identity.contiguous(), torch.stack((lo, hi)).contiguous()


# def _learned_absolute_class_proximity(identity: Tensor, absolute_deviation: Tensor | None = None) -> Tensor:
#     """Inter-class I_Q proximity relative to observed absolute within-class A_Q."""
#     center = torch.as_tensor(identity)
#     c, k, d = center.shape
#     delta = center[:, None, :, :] - center[None, :, :, :]
#     base = (1.0 / (1.0 + delta.norm(dim=-1) * math.sqrt(k))).clamp(0, 1)
#     if absolute_deviation is None:
#         return base
#     a = torch.as_tensor(absolute_deviation,device=center.device,dtype=center.dtype)
#     if a.shape != center.shape or not bool(torch.isfinite(a).all()) or bool((a < 0).any()):
#         raise ValueError("A_Q must be finite, nonnegative [C,K,D]")
#     tolerance = math.sqrt(torch.finfo(center.dtype).eps) / math.sqrt(k*d)
#     scale = a[:, None] + a[None, :] + tolerance
#     overlap = 1.0 / (1.0 + (delta.abs() / scale).mean(dim=-1))
#     return torch.sqrt(base * overlap.clamp(0,1))


# def _learned_relational_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
#     """Symmetric cross-class quantile-interval intrusion; high means rivals."""
#     center = torch.as_tensor(identity)
#     c, k, d = center.shape
#     v = validate_relational_variation(variation, length=k, features=d)
#     if v.shape[0] != c:
#         raise ValueError("learned identity/V_Q class counts disagree")
#     relation = center[:, 1:, :] - center[:, :-1, :]
#     delta = relation[None, :, :, :] - relation[:, None, :, :]
#     lo_c, hi_c = v[:, 0, :, :][:, None], v[:, 1, :, :][:, None]
#     lo_j, hi_j = v[:, 0, :, :][None, :], v[:, 1, :, :][None, :]
#     violation_c = (lo_c - delta).relu() + (delta - hi_c).relu()
#     violation_j = (lo_j + delta).relu() + (-delta - hi_j).relu()
#     scale = math.sqrt(k * d)
#     support_c = 1.0 / (1.0 + scale * violation_c.mean(dim=(-1, -2)))
#     support_j = 1.0 / (1.0 + scale * violation_j.mean(dim=(-1, -2)))
#     result = 0.5 * (support_c + support_j)
#     result.diagonal().fill_(1.0)
#     return result.clamp(0, 1)


# def _physical_cross_class_support(
#     identity: Tensor, variation: Tensor, *, adjacent: bool
# ) -> Tensor:
#     """Symmetric rival compatibility of physical identities in stored intervals."""
#     center = torch.as_tensor(identity)
#     bounds = sanitize_ordered_variation(variation)
#     if (
#         center.ndim != 3
#         or center.size(-1) != 1
#         or bounds.shape != (center.size(0), 2, 2 * center.size(1) - 1, 1)
#     ):
#         raise ValueError("physical identity/variation shape mismatch")
#     b = center.size(1)
#     if adjacent:
#         center = center[:, 1:] - center[:, :-1]
#         lo, hi = bounds[:, 0, b:], bounds[:, 1, b:]
#     else:
#         lo, hi = bounds[:, 0, :b], bounds[:, 1, :b]
#     # delta[c,j] is the observation from class c, evaluated by class j.
#     delta = center[:, None] - center[None, :]
#     as_j = _interval_compatibility(delta, lo[None], hi[None], center[None])
#     as_c = _interval_compatibility(-delta, lo[:, None], hi[:, None], center[:, None])
#     result = 0.5 * (as_j + as_c)
#     diagonal = torch.arange(center.size(0), device=center.device)
#     result[diagonal, diagonal] = 1.0
#     return result.mean(dim=-1).clamp(0, 1)


# def _absolute_class_proximity_by_slot(identity: Tensor, variation: Tensor) -> Tensor:
#     return _physical_cross_class_support(identity, variation, adjacent=False)


# def _relation_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
#     return _physical_cross_class_support(identity, variation, adjacent=True).mean(dim=2)


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     physical_identity: Tensor
#     physical_variation: Tensor
#     physical_variance: Tensor  # [C,B,1], exact native-band population variance
#     physical_delta_variance: Tensor  # [C,B-1,1], adjacent-band difference variance

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     learned_identity: Tensor  # [C,K,D], arithmetic mean in normalized Q
#     learned_variation: Tensor  # [C,2,K-1,D], signed V_Q
#     learned_absolute_deviation: Tensor  # [C,K,D], mean |Q - I_Q|; NOT covariance
#     physical_identity: Tensor  # [C,B,1], mean native spectrum
#     physical_variation: Tensor  # [C,2,2B-1,1], retained native-band diagnostics
#     physical_variance: Tensor  # [C,B,1]
#     physical_delta_variance: Tensor  # [C,B-1,1]
#     learned_covariance_basis: Tensor  # [C,F,R], ONE learned distribution per class
#     learned_covariance_eigenvalues: Tensor  # [C,R]
#     learned_covariance_residual: Tensor  # [C], isotropic residual variance

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_proximity_by_region: Tensor
#     learned_relation_proximity: Tensor
#     physical_identity_proximity_by_band: Tensor
#     physical_relation_proximity: Tensor
#     learned_prototype_cosine: Tensor  # PCDS-style angular inter-class structure

#     @property
#     def learned_identity_proximity(self):
#         return self.learned_identity_proximity_by_region.mean(dim=2)

#     @property
#     def learned_combined_proximity(self):
#         return torch.sqrt(
#             self.learned_identity_proximity.clamp(0, 1)
#             * self.learned_relation_proximity.clamp(0, 1)
#         )

#     @property
#     def physical_identity_proximity(self):
#         discrimination = (1 - self.physical_identity_proximity_by_band).clamp(0, 1)
#         mass = discrimination.sum(dim=2)
#         strength = discrimination.square().sum(dim=2) / mass.clamp_min(
#             torch.finfo(discrimination.dtype).eps
#         )
#         return (1 - strength).clamp(0, 1)

#     @property
#     def physical_combined_proximity(self):
#         return torch.sqrt(
#             self.physical_identity_proximity.clamp(0, 1)
#             * self.physical_relation_proximity.clamp(0, 1)
#         )

#     @property
#     def physical_band_discriminability(self):
#         return (1 - self.physical_identity_proximity).clamp(0, 1)


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor
#     learned_moments: dict[str, Tensor] | None = None

#     @property
#     def conditional_moments(self):
#         # Read-only compatibility for existing diagnostic callers. No conditions.
#         return self.learned_moments


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     learned_identity: Tensor
#     learned_variation: Tensor
#     learned_absolute_deviation: Tensor
#     physical_identity: Tensor
#     physical_variation: Tensor
#     physical_variance: Tensor
#     physical_delta_variance: Tensor
#     learned_covariance_basis: Tensor
#     learned_covariance_eigenvalues: Tensor
#     learned_covariance_residual: Tensor


# def estimate_physical_structure_batch(
#     physical_evidence: Tensor, labels: Tensor, *, class_ids
# ) -> PhysicalSpectralStructureBatch:
#     p = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels, device=p.device, dtype=torch.long).flatten()
#     ids = _class_id_tuple(class_ids, name="class_ids")
#     if y.shape != (p.size(0),) or set(map(int, y.unique().tolist())) != set(ids):
#         raise ValueError("physical training labels must match class IDs")
#     identity, variation, count, variance, delta_variance = [], [], [], [], []
#     for cid in ids:
#         x = p[y == cid]
#         i, v = _ordered_identity_and_variation(x)
#         identity.append(i)
#         variation.append(v)
#         count.append(x.size(0))
#         variance.append(x.var(dim=0, unbiased=False))
#         delta_variance.append((x[:, 1:] - x[:, :-1]).var(dim=0, unbiased=False))
#     return PhysicalSpectralStructureBatch(
#         torch.tensor(ids, device=p.device, dtype=torch.long),
#         torch.tensor(count, device=p.device, dtype=torch.long),
#         torch.stack(identity), torch.stack(variation),
#         torch.stack(variance), torch.stack(delta_variance),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical_structure: PhysicalSpectralStructureBatch,
#     *,
#     coupled_rank=None,
#     coupled_quantile=1.0,
#     moment_rank=8,
#     moment_covariance_shrinkage=0.10,
# ) -> ClassStructureBatch:
#     q = normalize_ordered_decision_features(ordered_features)
#     y = torch.as_tensor(labels, device=q.device, dtype=torch.long).flatten()
#     ids = physical_structure.class_ids.tolist()
#     if set(y.unique().tolist()) != set(ids) or y.numel() != q.size(0):
#         raise ValueError("Q class IDs mismatch")
#     identity = []
#     variation = []
#     dispersion, basis, eigenvalues, residuals = [], [], [], []
#     for cid in ids:
#         x = q[y == cid]
#         mean = x.mean(dim=0)
#         identity.append(mean)
#         # Old V_Q/A_Q remain read-only diagnostics for existing reports.
#         variation.append(estimate_relational_variation(x, mean))
#         dispersion.append((x - mean).abs().mean(dim=0))
#         u, lam, res = _compact_covariance(
#             x.flatten(1), rank=moment_rank, class_seed=73 + 997 * int(cid),
#             shrinkage=moment_covariance_shrinkage,
#         )
#         basis.append(u); eigenvalues.append(lam); residuals.append(res)
#     return ClassStructureBatch(
#         physical_structure.class_ids,
#         physical_structure.sample_counts,
#         torch.stack(identity), torch.stack(variation), torch.stack(dispersion),
#         physical_structure.physical_identity, physical_structure.physical_variation,
#         physical_structure.physical_variance, physical_structure.physical_delta_variance,
#         torch.stack(basis), torch.stack(eigenvalues), torch.stack(residuals),
#     )


# def estimate_structure_batch(
#     ordered_features,
#     physical_evidence,
#     labels,
#     *,
#     class_ids,
#     coupled_rank=None,
#     coupled_quantile=1.0,
#     moment_rank=8,
#     moment_covariance_shrinkage=0.10,
# ) -> ClassStructureBatch:
#     return combine_learned_and_physical_structure(
#         ordered_features,
#         labels,
#         estimate_physical_structure_batch(
#             physical_evidence, labels, class_ids=class_ids
#         ),
#     )


# def merge_class_structure_batches(structures) -> ClassStructureBatch:
#     xs = list(structures)
#     if not xs:
#         raise ValueError("missing class structures")
#     ids = [i for x in xs for i in x.class_ids.tolist()]
#     if len(ids) != len(set(ids)):
#         raise ValueError("duplicate class IDs")
#     values = {
#         name: torch.cat([getattr(x, name) for x in xs], dim=0)
#         for name in ClassStructureBatch.__dataclass_fields__
#     }
#     return ClassStructureBatch(**values)


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     return ClassRelationState(
#         structure.class_ids,
#         _learned_absolute_class_proximity(structure.learned_identity,
#                                           structure.learned_absolute_deviation),
#         _learned_relational_class_proximity(
#             structure.learned_identity, structure.learned_variation
#         ),
#         _absolute_class_proximity_by_slot(
#             structure.physical_identity, structure.physical_variation
#         ),
#         _relation_class_proximity(
#             structure.physical_identity, structure.physical_variation
#         ),
#         torch.nn.functional.normalize(structure.learned_identity.flatten(1), dim=1)
#         @ torch.nn.functional.normalize(structure.learned_identity.flatten(1), dim=1).T,
#     )


# def normalized_moment_translation_feasible(moments, class_translation):
#     """Per-class gate for a pure translation of unit-normalized Q moments.

#     Historical within-class covariance is unchanged by a common Euclidean
#     translation. A change in the squared mean norm must therefore not worsen
#     the existing unit-sphere second-moment identity. We compare against the
#     original compact covariance approximation error, not an exact dense model.
#     """
#     mu = moments["mean"]
#     basis, eig, residual = (moments[k] for k in
#         ("basis", "eigenvalues", "residual_variance"))
#     counts = moments["counts"]
#     shift = class_translation.reshape(mu.size(0), 1, -1).to(mu)
#     if shift.shape != (mu.size(0), 1, mu.size(-1)):
#         raise ValueError("historical class translation and moments disagree")
#     if not bool(torch.isfinite(shift).all()):
#         raise ValueError("non-finite proposed historical translation")
#     active_rank = (basis.square().sum(-2) > 1e-10).sum(-1)
#     trace = eig.sum(-1) + (mu.size(-1) - active_rank) * residual
#     factor = (counts - 1).clamp_min(0).to(mu) / counts.clamp_min(1).to(mu)
#     old_energy = mu.square().sum(-1) + factor * trace
#     new_energy = (mu + shift).square().sum(-1) + factor * trace
#     valid = counts > 0
#     # Numerical slack is only for floating-point accumulation; the original
#     # compact approximation's own deviation is not silently treated as zero.
#     slack = 128 * torch.finfo(mu.dtype).eps * (1.0 + old_energy.abs())
#     acceptable = ((new_energy - 1.0).abs() <= (old_energy - 1.0).abs() + slack)
#     acceptable &= (mu + shift).square().sum(-1) <= 1.0 + slack
#     return (acceptable | ~valid).all(dim=1)


# def whole_cds_evidence_masks(classes: ClassStructureBatch):
#     n = classes.sample_counts
#     v = (n[:, None] > 1) & (n[None, :] > 1)
#     return v, v


# def whole_cds_pair_risk(cds: ClassDiscriminativeStructureState):
#     """One inter-class risk from within-class learned variance and native spectra.

#     Computation is descriptive (not calibrated overlap probability). All class
#     pair scores are symmetric and detached when used as weighting evidence.
#     V_Q and absolute-deviation diagnostics do not enter this training risk.
#     """
#     st = cds.classes
#     mu = st.learned_identity.flatten(1)
#     # Use covariance ORIENTATION, not just a trace or scalar radius.
#     # Project each historical class covariance along the actual pairwise
#     # mean-separation direction; that is the ambiguous direction of the pair.
#     delta = mu[:, None, :] - mu[None, :, :]
#     distance2 = delta.square().sum(-1)
#     direction = delta / (distance2[..., None] + 1e-8).sqrt()
#     basis = st.learned_covariance_basis
#     eigen = st.learned_covariance_eigenvalues
#     residual = st.learned_covariance_residual
#     source_proj = torch.einsum("ijf,ifr->ijr",direction,basis)
#     source_var = (source_proj.square()*eigen[:,None,:]).sum(-1)
#     source_var += residual[:,None]*(direction.square().sum(-1)-source_proj.square().sum(-1)).clamp_min(0)
#     # Same directional quadratic form, with the *rival's* own covariance.
#     rival_proj = torch.einsum("ijf,jfr->ijr",direction,basis)
#     rival_var = (rival_proj.square()*eigen[None,:,:]).sum(-1)
#     rival_var += residual[None,:]*(direction.square().sum(-1)-rival_proj.square().sum(-1)).clamp_min(0)
#     counts = st.sample_counts.to(mu).clamp_min(1)
#     variance = (source_var * ((counts-1)/counts)[:,None] +
#                 rival_var * ((counts-1)/counts)[None,:]).clamp_min(0)
#     learned = variance / (variance + distance2 + 1e-6)
#     p = st.physical_identity[..., 0]
#     p_var = st.physical_variance[..., 0]
#     slope = p[:, 1:] - p[:, :-1]
#     slope_var = st.physical_delta_variance[..., 0]
#     def similarity(value, variance):
#         delta = value[:, None] - value[None]
#         spread_pair = variance[:, None] + variance[None] + 1e-4
#         return torch.exp(-0.5 * (delta.square() / spread_pair).mean(-1))
#     physical = (similarity(p, p_var) * similarity(slope, slope_var)).sqrt()
#     # A spectral-only similarity cannot override an already well-separated
#     # learned pair. Its purpose is to prioritize supported learned rivalry.
#     effective = combine_learned_physical_relevance(learned, physical)
#     eye = torch.eye(len(mu), device=mu.device, dtype=torch.bool)
#     return tuple(v.masked_fill(eye, 0.0).clamp(0, 1).contiguous()
#                  for v in (effective, learned, physical))


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Canonical exemplar-free inter/intra/spectral class statistics.

#     The bank alone owns one low-rank learned covariance per class and native spectral variation.
#     No raw HSI samples, feature exemplars or dense covariance matrices persist.
#     """

#     def __init__(
#         self,
#         *,
#         spectral_token_count,
#         spectral_token_dim,
#         spectral_bands,
#         spectral_band_positions=None,
#         coupled_rank=None,
#         coupled_quantile=1.0,
#         moment_rank=8,
#         moment_min_per_group=24,
#         moment_covariance_shrinkage=0.10,
#         device=None,
#         dtype=torch.float32,
#     ):
#         super().__init__()
#         self.spectral_token_count = int(spectral_token_count)
#         self.spectral_token_dim = int(spectral_token_dim)
#         self.spectral_bands = int(spectral_bands)
#         self.moment_rank = _integer(moment_rank, "moment_rank", minimum=1)
#         self.moment_min_per_group = _integer(moment_min_per_group, "moment_min_per_group", minimum=2)
#         self.moment_covariance_shrinkage = float(moment_covariance_shrinkage)
#         if not 0 < self.moment_covariance_shrinkage < 1:
#             raise ValueError("moment covariance shrinkage must be in (0,1)")
#         k = self.spectral_token_count
#         d = self.spectral_token_dim
#         b = self.spectral_bands
#         self.register_buffer(
#             "spectral_band_positions",
#             _band_positions(
#                 spectral_band_positions, count=b, device=device, dtype=dtype
#             ),
#         )
#         for key, x in {
#             "class_ids": torch.empty(0, device=device, dtype=torch.long),
#             "sample_count_bank": torch.empty(0, device=device, dtype=torch.long),
#             "learned_identity_bank": torch.empty(0, k, d, device=device, dtype=dtype),
#             "learned_variation_bank": torch.empty(
#                 0, 2, k - 1, d, device=device, dtype=dtype
#             ),
#             "learned_absolute_deviation_bank": torch.empty(
#                 0, k, d, device=device, dtype=dtype
#             ),
#             "physical_identity_bank": torch.empty(0, b, 1, device=device, dtype=dtype),
#             "physical_variation_bank": torch.empty(
#                 0, 2, 2 * b - 1, 1, device=device, dtype=dtype
#             ),
#             "physical_variance_bank": torch.empty(0, b, 1, device=device, dtype=dtype),
#             "physical_delta_variance_bank": torch.empty(0, b - 1, 1, device=device, dtype=dtype),
#         }.items():
#             self.register_buffer(key, x, persistent=True)
#         f = k * d
#         r = self.moment_rank
#         # Canonical per-class low-rank moments. No spectral-condition groups.
#         for key, x in {
#             "moment_mean_bank": torch.empty(0, f, device=device, dtype=dtype),
#             "moment_basis_bank": torch.empty(0, f, r, device=device, dtype=dtype),
#             "moment_eigenvalues_bank": torch.empty(0, r, device=device, dtype=dtype),
#             "moment_residual_variance_bank": torch.empty(0, device=device, dtype=dtype),
#             "moment_counts_bank": torch.empty(0, device=device, dtype=torch.long),
#         }.items():
#             self.register_buffer(key, x, persistent=True)

#     def __len__(self):
#         return int(self.class_ids.numel())

#     @property
#     def device(self):
#         return self.learned_identity_bank.device

#     @property
#     def dtype(self):
#         return self.learned_identity_bank.dtype

#     @property
#     def seen_class_ids(self):
#         return tuple(map(int, self.class_ids.tolist()))

#     def validate_state(self):
#         c = len(self)
#         k = self.spectral_token_count
#         d = self.spectral_token_dim
#         b = self.spectral_bands
#         for name, shape in [
#             ("sample_count_bank", (c,)),
#             ("learned_identity_bank", (c, k, d)),
#             ("learned_variation_bank", (c, 2, k - 1, d)),
#             ("learned_absolute_deviation_bank", (c, k, d)),
#             ("physical_identity_bank", (c, b, 1)),
#             ("physical_variation_bank", (c, 2, 2 * b - 1, 1)),
#             ("physical_variance_bank", (c,b,1)),
#             ("physical_delta_variance_bank", (c,b-1,1)),
#         ]:
#             t = getattr(self, name)
#             if tuple(t.shape) != shape:
#                 raise RuntimeError(f"Whole-CDS {name} shape mismatch")
#             if torch.is_floating_point(t) and not bool(torch.isfinite(t).all()):
#                 raise RuntimeError(f"{name} contains nonfinite statistics")
#         if c and (
#             (self.class_ids.unique().numel() != c)
#             or bool((self.sample_count_bank <= 0).any())
#         ):
#             raise RuntimeError("invalid CDS classes")
#         f = k * d
#         r = self.moment_rank
#         moments = {
#             "moment_mean_bank": (c, f),
#             "moment_basis_bank": (c, f, r),
#             "moment_eigenvalues_bank": (c, r),
#             "moment_residual_variance_bank": (c,),
#             "moment_counts_bank": (c,),
#             "physical_variance_bank": (c,b,1),
#             "physical_delta_variance_bank": (c,b-1,1),
#         }
#         for name, shape in moments.items():
#             value = getattr(self, name)
#             if tuple(value.shape) != shape:
#                 raise RuntimeError(f"Unified CDS {name} shape mismatch: {value.shape} != {shape}")
#             if torch.is_floating_point(value) and not bool(torch.isfinite(value).all()):
#                 raise RuntimeError(f"Unified CDS {name} contains non-finite statistics")
#         if c:
#             if not torch.equal(self.moment_counts_bank, self.sample_count_bank):
#                 raise RuntimeError("one CDS covariance requires one count per class")
#             if not torch.allclose(self.moment_mean_bank, self.learned_identity_bank.flatten(1), rtol=2e-5,atol=2e-5):
#                 raise RuntimeError("unified class means disagree")
#             for name in ("moment_eigenvalues_bank","moment_residual_variance_bank","physical_variance_bank","physical_delta_variance_bank"):
#                 if bool((getattr(self,name) < 0).any()):
#                     raise RuntimeError(f"negative unified variance in {name}")
#         if bool((self.learned_absolute_deviation_bank < 0).any()):
#             raise RuntimeError("negative dispersion")
#         if bool(
#             (
#                 self.learned_variation_bank[:, 0] > self.learned_variation_bank[:, 1]
#             ).any()
#         ):
#             raise RuntimeError("V_Q interval inverted")
#         if bool(
#             (
#                 self.physical_variation_bank[:, 0] > self.physical_variation_bank[:, 1]
#             ).any()
#         ):
#             raise RuntimeError("V_P interval inverted")
#         return True

#     def _validate_batch(self, structure):
#         if not isinstance(structure, ClassStructureBatch):
#             raise TypeError("expected ClassStructureBatch")
#         if structure.learned_identity.shape[1:] != (
#             self.spectral_token_count,
#             self.spectral_token_dim,
#         ):
#             raise ValueError("learned feature size mismatch")
#         if structure.physical_identity.shape[1:] != (self.spectral_bands, 1):
#             raise ValueError("physical band count mismatch")
#         if not bool((structure.learned_absolute_deviation >= 0).all()):
#             raise ValueError("negative dispersion")
#         for name in ClassStructureBatch.__dataclass_fields__:
#             x = getattr(structure, name)
#             if x.device != self.device:
#                 raise ValueError(f"{name} wrong device")
#             if torch.is_floating_point(x) and x.dtype != self.dtype:
#                 raise ValueError(f"{name} wrong dtype")

#     def estimate_physical(self, physical_evidence, labels, *, class_ids):
#         return estimate_physical_structure_batch(
#             physical_evidence, labels, class_ids=class_ids
#         )

#     def estimate_classes(
#         self,
#         ordered_features,
#         physical_evidence,
#         labels,
#         *,
#         class_ids,
#         physical_structure=None,
#     ):
#         if physical_structure is None:
#             if physical_evidence is None:
#                 raise ValueError("missing physical evidence")
#             physical_structure = self.estimate_physical(
#                 physical_evidence, labels, class_ids=class_ids
#             )
#         return combine_learned_and_physical_structure(
#             ordered_features, labels, physical_structure,
#             moment_rank=self.moment_rank,
#             moment_covariance_shrinkage=self.moment_covariance_shrinkage,
#         )

#     @torch.no_grad()
#     def estimate_compact_moments(self, ordered_features, physical_evidence, labels, *, class_ids):
#         """One unconditional compact distribution per observed class; no synthesis."""
#         q = normalize_ordered_decision_features(ordered_features).detach()
#         p = _validate_physical_evidence(torch.as_tensor(
#             physical_evidence, device=q.device, dtype=q.dtype)).detach()
#         y = torch.as_tensor(labels, device=q.device, dtype=torch.long).flatten()
#         ids = _class_id_tuple(class_ids, name="class_ids")
#         if q.shape[1:] != (self.spectral_token_count,self.spectral_token_dim):
#             raise ValueError("Q coordinate mismatch")
#         if p.shape != (len(q),self.spectral_bands,1) or y.shape != (len(q),):
#             raise ValueError("paired Q/P and labels required")
#         if set(y.unique().tolist()) != set(ids):
#             raise ValueError("moment estimation must use exactly current training classes")
#         means, axes, eig, residual, counts = [],[],[],[],[]
#         for cid in ids:
#             x=q[y==cid].flatten(1)
#             mu=x.mean(0)
#             u,l,r=_compact_covariance(x,rank=self.moment_rank,
#                 class_seed=73+997*cid,shrinkage=self.moment_covariance_shrinkage)
#             means.append(mu); axes.append(u); eig.append(l); residual.append(r)
#             counts.append(len(x))
#         return dict(moment_mean_bank=torch.stack(means),
#                     moment_basis_bank=torch.stack(axes),
#                     moment_eigenvalues_bank=torch.stack(eig),
#                     moment_residual_variance_bank=torch.stack(residual),
#                     moment_counts_bank=torch.tensor(counts,device=q.device,dtype=torch.long))

#     @torch.no_grad()
#     def replace_all_with_moments(self, structure, ordered_features, physical_evidence, labels):
#         """Atomically replace CDS and moments, retaining prior old-class rows.

#         No old examples are accessed. An old CDS coordinate change is rejected
#         until a scientifically valid companion moment transport is implemented.
#         """
#         self._validate_batch(structure)
#         all_ids = tuple(map(int, structure.class_ids.tolist()))
#         current = set(self.seen_class_ids)
#         incoming = tuple(cid for cid in all_ids if cid not in current)
#         if not incoming:
#             raise ValueError("commit needs at least one genuinely new class")
#         y = torch.as_tensor(labels, device=self.device, dtype=torch.long).flatten()
#         if set(y.tolist()) != set(incoming):
#             raise ValueError("moment fitting may only read newly introduced classes")
#         if set(self.seen_class_ids) - set(all_ids):
#             raise ValueError("cannot drop committed historical classes")
#         for cid in self.seen_class_ids:
#             row = self._row_for_class(cid)
#             next_row = all_ids.index(cid)
#             for source, target in (
#                 ("learned_identity", "learned_identity_bank"),
#                 ("learned_variation", "learned_variation_bank"),
#                 ("learned_absolute_deviation", "learned_absolute_deviation_bank"),
#                 ("physical_identity", "physical_identity_bank"),
#                 ("physical_variation", "physical_variation_bank"),
#                 ("physical_variance", "physical_variance_bank"),
#                 ("physical_delta_variance", "physical_delta_variance_bank"),
#                 ("learned_covariance_basis", "moment_basis_bank"),
#                 ("learned_covariance_eigenvalues", "moment_eigenvalues_bank"),
#                 ("learned_covariance_residual", "moment_residual_variance_bank"),
#             ):
#                 if not torch.equal(getattr(structure, source)[next_row], getattr(self, target)[row]):
#                     raise RuntimeError("historical CDS has shifted without joint moment transport")
#         estimate = self.estimate_compact_moments(
#             ordered_features, physical_evidence, labels, class_ids=incoming
#         )
#         # A caller cannot silently commit contradictory class covariance state.
#         # The same observed training features must produce the stored learned
#         # covariance used in inter-class risk, historical risk and Q support.
#         incoming_indices = torch.tensor([all_ids.index(cid) for cid in incoming],
#                                         device=self.device,dtype=torch.long)
#         for structure_name, moment_name in (
#             ("learned_identity", "moment_mean_bank"),
#             ("learned_covariance_basis", "moment_basis_bank"),
#             ("learned_covariance_eigenvalues", "moment_eigenvalues_bank"),
#             ("learned_covariance_residual", "moment_residual_variance_bank"),
#         ):
#             stored = getattr(structure,structure_name).index_select(0,incoming_indices)
#             if structure_name == "learned_identity":
#                 stored = stored.flatten(1)
#             derived = estimate[moment_name]
#             if not torch.allclose(stored,derived,rtol=2e-5,atol=2e-5):
#                 raise RuntimeError(f"unified covariance mismatch for {structure_name}")
#         pp = _validate_physical_evidence(torch.as_tensor(
#             physical_evidence,device=self.device,dtype=self.dtype))
#         for i,cid in enumerate(incoming):
#             obs=pp[y==cid]
#             expected_pv=obs.var(dim=0,unbiased=False)
#             expected_dv=(obs[:,1:]-obs[:,:-1]).var(dim=0,unbiased=False)
#             j=all_ids.index(cid)
#             if not torch.allclose(structure.physical_variance[j],expected_pv,rtol=2e-5,atol=2e-5) or not torch.allclose(
#                 structure.physical_delta_variance[j],expected_dv,rtol=2e-5,atol=2e-5):
#                 raise RuntimeError("unified physical spectral variance mismatch")
#         expected = torch.tensor([int((y == cid).sum()) for cid in incoming], device=y.device)
#         incoming_rows = torch.tensor([all_ids.index(cid) for cid in incoming], device=y.device)
#         if not torch.equal(structure.sample_counts[incoming_rows], expected):
#             raise RuntimeError("class counts differ from moment training samples")
#         keys = tuple(estimate)
#         combined = {}
#         for key in keys:
#             row_values = []
#             for cid in all_ids:
#                 if cid in current:
#                     row_values.append(getattr(self, key)[self._row_for_class(cid)])
#                 else:
#                     row_values.append(estimate[key][incoming.index(cid)])
#             combined[key] = torch.stack(row_values, dim=0)
#         # Validate candidate in a temporary bank before changing this instance.
#         candidate = HSIClassDiscriminativeStructureBank(
#             spectral_token_count=self.spectral_token_count,
#             spectral_token_dim=self.spectral_token_dim,
#             spectral_bands=self.spectral_bands,
#             spectral_band_positions=self.spectral_band_positions,
#             moment_rank=self.moment_rank,
#             moment_min_per_group=self.moment_min_per_group,
#             moment_covariance_shrinkage=self.moment_covariance_shrinkage,
#             device=self.device, dtype=self.dtype,
#         )
#         candidate._replace_base_unchecked(structure)
#         for key, value in combined.items():
#             setattr(candidate, key, value.detach().clone())
#         candidate.validate_state()
#         self._replace_base_unchecked(structure)
#         for key, value in combined.items():
#             setattr(self, key, value.detach().clone())
#         self.validate_state()

#     @torch.no_grad()
#     def apply_historical_orthogonal_transform(self,class_ids,transform):
#         """Atomic consistent historical mean/U update; spectral state untouched.

#         The historical class-mean and covariance are BOTH rotated using the
#         same unit-sphere isometry. No training features are stored or generated.
#         """
#         from models.phase_coordinate_alignment import rotate_historical_flat_rows
#         if transform.rotation_basis is None:
#             return
#         ids=_class_id_tuple(class_ids,name="class_ids")
#         rows=torch.tensor([self._row_for_class(i) for i in ids],device=self.device)
#         previous_mean=self.moment_mean_bank.clone()
#         previous_basis=self.moment_basis_bank.clone()
#         proposed_mean=rotate_historical_flat_rows(previous_mean[rows],transform)
#         proposed_basis=rotate_historical_flat_rows(previous_basis[rows].transpose(1,2),transform).transpose(1,2)
#         before_norm=previous_mean[rows].norm(dim=1)
#         if not bool(torch.allclose(proposed_mean.norm(dim=1),before_norm,rtol=1e-5,atol=1e-6)):
#             raise RuntimeError("historical orthogonal transport did not preserve mean norm")
#         self.moment_mean_bank[rows]=proposed_mean
#         self.moment_basis_bank[rows]=proposed_basis
#         self.learned_identity_bank[rows]=proposed_mean.view(-1,self.spectral_token_count,self.spectral_token_dim)
#         try:
#             self.validate_state()
#         except Exception:
#             self.moment_mean_bank.copy_(previous_mean)
#             self.moment_basis_bank.copy_(previous_basis)
#             self.learned_identity_bank[rows]=previous_mean[rows].view(-1,self.spectral_token_count,self.spectral_token_dim)
#             raise

#     @torch.no_grad()
#     def apply_historical_translation(self,class_ids,class_translation):
#         """Legacy unsafe migration stays prohibited; use orthogonal transport."""
#         delta=torch.as_tensor(class_translation,device=self.device,dtype=self.dtype)
#         if bool((delta != 0).any()):
#             raise RuntimeError("Nonzero mean-only transport is prohibited; use apply_historical_orthogonal_transform")

#     def class_moments(self, class_ids, *, device=None, dtype=None):
#         """Detached, canonical, one-class-one-covariance historical description."""
#         ids = _class_id_tuple(class_ids, name="class_ids")
#         rows = torch.tensor([self._row_for_class(cid) for cid in ids], device=self.device)
#         def take(key):
#             v=getattr(self,key).index_select(0,rows).detach().clone()
#             return v.to(device=device or self.device,dtype=(dtype or self.dtype) if v.is_floating_point() else torch.long)
#         return dict(mean=take("moment_mean_bank"),basis=take("moment_basis_bank"),
#                     eigenvalues=take("moment_eigenvalues_bank"),
#                     residual_variance=take("moment_residual_variance_bank"),
#                     counts=take("moment_counts_bank"),
#                     physical_variance=take("physical_variance_bank")[...,0],
#                     physical_delta_variance=take("physical_delta_variance_bank")[...,0])

#     @staticmethod
#     def _moment_keys():
#         return ("moment_mean_bank","moment_basis_bank","moment_eigenvalues_bank",
#                 "moment_residual_variance_bank","moment_counts_bank")

#     @torch.no_grad()
#     def append_estimate(self, structure, *, ordered_features=None, physical_evidence=None, labels=None):
#         """Commit additional observed classes; paired Q/P evidence is mandatory."""
#         if ordered_features is None or physical_evidence is None or labels is None:
#             raise ValueError("CDS inter/intra/spectral commit requires observed Q, P, labels")
#         if set(self.seen_class_ids) & set(structure.class_ids.tolist()):
#             raise ValueError("class already present")
#         all_classes = structure if not len(self) else merge_class_structure_batches(
#             [self.class_structures(), structure]
#         )
#         self.replace_all_with_moments(all_classes, ordered_features, physical_evidence, labels)

#     @torch.no_grad()
#     def replace_all(self, structure, *, ordered_features=None, physical_evidence=None, labels=None):
#         if ordered_features is None or physical_evidence is None or labels is None:
#             raise ValueError("CDS replacement requires observed Q, P, labels for new classes")
#         self.replace_all_with_moments(structure, ordered_features, physical_evidence, labels)

#     @torch.no_grad()
#     def _replace_base_unchecked(self, structure):
#         self._validate_batch(structure)
#         mapping = {
#             "class_ids": "class_ids",
#             "sample_counts": "sample_count_bank",
#             "learned_identity": "learned_identity_bank",
#             "learned_variation": "learned_variation_bank",
#             "learned_absolute_deviation": "learned_absolute_deviation_bank",
#             "physical_identity": "physical_identity_bank",
#             "physical_variation": "physical_variation_bank",
#             "physical_variance": "physical_variance_bank",
#             "physical_delta_variance": "physical_delta_variance_bank",
#         }
#         for source, target in mapping.items():
#             setattr(self, target, getattr(structure, source).detach().clone())
#         # Caller must attach the matching moment buffers and validate atomically.

#     def _row_for_class(self, class_id):
#         a = (self.class_ids == int(class_id)).nonzero().flatten()
#         if a.numel() != 1:
#             raise KeyError(class_id)
#         return int(a.item())

#     def class_structure(self, class_id):
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(class_id),
#             sample_count=int(self.sample_count_bank[row]),
#             learned_identity=self.learned_identity_bank[row].clone(),
#             learned_variation=self.learned_variation_bank[row].clone(),
#             learned_absolute_deviation=self.learned_absolute_deviation_bank[
#                 row
#             ].clone(),
#             physical_identity=self.physical_identity_bank[row].clone(),
#             physical_variation=self.physical_variation_bank[row].clone(),
#             physical_variance=self.physical_variance_bank[row].clone(),
#             physical_delta_variance=self.physical_delta_variance_bank[row].clone(),
#             learned_covariance_basis=self.moment_basis_bank[row].clone(),
#             learned_covariance_eigenvalues=self.moment_eigenvalues_bank[row].clone(),
#             learned_covariance_residual=self.moment_residual_variance_bank[row].clone(),
#         )

#     def class_structures(self, class_ids=None):
#         if not len(self):
#             raise RuntimeError("empty CDS bank")
#         ids = (
#             self.seen_class_ids
#             if class_ids is None
#             else _class_id_tuple(class_ids, name="class_ids")
#         )
#         rows = torch.tensor([self._row_for_class(x) for x in ids], device=self.device)
#         return ClassStructureBatch(
#             class_ids=self.class_ids[rows].clone(),
#             sample_counts=self.sample_count_bank[rows].clone(),
#             learned_identity=self.learned_identity_bank[rows].clone(),
#             learned_variation=self.learned_variation_bank[rows].clone(),
#             learned_absolute_deviation=self.learned_absolute_deviation_bank[
#                 rows
#             ].clone(),
#             physical_identity=self.physical_identity_bank[rows].clone(),
#             physical_variation=self.physical_variation_bank[rows].clone(),
#             physical_variance=self.physical_variance_bank[rows].clone(),
#             physical_delta_variance=self.physical_delta_variance_bank[rows].clone(),
#             learned_covariance_basis=self.moment_basis_bank[rows].clone(),
#             learned_covariance_eigenvalues=self.moment_eigenvalues_bank[rows].clone(),
#             learned_covariance_residual=self.moment_residual_variance_bank[rows].clone(),
#         )

#     def prototype_cosine(self, class_ids=None):
#         """PCDS-style inter-class cosine of normalized learned class prototypes."""
#         cls = self.class_structures(class_ids)
#         mu = torch.nn.functional.normalize(cls.learned_identity.flatten(1), dim=1)
#         return mu @ mu.T

#     def build_cds(self, structure):
#         self._validate_batch(structure)
#         ids = tuple(map(int, structure.class_ids.detach().cpu().tolist()))
#         committed = all(class_id in self.seen_class_ids for class_id in ids)
#         if committed:
#             for index, class_id in enumerate(ids):
#                 bank_row = self._row_for_class(class_id)
#                 for name, buffer_name in (
#                     ("learned_identity", "learned_identity_bank"),
#                     ("learned_variation", "learned_variation_bank"),
#                     ("physical_identity", "physical_identity_bank"),
#                     ("physical_variation", "physical_variation_bank"),
#                 ("physical_variance", "physical_variance_bank"),
#                 ("physical_delta_variance", "physical_delta_variance_bank"),
#                 ("learned_covariance_basis", "moment_basis_bank"),
#                 ("learned_covariance_eigenvalues", "moment_eigenvalues_bank"),
#                 ("learned_covariance_residual", "moment_residual_variance_bank"),
#                 ):
#                     if not torch.equal(
#                         getattr(structure, name)[index], getattr(self, buffer_name)[bank_row]
#                     ):
#                         committed = False
#                         break
#                 if not committed:
#                     break
#         # The auxiliary classifier supports use moments only when the supplied
#         # structure exactly matches one committed coordinate system. Raw cosine
#         # logits never depend on these nonparametric statistics.
#         moments = self.class_moments(ids) if committed else None
#         return ClassDiscriminativeStructureState(
#             classes=structure,
#             relations=derive_class_relations(structure),
#             spectral_band_positions=self.spectral_band_positions.detach().clone(),
#             learned_moments=moments,
#         )

#     def _load_from_state_dict(
#         self,
#         state_dict,
#         prefix,
#         local_metadata,
#         strict,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     ):
#         required = (*self._moment_keys(), "physical_variance_bank", "physical_delta_variance_bank")
#         mean_state = state_dict.get(prefix + "moment_mean_bank")
#         if (any(prefix + key not in state_dict for key in required)
#                 or mean_state is not None and mean_state.ndim != 2):
#             error_msgs.append("incompatible conditional or incomplete CDS checkpoint; rebuild Phase-0 from its training data under the unified CDS contract")
#             return
#         if any(prefix + key not in state_dict for key in self._moment_keys()):
#             error_msgs.append("legacy checkpoint has no bank-owned moments; rebuild Phase-0 CDS from original training data")
#             return
#         if any(
#             k.startswith(prefix + "learned_coupled_")
#             or k == prefix + "learned_residual_scale_bank"
#             for k in state_dict
#         ):
#             error_msgs.append(
#                 "G_Q checkpoint incompatible: rebuild class statistics from original phase TRAIN data"
#             )
#             return
#         for key in (
#             "class_ids",
#             "sample_count_bank",
#             "learned_identity_bank",
#             "learned_variation_bank",
#             "learned_absolute_deviation_bank",
#             "physical_identity_bank",
#             "physical_variation_bank",
#             "physical_variance_bank", "physical_delta_variance_bank",
#             *self._moment_keys(),
#         ):
#             k = prefix + key
#             if k in state_dict and getattr(self, key).shape != state_dict[k].shape:
#                 x = state_dict[k]
#                 setattr(
#                     self,
#                     key,
#                     torch.empty(
#                         x.shape, device=self.device, dtype=getattr(self, key).dtype
#                     ),
#                 )
#         super()._load_from_state_dict(
#             state_dict,
#             prefix,
#             local_metadata,
#             strict,
#             missing_keys,
#             unexpected_keys,
#             error_msgs,
#         )
#         if not error_msgs:
#             self.validate_state()


# __all__ = [
#     "PhysicalSpectralStructureBatch",
#     "ClassStructureBatch",
#     "ClassRelationState",
#     "ClassDiscriminativeStructureState",
#     "ClassStructureView",
#     "HSIClassDiscriminativeStructureBank",
#     "estimate_physical_structure_batch",
#     "combine_learned_and_physical_structure",
#     "estimate_structure_batch",
#     "merge_class_structure_batches",
#     "normalize_ordered_decision_features",
#     "sanitize_ordered_variation",
#     "ordered_variation_operator",
#     "ordered_cds_support_components",
#     "estimate_relational_variation",
#     "validate_relational_variation",
#     "learned_relational_support",
#     "combine_learned_physical_relevance",
#     "combine_class_pair_rivalry",
#     "derive_class_relations",
#     "whole_cds_pair_risk",
#     "whole_cds_evidence_masks",
#     "physical_support_components",
#     "spectral_conditioned_learned_support",
#     "native_spectral_variation_support",
# ]
# ############## MEAN ENDS############
