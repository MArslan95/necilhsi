from __future__ import annotations

"""HSI Class-Discriminative Structure (CDS) bank.

Persistent class state
----------------------
For every class c the bank stores exactly

    CDS_c = {I_Q, V_Q, I_P, V_P, N}

I_Q [K,D]
    Robust ordered learned spectral identity.  It is the deterministic
    coordinate-wise class median in the single Q space.

V_Q [K,K,D]
    Ordered learned intra-class spectral-relation variation.  For every
    ordered token pair (k,l), it stores the robust interquartile spread of the
    direct sample relation Q[k,d]-Q[l,d] across real class samples.  V_Q is
    estimated independently of I_Q: no class-center subtraction or residual
    construction is used.  Identical class samples therefore produce V_Q == 0.

I_P [B,3]
    Robust native-band physical spectral identity for
    [level, slope, curvature].

V_P [B,B,3]
    Native-band intra-class spectral-relation variation, defined by the same
    direct interquartile-spread principle as V_Q but on the physical full-band
    level/slope/curvature evidence.

N
    Number of real TRAIN samples used to build the class CDS.

Inter-class CDS is derived on demand from the stored class states.  No pairwise
memory bank, exemplar, token buffer, Gaussian model, second classifier, or
phase-specific head is stored.

Compatibility note
------------------
The historical public tensor field names ``identity`` and ``variation`` are
retained so the surrounding repository does not need a destructive API change.
Their semantics are the HSI-CDS definitions above.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


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


def _positions(
    values: Sequence[float] | Tensor | None,
    *,
    count: int,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> Tensor:
    result = (
        torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
        if values is None
        else torch.as_tensor(values, device=device, dtype=dtype).flatten()
    )
    if result.shape != (count,):
        raise ValueError(f"spectral_band_positions must contain {count} values")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("spectral_band_positions contain NaN or Inf")
    if count > 1 and bool((result[1:] <= result[:-1]).any()):
        raise ValueError("spectral_band_positions must be strictly increasing")
    return result


def _deterministic_lower_median_dim0(values: Tensor) -> Tensor:
    """Deterministic lower median over dimension 0.

    CDS construction is persistent state estimation, not a gradient operation.
    The order statistic is computed on CPU so deterministic mode behaves the
    same on CPU and CUDA.  For even N, index (N-1)//2 is used deliberately.
    """
    value = torch.as_tensor(values)
    if value.ndim < 1 or value.size(0) <= 0:
        raise ValueError("median input must have a non-empty leading dimension")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError("median input must be real floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("median input contains NaN or Inf")

    original_device = value.device
    cpu = value.detach().to(device="cpu")
    ordered = torch.sort(cpu, dim=0, stable=True).values
    result = ordered[(int(cpu.size(0)) - 1) // 2]
    return result.to(device=original_device, dtype=value.dtype)


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


# -----------------------------------------------------------------------------
# Intra-class HSI-CDS extraction
# -----------------------------------------------------------------------------


def _deterministic_quantile_dim0(values: Tensor, q: float) -> Tensor:
    """Deterministic linear quantile over dimension 0.

    This is an order-statistic operation used only for persistent CDS
    construction.  It is not a class mean, covariance, or learned parameter.
    CPU sorting avoids CUDA nondeterminism under deterministic training.
    """
    value = torch.as_tensor(values)
    if value.ndim < 1 or value.size(0) <= 0:
        raise ValueError("quantile input must have a non-empty leading dimension")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError("quantile input must be real floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quantile input contains NaN or Inf")
    q_value = float(q)
    if not 0.0 <= q_value <= 1.0:
        raise ValueError("q must be in [0,1]")

    original_device = value.device
    cpu = value.detach().to(device="cpu")
    ordered = torch.sort(cpu, dim=0, stable=True).values
    if int(cpu.size(0)) == 1:
        result = ordered[0]
    else:
        position = q_value * float(int(cpu.size(0)) - 1)
        lower = int(position)
        upper = min(lower + 1, int(cpu.size(0)) - 1)
        weight = position - float(lower)
        if upper == lower or weight == 0.0:
            result = ordered[lower]
        else:
            result = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return result.to(device=original_device, dtype=value.dtype)


def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
    """Robust middle-50% spread over dimension 0."""
    lower = _deterministic_quantile_dim0(values, 0.25)
    upper = _deterministic_quantile_dim0(values, 0.75)
    return (upper - lower).clamp_min(0.0)


def _ordered_identity_and_deformation(
    values: Tensor,
    *,
    row_chunk: int = 8,
) -> tuple[Tensor, Tensor]:
    """Build HSI identity and intra-class ordered relation variation.

    ``I`` and ``V`` are estimated from different information:

    * identity [L,F] is the deterministic coordinate-wise spectral consensus
      (median) of the real class samples;
    * variation [L,L,F] is the interquartile spread across samples of the direct
      ordered relation ``values[a,f] - values[b,f]``.

    Crucially, V is **not** formed by subtracting I, subtracting a class center,
    fitting a variance/covariance, or constructing residual features.  It asks
    only how stable each ordered spectral relation is inside the class.

    Identical class samples have zero relation spread and therefore V == 0.
    """
    value = _validate_ordered_samples(values)
    identity = _deterministic_lower_median_dim0(value)

    _, length, _ = value.shape
    chunk = _integer(row_chunk, "row_chunk", minimum=1)
    blocks: list[Tensor] = []

    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        # Direct per-sample ordered spectral relation [N,R,L,F].
        relation = value[:, start:stop, None, :] - value[:, None, :, :]
        # V stores relation spread across samples directly.  No identity/center
        # enters this calculation.
        block = _deterministic_iqr_dim0(relation)
        blocks.append(block)

    deformation = torch.cat(blocks, dim=0)
    if deformation.shape[0] != length or deformation.shape[1] != length:
        raise RuntimeError("ordered CDS variation has an invalid shape")

    # IQR(R[a,b]) == IQR(-R[b,a]) mathematically.  Maximum only removes tiny
    # numerical asymmetry while preserving the measured spread magnitude.
    deformation = torch.maximum(deformation, deformation.transpose(0, 1))
    diagonal = torch.arange(length, device=value.device)
    deformation = deformation.clone()
    deformation[diagonal, diagonal, :] = 0.0

    if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(deformation).all()):
        raise RuntimeError("ordered CDS extraction produced NaN or Inf")
    if bool((deformation < 0).any()):
        raise RuntimeError("ordered CDS variation must be non-negative")

    return identity.contiguous(), deformation.contiguous()


def sanitize_ordered_variation(variation: Tensor) -> Tensor:
    """Restore valid V_Q/V_P semantics after deterministic coordinate transport.

    Stored V is a non-negative symmetric relation-variation tensor with a zero
    diagonal.  No spectral decomposition, distribution fitting, or whitening is
    performed here.
    """
    value = torch.as_tensor(variation)
    if value.ndim != 4 or value.size(1) != value.size(2):
        raise ValueError("variation must be [C,L,L,F]")
    if (
        not torch.is_floating_point(value)
        or value.is_complex()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("variation must be finite real floating point")

    clean = value.clamp_min(0.0)
    clean = torch.maximum(clean, clean.transpose(1, 2))
    length = clean.size(1)
    diagonal = torch.arange(length, device=clean.device)
    clean = clean.clone()
    clean[:, diagonal, diagonal, :] = 0.0
    return clean.contiguous()


# Legacy API names retained because the surrounding repository may import them.
# They no longer define the mathematics of the CDS.
def project_ordered_variation_to_psd(variation: Tensor) -> Tensor:
    return sanitize_ordered_variation(variation)


def minimum_ordered_variation_eigenvalue(variation: Tensor) -> Tensor:
    """Legacy reporting hook: return the minimum stored deformation value.

    The historical function name is retained only for import compatibility.
    V_Q/V_P are not interpreted through eigenvalues by this CDS implementation.
    """
    value = sanitize_ordered_variation(variation)
    return value.amin()


def ordered_variation_operator(variation: Tensor) -> Tensor:
    """Return the complete stored ordered intra-class deformation tensor."""
    return sanitize_ordered_variation(variation)


def _estimate_ordered_variation(values: Tensor, identity: Tensor | None = None) -> Tensor:
    """Compatibility wrapper returning direct intra-class relation spread.

    ``identity`` is accepted only for call-site compatibility and deliberately
    does not participate in V estimation.  V_Q/V_P are independent of I_Q/I_P.
    """
    value = _validate_ordered_samples(values)
    if identity is not None:
        identity_value = torch.as_tensor(identity, device=value.device, dtype=value.dtype)
        if identity_value.shape != value.shape[1:]:
            raise ValueError("identity must have shape [L,F]")
        if not bool(torch.isfinite(identity_value).all()):
            raise ValueError("identity contains NaN or Inf")
    _, deformation = _ordered_identity_and_deformation(value)
    return deformation


# -----------------------------------------------------------------------------
# Inter-class CDS relations
# -----------------------------------------------------------------------------


def _pairwise_l1_similarity_global(value: Tensor) -> Tensor:
    """Scale-aware normalized L1 similarity for complete class tensors [C,...]."""
    x = torch.as_tensor(value)
    if x.ndim < 2 or x.size(0) <= 0:
        raise ValueError("value must be [C,...]")
    if not torch.is_floating_point(x) or x.is_complex() or not bool(torch.isfinite(x).all()):
        raise ValueError("value must be finite real floating point")

    flat = x.reshape(x.size(0), -1)
    diff = (flat[:, None, :] - flat[None, :, :]).abs().sum(dim=2)
    strength = flat.abs().sum(dim=1)
    denom = strength[:, None] + strength[None, :]
    eps = torch.finfo(x.dtype).eps
    similarity = 1.0 - diff / denom.clamp_min(eps)
    similarity = similarity.clamp(0.0, 1.0)
    both_zero = denom.le(eps)
    similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
    diagonal = torch.arange(x.size(0), device=x.device)
    similarity = similarity.clone()
    similarity[diagonal, diagonal] = 1.0
    return similarity


def _pairwise_l1_similarity_by_slot(value: Tensor) -> Tensor:
    """Normalized L1 similarity for [C,S,F], returned as [C,C,S]."""
    x = torch.as_tensor(value)
    if x.ndim != 3 or x.size(0) <= 0 or x.size(1) <= 0 or x.size(2) <= 0:
        raise ValueError("value must be non-empty [C,S,F]")
    if not torch.is_floating_point(x) or x.is_complex() or not bool(torch.isfinite(x).all()):
        raise ValueError("value must be finite real floating point")

    diff = (x[:, None, :, :] - x[None, :, :, :]).abs().sum(dim=3)
    strength = x.abs().sum(dim=2)
    denom = strength[:, None, :] + strength[None, :, :]
    eps = torch.finfo(x.dtype).eps
    similarity = 1.0 - diff / denom.clamp_min(eps)
    similarity = similarity.clamp(0.0, 1.0)
    similarity = torch.where(denom.le(eps), torch.ones_like(similarity), similarity)
    diagonal = torch.arange(x.size(0), device=x.device)
    similarity = similarity.clone()
    similarity[diagonal, diagonal, :] = 1.0
    return similarity


def _relation_overlap(relation_operator: Tensor) -> Tensor:
    """Complete V_Q/V_P similarity [C,C]."""
    value = ordered_variation_operator(relation_operator)
    return _pairwise_l1_similarity_global(value)


def _variation_overlap(variation: Tensor) -> Tensor:
    return _relation_overlap(variation)


def _component_relation_overlap(relation_operator: Tensor) -> Tensor:
    """Per-component V similarity [C,C,F] for [C,L,L,F]."""
    value = ordered_variation_operator(relation_operator)
    c, length, _, features = value.shape
    by_component = value.permute(0, 3, 1, 2).reshape(c, features, length * length)
    diff = (
        by_component[:, None, :, :]
        - by_component[None, :, :, :]
    ).abs().sum(dim=3)
    strength = by_component.abs().sum(dim=2)
    denom = strength[:, None, :] + strength[None, :, :]
    eps = torch.finfo(value.dtype).eps
    similarity = 1.0 - diff / denom.clamp_min(eps)
    similarity = similarity.clamp(0.0, 1.0)
    similarity = torch.where(denom.le(eps), torch.ones_like(similarity), similarity)
    diagonal = torch.arange(c, device=value.device)
    similarity = similarity.clone()
    similarity[diagonal, diagonal, :] = 1.0
    return similarity


# -----------------------------------------------------------------------------
# Public CDS views
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalSpectralStructureBatch:
    class_ids: Tensor                    # [C]
    sample_counts: Tensor                # [C]
    identity: Tensor                     # [C,B,3] I_P
    variation: Tensor                    # [C,B,B,3] V_P

    @property
    def class_count(self) -> int:
        return int(self.class_ids.numel())

    @property
    def spectral_identity(self) -> Tensor:
        return self.identity

    @property
    def intra_spectral_relation(self) -> Tensor:
        return self.variation


@dataclass(frozen=True)
class ClassStructureBatch:
    class_ids: Tensor                    # [C]
    sample_counts: Tensor                # [C]
    identity: Tensor                     # [C,K,D] I_Q
    variation: Tensor                    # [C,K,K,D] V_Q
    spectral_identity: Tensor            # [C,B,3] I_P
    spectral_variation: Tensor           # [C,B,B,3] V_P

    @property
    def class_count(self) -> int:
        return int(self.class_ids.numel())

    @property
    def learned_identity(self) -> Tensor:
        return self.identity

    @property
    def learned_intra_relation(self) -> Tensor:
        return self.variation

    @property
    def physical_identity(self) -> Tensor:
        return self.spectral_identity

    @property
    def physical_intra_relation(self) -> Tensor:
        return self.spectral_variation

    # Historical property names retained only for repository compatibility.
    @property
    def learned_mode(self) -> Tensor:
        return self.identity

    @property
    def physical_mode(self) -> Tensor:
        return self.spectral_identity

    @property
    def feature_variation(self) -> Tensor:
        """Legacy diagnostic view: strongest relation variation per token.

        The diagonal of V_Q is structurally zero, so returning the diagonal
        (the old covariance-era behavior) would be meaningless.  This derived
        view keeps shape [C,K,D] without changing persistent CDS semantics.
        """
        return self.variation.amax(dim=2)

    @property
    def spectral_feature_variation(self) -> Tensor:
        """Legacy diagnostic view: strongest physical relation variation."""
        return self.spectral_variation.amax(dim=2)


@dataclass(frozen=True)
class ClassRelationState:
    class_ids: Tensor
    learned_identity_cosine_by_region: Tensor      # [C,C,K] compatibility field; normalized L1 identity relation
    learned_variation_overlap: Tensor              # [C,C] complete V_Q relation
    spectral_identity_cosine_by_component: Tensor  # [C,C,3] compatibility field; normalized L1 physical identity relation
    spectral_variation_overlap: Tensor             # [C,C] complete V_P relation
    spectral_confusability: Tensor                 # [C,C] physical identity+deformation relation

    @property
    def learned_identity_relation_by_region(self) -> Tensor:
        return self.learned_identity_cosine_by_region

    @property
    def learned_intra_relation_overlap(self) -> Tensor:
        return self.learned_variation_overlap

    @property
    def physical_identity_relation_by_component(self) -> Tensor:
        return self.spectral_identity_cosine_by_component

    @property
    def physical_intra_relation_overlap(self) -> Tensor:
        return self.spectral_variation_overlap

    # Historical aliases.
    @property
    def learned_mode_relation_by_region(self) -> Tensor:
        return self.learned_identity_cosine_by_region

    @property
    def physical_mode_relation_by_component(self) -> Tensor:
        return self.spectral_identity_cosine_by_component


@dataclass(frozen=True)
class ClassDiscriminativeStructureState:
    classes: ClassStructureBatch
    relations: ClassRelationState
    spectral_band_positions: Tensor

    @property
    def class_ids(self) -> Tensor:
        return self.classes.class_ids

    @property
    def spectral_confusability(self) -> Tensor:
        return self.relations.spectral_confusability


@dataclass(frozen=True)
class ClassStructureView:
    class_id: int
    sample_count: int
    identity: Tensor
    variation: Tensor
    spectral_identity: Tensor
    spectral_variation: Tensor

    @property
    def learned_identity(self) -> Tensor:
        return self.identity

    @property
    def learned_intra_relation(self) -> Tensor:
        return self.variation

    @property
    def physical_identity(self) -> Tensor:
        return self.spectral_identity

    @property
    def physical_intra_relation(self) -> Tensor:
        return self.spectral_variation

    # Historical aliases.
    @property
    def learned_mode(self) -> Tensor:
        return self.identity

    @property
    def physical_mode(self) -> Tensor:
        return self.spectral_identity


# -----------------------------------------------------------------------------
# Class-state construction
# -----------------------------------------------------------------------------


def estimate_physical_structure_batch(
    spectral_evidence: Tensor,
    labels: Tensor,
    *,
    class_ids: Sequence[int] | Tensor,
) -> PhysicalSpectralStructureBatch:
    p = torch.as_tensor(spectral_evidence)
    y = torch.as_tensor(labels)
    if p.ndim != 3 or p.size(0) == 0 or p.size(1) < 2 or p.size(2) != 3:
        raise ValueError("spectral_evidence must be full-band [N,B,3], B>=2")
    if y.shape != (p.size(0),) or y.dtype != torch.long or y.device != p.device:
        raise ValueError("labels must be torch.long [N] on the spectral-evidence device")
    if (
        not torch.is_floating_point(p)
        or p.is_complex()
        or not bool(torch.isfinite(p).all())
    ):
        raise ValueError("spectral_evidence must be finite real floating point")

    requested = _class_id_tuple(class_ids, name="class_ids")
    observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
    if observed != tuple(sorted(requested)):
        raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

    counts: list[int] = []
    identities: list[Tensor] = []
    deformations: list[Tensor] = []
    for class_id in requested:
        mask = y.eq(class_id)
        count = int(mask.sum().item())
        if count <= 0:
            raise RuntimeError(f"class {class_id} has no samples")
        identity, deformation = _ordered_identity_and_deformation(p[mask])
        counts.append(count)
        identities.append(identity)
        deformations.append(deformation)

    return PhysicalSpectralStructureBatch(
        class_ids=torch.tensor(requested, device=p.device, dtype=torch.long),
        sample_counts=torch.tensor(counts, device=p.device, dtype=torch.long),
        identity=torch.stack(identities, dim=0),
        variation=torch.stack(deformations, dim=0),
    )


def combine_learned_and_physical_structure(
    ordered_features: Tensor,
    labels: Tensor,
    physical: PhysicalSpectralStructureBatch,
) -> ClassStructureBatch:
    q = torch.as_tensor(ordered_features)
    y = torch.as_tensor(labels)
    if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
        raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
    if y.shape != (q.size(0),) or y.dtype != torch.long or y.device != q.device:
        raise ValueError("labels must be torch.long [N] on the Q device")
    if not isinstance(physical, PhysicalSpectralStructureBatch) or physical.class_count <= 0:
        raise TypeError("physical must be a non-empty PhysicalSpectralStructureBatch")
    if physical.identity.device != q.device or physical.identity.dtype != q.dtype:
        raise ValueError("Q and physical CDS must share device/dtype")

    requested = tuple(int(v) for v in physical.class_ids.detach().cpu().tolist())
    observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
    if observed != tuple(sorted(requested)):
        raise ValueError("Q labels do not match cached physical CDS classes")

    identities: list[Tensor] = []
    deformations: list[Tensor] = []
    counts: list[int] = []
    for class_id in requested:
        mask = y.eq(class_id)
        count = int(mask.sum().item())
        if count <= 0:
            raise RuntimeError(f"class {class_id} has no Q samples")
        identity, deformation = _ordered_identity_and_deformation(q[mask])
        counts.append(count)
        identities.append(identity)
        deformations.append(deformation)

    count_tensor = torch.tensor(counts, device=q.device, dtype=torch.long)
    if not torch.equal(count_tensor, physical.sample_counts):
        raise RuntimeError("learned and physical CDS sample counts disagree")

    return ClassStructureBatch(
        class_ids=physical.class_ids,
        sample_counts=physical.sample_counts,
        identity=torch.stack(identities, dim=0),
        variation=torch.stack(deformations, dim=0),
        spectral_identity=physical.identity,
        spectral_variation=physical.variation,
    )


def estimate_structure_batch(
    ordered_features: Tensor,
    spectral_evidence: Tensor,
    labels: Tensor,
    *,
    class_ids: Sequence[int] | Tensor,
) -> ClassStructureBatch:
    physical = estimate_physical_structure_batch(
        spectral_evidence,
        labels,
        class_ids=class_ids,
    )
    return combine_learned_and_physical_structure(
        ordered_features,
        labels,
        physical,
    )


def merge_class_structure_batches(
    batches: Sequence[ClassStructureBatch],
) -> ClassStructureBatch:
    values = list(batches)
    if not values:
        raise ValueError("batches cannot be empty")

    device = values[0].identity.device
    dtype = values[0].identity.dtype
    learned_shape = tuple(values[0].identity.shape[1:])
    variation_shape = tuple(values[0].variation.shape[1:])
    spectral_shape = tuple(values[0].spectral_identity.shape[1:])
    spectral_variation_shape = tuple(values[0].spectral_variation.shape[1:])
    seen: set[int] = set()

    for batch in values:
        if batch.class_count <= 0:
            raise ValueError("structure batches cannot be empty")
        if batch.identity.device != device or batch.identity.dtype != dtype:
            raise ValueError("all structure batches must share device/dtype")
        if (
            tuple(batch.identity.shape[1:]) != learned_shape
            or tuple(batch.variation.shape[1:]) != variation_shape
            or tuple(batch.spectral_identity.shape[1:]) != spectral_shape
            or tuple(batch.spectral_variation.shape[1:]) != spectral_variation_shape
        ):
            raise ValueError("all structure batches must share CDS dimensions")
        ids = [int(v) for v in batch.class_ids.detach().cpu().tolist()]
        duplicate = seen.intersection(ids)
        if duplicate:
            raise ValueError(f"duplicate structure classes {sorted(duplicate)}")
        seen.update(ids)

    return ClassStructureBatch(
        class_ids=torch.cat([value.class_ids for value in values], dim=0),
        sample_counts=torch.cat([value.sample_counts for value in values], dim=0),
        identity=torch.cat([value.identity for value in values], dim=0),
        variation=torch.cat([value.variation for value in values], dim=0),
        spectral_identity=torch.cat([value.spectral_identity for value in values], dim=0),
        spectral_variation=torch.cat([value.spectral_variation for value in values], dim=0),
    )


# -----------------------------------------------------------------------------
# Derived inter-class CDS
# -----------------------------------------------------------------------------


def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
    """Derive learned + physical inter-class CDS from stored class states."""
    if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
        raise ValueError("structure must be a non-empty ClassStructureBatch")

    learned_identity_by_region = _pairwise_l1_similarity_by_slot(structure.identity)
    learned_variation_relation = _pairwise_l1_similarity_global(
        ordered_variation_operator(structure.variation)
    )

    # [C,B,3] -> [C,3,B] so the 3 physical components are the slots.
    physical_identity_by_component = _pairwise_l1_similarity_by_slot(
        structure.spectral_identity.permute(0, 2, 1)
    )
    physical_variation_relation = _pairwise_l1_similarity_global(
        ordered_variation_operator(structure.spectral_variation)
    )

    physical_identity_scalar = physical_identity_by_component.sum(dim=2) / 3.0
    spectral_confusability = (
        0.5 * (physical_identity_scalar + physical_variation_relation)
    ).clamp(0.0, 1.0)

    c = structure.class_count
    diagonal = torch.arange(c, device=structure.identity.device)
    spectral_confusability = spectral_confusability.clone()
    spectral_confusability[diagonal, diagonal] = 1.0

    values = (
        learned_identity_by_region,
        learned_variation_relation,
        physical_identity_by_component,
        physical_variation_relation,
        spectral_confusability,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("derived HSI-CDS relations contain NaN or Inf")

    return ClassRelationState(
        class_ids=structure.class_ids,
        learned_identity_cosine_by_region=learned_identity_by_region,
        learned_variation_overlap=learned_variation_relation,
        spectral_identity_cosine_by_component=physical_identity_by_component,
        spectral_variation_overlap=physical_variation_relation,
        spectral_confusability=spectral_confusability,
    )


def learned_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
    """Learned HSI-CDS confusability from I_Q and V_Q [C,C]."""
    relations = derive_class_relations(structure)
    identity_scalar = (
        relations.learned_identity_cosine_by_region.sum(dim=2)
        / float(relations.learned_identity_cosine_by_region.size(2))
    )
    result = 0.5 * (identity_scalar + relations.learned_variation_overlap)
    return result.clamp(0.0, 1.0)


def learned_boundary_overlap_matrix(structure: ClassStructureBatch) -> Tensor:
    """Legacy alias; this is CDS confusability, not a classifier boundary."""
    return learned_cds_confusability_matrix(structure)


def physical_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
    """Physical HSI-CDS confusability from I_P and V_P [C,C]."""
    relations = derive_class_relations(structure)
    return relations.spectral_confusability


def physical_region_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
    """Legacy alias for physical HSI-CDS confusability."""
    return physical_cds_confusability_matrix(structure)


def whole_cds_pair_risk(
    cds: ClassDiscriminativeStructureState,
) -> tuple[Tensor, Tensor, Tensor]:
    """One derived HSI risk from learned and physical intra/inter CDS.

    learned  = 0.5 * (I_Q relation + V_Q relation)
    physical = 0.5 * (I_P relation + V_P relation)
    risk     = sqrt(learned * physical)

    The diagonal is zero because pair risk is only used between different
    classes.  No class-set normalization or hand-selected threshold is used.
    """
    if not isinstance(cds, ClassDiscriminativeStructureState):
        raise TypeError("cds must be ClassDiscriminativeStructureState")

    rel = cds.relations
    learned_identity = (
        rel.learned_identity_cosine_by_region.sum(dim=2)
        / float(rel.learned_identity_cosine_by_region.size(2))
    )
    learned = 0.5 * (learned_identity + rel.learned_variation_overlap)

    physical_identity = (
        rel.spectral_identity_cosine_by_component.sum(dim=2)
        / float(rel.spectral_identity_cosine_by_component.size(2))
    )
    physical = 0.5 * (physical_identity + rel.spectral_variation_overlap)

    risk = torch.sqrt(
        learned.clamp_min(0.0) * physical.clamp_min(0.0)
    ).clamp(0.0, 1.0)

    c = int(risk.size(0))
    if c < 2:
        raise ValueError("pair risk requires at least two classes")
    diagonal = torch.arange(c, device=risk.device)
    risk = risk.clone()
    risk[diagonal, diagonal] = 0.0

    if not bool(torch.isfinite(risk).all()):
        raise RuntimeError("whole-CDS pair risk became NaN or Inf")
    return risk, learned.clamp(0.0, 1.0), physical.clamp(0.0, 1.0)


def spectral_correspondence_metric(structure: ClassStructureBatch) -> Tensor:
    """Robust native-band class-balanced V_P reference [3,B,B]."""
    if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
        raise ValueError("structure must be a non-empty ClassStructureBatch")
    physical = ordered_variation_operator(structure.spectral_variation)
    # [C,B,B,3] -> median over class dimension -> [B,B,3] -> [3,B,B]
    relation = _deterministic_lower_median_dim0(physical)
    result = relation.permute(2, 0, 1).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("spectral correspondence metric contains NaN or Inf")
    return result


# -----------------------------------------------------------------------------
# Persistent bank
# -----------------------------------------------------------------------------


class HSIClassDiscriminativeStructureBank(nn.Module):
    """Persistent HSI-CDS bank storing {I_Q,V_Q,I_P,V_P,N}."""

    def __init__(
        self,
        *,
        spectral_token_count: int,
        spectral_token_dim: int,
        spectral_bands: int,
        spectral_band_positions: Sequence[float] | Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.spectral_token_count = _integer(
            spectral_token_count,
            "spectral_token_count",
            minimum=2,
        )
        self.spectral_token_dim = _integer(
            spectral_token_dim,
            "spectral_token_dim",
            minimum=1,
        )
        self.spectral_bands = _integer(
            spectral_bands,
            "spectral_bands",
            minimum=2,
        )
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("structure dtype must be float32 or float64")

        positions = _positions(
            spectral_band_positions,
            count=self.spectral_bands,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "_floating_reference",
            torch.empty(0, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer("spectral_band_positions", positions, persistent=True)
        self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long))
        self.register_buffer("sample_counts", torch.empty(0, device=device, dtype=torch.long))
        self.register_buffer(
            "identity_bank",
            torch.empty(
                0,
                self.spectral_token_count,
                self.spectral_token_dim,
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "variation_bank",
            torch.empty(
                0,
                self.spectral_token_count,
                self.spectral_token_count,
                self.spectral_token_dim,
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "spectral_identity_bank",
            torch.empty(
                0,
                self.spectral_bands,
                3,
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer(
            "spectral_variation_bank",
            torch.empty(
                0,
                self.spectral_bands,
                self.spectral_bands,
                3,
                device=device,
                dtype=dtype,
            ),
        )
        self.validate_state()

    @property
    def device(self) -> torch.device:
        return self._floating_reference.device

    @property
    def dtype(self) -> torch.dtype:
        return self._floating_reference.dtype

    def __len__(self) -> int:
        return int(self.class_ids.numel())

    @property
    def seen_class_ids(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.class_ids.detach().cpu().tolist())

    @staticmethod
    def _validate_deformation_semantics(value: Tensor, *, name: str) -> None:
        if value.ndim != 4 or value.size(1) != value.size(2):
            raise RuntimeError(f"{name} must be [C,L,L,F]")
        if value.numel() == 0:
            return

        tolerance = 32.0 * torch.finfo(value.dtype).eps
        if bool((value < -tolerance).any()):
            raise RuntimeError(f"{name} contains negative variation")
        if not bool(torch.allclose(value, value.transpose(1, 2), rtol=0.0, atol=tolerance)):
            raise RuntimeError(f"{name} is not symmetric across ordered spectral pairs")

        diagonal = value.diagonal(dim1=1, dim2=2)
        if bool((diagonal.abs() > tolerance).any()):
            raise RuntimeError(f"{name} diagonal must be zero")

    def validate_state(self) -> bool:
        c = len(self)
        expected = {
            "class_ids": (c,),
            "sample_counts": (c,),
            "identity_bank": (c, self.spectral_token_count, self.spectral_token_dim),
            "variation_bank": (
                c,
                self.spectral_token_count,
                self.spectral_token_count,
                self.spectral_token_dim,
            ),
            "spectral_identity_bank": (c, self.spectral_bands, 3),
            "spectral_variation_bank": (
                c,
                self.spectral_bands,
                self.spectral_bands,
                3,
            ),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise RuntimeError(
                    f"{name} state is invalid: "
                    f"{tuple(getattr(self, name).shape)} != {shape}"
                )

        if self.class_ids.dtype != torch.long or self.sample_counts.dtype != torch.long:
            raise RuntimeError("class IDs/counts must be torch.long")
        if c:
            if bool((self.class_ids < 0).any()) or self.class_ids.unique().numel() != c:
                raise RuntimeError("class IDs must be unique non-negative values")
            if bool((self.sample_counts <= 0).any()):
                raise RuntimeError("class sample counts must be positive")

        for name in (
            "identity_bank",
            "variation_bank",
            "spectral_identity_bank",
            "spectral_variation_bank",
        ):
            value = getattr(self, name)
            if value.device != self.device or value.dtype != self.dtype:
                raise RuntimeError(f"{name} has wrong device/dtype")
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"{name} contains NaN or Inf")

        self._validate_deformation_semantics(self.variation_bank, name="variation_bank")
        self._validate_deformation_semantics(
            self.spectral_variation_bank,
            name="spectral_variation_bank",
        )
        return True

    def _validate_batch(self, batch: ClassStructureBatch) -> None:
        if not isinstance(batch, ClassStructureBatch) or batch.class_count <= 0:
            raise TypeError("batch must be a non-empty ClassStructureBatch")
        c = batch.class_count
        shapes = (
            (batch.class_ids, (c,)),
            (batch.sample_counts, (c,)),
            (batch.identity, (c, self.spectral_token_count, self.spectral_token_dim)),
            (
                batch.variation,
                (
                    c,
                    self.spectral_token_count,
                    self.spectral_token_count,
                    self.spectral_token_dim,
                ),
            ),
            (batch.spectral_identity, (c, self.spectral_bands, 3)),
            (
                batch.spectral_variation,
                (c, self.spectral_bands, self.spectral_bands, 3),
            ),
        )
        if any(value.shape != shape for value, shape in shapes):
            raise ValueError("class structure batch shape is invalid")
        if batch.class_ids.dtype != torch.long or batch.sample_counts.dtype != torch.long:
            raise ValueError("class IDs/counts must be torch.long")

        tensors = (
            batch.identity,
            batch.variation,
            batch.spectral_identity,
            batch.spectral_variation,
        )
        if any(value.device != self.device or value.dtype != self.dtype for value in tensors):
            raise ValueError("batch and bank must share device/dtype")
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("HSI-CDS contains NaN or Inf")
        if bool((batch.sample_counts <= 0).any()):
            raise ValueError("class sample counts must be positive")

        self._validate_deformation_semantics(batch.variation, name="batch.variation")
        self._validate_deformation_semantics(
            batch.spectral_variation,
            name="batch.spectral_variation",
        )

    def estimate_physical(
        self,
        spectral_evidence: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> PhysicalSpectralStructureBatch:
        p = torch.as_tensor(spectral_evidence)
        if p.device != self.device or p.dtype != self.dtype:
            raise ValueError("spectral evidence and bank must share device/dtype")
        if p.ndim != 3 or p.shape[1:] != (self.spectral_bands, 3):
            raise ValueError("spectral evidence does not match dataset-native band count")
        return estimate_physical_structure_batch(
            p,
            labels,
            class_ids=class_ids,
        )

    def estimate_classes(
        self,
        ordered_features: Tensor,
        spectral_evidence: Tensor | None,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
        physical_structure: PhysicalSpectralStructureBatch | None = None,
    ) -> ClassStructureBatch:
        q = torch.as_tensor(ordered_features)
        if q.device != self.device or q.dtype != self.dtype:
            raise ValueError("Q and bank must share device/dtype")

        if physical_structure is None:
            if spectral_evidence is None:
                raise ValueError("spectral_evidence is required when physical_structure is absent")
            physical_structure = self.estimate_physical(
                spectral_evidence,
                labels,
                class_ids=class_ids,
            )
        else:
            requested = _class_id_tuple(class_ids, name="class_ids")
            actual = tuple(
                int(v)
                for v in physical_structure.class_ids.detach().cpu().tolist()
            )
            if actual != requested:
                raise ValueError("physical_structure class order does not match class_ids")

        return combine_learned_and_physical_structure(
            q,
            labels,
            physical_structure,
        )

    @torch.no_grad()
    def append_estimate(self, estimate: ClassStructureBatch) -> None:
        self._validate_batch(estimate)
        overlap = set(
            int(v) for v in estimate.class_ids.detach().cpu().tolist()
        ).intersection(self.seen_class_ids)
        if overlap:
            raise ValueError(f"class structure already exists for {sorted(overlap)}")
        combined = (
            estimate
            if len(self) == 0
            else merge_class_structure_batches((self.class_structures(), estimate))
        )
        self.replace_all(combined)

    @torch.no_grad()
    def replace_all(self, structure: ClassStructureBatch) -> None:
        self._validate_batch(structure)
        self.class_ids = structure.class_ids.detach().clone()
        self.sample_counts = structure.sample_counts.detach().clone()
        self.identity_bank = structure.identity.detach().clone()
        self.variation_bank = sanitize_ordered_variation(structure.variation.detach().clone())
        self.spectral_identity_bank = structure.spectral_identity.detach().clone()
        self.spectral_variation_bank = sanitize_ordered_variation(
            structure.spectral_variation.detach().clone()
        )
        self.validate_state()

    def _row_for_class(self, class_id: int) -> int:
        cid = _integer(class_id, "class_id", minimum=0)
        rows = torch.nonzero(self.class_ids.eq(cid), as_tuple=False).flatten()
        if rows.numel() != 1:
            raise KeyError(f"class structure is not stored uniquely for class {cid}")
        return int(rows.item())

    def class_structure(self, class_id: int) -> ClassStructureView:
        row = self._row_for_class(class_id)
        return ClassStructureView(
            class_id=int(self.class_ids[row].item()),
            sample_count=int(self.sample_counts[row].item()),
            identity=self.identity_bank[row].detach().clone(),
            variation=self.variation_bank[row].detach().clone(),
            spectral_identity=self.spectral_identity_bank[row].detach().clone(),
            spectral_variation=self.spectral_variation_bank[row].detach().clone(),
        )

    def class_structures(
        self,
        class_ids: Sequence[int] | Tensor | None = None,
    ) -> ClassStructureBatch:
        self.validate_state()
        if len(self) == 0:
            raise RuntimeError("class structure bank is empty")
        requested = (
            self.seen_class_ids
            if class_ids is None
            else _class_id_tuple(class_ids, name="class_ids")
        )
        rows = torch.tensor(
            [self._row_for_class(cid) for cid in requested],
            device=self.device,
            dtype=torch.long,
        )
        return ClassStructureBatch(
            class_ids=self.class_ids.index_select(0, rows).detach().clone(),
            sample_counts=self.sample_counts.index_select(0, rows).detach().clone(),
            identity=self.identity_bank.index_select(0, rows).detach().clone(),
            variation=self.variation_bank.index_select(0, rows).detach().clone(),
            spectral_identity=self.spectral_identity_bank.index_select(0, rows).detach().clone(),
            spectral_variation=self.spectral_variation_bank.index_select(0, rows).detach().clone(),
        )

    def build_cds(self, batch: ClassStructureBatch) -> ClassDiscriminativeStructureState:
        self._validate_batch(batch)
        return ClassDiscriminativeStructureState(
            classes=batch,
            relations=derive_class_relations(batch),
            spectral_band_positions=self.spectral_band_positions.detach().clone(),
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
        dynamic = {
            "class_ids": torch.long,
            "sample_counts": torch.long,
            "identity_bank": self.dtype,
            "variation_bank": self.dtype,
            "spectral_identity_bank": self.dtype,
            "spectral_variation_bank": self.dtype,
        }
        for name, dtype in dynamic.items():
            incoming = state_dict.get(prefix + name)
            if incoming is not None and getattr(self, name).shape != incoming.shape:
                setattr(
                    self,
                    name,
                    torch.empty(incoming.shape, device=self.device, dtype=dtype),
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
            # Old checkpoints may contain the previous relation semantics.  Shape
            # compatibility is preserved, but a new run should rebuild CDS from
            # real TRAIN samples before relying on the state.
            self.validate_state()


__all__ = [
    "PhysicalSpectralStructureBatch",
    "ClassStructureBatch",
    "ClassRelationState",
    "ClassDiscriminativeStructureState",
    "ClassStructureView",
    "HSIClassDiscriminativeStructureBank",
    "merge_class_structure_batches",
    "estimate_physical_structure_batch",
    "combine_learned_and_physical_structure",
    "estimate_structure_batch",
    "derive_class_relations",
    "learned_cds_confusability_matrix",
    "learned_boundary_overlap_matrix",
    "sanitize_ordered_variation",
    "project_ordered_variation_to_psd",
    "minimum_ordered_variation_eigenvalue",
    "ordered_variation_operator",
    "spectral_correspondence_metric",
    "physical_cds_confusability_matrix",
    "physical_region_confusability_matrix",
    "whole_cds_pair_risk",
]












# from __future__ import annotations

# """HSI Class-Discriminative Structure (CDS) bank.

# Persistent class state
# ----------------------
# For every class c the bank stores exactly

#     CDS_c = {I_Q, V_Q, I_P, V_P, N}

# I_Q [K,D]
#     Robust ordered learned spectral identity.  It is the deterministic
#     coordinate-wise class median in the single Q space.

# V_Q [K,K,D]
#     Ordered learned intra-class spectral-relation variation.  For every
#     ordered token pair (k,l), it stores the robust interquartile spread of the
#     direct sample relation Q[k,d]-Q[l,d] across real class samples.  V_Q is
#     estimated independently of I_Q: no class-center subtraction or residual
#     construction is used.  Identical class samples therefore produce V_Q == 0.

# I_P [B,3]
#     Robust native-band physical spectral identity for
#     [level, slope, curvature].

# V_P [B,B,3]
#     Native-band intra-class spectral-relation variation, defined by the same
#     direct interquartile-spread principle as V_Q but on the physical full-band
#     level/slope/curvature evidence.

# N
#     Number of real TRAIN samples used to build the class CDS.

# Inter-class CDS is derived on demand from the stored class states.  No pairwise
# memory bank, exemplar, token buffer, Gaussian model, second classifier, or
# phase-specific head is stored.

# Compatibility note
# ------------------
# The historical public tensor field names ``identity`` and ``variation`` are
# retained so the surrounding repository does not need a destructive API change.
# Their semantics are the HSI-CDS definitions above.
# """

# from dataclasses import dataclass
# from numbers import Integral
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# # -----------------------------------------------------------------------------
# # Generic helpers
# # -----------------------------------------------------------------------------


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


# def _positions(
#     values: Sequence[float] | Tensor | None,
#     *,
#     count: int,
#     device: torch.device | str | None,
#     dtype: torch.dtype,
# ) -> Tensor:
#     result = (
#         torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
#         if values is None
#         else torch.as_tensor(values, device=device, dtype=dtype).flatten()
#     )
#     if result.shape != (count,):
#         raise ValueError(f"spectral_band_positions must contain {count} values")
#     if not bool(torch.isfinite(result).all()):
#         raise ValueError("spectral_band_positions contain NaN or Inf")
#     if count > 1 and bool((result[1:] <= result[:-1]).any()):
#         raise ValueError("spectral_band_positions must be strictly increasing")
#     return result


# def _deterministic_lower_median_dim0(values: Tensor) -> Tensor:
#     """Deterministic lower median over dimension 0.

#     CDS construction is persistent state estimation, not a gradient operation.
#     The order statistic is computed on CPU so deterministic mode behaves the
#     same on CPU and CUDA.  For even N, index (N-1)//2 is used deliberately.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("median input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("median input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("median input contains NaN or Inf")

#     original_device = value.device
#     cpu = value.detach().to(device="cpu")
#     ordered = torch.sort(cpu, dim=0, stable=True).values
#     result = ordered[(int(cpu.size(0)) - 1) // 2]
#     return result.to(device=original_device, dtype=value.dtype)


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


# # -----------------------------------------------------------------------------
# # Intra-class HSI-CDS extraction
# # -----------------------------------------------------------------------------


# def _deterministic_quantile_dim0(values: Tensor, q: float) -> Tensor:
#     """Deterministic linear quantile over dimension 0.

#     This is an order-statistic operation used only for persistent CDS
#     construction.  It is not a class mean, covariance, or learned parameter.
#     CPU sorting avoids CUDA nondeterminism under deterministic training.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("quantile input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("quantile input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("quantile input contains NaN or Inf")
#     q_value = float(q)
#     if not 0.0 <= q_value <= 1.0:
#         raise ValueError("q must be in [0,1]")

#     original_device = value.device
#     cpu = value.detach().to(device="cpu")
#     ordered = torch.sort(cpu, dim=0, stable=True).values
#     if int(cpu.size(0)) == 1:
#         result = ordered[0]
#     else:
#         position = q_value * float(int(cpu.size(0)) - 1)
#         lower = int(position)
#         upper = min(lower + 1, int(cpu.size(0)) - 1)
#         weight = position - float(lower)
#         if upper == lower or weight == 0.0:
#             result = ordered[lower]
#         else:
#             result = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
#     return result.to(device=original_device, dtype=value.dtype)


# def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
#     """Robust middle-50% spread over dimension 0."""
#     lower = _deterministic_quantile_dim0(values, 0.25)
#     upper = _deterministic_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _ordered_identity_and_deformation(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     """Build HSI identity and intra-class ordered relation variation.

#     ``I`` and ``V`` are estimated from different information:

#     * identity [L,F] is the deterministic coordinate-wise spectral consensus
#       (median) of the real class samples;
#     * variation [L,L,F] is the interquartile spread across samples of the direct
#       ordered relation ``values[a,f] - values[b,f]``.

#     Crucially, V is **not** formed by subtracting I, subtracting a class center,
#     fitting a variance/covariance, or constructing residual features.  It asks
#     only how stable each ordered spectral relation is inside the class.

#     Identical class samples have zero relation spread and therefore V == 0.
#     """
#     value = _validate_ordered_samples(values)
#     identity = _deterministic_lower_median_dim0(value)

#     _, length, _ = value.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)
#     blocks: list[Tensor] = []

#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         # Direct per-sample ordered spectral relation [N,R,L,F].
#         relation = value[:, start:stop, None, :] - value[:, None, :, :]
#         # V stores relation spread across samples directly.  No identity/center
#         # enters this calculation.
#         block = _deterministic_iqr_dim0(relation)
#         blocks.append(block)

#     deformation = torch.cat(blocks, dim=0)
#     if deformation.shape[0] != length or deformation.shape[1] != length:
#         raise RuntimeError("ordered CDS variation has an invalid shape")

#     # IQR(R[a,b]) == IQR(-R[b,a]) mathematically.  Maximum only removes tiny
#     # numerical asymmetry while preserving the measured spread magnitude.
#     deformation = torch.maximum(deformation, deformation.transpose(0, 1))
#     diagonal = torch.arange(length, device=value.device)
#     deformation = deformation.clone()
#     deformation[diagonal, diagonal, :] = 0.0

#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(deformation).all()):
#         raise RuntimeError("ordered CDS extraction produced NaN or Inf")
#     if bool((deformation < 0).any()):
#         raise RuntimeError("ordered CDS variation must be non-negative")

#     return identity.contiguous(), deformation.contiguous()


# def sanitize_ordered_variation(variation: Tensor) -> Tensor:
#     """Restore valid V_Q/V_P semantics after deterministic coordinate transport.

#     Stored V is a non-negative symmetric relation-variation tensor with a zero
#     diagonal.  No spectral decomposition, distribution fitting, or whitening is
#     performed here.
#     """
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     if (
#         not torch.is_floating_point(value)
#         or value.is_complex()
#         or not bool(torch.isfinite(value).all())
#     ):
#         raise ValueError("variation must be finite real floating point")

#     clean = value.clamp_min(0.0)
#     clean = torch.maximum(clean, clean.transpose(1, 2))
#     length = clean.size(1)
#     diagonal = torch.arange(length, device=clean.device)
#     clean = clean.clone()
#     clean[:, diagonal, diagonal, :] = 0.0
#     return clean.contiguous()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     """Return the complete stored ordered intra-class deformation tensor."""
#     return sanitize_ordered_variation(variation)


# # -----------------------------------------------------------------------------
# # Inter-class CDS relations
# # -----------------------------------------------------------------------------


# def _pairwise_l1_similarity_global(value: Tensor) -> Tensor:
#     """Scale-aware normalized L1 similarity for complete class tensors [C,...]."""
#     x = torch.as_tensor(value)
#     if x.ndim < 2 or x.size(0) <= 0:
#         raise ValueError("value must be [C,...]")
#     if not torch.is_floating_point(x) or x.is_complex() or not bool(torch.isfinite(x).all()):
#         raise ValueError("value must be finite real floating point")

#     flat = x.reshape(x.size(0), -1)
#     diff = (flat[:, None, :] - flat[None, :, :]).abs().sum(dim=2)
#     strength = flat.abs().sum(dim=1)
#     denom = strength[:, None] + strength[None, :]
#     eps = torch.finfo(x.dtype).eps
#     similarity = 1.0 - diff / denom.clamp_min(eps)
#     similarity = similarity.clamp(0.0, 1.0)
#     both_zero = denom.le(eps)
#     similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
#     diagonal = torch.arange(x.size(0), device=x.device)
#     similarity = similarity.clone()
#     similarity[diagonal, diagonal] = 1.0
#     return similarity


# def _pairwise_l1_similarity_by_slot(value: Tensor) -> Tensor:
#     """Normalized L1 similarity for [C,S,F], returned as [C,C,S]."""
#     x = torch.as_tensor(value)
#     if x.ndim != 3 or x.size(0) <= 0 or x.size(1) <= 0 or x.size(2) <= 0:
#         raise ValueError("value must be non-empty [C,S,F]")
#     if not torch.is_floating_point(x) or x.is_complex() or not bool(torch.isfinite(x).all()):
#         raise ValueError("value must be finite real floating point")

#     diff = (x[:, None, :, :] - x[None, :, :, :]).abs().sum(dim=3)
#     strength = x.abs().sum(dim=2)
#     denom = strength[:, None, :] + strength[None, :, :]
#     eps = torch.finfo(x.dtype).eps
#     similarity = 1.0 - diff / denom.clamp_min(eps)
#     similarity = similarity.clamp(0.0, 1.0)
#     similarity = torch.where(denom.le(eps), torch.ones_like(similarity), similarity)
#     diagonal = torch.arange(x.size(0), device=x.device)
#     similarity = similarity.clone()
#     similarity[diagonal, diagonal, :] = 1.0
#     return similarity


# def _relation_overlap(relation_operator: Tensor) -> Tensor:
#     """Complete V_Q/V_P similarity [C,C]."""
#     value = ordered_variation_operator(relation_operator)
#     return _pairwise_l1_similarity_global(value)


# def _variation_overlap(variation: Tensor) -> Tensor:
#     return _relation_overlap(variation)


# def _component_relation_overlap(relation_operator: Tensor) -> Tensor:
#     """Per-component V similarity [C,C,F] for [C,L,L,F]."""
#     value = ordered_variation_operator(relation_operator)
#     c, length, _, features = value.shape
#     by_component = value.permute(0, 3, 1, 2).reshape(c, features, length * length)
#     diff = (
#         by_component[:, None, :, :]
#         - by_component[None, :, :, :]
#     ).abs().sum(dim=3)
#     strength = by_component.abs().sum(dim=2)
#     denom = strength[:, None, :] + strength[None, :, :]
#     eps = torch.finfo(value.dtype).eps
#     similarity = 1.0 - diff / denom.clamp_min(eps)
#     similarity = similarity.clamp(0.0, 1.0)
#     similarity = torch.where(denom.le(eps), torch.ones_like(similarity), similarity)
#     diagonal = torch.arange(c, device=value.device)
#     similarity = similarity.clone()
#     similarity[diagonal, diagonal, :] = 1.0
#     return similarity


# # -----------------------------------------------------------------------------
# # Public CDS views
# # -----------------------------------------------------------------------------


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor                    # [C]
#     sample_counts: Tensor                # [C]
#     identity: Tensor                     # [C,B,3] I_P
#     variation: Tensor                    # [C,B,B,3] V_P

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def spectral_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def intra_spectral_relation(self) -> Tensor:
#         return self.variation


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor                    # [C]
#     sample_counts: Tensor                # [C]
#     identity: Tensor                     # [C,K,D] I_Q
#     variation: Tensor                    # [C,K,K,D] V_Q
#     spectral_identity: Tensor            # [C,B,3] I_P
#     spectral_variation: Tensor           # [C,B,B,3] V_P

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def learned_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def learned_intra_relation(self) -> Tensor:
#         return self.variation

#     @property
#     def physical_identity(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def physical_intra_relation(self) -> Tensor:
#         return self.spectral_variation



# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_similarity_by_region: Tensor      # [C,C,K] compatibility field; normalized L1 identity relation
#     learned_variation_overlap: Tensor              # [C,C] complete V_Q relation
#     spectral_identity_similarity_by_component: Tensor  # [C,C,3] compatibility field; normalized L1 physical identity relation
#     spectral_variation_overlap: Tensor             # [C,C] complete V_P relation
#     spectral_confusability: Tensor                 # [C,C] physical identity+deformation relation

#     @property
#     def learned_identity_relation_by_region(self) -> Tensor:
#         return self.learned_identity_similarity_by_region

#     @property
#     def learned_intra_relation_overlap(self) -> Tensor:
#         return self.learned_variation_overlap

#     @property
#     def physical_identity_relation_by_component(self) -> Tensor:
#         return self.spectral_identity_similarity_by_component

#     @property
#     def physical_intra_relation_overlap(self) -> Tensor:
#         return self.spectral_variation_overlap



# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor

#     @property
#     def class_ids(self) -> Tensor:
#         return self.classes.class_ids

#     @property
#     def spectral_confusability(self) -> Tensor:
#         return self.relations.spectral_confusability


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     identity: Tensor
#     variation: Tensor
#     spectral_identity: Tensor
#     spectral_variation: Tensor

#     @property
#     def learned_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def learned_intra_relation(self) -> Tensor:
#         return self.variation

#     @property
#     def physical_identity(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def physical_intra_relation(self) -> Tensor:
#         return self.spectral_variation



# # -----------------------------------------------------------------------------
# # Class-state construction
# # -----------------------------------------------------------------------------


# def estimate_physical_structure_batch(
#     spectral_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> PhysicalSpectralStructureBatch:
#     p = torch.as_tensor(spectral_evidence)
#     y = torch.as_tensor(labels)
#     if p.ndim != 3 or p.size(0) == 0 or p.size(1) < 2 or p.size(2) != 3:
#         raise ValueError("spectral_evidence must be full-band [N,B,3], B>=2")
#     if y.shape != (p.size(0),) or y.dtype != torch.long or y.device != p.device:
#         raise ValueError("labels must be torch.long [N] on the spectral-evidence device")
#     if (
#         not torch.is_floating_point(p)
#         or p.is_complex()
#         or not bool(torch.isfinite(p).all())
#     ):
#         raise ValueError("spectral_evidence must be finite real floating point")

#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     deformations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no samples")
#         identity, deformation = _ordered_identity_and_deformation(p[mask])
#         counts.append(count)
#         identities.append(identity)
#         deformations.append(deformation)

#     return PhysicalSpectralStructureBatch(
#         class_ids=torch.tensor(requested, device=p.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=p.device, dtype=torch.long),
#         identity=torch.stack(identities, dim=0),
#         variation=torch.stack(deformations, dim=0),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical: PhysicalSpectralStructureBatch,
# ) -> ClassStructureBatch:
#     q = torch.as_tensor(ordered_features)
#     y = torch.as_tensor(labels)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if y.shape != (q.size(0),) or y.dtype != torch.long or y.device != q.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if not isinstance(physical, PhysicalSpectralStructureBatch) or physical.class_count <= 0:
#         raise TypeError("physical must be a non-empty PhysicalSpectralStructureBatch")
#     if physical.identity.device != q.device or physical.identity.dtype != q.dtype:
#         raise ValueError("Q and physical CDS must share device/dtype")

#     requested = tuple(int(v) for v in physical.class_ids.detach().cpu().tolist())
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("Q labels do not match cached physical CDS classes")

#     identities: list[Tensor] = []
#     deformations: list[Tensor] = []
#     counts: list[int] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no Q samples")
#         identity, deformation = _ordered_identity_and_deformation(q[mask])
#         counts.append(count)
#         identities.append(identity)
#         deformations.append(deformation)

#     count_tensor = torch.tensor(counts, device=q.device, dtype=torch.long)
#     if not torch.equal(count_tensor, physical.sample_counts):
#         raise RuntimeError("learned and physical CDS sample counts disagree")

#     return ClassStructureBatch(
#         class_ids=physical.class_ids,
#         sample_counts=physical.sample_counts,
#         identity=torch.stack(identities, dim=0),
#         variation=torch.stack(deformations, dim=0),
#         spectral_identity=physical.identity,
#         spectral_variation=physical.variation,
#     )


# def estimate_structure_batch(
#     ordered_features: Tensor,
#     spectral_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     physical = estimate_physical_structure_batch(
#         spectral_evidence,
#         labels,
#         class_ids=class_ids,
#     )
#     return combine_learned_and_physical_structure(
#         ordered_features,
#         labels,
#         physical,
#     )


# def merge_class_structure_batches(
#     batches: Sequence[ClassStructureBatch],
# ) -> ClassStructureBatch:
#     values = list(batches)
#     if not values:
#         raise ValueError("batches cannot be empty")

#     device = values[0].identity.device
#     dtype = values[0].identity.dtype
#     learned_shape = tuple(values[0].identity.shape[1:])
#     variation_shape = tuple(values[0].variation.shape[1:])
#     spectral_shape = tuple(values[0].spectral_identity.shape[1:])
#     spectral_variation_shape = tuple(values[0].spectral_variation.shape[1:])
#     seen: set[int] = set()

#     for batch in values:
#         if batch.class_count <= 0:
#             raise ValueError("structure batches cannot be empty")
#         if batch.identity.device != device or batch.identity.dtype != dtype:
#             raise ValueError("all structure batches must share device/dtype")
#         if (
#             tuple(batch.identity.shape[1:]) != learned_shape
#             or tuple(batch.variation.shape[1:]) != variation_shape
#             or tuple(batch.spectral_identity.shape[1:]) != spectral_shape
#             or tuple(batch.spectral_variation.shape[1:]) != spectral_variation_shape
#         ):
#             raise ValueError("all structure batches must share CDS dimensions")
#         ids = [int(v) for v in batch.class_ids.detach().cpu().tolist()]
#         duplicate = seen.intersection(ids)
#         if duplicate:
#             raise ValueError(f"duplicate structure classes {sorted(duplicate)}")
#         seen.update(ids)

#     return ClassStructureBatch(
#         class_ids=torch.cat([value.class_ids for value in values], dim=0),
#         sample_counts=torch.cat([value.sample_counts for value in values], dim=0),
#         identity=torch.cat([value.identity for value in values], dim=0),
#         variation=torch.cat([value.variation for value in values], dim=0),
#         spectral_identity=torch.cat([value.spectral_identity for value in values], dim=0),
#         spectral_variation=torch.cat([value.spectral_variation for value in values], dim=0),
#     )


# # -----------------------------------------------------------------------------
# # Derived inter-class CDS
# # -----------------------------------------------------------------------------


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     """Derive learned + physical inter-class CDS from stored class states."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")

#     learned_identity_by_region = _pairwise_l1_similarity_by_slot(structure.identity)
#     learned_variation_relation = _pairwise_l1_similarity_global(
#         ordered_variation_operator(structure.variation)
#     )

#     # [C,B,3] -> [C,3,B] so the 3 physical components are the slots.
#     physical_identity_by_component = _pairwise_l1_similarity_by_slot(
#         structure.spectral_identity.permute(0, 2, 1)
#     )
#     physical_variation_relation = _pairwise_l1_similarity_global(
#         ordered_variation_operator(structure.spectral_variation)
#     )

#     physical_identity_scalar = physical_identity_by_component.sum(dim=2) / 3.0
#     spectral_confusability = (
#         0.5 * (physical_identity_scalar + physical_variation_relation)
#     ).clamp(0.0, 1.0)

#     c = structure.class_count
#     diagonal = torch.arange(c, device=structure.identity.device)
#     spectral_confusability = spectral_confusability.clone()
#     spectral_confusability[diagonal, diagonal] = 1.0

#     values = (
#         learned_identity_by_region,
#         learned_variation_relation,
#         physical_identity_by_component,
#         physical_variation_relation,
#         spectral_confusability,
#     )
#     if not all(bool(torch.isfinite(value).all()) for value in values):
#         raise RuntimeError("derived HSI-CDS relations contain NaN or Inf")

#     return ClassRelationState(
#         class_ids=structure.class_ids,
#         learned_identity_similarity_by_region=learned_identity_by_region,
#         learned_variation_overlap=learned_variation_relation,
#         spectral_identity_similarity_by_component=physical_identity_by_component,
#         spectral_variation_overlap=physical_variation_relation,
#         spectral_confusability=spectral_confusability,
#     )


# def learned_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     """Learned HSI-CDS confusability from I_Q and V_Q [C,C]."""
#     relations = derive_class_relations(structure)
#     identity_scalar = (
#         relations.learned_identity_similarity_by_region.sum(dim=2)
#         / float(relations.learned_identity_similarity_by_region.size(2))
#     )
#     result = 0.5 * (identity_scalar + relations.learned_variation_overlap)
#     return result.clamp(0.0, 1.0)


# def physical_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     """Physical HSI-CDS confusability from I_P and V_P [C,C]."""
#     relations = derive_class_relations(structure)
#     return relations.spectral_confusability


# def whole_cds_pair_risk(
#     cds: ClassDiscriminativeStructureState,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """One derived HSI risk from learned and physical intra/inter CDS.

#     learned  = 0.5 * (I_Q relation + V_Q relation)
#     physical = 0.5 * (I_P relation + V_P relation)
#     risk     = sqrt(learned * physical)

#     The diagonal is zero because pair risk is only used between different
#     classes.  No class-set normalization or hand-selected threshold is used.
#     """
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")

#     rel = cds.relations
#     learned_identity = (
#         rel.learned_identity_similarity_by_region.sum(dim=2)
#         / float(rel.learned_identity_similarity_by_region.size(2))
#     )
#     learned = 0.5 * (learned_identity + rel.learned_variation_overlap)

#     physical_identity = (
#         rel.spectral_identity_similarity_by_component.sum(dim=2)
#         / float(rel.spectral_identity_similarity_by_component.size(2))
#     )
#     physical = 0.5 * (physical_identity + rel.spectral_variation_overlap)

#     risk = torch.sqrt(
#         learned.clamp_min(0.0) * physical.clamp_min(0.0)
#     ).clamp(0.0, 1.0)

#     c = int(risk.size(0))
#     if c < 2:
#         raise ValueError("pair risk requires at least two classes")
#     diagonal = torch.arange(c, device=risk.device)
#     risk = risk.clone()
#     risk[diagonal, diagonal] = 0.0

#     if not bool(torch.isfinite(risk).all()):
#         raise RuntimeError("whole-CDS pair risk became NaN or Inf")
#     return risk, learned.clamp(0.0, 1.0), physical.clamp(0.0, 1.0)


# def spectral_correspondence_metric(structure: ClassStructureBatch) -> Tensor:
#     """Robust native-band class-balanced V_P reference [3,B,B]."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     physical = ordered_variation_operator(structure.spectral_variation)
#     # [C,B,B,3] -> median over class dimension -> [B,B,3] -> [3,B,B]
#     relation = _deterministic_lower_median_dim0(physical)
#     result = relation.permute(2, 0, 1).contiguous()
#     if not bool(torch.isfinite(result).all()):
#         raise RuntimeError("spectral correspondence metric contains NaN or Inf")
#     return result


# # -----------------------------------------------------------------------------
# # Persistent bank
# # -----------------------------------------------------------------------------


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Persistent HSI-CDS bank storing {I_Q,V_Q,I_P,V_P,N}."""

#     def __init__(
#         self,
#         *,
#         spectral_token_count: int,
#         spectral_token_dim: int,
#         spectral_bands: int,
#         spectral_band_positions: Sequence[float] | Tensor | None = None,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype = torch.float32,
#     ) -> None:
#         super().__init__()
#         self.spectral_token_count = _integer(
#             spectral_token_count,
#             "spectral_token_count",
#             minimum=2,
#         )
#         self.spectral_token_dim = _integer(
#             spectral_token_dim,
#             "spectral_token_dim",
#             minimum=1,
#         )
#         self.spectral_bands = _integer(
#             spectral_bands,
#             "spectral_bands",
#             minimum=2,
#         )
#         if dtype not in (torch.float32, torch.float64):
#             raise TypeError("structure dtype must be float32 or float64")

#         positions = _positions(
#             spectral_band_positions,
#             count=self.spectral_bands,
#             device=device,
#             dtype=dtype,
#         )
#         self.register_buffer(
#             "_floating_reference",
#             torch.empty(0, device=device, dtype=dtype),
#             persistent=False,
#         )
#         self.register_buffer("spectral_band_positions", positions, persistent=True)
#         self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long))
#         self.register_buffer("sample_counts", torch.empty(0, device=device, dtype=torch.long))
#         self.register_buffer(
#             "identity_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "spectral_identity_bank",
#             torch.empty(
#                 0,
#                 self.spectral_bands,
#                 3,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "spectral_variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_bands,
#                 self.spectral_bands,
#                 3,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.validate_state()

#     @property
#     def device(self) -> torch.device:
#         return self._floating_reference.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self._floating_reference.dtype

#     def __len__(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(v) for v in self.class_ids.detach().cpu().tolist())

#     @staticmethod
#     def _validate_deformation_semantics(value: Tensor, *, name: str) -> None:
#         if value.ndim != 4 or value.size(1) != value.size(2):
#             raise RuntimeError(f"{name} must be [C,L,L,F]")
#         if value.numel() == 0:
#             return

#         tolerance = 32.0 * torch.finfo(value.dtype).eps
#         if bool((value < -tolerance).any()):
#             raise RuntimeError(f"{name} contains negative variation")
#         if not bool(torch.allclose(value, value.transpose(1, 2), rtol=0.0, atol=tolerance)):
#             raise RuntimeError(f"{name} is not symmetric across ordered spectral pairs")

#         diagonal = value.diagonal(dim1=1, dim2=2)
#         if bool((diagonal.abs() > tolerance).any()):
#             raise RuntimeError(f"{name} diagonal must be zero")

#     def validate_state(self) -> bool:
#         c = len(self)
#         expected = {
#             "class_ids": (c,),
#             "sample_counts": (c,),
#             "identity_bank": (c, self.spectral_token_count, self.spectral_token_dim),
#             "variation_bank": (
#                 c,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#             ),
#             "spectral_identity_bank": (c, self.spectral_bands, 3),
#             "spectral_variation_bank": (
#                 c,
#                 self.spectral_bands,
#                 self.spectral_bands,
#                 3,
#             ),
#         }
#         for name, shape in expected.items():
#             if getattr(self, name).shape != shape:
#                 raise RuntimeError(
#                     f"{name} state is invalid: "
#                     f"{tuple(getattr(self, name).shape)} != {shape}"
#                 )

#         if self.class_ids.dtype != torch.long or self.sample_counts.dtype != torch.long:
#             raise RuntimeError("class IDs/counts must be torch.long")
#         if c:
#             if bool((self.class_ids < 0).any()) or self.class_ids.unique().numel() != c:
#                 raise RuntimeError("class IDs must be unique non-negative values")
#             if bool((self.sample_counts <= 0).any()):
#                 raise RuntimeError("class sample counts must be positive")

#         for name in (
#             "identity_bank",
#             "variation_bank",
#             "spectral_identity_bank",
#             "spectral_variation_bank",
#         ):
#             value = getattr(self, name)
#             if value.device != self.device or value.dtype != self.dtype:
#                 raise RuntimeError(f"{name} has wrong device/dtype")
#             if not bool(torch.isfinite(value).all()):
#                 raise RuntimeError(f"{name} contains NaN or Inf")

#         self._validate_deformation_semantics(self.variation_bank, name="variation_bank")
#         self._validate_deformation_semantics(
#             self.spectral_variation_bank,
#             name="spectral_variation_bank",
#         )
#         return True

#     def _validate_batch(self, batch: ClassStructureBatch) -> None:
#         if not isinstance(batch, ClassStructureBatch) or batch.class_count <= 0:
#             raise TypeError("batch must be a non-empty ClassStructureBatch")
#         c = batch.class_count
#         shapes = (
#             (batch.class_ids, (c,)),
#             (batch.sample_counts, (c,)),
#             (batch.identity, (c, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 batch.variation,
#                 (
#                     c,
#                     self.spectral_token_count,
#                     self.spectral_token_count,
#                     self.spectral_token_dim,
#                 ),
#             ),
#             (batch.spectral_identity, (c, self.spectral_bands, 3)),
#             (
#                 batch.spectral_variation,
#                 (c, self.spectral_bands, self.spectral_bands, 3),
#             ),
#         )
#         if any(value.shape != shape for value, shape in shapes):
#             raise ValueError("class structure batch shape is invalid")
#         if batch.class_ids.dtype != torch.long or batch.sample_counts.dtype != torch.long:
#             raise ValueError("class IDs/counts must be torch.long")

#         tensors = (
#             batch.identity,
#             batch.variation,
#             batch.spectral_identity,
#             batch.spectral_variation,
#         )
#         if any(value.device != self.device or value.dtype != self.dtype for value in tensors):
#             raise ValueError("batch and bank must share device/dtype")
#         if not all(bool(torch.isfinite(value).all()) for value in tensors):
#             raise ValueError("HSI-CDS contains NaN or Inf")
#         if bool((batch.sample_counts <= 0).any()):
#             raise ValueError("class sample counts must be positive")

#         self._validate_deformation_semantics(batch.variation, name="batch.variation")
#         self._validate_deformation_semantics(
#             batch.spectral_variation,
#             name="batch.spectral_variation",
#         )

#     def estimate_physical(
#         self,
#         spectral_evidence: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         p = torch.as_tensor(spectral_evidence)
#         if p.device != self.device or p.dtype != self.dtype:
#             raise ValueError("spectral evidence and bank must share device/dtype")
#         if p.ndim != 3 or p.shape[1:] != (self.spectral_bands, 3):
#             raise ValueError("spectral evidence does not match dataset-native band count")
#         return estimate_physical_structure_batch(
#             p,
#             labels,
#             class_ids=class_ids,
#         )

#     def estimate_classes(
#         self,
#         ordered_features: Tensor,
#         spectral_evidence: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         q = torch.as_tensor(ordered_features)
#         if q.device != self.device or q.dtype != self.dtype:
#             raise ValueError("Q and bank must share device/dtype")

#         if physical_structure is None:
#             if spectral_evidence is None:
#                 raise ValueError("spectral_evidence is required when physical_structure is absent")
#             physical_structure = self.estimate_physical(
#                 spectral_evidence,
#                 labels,
#                 class_ids=class_ids,
#             )
#         else:
#             requested = _class_id_tuple(class_ids, name="class_ids")
#             actual = tuple(
#                 int(v)
#                 for v in physical_structure.class_ids.detach().cpu().tolist()
#             )
#             if actual != requested:
#                 raise ValueError("physical_structure class order does not match class_ids")

#         return combine_learned_and_physical_structure(
#             q,
#             labels,
#             physical_structure,
#         )

#     @torch.no_grad()
#     def append_estimate(self, estimate: ClassStructureBatch) -> None:
#         self._validate_batch(estimate)
#         overlap = set(
#             int(v) for v in estimate.class_ids.detach().cpu().tolist()
#         ).intersection(self.seen_class_ids)
#         if overlap:
#             raise ValueError(f"class structure already exists for {sorted(overlap)}")
#         combined = (
#             estimate
#             if len(self) == 0
#             else merge_class_structure_batches((self.class_structures(), estimate))
#         )
#         self.replace_all(combined)

#     @torch.no_grad()
#     def replace_all(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         self.class_ids = structure.class_ids.detach().clone()
#         self.sample_counts = structure.sample_counts.detach().clone()
#         self.identity_bank = structure.identity.detach().clone()
#         self.variation_bank = sanitize_ordered_variation(structure.variation.detach().clone())
#         self.spectral_identity_bank = structure.spectral_identity.detach().clone()
#         self.spectral_variation_bank = sanitize_ordered_variation(
#             structure.spectral_variation.detach().clone()
#         )
#         self.validate_state()

#     def _row_for_class(self, class_id: int) -> int:
#         cid = _integer(class_id, "class_id", minimum=0)
#         rows = torch.nonzero(self.class_ids.eq(cid), as_tuple=False).flatten()
#         if rows.numel() != 1:
#             raise KeyError(f"class structure is not stored uniquely for class {cid}")
#         return int(rows.item())

#     def class_structure(self, class_id: int) -> ClassStructureView:
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(self.class_ids[row].item()),
#             sample_count=int(self.sample_counts[row].item()),
#             identity=self.identity_bank[row].detach().clone(),
#             variation=self.variation_bank[row].detach().clone(),
#             spectral_identity=self.spectral_identity_bank[row].detach().clone(),
#             spectral_variation=self.spectral_variation_bank[row].detach().clone(),
#         )

#     def class_structures(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassStructureBatch:
#         self.validate_state()
#         if len(self) == 0:
#             raise RuntimeError("class structure bank is empty")
#         requested = (
#             self.seen_class_ids
#             if class_ids is None
#             else _class_id_tuple(class_ids, name="class_ids")
#         )
#         rows = torch.tensor(
#             [self._row_for_class(cid) for cid in requested],
#             device=self.device,
#             dtype=torch.long,
#         )
#         return ClassStructureBatch(
#             class_ids=self.class_ids.index_select(0, rows).detach().clone(),
#             sample_counts=self.sample_counts.index_select(0, rows).detach().clone(),
#             identity=self.identity_bank.index_select(0, rows).detach().clone(),
#             variation=self.variation_bank.index_select(0, rows).detach().clone(),
#             spectral_identity=self.spectral_identity_bank.index_select(0, rows).detach().clone(),
#             spectral_variation=self.spectral_variation_bank.index_select(0, rows).detach().clone(),
#         )

#     def build_cds(self, batch: ClassStructureBatch) -> ClassDiscriminativeStructureState:
#         self._validate_batch(batch)
#         return ClassDiscriminativeStructureState(
#             classes=batch,
#             relations=derive_class_relations(batch),
#             spectral_band_positions=self.spectral_band_positions.detach().clone(),
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
#         dynamic = {
#             "class_ids": torch.long,
#             "sample_counts": torch.long,
#             "identity_bank": self.dtype,
#             "variation_bank": self.dtype,
#             "spectral_identity_bank": self.dtype,
#             "spectral_variation_bank": self.dtype,
#         }
#         for name, dtype in dynamic.items():
#             incoming = state_dict.get(prefix + name)
#             if incoming is not None and getattr(self, name).shape != incoming.shape:
#                 setattr(
#                     self,
#                     name,
#                     torch.empty(incoming.shape, device=self.device, dtype=dtype),
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
#             # Old checkpoints may contain the previous relation semantics.  Shape
#             # compatibility is preserved, but a new run should rebuild CDS from
#             # real TRAIN samples before relying on the state.
#             self.validate_state()


# __all__ = [
#     "PhysicalSpectralStructureBatch",
#     "ClassStructureBatch",
#     "ClassRelationState",
#     "ClassDiscriminativeStructureState",
#     "ClassStructureView",
#     "HSIClassDiscriminativeStructureBank",
#     "merge_class_structure_batches",
#     "estimate_physical_structure_batch",
#     "combine_learned_and_physical_structure",
#     "estimate_structure_batch",
#     "derive_class_relations",
#     "learned_cds_confusability_matrix",
#     "sanitize_ordered_variation",
#     "ordered_variation_operator",
#     "spectral_correspondence_metric",
#     "physical_cds_confusability_matrix",
#     "whole_cds_pair_risk",
# ]























# from __future__ import annotations

# """HSI Class-Discriminative Structure (CDS) bank.

# Persistent class state
# ----------------------
# For every class c the bank stores exactly

#     CDS_c = {I_Q, V_Q, I_P, V_P, N}

# I_Q [K,D]
#     Robust ordered learned spectral identity.  It is the deterministic
#     coordinate-wise class median in the single Q space.

# V_Q [K,K,D]
#     Ordered learned intra-class spectral-relation variation.  Its diagonal stores absolute coordinate IQR and its off-diagonal stores
#     robust IQR of the direct ordered relation Q[k,d]-Q[l,d].  V_Q is
#     estimated independently of I_Q: no class-center subtraction or residual
#     construction is used.  Identical class samples therefore produce V_Q == 0.

# I_P [B,3]
#     Robust native-band physical spectral identity for
#     [level, slope, curvature].

# V_P [B,B,3]
#     Native-band absolute-plus-relation intra-class spectral variation, defined by the same
#     robust interquartile-spread principle as V_Q but on the physical full-band
#     level/slope/curvature evidence.

# N
#     Number of real TRAIN samples used to build the class CDS.

# Inter-class CDS is derived on demand from the stored class states.  No pairwise
# memory bank, exemplar, token buffer, Gaussian model, second classifier, or
# phase-specific head is stored.

# Compatibility note
# ------------------
# The historical public tensor field names ``identity`` and ``variation`` are
# retained so the surrounding repository does not need a destructive API change.
# Their semantics are the HSI-CDS definitions above.
# """

# from dataclasses import dataclass
# from numbers import Integral
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# # -----------------------------------------------------------------------------
# # Generic helpers
# # -----------------------------------------------------------------------------


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


# def _positions(
#     values: Sequence[float] | Tensor | None,
#     *,
#     count: int,
#     device: torch.device | str | None,
#     dtype: torch.dtype,
# ) -> Tensor:
#     result = (
#         torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
#         if values is None
#         else torch.as_tensor(values, device=device, dtype=dtype).flatten()
#     )
#     if result.shape != (count,):
#         raise ValueError(f"spectral_band_positions must contain {count} values")
#     if not bool(torch.isfinite(result).all()):
#         raise ValueError("spectral_band_positions contain NaN or Inf")
#     if count > 1 and bool((result[1:] <= result[:-1]).any()):
#         raise ValueError("spectral_band_positions must be strictly increasing")
#     return result


# def _deterministic_lower_median_dim0(values: Tensor) -> Tensor:
#     """Deterministic lower median over dimension 0.

#     CDS construction is persistent state estimation, not a gradient operation.
#     The order statistic is computed on CPU so deterministic mode behaves the
#     same on CPU and CUDA.  For even N, index (N-1)//2 is used deliberately.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("median input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("median input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("median input contains NaN or Inf")

#     original_device = value.device
#     cpu = value.detach().to(device="cpu")
#     ordered = torch.sort(cpu, dim=0, stable=True).values
#     result = ordered[(int(cpu.size(0)) - 1) // 2]
#     return result.to(device=original_device, dtype=value.dtype)


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


# # -----------------------------------------------------------------------------
# # Intra-class HSI-CDS extraction
# # -----------------------------------------------------------------------------


# def _deterministic_quantile_dim0(values: Tensor, q: float) -> Tensor:
#     """Deterministic linear quantile over dimension 0.

#     This is an order-statistic operation used only for persistent CDS
#     construction.  It is not a class mean, covariance, or learned parameter.
#     CPU sorting avoids CUDA nondeterminism under deterministic training.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("quantile input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("quantile input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("quantile input contains NaN or Inf")
#     q_value = float(q)
#     if not 0.0 <= q_value <= 1.0:
#         raise ValueError("q must be in [0,1]")

#     original_device = value.device
#     cpu = value.detach().to(device="cpu")
#     ordered = torch.sort(cpu, dim=0, stable=True).values
#     if int(cpu.size(0)) == 1:
#         result = ordered[0]
#     else:
#         position = q_value * float(int(cpu.size(0)) - 1)
#         lower = int(position)
#         upper = min(lower + 1, int(cpu.size(0)) - 1)
#         weight = position - float(lower)
#         if upper == lower or weight == 0.0:
#             result = ordered[lower]
#         else:
#             result = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
#     return result.to(device=original_device, dtype=value.dtype)


# def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
#     """Robust middle-50% spread over dimension 0."""
#     lower = _deterministic_quantile_dim0(values, 0.25)
#     upper = _deterministic_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _ordered_identity_and_deformation(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     """Build robust identity and complete intra-class ordered structure.

#     The diagonal and off-diagonal entries intentionally have different but
#     complementary semantics:

#     * ``V[l,l,f]`` is the IQR of the absolute coordinate ``values[:,l,f]``;
#     * ``V[l,m,f]`` for ``l != m`` is the IQR of the ordered relation
#       ``values[:,l,f] - values[:,m,f]``.

#     This prevents common-mode class variation from disappearing while retaining
#     the ordered relation deformation that is important for HSI spectral shape.
#     """
#     value = _validate_ordered_samples(values)
#     identity = _deterministic_lower_median_dim0(value)

#     _, length, _ = value.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)
#     blocks: list[Tensor] = []
#     for first in range(0, length, chunk):
#         stop = min(first + chunk, length)
#         relation = value[:, first:stop, None, :] - value[:, None, :, :]
#         blocks.append(_deterministic_iqr_dim0(relation))

#     structure = torch.cat(blocks, dim=0)
#     structure = torch.maximum(structure, structure.transpose(0, 1))
#     diagonal = torch.arange(length, device=value.device)
#     structure = structure.clone()
#     structure[diagonal, diagonal, :] = _deterministic_iqr_dim0(value)

#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(structure).all()):
#         raise RuntimeError("ordered CDS extraction produced NaN or Inf")
#     if bool((structure < 0).any()):
#         raise RuntimeError("ordered CDS variation must be non-negative")
#     return identity.contiguous(), structure.contiguous()

# def sanitize_ordered_variation(variation: Tensor) -> Tensor:
#     """Restore valid V_Q/V_P semantics after coordinate transport.

#     Off-diagonal relation spreads are symmetric. Diagonal absolute spreads are
#     non-negative and are preserved rather than zeroed.
#     """
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("variation must be finite real floating point")
#     clean = value.clamp_min(0.0).clone()
#     length = clean.size(1)
#     diagonal = torch.arange(length, device=clean.device)
#     diag = clean[:, diagonal, diagonal, :].clone()
#     symmetric = torch.maximum(clean, clean.transpose(1, 2))
#     symmetric[:, diagonal, diagonal, :] = diag
#     return symmetric.contiguous()


# # Legacy API names retained because the surrounding repository may import them.
# # They no longer define the mathematics of the CDS.
# def project_ordered_variation_to_psd(variation: Tensor) -> Tensor:
#     return sanitize_ordered_variation(variation)


# def minimum_ordered_variation_eigenvalue(variation: Tensor) -> Tensor:
#     """Legacy reporting hook: return the minimum stored deformation value.

#     The historical function name is retained only for import compatibility.
#     V_Q/V_P are not interpreted through eigenvalues by this CDS implementation.
#     """
#     value = sanitize_ordered_variation(variation)
#     return value.amin()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     """Return the complete stored ordered intra-class deformation tensor."""
#     return sanitize_ordered_variation(variation)


# def _estimate_ordered_variation(values: Tensor, identity: Tensor | None = None) -> Tensor:
#     """Compatibility wrapper returning direct intra-class relation spread.

#     ``identity`` is accepted only for call-site compatibility and deliberately
#     does not participate in V estimation.  V_Q/V_P are independent of I_Q/I_P.
#     """
#     value = _validate_ordered_samples(values)
#     if identity is not None:
#         identity_value = torch.as_tensor(identity, device=value.device, dtype=value.dtype)
#         if identity_value.shape != value.shape[1:]:
#             raise ValueError("identity must have shape [L,F]")
#         if not bool(torch.isfinite(identity_value).all()):
#             raise ValueError("identity contains NaN or Inf")
#     _, deformation = _ordered_identity_and_deformation(value)
#     return deformation


# # -----------------------------------------------------------------------------
# # Inter-class CDS relations
# # -----------------------------------------------------------------------------


# def _diagonal_extent(variation: Tensor) -> Tensor:
#     value = sanitize_ordered_variation(variation)
#     length = value.size(1)
#     index = torch.arange(length, device=value.device)
#     return value[:, index, index, :]


# def _support_relative_identity_confusability(identity: Tensor, variation: Tensor) -> tuple[Tensor, Tensor]:
#     """Return scalar and slot-wise overlap from identity distance / class extent."""
#     center = torch.as_tensor(identity)
#     extent = _diagonal_extent(variation)
#     eps = torch.finfo(center.dtype).eps
#     diff = (center[:, None, :, :] - center[None, :, :, :]).abs()
#     denom = extent[:, None, :, :] + extent[None, :, :, :]
#     standardized = diff / denom.clamp_min(eps)
#     slot_distance = standardized.mean(dim=3)
#     by_slot = 1.0 / (1.0 + slot_distance)
#     scalar = by_slot.mean(dim=2)
#     diagonal = torch.arange(center.size(0), device=center.device)
#     by_slot = by_slot.clone(); scalar = scalar.clone()
#     by_slot[diagonal, diagonal, :] = 1.0
#     scalar[diagonal, diagonal] = 1.0
#     return scalar.clamp(0.0, 1.0), by_slot.clamp(0.0, 1.0)


# def _support_relative_relation_confusability(identity: Tensor, variation: Tensor) -> Tensor:
#     """Compare class relation identities relative to their legitimate relation spread."""
#     center = torch.as_tensor(identity)
#     v = sanitize_ordered_variation(variation)
#     c, length, features = center.shape
#     relation = center[:, :, None, :] - center[:, None, :, :]
#     diff = (relation[:, None] - relation[None, :]).abs()
#     denom = v[:, None] + v[None, :]
#     eps = torch.finfo(center.dtype).eps
#     standardized = diff / denom.clamp_min(eps)
#     upper = torch.triu(torch.ones(length, length, device=center.device, dtype=torch.bool), diagonal=1)
#     if not bool(upper.any()):
#         out = center.new_ones((c, c))
#     else:
#         distance = standardized[:, :, upper, :].mean(dim=(2, 3))
#         out = 1.0 / (1.0 + distance)
#     diagonal = torch.arange(c, device=center.device)
#     out = out.clone(); out[diagonal, diagonal] = 1.0
#     return out.clamp(0.0, 1.0)


# def _support_relative_component_confusability(identity: Tensor, variation: Tensor) -> Tensor:
#     """Physical identity overlap by component [C,C,F]."""
#     center = torch.as_tensor(identity)
#     extent = _diagonal_extent(variation)
#     eps = torch.finfo(center.dtype).eps
#     diff = (center[:, None] - center[None, :]).abs()
#     denom = extent[:, None] + extent[None, :]
#     distance = (diff / denom.clamp_min(eps)).mean(dim=2)  # [C,C,F]
#     out = 1.0 / (1.0 + distance)
#     diagonal = torch.arange(center.size(0), device=center.device)
#     out = out.clone(); out[diagonal, diagonal, :] = 1.0
#     return out.clamp(0.0, 1.0)


# def _variation_overlap(variation: Tensor) -> Tensor:
#     # Compatibility name: relation overlap is now support-relative and requires
#     # identity. Callers should use derive_class_relations instead.
#     value = sanitize_ordered_variation(variation)
#     flat = value.reshape(value.size(0), -1)
#     diff = (flat[:, None] - flat[None, :]).abs().mean(dim=2)
#     scale = flat.abs().mean(dim=1)
#     return (1.0 / (1.0 + diff / (scale[:, None] + scale[None, :] + torch.finfo(value.dtype).eps))).clamp(0, 1)


# def _relation_overlap(relation_operator: Tensor) -> Tensor:
#     return _variation_overlap(relation_operator)


# def _component_relation_overlap(relation_operator: Tensor) -> Tensor:
#     value = sanitize_ordered_variation(relation_operator)
#     c, l, _, f = value.shape
#     out = []
#     for index in range(f):
#         out.append(_variation_overlap(value[..., index:index+1]))
#     return torch.stack(out, dim=2)


# # -----------------------------------------------------------------------------
# # Public CDS views
# # -----------------------------------------------------------------------------


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor                    # [C]
#     sample_counts: Tensor                # [C]
#     identity: Tensor                     # [C,B,3] I_P
#     variation: Tensor                    # [C,B,B,3] V_P

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def spectral_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def intra_spectral_relation(self) -> Tensor:
#         return self.variation


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor                    # [C]
#     sample_counts: Tensor                # [C]
#     identity: Tensor                     # [C,K,D] I_Q
#     variation: Tensor                    # [C,K,K,D] V_Q
#     spectral_identity: Tensor            # [C,B,3] I_P
#     spectral_variation: Tensor           # [C,B,B,3] V_P

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def learned_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def learned_intra_relation(self) -> Tensor:
#         return self.variation

#     @property
#     def physical_identity(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def physical_intra_relation(self) -> Tensor:
#         return self.spectral_variation

#     # Historical property names retained only for repository compatibility.
#     @property
#     def learned_mode(self) -> Tensor:
#         return self.identity

#     @property
#     def physical_mode(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def feature_variation(self) -> Tensor:
#         """Legacy diagnostic view: strongest relation variation per token.

#         The diagonal stores absolute robust feature extent.
#         """
#         idx = torch.arange(self.variation.size(1), device=self.variation.device)
#         return self.variation[:, idx, idx, :]

#     @property
#     def spectral_feature_variation(self) -> Tensor:
#         """Legacy diagnostic view: absolute physical spectral extent."""
#         idx = torch.arange(self.spectral_variation.size(1), device=self.spectral_variation.device)
#         return self.spectral_variation[:, idx, idx, :]


# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_cosine_by_region: Tensor      # [C,C,K] compatibility field; normalized L1 identity relation
#     learned_variation_overlap: Tensor              # [C,C] complete V_Q relation
#     spectral_identity_cosine_by_component: Tensor  # [C,C,3] compatibility field; normalized L1 physical identity relation
#     spectral_variation_overlap: Tensor             # [C,C] complete V_P relation
#     spectral_confusability: Tensor                 # [C,C] physical identity+deformation relation

#     @property
#     def learned_identity_relation_by_region(self) -> Tensor:
#         return self.learned_identity_cosine_by_region

#     @property
#     def learned_intra_relation_overlap(self) -> Tensor:
#         return self.learned_variation_overlap

#     @property
#     def physical_identity_relation_by_component(self) -> Tensor:
#         return self.spectral_identity_cosine_by_component

#     @property
#     def physical_intra_relation_overlap(self) -> Tensor:
#         return self.spectral_variation_overlap

#     # Historical aliases.
#     @property
#     def learned_mode_relation_by_region(self) -> Tensor:
#         return self.learned_identity_cosine_by_region

#     @property
#     def physical_mode_relation_by_component(self) -> Tensor:
#         return self.spectral_identity_cosine_by_component


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor

#     @property
#     def class_ids(self) -> Tensor:
#         return self.classes.class_ids

#     @property
#     def spectral_confusability(self) -> Tensor:
#         return self.relations.spectral_confusability


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     identity: Tensor
#     variation: Tensor
#     spectral_identity: Tensor
#     spectral_variation: Tensor

#     @property
#     def learned_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def learned_intra_relation(self) -> Tensor:
#         return self.variation

#     @property
#     def physical_identity(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def physical_intra_relation(self) -> Tensor:
#         return self.spectral_variation

#     # Historical aliases.
#     @property
#     def learned_mode(self) -> Tensor:
#         return self.identity

#     @property
#     def physical_mode(self) -> Tensor:
#         return self.spectral_identity


# # -----------------------------------------------------------------------------
# # Class-state construction
# # -----------------------------------------------------------------------------


# def estimate_physical_structure_batch(
#     spectral_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> PhysicalSpectralStructureBatch:
#     p = torch.as_tensor(spectral_evidence)
#     y = torch.as_tensor(labels)
#     if p.ndim != 3 or p.size(0) == 0 or p.size(1) < 2 or p.size(2) != 3:
#         raise ValueError("spectral_evidence must be full-band [N,B,3], B>=2")
#     if y.shape != (p.size(0),) or y.dtype != torch.long or y.device != p.device:
#         raise ValueError("labels must be torch.long [N] on the spectral-evidence device")
#     if (
#         not torch.is_floating_point(p)
#         or p.is_complex()
#         or not bool(torch.isfinite(p).all())
#     ):
#         raise ValueError("spectral_evidence must be finite real floating point")

#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     deformations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no samples")
#         identity, deformation = _ordered_identity_and_deformation(p[mask])
#         counts.append(count)
#         identities.append(identity)
#         deformations.append(deformation)

#     return PhysicalSpectralStructureBatch(
#         class_ids=torch.tensor(requested, device=p.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=p.device, dtype=torch.long),
#         identity=torch.stack(identities, dim=0),
#         variation=torch.stack(deformations, dim=0),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical: PhysicalSpectralStructureBatch,
# ) -> ClassStructureBatch:
#     q = torch.as_tensor(ordered_features)
#     y = torch.as_tensor(labels)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if y.shape != (q.size(0),) or y.dtype != torch.long or y.device != q.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if not isinstance(physical, PhysicalSpectralStructureBatch) or physical.class_count <= 0:
#         raise TypeError("physical must be a non-empty PhysicalSpectralStructureBatch")
#     if physical.identity.device != q.device or physical.identity.dtype != q.dtype:
#         raise ValueError("Q and physical CDS must share device/dtype")

#     requested = tuple(int(v) for v in physical.class_ids.detach().cpu().tolist())
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("Q labels do not match cached physical CDS classes")

#     identities: list[Tensor] = []
#     deformations: list[Tensor] = []
#     counts: list[int] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no Q samples")
#         identity, deformation = _ordered_identity_and_deformation(q[mask])
#         counts.append(count)
#         identities.append(identity)
#         deformations.append(deformation)

#     count_tensor = torch.tensor(counts, device=q.device, dtype=torch.long)
#     if not torch.equal(count_tensor, physical.sample_counts):
#         raise RuntimeError("learned and physical CDS sample counts disagree")

#     return ClassStructureBatch(
#         class_ids=physical.class_ids,
#         sample_counts=physical.sample_counts,
#         identity=torch.stack(identities, dim=0),
#         variation=torch.stack(deformations, dim=0),
#         spectral_identity=physical.identity,
#         spectral_variation=physical.variation,
#     )


# def estimate_structure_batch(
#     ordered_features: Tensor,
#     spectral_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     physical = estimate_physical_structure_batch(
#         spectral_evidence,
#         labels,
#         class_ids=class_ids,
#     )
#     return combine_learned_and_physical_structure(
#         ordered_features,
#         labels,
#         physical,
#     )


# def merge_class_structure_batches(
#     batches: Sequence[ClassStructureBatch],
# ) -> ClassStructureBatch:
#     values = list(batches)
#     if not values:
#         raise ValueError("batches cannot be empty")

#     device = values[0].identity.device
#     dtype = values[0].identity.dtype
#     learned_shape = tuple(values[0].identity.shape[1:])
#     variation_shape = tuple(values[0].variation.shape[1:])
#     spectral_shape = tuple(values[0].spectral_identity.shape[1:])
#     spectral_variation_shape = tuple(values[0].spectral_variation.shape[1:])
#     seen: set[int] = set()

#     for batch in values:
#         if batch.class_count <= 0:
#             raise ValueError("structure batches cannot be empty")
#         if batch.identity.device != device or batch.identity.dtype != dtype:
#             raise ValueError("all structure batches must share device/dtype")
#         if (
#             tuple(batch.identity.shape[1:]) != learned_shape
#             or tuple(batch.variation.shape[1:]) != variation_shape
#             or tuple(batch.spectral_identity.shape[1:]) != spectral_shape
#             or tuple(batch.spectral_variation.shape[1:]) != spectral_variation_shape
#         ):
#             raise ValueError("all structure batches must share CDS dimensions")
#         ids = [int(v) for v in batch.class_ids.detach().cpu().tolist()]
#         duplicate = seen.intersection(ids)
#         if duplicate:
#             raise ValueError(f"duplicate structure classes {sorted(duplicate)}")
#         seen.update(ids)

#     return ClassStructureBatch(
#         class_ids=torch.cat([value.class_ids for value in values], dim=0),
#         sample_counts=torch.cat([value.sample_counts for value in values], dim=0),
#         identity=torch.cat([value.identity for value in values], dim=0),
#         variation=torch.cat([value.variation for value in values], dim=0),
#         spectral_identity=torch.cat([value.spectral_identity for value in values], dim=0),
#         spectral_variation=torch.cat([value.spectral_variation for value in values], dim=0),
#     )


# # -----------------------------------------------------------------------------
# # Derived inter-class CDS
# # -----------------------------------------------------------------------------


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     """Derive inter-class CDS from separation relative to intra-class support."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     learned_identity_scalar, learned_by_region = _support_relative_identity_confusability(
#         structure.identity, structure.variation
#     )
#     learned_relation = _support_relative_relation_confusability(
#         structure.identity, structure.variation
#     )
#     physical_component = _support_relative_component_confusability(
#         structure.spectral_identity, structure.spectral_variation
#     )
#     physical_identity_scalar, _physical_by_band = _support_relative_identity_confusability(
#         structure.spectral_identity, structure.spectral_variation
#     )
#     physical_relation = _support_relative_relation_confusability(
#         structure.spectral_identity, structure.spectral_variation
#     )
#     # Relation similarity can increase concern only when absolute class supports
#     # are already close; identical deformation patterns must not make far-apart
#     # classes look overlapping.
#     spectral_confusability = physical_identity_scalar * (0.5 + 0.5 * physical_relation)
#     return ClassRelationState(
#         class_ids=structure.class_ids,
#         learned_identity_cosine_by_region=learned_by_region,
#         learned_variation_overlap=learned_relation,
#         spectral_identity_cosine_by_component=physical_component,
#         spectral_variation_overlap=physical_relation,
#         spectral_confusability=spectral_confusability.clamp(0.0, 1.0),
#     )


# def learned_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     rel = derive_class_relations(structure)
#     identity = rel.learned_identity_cosine_by_region.mean(dim=2)
#     return (identity * (0.5 + 0.5 * rel.learned_variation_overlap)).clamp(0.0, 1.0)


# def learned_boundary_overlap_matrix(structure: ClassStructureBatch) -> Tensor:
#     return learned_cds_confusability_matrix(structure)


# def physical_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     return derive_class_relations(structure).spectral_confusability


# def physical_region_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     return physical_cds_confusability_matrix(structure)


# def whole_cds_pair_risk(cds: ClassDiscriminativeStructureState) -> tuple[Tensor, Tensor, Tensor]:
#     """Return learned decision risk and spectral positive prioritization.

#     Learned overlap is the actual representation-space danger. Physical
#     confusability can only increase its priority; it never gates a learned
#     danger off. All caller losses should add a positive baseline (1 + risk).
#     """
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")
#     learned = learned_cds_confusability_matrix(cds.classes)
#     physical = physical_cds_confusability_matrix(cds.classes)
#     risk = learned * (1.0 + physical) / 2.0
#     c = risk.size(0)
#     diagonal = torch.arange(c, device=risk.device)
#     risk = risk.clone(); learned = learned.clone(); physical = physical.clone()
#     risk[diagonal, diagonal] = 0.0
#     learned[diagonal, diagonal] = 0.0
#     physical[diagonal, diagonal] = 0.0
#     return risk.clamp(0.0, 1.0), learned.clamp(0.0, 1.0), physical.clamp(0.0, 1.0)


# def learned_sample_class_support(
#     ordered_features: Tensor,
#     cds: ClassDiscriminativeStructureState,
# ) -> Tensor:
#     """Return complete learned I_Q/V_Q sample-to-class support [N,C].

#     The diagonal of V_Q is interpreted as absolute coordinate IQR extent, while
#     off-diagonal entries are direct ordered-relation IQR extents.  A sample must
#     therefore agree with both the class's absolute spectral-region identity and
#     its legitimate cross-region deformation.  The geometric mean gives both
#     structures equal logical responsibility without introducing a tunable fusion
#     coefficient.
#     """
#     q = torch.as_tensor(ordered_features)
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")
#     classes = cds.classes
#     center = classes.identity.to(device=q.device, dtype=q.dtype)
#     variation = sanitize_ordered_variation(
#         classes.variation.to(device=q.device, dtype=q.dtype)
#     )
#     if q.ndim != 3 or q.size(0) == 0 or q.shape[1:] != center.shape[1:]:
#         raise ValueError("ordered_features must match CDS [K,D]")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")

#     eps = torch.finfo(q.dtype).eps
#     length = q.size(1)
#     index = torch.arange(length, device=q.device)

#     absolute_extent = variation[:, index, index, :]
#     observed_abs = (q[:, None] - center[None]).abs()
#     abs_support = absolute_extent[None] / (absolute_extent[None] + observed_abs + eps)
#     both_zero = (absolute_extent[None] <= eps) & (observed_abs <= eps)
#     abs_support = torch.where(both_zero, torch.ones_like(abs_support), abs_support)
#     abs_support = abs_support.mean(dim=(2, 3))

#     upper = torch.triu(
#         torch.ones(length, length, device=q.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     if bool(upper.any()):
#         sample_rel = q[:, :, None, :] - q[:, None, :, :]
#         center_rel = center[:, :, None, :] - center[:, None, :, :]
#         observed_rel = (sample_rel[:, None] - center_rel[None]).abs()
#         tolerance = variation[None]
#         rel_support = tolerance / (tolerance + observed_rel + eps)
#         both_zero_rel = (tolerance <= eps) & (observed_rel <= eps)
#         rel_support = torch.where(
#             both_zero_rel, torch.ones_like(rel_support), rel_support
#         )
#         rel_support = rel_support[:, :, upper, :].mean(dim=(2, 3))
#     else:
#         rel_support = abs_support

#     support = torch.sqrt((abs_support * rel_support).clamp_min(0.0)).clamp(0.0, 1.0)
#     if not bool(torch.isfinite(support).all()):
#         raise RuntimeError("learned sample-to-CDS support became NaN or Inf")
#     return support


# def spectral_sample_class_support(
#     spectral_evidence: Tensor,
#     cds: ClassDiscriminativeStructureState,
# ) -> Tensor:
#     """Full I_P/V_P sample-to-class compatibility [N,C] in [0,1]."""
#     p = torch.as_tensor(spectral_evidence)
#     classes = cds.classes
#     center = classes.spectral_identity.to(device=p.device, dtype=p.dtype)
#     variation = sanitize_ordered_variation(classes.spectral_variation.to(device=p.device, dtype=p.dtype))
#     if p.ndim != 3 or p.shape[1:] != center.shape[1:]:
#         raise ValueError("spectral_evidence must match CDS [B,3]")
#     eps = torch.finfo(p.dtype).eps
#     length = p.size(1)
#     index = torch.arange(length, device=p.device)
#     absolute_extent = variation[:, index, index, :]
#     observed_abs = (p[:, None] - center[None]).abs()
#     abs_support = absolute_extent[None] / (absolute_extent[None] + observed_abs + eps)
#     both_zero = (absolute_extent[None] <= eps) & (observed_abs <= eps)
#     abs_support = torch.where(both_zero, torch.ones_like(abs_support), abs_support).mean(dim=(2,3))

#     upper = torch.triu(torch.ones(length, length, device=p.device, dtype=torch.bool), diagonal=1)
#     if bool(upper.any()):
#         sample_rel = p[:, :, None, :] - p[:, None, :, :]
#         center_rel = center[:, :, None, :] - center[:, None, :, :]
#         observed_rel = (sample_rel[:, None] - center_rel[None]).abs()
#         tol = variation[None]
#         rel_support = tol / (tol + observed_rel + eps)
#         both_zero = (tol <= eps) & (observed_rel <= eps)
#         rel_support = torch.where(both_zero, torch.ones_like(rel_support), rel_support)
#         rel_support = rel_support[:, :, upper, :].mean(dim=(2,3))
#     else:
#         rel_support = abs_support
#     support = torch.sqrt((abs_support * rel_support).clamp_min(0.0)).clamp(0.0, 1.0)
#     if not bool(torch.isfinite(support).all()):
#         raise RuntimeError("physical sample-to-CDS support became NaN or Inf")
#     return support


# def whole_cds_sample_class_support(
#     ordered_features: Tensor,
#     spectral_evidence: Tensor,
#     cds: ClassDiscriminativeStructureState,
# ) -> Tensor:
#     """Return complete HSI-CDS sample-to-class compatibility [N,C].

#     Learned support consumes ``I_Q,V_Q`` and physical support consumes
#     ``I_P,V_P``.  Their geometric mean makes both structures necessary without
#     introducing a tunable fusion coefficient. ``N`` remains class-state
#     accounting/reliability metadata and is deliberately not used as a class
#     prior because class frequency must not bias the decision geometry.
#     """
#     learned = learned_sample_class_support(ordered_features, cds)
#     physical = spectral_sample_class_support(spectral_evidence, cds)
#     if learned.shape != physical.shape:
#         raise RuntimeError("learned and physical CDS support shapes disagree")
#     physical = physical.to(device=learned.device, dtype=learned.dtype)
#     support = torch.sqrt((learned * physical).clamp_min(0.0)).clamp(0.0, 1.0)
#     if not bool(torch.isfinite(support).all()):
#         raise RuntimeError("whole HSI-CDS sample support became NaN or Inf")
#     return support


# def spectral_region_discriminability(
#     cds: ClassDiscriminativeStructureState,
#     *,
#     region_count: int,
# ) -> Tensor:
#     """Map native-band pair discriminability to contiguous learned regions [C,C,K]."""
#     classes = cds.classes
#     identity = classes.spectral_identity
#     variation = sanitize_ordered_variation(classes.spectral_variation)
#     c, bands, _ = identity.shape
#     if region_count < 1 or region_count > bands:
#         raise ValueError("region_count must be in [1,B]")
#     index = torch.arange(bands, device=identity.device)
#     extent = variation[:, index, index, :]
#     eps = torch.finfo(identity.dtype).eps
#     diff = (identity[:, None] - identity[None]).abs().mean(dim=3)
#     denom = (extent[:, None] + extent[None]).mean(dim=3)
#     band_disc = diff / denom.clamp_min(eps)
#     regions = []
#     for k in range(region_count):
#         lo = k * bands // region_count
#         hi = (k + 1) * bands // region_count
#         regions.append(band_disc[:, :, lo:hi].mean(dim=2))
#     out = torch.stack(regions, dim=2)
#     mean = out.mean(dim=2, keepdim=True).clamp_min(eps)
#     return (out / mean).clamp_min(0.0)

# def spectral_correspondence_metric(structure: ClassStructureBatch) -> Tensor:
#     """Robust native-band class-balanced V_P reference [3,B,B]."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     physical = ordered_variation_operator(structure.spectral_variation)
#     # [C,B,B,3] -> median over class dimension -> [B,B,3] -> [3,B,B]
#     relation = _deterministic_lower_median_dim0(physical)
#     result = relation.permute(2, 0, 1).contiguous()
#     if not bool(torch.isfinite(result).all()):
#         raise RuntimeError("spectral correspondence metric contains NaN or Inf")
#     return result


# # -----------------------------------------------------------------------------
# # Persistent bank
# # -----------------------------------------------------------------------------


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Persistent HSI-CDS bank storing {I_Q,V_Q,I_P,V_P,N}."""

#     def __init__(
#         self,
#         *,
#         spectral_token_count: int,
#         spectral_token_dim: int,
#         spectral_bands: int,
#         spectral_band_positions: Sequence[float] | Tensor | None = None,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype = torch.float32,
#     ) -> None:
#         super().__init__()
#         self.spectral_token_count = _integer(
#             spectral_token_count,
#             "spectral_token_count",
#             minimum=2,
#         )
#         self.spectral_token_dim = _integer(
#             spectral_token_dim,
#             "spectral_token_dim",
#             minimum=1,
#         )
#         self.spectral_bands = _integer(
#             spectral_bands,
#             "spectral_bands",
#             minimum=2,
#         )
#         if dtype not in (torch.float32, torch.float64):
#             raise TypeError("structure dtype must be float32 or float64")

#         positions = _positions(
#             spectral_band_positions,
#             count=self.spectral_bands,
#             device=device,
#             dtype=dtype,
#         )
#         self.register_buffer(
#             "_floating_reference",
#             torch.empty(0, device=device, dtype=dtype),
#             persistent=False,
#         )
#         self.register_buffer("spectral_band_positions", positions, persistent=True)
#         self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long))
#         self.register_buffer("sample_counts", torch.empty(0, device=device, dtype=torch.long))
#         self.register_buffer(
#             "identity_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "spectral_identity_bank",
#             torch.empty(
#                 0,
#                 self.spectral_bands,
#                 3,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "spectral_variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_bands,
#                 self.spectral_bands,
#                 3,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.validate_state()

#     @property
#     def device(self) -> torch.device:
#         return self._floating_reference.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self._floating_reference.dtype

#     def __len__(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(v) for v in self.class_ids.detach().cpu().tolist())

#     @staticmethod
#     def _validate_deformation_semantics(value: Tensor, *, name: str) -> None:
#         if value.ndim != 4 or value.size(1) != value.size(2):
#             raise RuntimeError(f"{name} must be [C,L,L,F]")
#         if value.numel() == 0:
#             return

#         tolerance = 32.0 * torch.finfo(value.dtype).eps
#         if bool((value < -tolerance).any()):
#             raise RuntimeError(f"{name} contains negative variation")
#         if not bool(torch.allclose(value, value.transpose(1, 2), rtol=0.0, atol=tolerance)):
#             raise RuntimeError(f"{name} is not symmetric across ordered spectral pairs")

#         diagonal = value.diagonal(dim1=1, dim2=2)
#         if bool((diagonal < -tolerance).any()):
#             raise RuntimeError(f"{name} diagonal absolute extent must be non-negative")

#     def validate_state(self) -> bool:
#         c = len(self)
#         expected = {
#             "class_ids": (c,),
#             "sample_counts": (c,),
#             "identity_bank": (c, self.spectral_token_count, self.spectral_token_dim),
#             "variation_bank": (
#                 c,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#             ),
#             "spectral_identity_bank": (c, self.spectral_bands, 3),
#             "spectral_variation_bank": (
#                 c,
#                 self.spectral_bands,
#                 self.spectral_bands,
#                 3,
#             ),
#         }
#         for name, shape in expected.items():
#             if getattr(self, name).shape != shape:
#                 raise RuntimeError(
#                     f"{name} state is invalid: "
#                     f"{tuple(getattr(self, name).shape)} != {shape}"
#                 )

#         if self.class_ids.dtype != torch.long or self.sample_counts.dtype != torch.long:
#             raise RuntimeError("class IDs/counts must be torch.long")
#         if c:
#             if bool((self.class_ids < 0).any()) or self.class_ids.unique().numel() != c:
#                 raise RuntimeError("class IDs must be unique non-negative values")
#             if bool((self.sample_counts <= 0).any()):
#                 raise RuntimeError("class sample counts must be positive")

#         for name in (
#             "identity_bank",
#             "variation_bank",
#             "spectral_identity_bank",
#             "spectral_variation_bank",
#         ):
#             value = getattr(self, name)
#             if value.device != self.device or value.dtype != self.dtype:
#                 raise RuntimeError(f"{name} has wrong device/dtype")
#             if not bool(torch.isfinite(value).all()):
#                 raise RuntimeError(f"{name} contains NaN or Inf")

#         self._validate_deformation_semantics(self.variation_bank, name="variation_bank")
#         self._validate_deformation_semantics(
#             self.spectral_variation_bank,
#             name="spectral_variation_bank",
#         )
#         return True

#     def _validate_batch(self, batch: ClassStructureBatch) -> None:
#         if not isinstance(batch, ClassStructureBatch) or batch.class_count <= 0:
#             raise TypeError("batch must be a non-empty ClassStructureBatch")
#         c = batch.class_count
#         shapes = (
#             (batch.class_ids, (c,)),
#             (batch.sample_counts, (c,)),
#             (batch.identity, (c, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 batch.variation,
#                 (
#                     c,
#                     self.spectral_token_count,
#                     self.spectral_token_count,
#                     self.spectral_token_dim,
#                 ),
#             ),
#             (batch.spectral_identity, (c, self.spectral_bands, 3)),
#             (
#                 batch.spectral_variation,
#                 (c, self.spectral_bands, self.spectral_bands, 3),
#             ),
#         )
#         if any(value.shape != shape for value, shape in shapes):
#             raise ValueError("class structure batch shape is invalid")
#         if batch.class_ids.dtype != torch.long or batch.sample_counts.dtype != torch.long:
#             raise ValueError("class IDs/counts must be torch.long")

#         tensors = (
#             batch.identity,
#             batch.variation,
#             batch.spectral_identity,
#             batch.spectral_variation,
#         )
#         if any(value.device != self.device or value.dtype != self.dtype for value in tensors):
#             raise ValueError("batch and bank must share device/dtype")
#         if not all(bool(torch.isfinite(value).all()) for value in tensors):
#             raise ValueError("HSI-CDS contains NaN or Inf")
#         if bool((batch.sample_counts <= 0).any()):
#             raise ValueError("class sample counts must be positive")

#         self._validate_deformation_semantics(batch.variation, name="batch.variation")
#         self._validate_deformation_semantics(
#             batch.spectral_variation,
#             name="batch.spectral_variation",
#         )

#     def estimate_physical(
#         self,
#         spectral_evidence: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         p = torch.as_tensor(spectral_evidence)
#         if p.device != self.device or p.dtype != self.dtype:
#             raise ValueError("spectral evidence and bank must share device/dtype")
#         if p.ndim != 3 or p.shape[1:] != (self.spectral_bands, 3):
#             raise ValueError("spectral evidence does not match dataset-native band count")
#         return estimate_physical_structure_batch(
#             p,
#             labels,
#             class_ids=class_ids,
#         )

#     def estimate_classes(
#         self,
#         ordered_features: Tensor,
#         spectral_evidence: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         q = torch.as_tensor(ordered_features)
#         if q.device != self.device or q.dtype != self.dtype:
#             raise ValueError("Q and bank must share device/dtype")

#         if physical_structure is None:
#             if spectral_evidence is None:
#                 raise ValueError("spectral_evidence is required when physical_structure is absent")
#             physical_structure = self.estimate_physical(
#                 spectral_evidence,
#                 labels,
#                 class_ids=class_ids,
#             )
#         else:
#             requested = _class_id_tuple(class_ids, name="class_ids")
#             actual = tuple(
#                 int(v)
#                 for v in physical_structure.class_ids.detach().cpu().tolist()
#             )
#             if actual != requested:
#                 raise ValueError("physical_structure class order does not match class_ids")

#         return combine_learned_and_physical_structure(
#             q,
#             labels,
#             physical_structure,
#         )

#     @torch.no_grad()
#     def append_estimate(self, estimate: ClassStructureBatch) -> None:
#         self._validate_batch(estimate)
#         overlap = set(
#             int(v) for v in estimate.class_ids.detach().cpu().tolist()
#         ).intersection(self.seen_class_ids)
#         if overlap:
#             raise ValueError(f"class structure already exists for {sorted(overlap)}")
#         combined = (
#             estimate
#             if len(self) == 0
#             else merge_class_structure_batches((self.class_structures(), estimate))
#         )
#         self.replace_all(combined)

#     @torch.no_grad()
#     def replace_all(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         self.class_ids = structure.class_ids.detach().clone()
#         self.sample_counts = structure.sample_counts.detach().clone()
#         self.identity_bank = structure.identity.detach().clone()
#         self.variation_bank = sanitize_ordered_variation(structure.variation.detach().clone())
#         self.spectral_identity_bank = structure.spectral_identity.detach().clone()
#         self.spectral_variation_bank = sanitize_ordered_variation(
#             structure.spectral_variation.detach().clone()
#         )
#         self.validate_state()

#     def _row_for_class(self, class_id: int) -> int:
#         cid = _integer(class_id, "class_id", minimum=0)
#         rows = torch.nonzero(self.class_ids.eq(cid), as_tuple=False).flatten()
#         if rows.numel() != 1:
#             raise KeyError(f"class structure is not stored uniquely for class {cid}")
#         return int(rows.item())

#     def class_structure(self, class_id: int) -> ClassStructureView:
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(self.class_ids[row].item()),
#             sample_count=int(self.sample_counts[row].item()),
#             identity=self.identity_bank[row].detach().clone(),
#             variation=self.variation_bank[row].detach().clone(),
#             spectral_identity=self.spectral_identity_bank[row].detach().clone(),
#             spectral_variation=self.spectral_variation_bank[row].detach().clone(),
#         )

#     def class_structures(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassStructureBatch:
#         self.validate_state()
#         if len(self) == 0:
#             raise RuntimeError("class structure bank is empty")
#         requested = (
#             self.seen_class_ids
#             if class_ids is None
#             else _class_id_tuple(class_ids, name="class_ids")
#         )
#         rows = torch.tensor(
#             [self._row_for_class(cid) for cid in requested],
#             device=self.device,
#             dtype=torch.long,
#         )
#         return ClassStructureBatch(
#             class_ids=self.class_ids.index_select(0, rows).detach().clone(),
#             sample_counts=self.sample_counts.index_select(0, rows).detach().clone(),
#             identity=self.identity_bank.index_select(0, rows).detach().clone(),
#             variation=self.variation_bank.index_select(0, rows).detach().clone(),
#             spectral_identity=self.spectral_identity_bank.index_select(0, rows).detach().clone(),
#             spectral_variation=self.spectral_variation_bank.index_select(0, rows).detach().clone(),
#         )

#     def build_cds(self, batch: ClassStructureBatch) -> ClassDiscriminativeStructureState:
#         self._validate_batch(batch)
#         return ClassDiscriminativeStructureState(
#             classes=batch,
#             relations=derive_class_relations(batch),
#             spectral_band_positions=self.spectral_band_positions.detach().clone(),
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
#         dynamic = {
#             "class_ids": torch.long,
#             "sample_counts": torch.long,
#             "identity_bank": self.dtype,
#             "variation_bank": self.dtype,
#             "spectral_identity_bank": self.dtype,
#             "spectral_variation_bank": self.dtype,
#         }
#         for name, dtype in dynamic.items():
#             incoming = state_dict.get(prefix + name)
#             if incoming is not None and getattr(self, name).shape != incoming.shape:
#                 setattr(
#                     self,
#                     name,
#                     torch.empty(incoming.shape, device=self.device, dtype=dtype),
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
#             # Old checkpoints may contain the previous relation semantics.  Shape
#             # compatibility is preserved, but a new run should rebuild CDS from
#             # real TRAIN samples before relying on the state.
#             self.validate_state()


# __all__ = [
#     "PhysicalSpectralStructureBatch",
#     "ClassStructureBatch",
#     "ClassRelationState",
#     "ClassDiscriminativeStructureState",
#     "ClassStructureView",
#     "HSIClassDiscriminativeStructureBank",
#     "merge_class_structure_batches",
#     "estimate_physical_structure_batch",
#     "combine_learned_and_physical_structure",
#     "estimate_structure_batch",
#     "derive_class_relations",
#     "learned_cds_confusability_matrix",
#     "learned_boundary_overlap_matrix",
#     "sanitize_ordered_variation",
#     "project_ordered_variation_to_psd",
#     "minimum_ordered_variation_eigenvalue",
#     "ordered_variation_operator",
#     "spectral_correspondence_metric",
#     "physical_cds_confusability_matrix",
#     "physical_region_confusability_matrix",
#     "whole_cds_pair_risk",
#     "learned_sample_class_support",
#     "spectral_sample_class_support",
#     "whole_cds_sample_class_support",
#     "spectral_region_discriminability",
# ]










# from __future__ import annotations

# """HSI Class-Discriminative Structure (CDS) bank.

# Persistent class state
# ----------------------
# For every class c the bank stores exactly

#     CDS_c = {I_Q, V_Q, I_P, V_P, N}

# I_Q [K,D]
#     Robust ordered learned spectral identity.  It is the deterministic
#     coordinate-wise class median in the single Q space.

# V_Q [K,K,D]
#     Ordered learned intra-class spectral-relation variation.  For every
#     ordered token pair (k,l), it stores the robust interquartile spread of the
#     direct sample relation Q[k,d]-Q[l,d] across real class samples.  V_Q is
#     estimated independently of I_Q: no class-center subtraction or residual
#     construction is used.  Identical class samples therefore produce V_Q == 0.

# I_P [B,3]
#     Robust native-band physical spectral identity for
#     [level, slope, curvature].

# V_P [B,B,3]
#     Native-band intra-class spectral-relation variation, defined by the same
#     direct interquartile-spread principle as V_Q but on the physical full-band
#     level/slope/curvature evidence.

# N
#     Number of real TRAIN samples used to build the class CDS.

# Inter-class CDS is derived on demand from the stored class states.  No pairwise
# memory bank, exemplar, token buffer, Gaussian model, second classifier, or
# phase-specific head is stored.

# Compatibility note
# ------------------
# The historical public tensor field names ``identity`` and ``variation`` are
# retained so the surrounding repository does not need a destructive API change.
# Their semantics are the HSI-CDS definitions above.
# """

# from dataclasses import dataclass
# from numbers import Integral
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


# # -----------------------------------------------------------------------------
# # Generic helpers
# # -----------------------------------------------------------------------------


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


# def _positions(
#     values: Sequence[float] | Tensor | None,
#     *,
#     count: int,
#     device: torch.device | str | None,
#     dtype: torch.dtype,
# ) -> Tensor:
#     result = (
#         torch.linspace(-1.0, 1.0, count, device=device, dtype=dtype)
#         if values is None
#         else torch.as_tensor(values, device=device, dtype=dtype).flatten()
#     )
#     if result.shape != (count,):
#         raise ValueError(f"spectral_band_positions must contain {count} values")
#     if not bool(torch.isfinite(result).all()):
#         raise ValueError("spectral_band_positions contain NaN or Inf")
#     if count > 1 and bool((result[1:] <= result[:-1]).any()):
#         raise ValueError("spectral_band_positions must be strictly increasing")
#     return result


# def _deterministic_lower_median_dim0(values: Tensor) -> Tensor:
#     """Deterministic lower median over dimension 0.

#     CDS construction is persistent state estimation, not a gradient operation.
#     The order statistic is computed on CPU so deterministic mode behaves the
#     same on CPU and CUDA.  For even N, index (N-1)//2 is used deliberately.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("median input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("median input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("median input contains NaN or Inf")

#     original_device = value.device
#     cpu = value.detach().to(device="cpu")
#     ordered = torch.sort(cpu, dim=0, stable=True).values
#     result = ordered[(int(cpu.size(0)) - 1) // 2]
#     return result.to(device=original_device, dtype=value.dtype)


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


# # -----------------------------------------------------------------------------
# # Intra-class HSI-CDS extraction
# # -----------------------------------------------------------------------------


# def _deterministic_quantile_dim0(values: Tensor, q: float) -> Tensor:
#     """Deterministic linear quantile over dimension 0.

#     This is an order-statistic operation used only for persistent CDS
#     construction.  It is not a class mean, covariance, or learned parameter.
#     CPU sorting avoids CUDA nondeterminism under deterministic training.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("quantile input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("quantile input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("quantile input contains NaN or Inf")
#     q_value = float(q)
#     if not 0.0 <= q_value <= 1.0:
#         raise ValueError("q must be in [0,1]")

#     original_device = value.device
#     cpu = value.detach().to(device="cpu")
#     ordered = torch.sort(cpu, dim=0, stable=True).values
#     if int(cpu.size(0)) == 1:
#         result = ordered[0]
#     else:
#         position = q_value * float(int(cpu.size(0)) - 1)
#         lower = int(position)
#         upper = min(lower + 1, int(cpu.size(0)) - 1)
#         weight = position - float(lower)
#         if upper == lower or weight == 0.0:
#             result = ordered[lower]
#         else:
#             result = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
#     return result.to(device=original_device, dtype=value.dtype)


# def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
#     """Robust middle-50% spread over dimension 0."""
#     lower = _deterministic_quantile_dim0(values, 0.25)
#     upper = _deterministic_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _ordered_identity_and_deformation(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     """Build HSI identity and intra-class ordered relation variation.

#     ``I`` and ``V`` are estimated from different information:

#     * identity [L,F] is the deterministic coordinate-wise spectral consensus
#       (median) of the real class samples;
#     * variation [L,L,F] is the interquartile spread across samples of the direct
#       ordered relation ``values[a,f] - values[b,f]``.

#     Crucially, V is **not** formed by subtracting I, subtracting a class center,
#     fitting a variance/covariance, or constructing residual features.  It asks
#     only how stable each ordered spectral relation is inside the class.

#     Identical class samples have zero relation spread and therefore V == 0.
#     """
#     value = _validate_ordered_samples(values)
#     identity = _deterministic_lower_median_dim0(value)

#     _, length, _ = value.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)
#     blocks: list[Tensor] = []

#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         # Direct per-sample ordered spectral relation [N,R,L,F].
#         relation = value[:, start:stop, None, :] - value[:, None, :, :]
#         # V stores relation spread across samples directly.  No identity/center
#         # enters this calculation.
#         block = _deterministic_iqr_dim0(relation)
#         blocks.append(block)

#     deformation = torch.cat(blocks, dim=0)
#     if deformation.shape[0] != length or deformation.shape[1] != length:
#         raise RuntimeError("ordered CDS variation has an invalid shape")

#     # IQR(R[a,b]) == IQR(-R[b,a]) mathematically.  Maximum only removes tiny
#     # numerical asymmetry while preserving the measured spread magnitude.
#     deformation = torch.maximum(deformation, deformation.transpose(0, 1))
#     diagonal = torch.arange(length, device=value.device)
#     deformation = deformation.clone()
#     deformation[diagonal, diagonal, :] = 0.0

#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(deformation).all()):
#         raise RuntimeError("ordered CDS extraction produced NaN or Inf")
#     if bool((deformation < 0).any()):
#         raise RuntimeError("ordered CDS variation must be non-negative")

#     return identity.contiguous(), deformation.contiguous()


# def sanitize_ordered_variation(variation: Tensor) -> Tensor:
#     """Restore valid V_Q/V_P semantics after deterministic coordinate transport.

#     Stored V is a non-negative symmetric relation-variation tensor with a zero
#     diagonal.  No spectral decomposition, distribution fitting, or whitening is
#     performed here.
#     """
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     if (
#         not torch.is_floating_point(value)
#         or value.is_complex()
#         or not bool(torch.isfinite(value).all())
#     ):
#         raise ValueError("variation must be finite real floating point")

#     clean = value.clamp_min(0.0)
#     clean = torch.maximum(clean, clean.transpose(1, 2))
#     length = clean.size(1)
#     diagonal = torch.arange(length, device=clean.device)
#     clean = clean.clone()
#     clean[:, diagonal, diagonal, :] = 0.0
#     return clean.contiguous()


# # Legacy API names retained because the surrounding repository may import them.
# # They no longer define the mathematics of the CDS.
# def project_ordered_variation_to_psd(variation: Tensor) -> Tensor:
#     return sanitize_ordered_variation(variation)


# def minimum_ordered_variation_eigenvalue(variation: Tensor) -> Tensor:
#     """Legacy reporting hook: return the minimum stored deformation value.

#     The historical function name is retained only for import compatibility.
#     V_Q/V_P are not interpreted through eigenvalues by this CDS implementation.
#     """
#     value = sanitize_ordered_variation(variation)
#     return value.amin()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     """Return the complete stored ordered intra-class deformation tensor."""
#     return sanitize_ordered_variation(variation)


# def _estimate_ordered_variation(values: Tensor, identity: Tensor | None = None) -> Tensor:
#     """Compatibility wrapper returning direct intra-class relation spread.

#     ``identity`` is accepted only for call-site compatibility and deliberately
#     does not participate in V estimation.  V_Q/V_P are independent of I_Q/I_P.
#     """
#     value = _validate_ordered_samples(values)
#     if identity is not None:
#         identity_value = torch.as_tensor(identity, device=value.device, dtype=value.dtype)
#         if identity_value.shape != value.shape[1:]:
#             raise ValueError("identity must have shape [L,F]")
#         if not bool(torch.isfinite(identity_value).all()):
#             raise ValueError("identity contains NaN or Inf")
#     _, deformation = _ordered_identity_and_deformation(value)
#     return deformation


# # -----------------------------------------------------------------------------
# # Inter-class CDS relations
# # -----------------------------------------------------------------------------


# def _pairwise_l1_similarity_global(value: Tensor) -> Tensor:
#     """Scale-aware normalized L1 similarity for complete class tensors [C,...]."""
#     x = torch.as_tensor(value)
#     if x.ndim < 2 or x.size(0) <= 0:
#         raise ValueError("value must be [C,...]")
#     if not torch.is_floating_point(x) or x.is_complex() or not bool(torch.isfinite(x).all()):
#         raise ValueError("value must be finite real floating point")

#     flat = x.reshape(x.size(0), -1)
#     diff = (flat[:, None, :] - flat[None, :, :]).abs().sum(dim=2)
#     strength = flat.abs().sum(dim=1)
#     denom = strength[:, None] + strength[None, :]
#     eps = torch.finfo(x.dtype).eps
#     similarity = 1.0 - diff / denom.clamp_min(eps)
#     similarity = similarity.clamp(0.0, 1.0)
#     both_zero = denom.le(eps)
#     similarity = torch.where(both_zero, torch.ones_like(similarity), similarity)
#     diagonal = torch.arange(x.size(0), device=x.device)
#     similarity = similarity.clone()
#     similarity[diagonal, diagonal] = 1.0
#     return similarity


# def _pairwise_l1_similarity_by_slot(value: Tensor) -> Tensor:
#     """Normalized L1 similarity for [C,S,F], returned as [C,C,S]."""
#     x = torch.as_tensor(value)
#     if x.ndim != 3 or x.size(0) <= 0 or x.size(1) <= 0 or x.size(2) <= 0:
#         raise ValueError("value must be non-empty [C,S,F]")
#     if not torch.is_floating_point(x) or x.is_complex() or not bool(torch.isfinite(x).all()):
#         raise ValueError("value must be finite real floating point")

#     diff = (x[:, None, :, :] - x[None, :, :, :]).abs().sum(dim=3)
#     strength = x.abs().sum(dim=2)
#     denom = strength[:, None, :] + strength[None, :, :]
#     eps = torch.finfo(x.dtype).eps
#     similarity = 1.0 - diff / denom.clamp_min(eps)
#     similarity = similarity.clamp(0.0, 1.0)
#     similarity = torch.where(denom.le(eps), torch.ones_like(similarity), similarity)
#     diagonal = torch.arange(x.size(0), device=x.device)
#     similarity = similarity.clone()
#     similarity[diagonal, diagonal, :] = 1.0
#     return similarity


# def _relation_overlap(relation_operator: Tensor) -> Tensor:
#     """Complete V_Q/V_P similarity [C,C]."""
#     value = ordered_variation_operator(relation_operator)
#     return _pairwise_l1_similarity_global(value)


# def _variation_overlap(variation: Tensor) -> Tensor:
#     return _relation_overlap(variation)


# def _component_relation_overlap(relation_operator: Tensor) -> Tensor:
#     """Per-component V similarity [C,C,F] for [C,L,L,F]."""
#     value = ordered_variation_operator(relation_operator)
#     c, length, _, features = value.shape
#     by_component = value.permute(0, 3, 1, 2).reshape(c, features, length * length)
#     diff = (
#         by_component[:, None, :, :]
#         - by_component[None, :, :, :]
#     ).abs().sum(dim=3)
#     strength = by_component.abs().sum(dim=2)
#     denom = strength[:, None, :] + strength[None, :, :]
#     eps = torch.finfo(value.dtype).eps
#     similarity = 1.0 - diff / denom.clamp_min(eps)
#     similarity = similarity.clamp(0.0, 1.0)
#     similarity = torch.where(denom.le(eps), torch.ones_like(similarity), similarity)
#     diagonal = torch.arange(c, device=value.device)
#     similarity = similarity.clone()
#     similarity[diagonal, diagonal, :] = 1.0
#     return similarity


# # -----------------------------------------------------------------------------
# # Public CDS views
# # -----------------------------------------------------------------------------


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor                    # [C]
#     sample_counts: Tensor                # [C]
#     identity: Tensor                     # [C,B,3] I_P
#     variation: Tensor                    # [C,B,B,3] V_P

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def spectral_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def intra_spectral_relation(self) -> Tensor:
#         return self.variation


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor                    # [C]
#     sample_counts: Tensor                # [C]
#     identity: Tensor                     # [C,K,D] I_Q
#     variation: Tensor                    # [C,K,K,D] V_Q
#     spectral_identity: Tensor            # [C,B,3] I_P
#     spectral_variation: Tensor           # [C,B,B,3] V_P

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def learned_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def learned_intra_relation(self) -> Tensor:
#         return self.variation

#     @property
#     def physical_identity(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def physical_intra_relation(self) -> Tensor:
#         return self.spectral_variation

#     # Historical property names retained only for repository compatibility.
#     @property
#     def learned_mode(self) -> Tensor:
#         return self.identity

#     @property
#     def physical_mode(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def feature_variation(self) -> Tensor:
#         """Legacy diagnostic view: strongest relation variation per token.

#         The diagonal of V_Q is structurally zero, so returning the diagonal
#         (the old covariance-era behavior) would be meaningless.  This derived
#         view keeps shape [C,K,D] without changing persistent CDS semantics.
#         """
#         return self.variation.amax(dim=2)

#     @property
#     def spectral_feature_variation(self) -> Tensor:
#         """Legacy diagnostic view: strongest physical relation variation."""
#         return self.spectral_variation.amax(dim=2)


# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_cosine_by_region: Tensor      # [C,C,K] compatibility field; normalized L1 identity relation
#     learned_variation_overlap: Tensor              # [C,C] complete V_Q relation
#     spectral_identity_cosine_by_component: Tensor  # [C,C,3] compatibility field; normalized L1 physical identity relation
#     spectral_variation_overlap: Tensor             # [C,C] complete V_P relation
#     spectral_confusability: Tensor                 # [C,C] physical identity+deformation relation

#     @property
#     def learned_identity_relation_by_region(self) -> Tensor:
#         return self.learned_identity_cosine_by_region

#     @property
#     def learned_intra_relation_overlap(self) -> Tensor:
#         return self.learned_variation_overlap

#     @property
#     def physical_identity_relation_by_component(self) -> Tensor:
#         return self.spectral_identity_cosine_by_component

#     @property
#     def physical_intra_relation_overlap(self) -> Tensor:
#         return self.spectral_variation_overlap

#     # Historical aliases.
#     @property
#     def learned_mode_relation_by_region(self) -> Tensor:
#         return self.learned_identity_cosine_by_region

#     @property
#     def physical_mode_relation_by_component(self) -> Tensor:
#         return self.spectral_identity_cosine_by_component


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor

#     @property
#     def class_ids(self) -> Tensor:
#         return self.classes.class_ids

#     @property
#     def spectral_confusability(self) -> Tensor:
#         return self.relations.spectral_confusability


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     identity: Tensor
#     variation: Tensor
#     spectral_identity: Tensor
#     spectral_variation: Tensor

#     @property
#     def learned_identity(self) -> Tensor:
#         return self.identity

#     @property
#     def learned_intra_relation(self) -> Tensor:
#         return self.variation

#     @property
#     def physical_identity(self) -> Tensor:
#         return self.spectral_identity

#     @property
#     def physical_intra_relation(self) -> Tensor:
#         return self.spectral_variation

#     # Historical aliases.
#     @property
#     def learned_mode(self) -> Tensor:
#         return self.identity

#     @property
#     def physical_mode(self) -> Tensor:
#         return self.spectral_identity


# # -----------------------------------------------------------------------------
# # Class-state construction
# # -----------------------------------------------------------------------------


# def estimate_physical_structure_batch(
#     spectral_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> PhysicalSpectralStructureBatch:
#     p = torch.as_tensor(spectral_evidence)
#     y = torch.as_tensor(labels)
#     if p.ndim != 3 or p.size(0) == 0 or p.size(1) < 2 or p.size(2) != 3:
#         raise ValueError("spectral_evidence must be full-band [N,B,3], B>=2")
#     if y.shape != (p.size(0),) or y.dtype != torch.long or y.device != p.device:
#         raise ValueError("labels must be torch.long [N] on the spectral-evidence device")
#     if (
#         not torch.is_floating_point(p)
#         or p.is_complex()
#         or not bool(torch.isfinite(p).all())
#     ):
#         raise ValueError("spectral_evidence must be finite real floating point")

#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     deformations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no samples")
#         identity, deformation = _ordered_identity_and_deformation(p[mask])
#         counts.append(count)
#         identities.append(identity)
#         deformations.append(deformation)

#     return PhysicalSpectralStructureBatch(
#         class_ids=torch.tensor(requested, device=p.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=p.device, dtype=torch.long),
#         identity=torch.stack(identities, dim=0),
#         variation=torch.stack(deformations, dim=0),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical: PhysicalSpectralStructureBatch,
# ) -> ClassStructureBatch:
#     q = torch.as_tensor(ordered_features)
#     y = torch.as_tensor(labels)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if y.shape != (q.size(0),) or y.dtype != torch.long or y.device != q.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if not isinstance(physical, PhysicalSpectralStructureBatch) or physical.class_count <= 0:
#         raise TypeError("physical must be a non-empty PhysicalSpectralStructureBatch")
#     if physical.identity.device != q.device or physical.identity.dtype != q.dtype:
#         raise ValueError("Q and physical CDS must share device/dtype")

#     requested = tuple(int(v) for v in physical.class_ids.detach().cpu().tolist())
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("Q labels do not match cached physical CDS classes")

#     identities: list[Tensor] = []
#     deformations: list[Tensor] = []
#     counts: list[int] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no Q samples")
#         identity, deformation = _ordered_identity_and_deformation(q[mask])
#         counts.append(count)
#         identities.append(identity)
#         deformations.append(deformation)

#     count_tensor = torch.tensor(counts, device=q.device, dtype=torch.long)
#     if not torch.equal(count_tensor, physical.sample_counts):
#         raise RuntimeError("learned and physical CDS sample counts disagree")

#     return ClassStructureBatch(
#         class_ids=physical.class_ids,
#         sample_counts=physical.sample_counts,
#         identity=torch.stack(identities, dim=0),
#         variation=torch.stack(deformations, dim=0),
#         spectral_identity=physical.identity,
#         spectral_variation=physical.variation,
#     )


# def estimate_structure_batch(
#     ordered_features: Tensor,
#     spectral_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     physical = estimate_physical_structure_batch(
#         spectral_evidence,
#         labels,
#         class_ids=class_ids,
#     )
#     return combine_learned_and_physical_structure(
#         ordered_features,
#         labels,
#         physical,
#     )


# def merge_class_structure_batches(
#     batches: Sequence[ClassStructureBatch],
# ) -> ClassStructureBatch:
#     values = list(batches)
#     if not values:
#         raise ValueError("batches cannot be empty")

#     device = values[0].identity.device
#     dtype = values[0].identity.dtype
#     learned_shape = tuple(values[0].identity.shape[1:])
#     variation_shape = tuple(values[0].variation.shape[1:])
#     spectral_shape = tuple(values[0].spectral_identity.shape[1:])
#     spectral_variation_shape = tuple(values[0].spectral_variation.shape[1:])
#     seen: set[int] = set()

#     for batch in values:
#         if batch.class_count <= 0:
#             raise ValueError("structure batches cannot be empty")
#         if batch.identity.device != device or batch.identity.dtype != dtype:
#             raise ValueError("all structure batches must share device/dtype")
#         if (
#             tuple(batch.identity.shape[1:]) != learned_shape
#             or tuple(batch.variation.shape[1:]) != variation_shape
#             or tuple(batch.spectral_identity.shape[1:]) != spectral_shape
#             or tuple(batch.spectral_variation.shape[1:]) != spectral_variation_shape
#         ):
#             raise ValueError("all structure batches must share CDS dimensions")
#         ids = [int(v) for v in batch.class_ids.detach().cpu().tolist()]
#         duplicate = seen.intersection(ids)
#         if duplicate:
#             raise ValueError(f"duplicate structure classes {sorted(duplicate)}")
#         seen.update(ids)

#     return ClassStructureBatch(
#         class_ids=torch.cat([value.class_ids for value in values], dim=0),
#         sample_counts=torch.cat([value.sample_counts for value in values], dim=0),
#         identity=torch.cat([value.identity for value in values], dim=0),
#         variation=torch.cat([value.variation for value in values], dim=0),
#         spectral_identity=torch.cat([value.spectral_identity for value in values], dim=0),
#         spectral_variation=torch.cat([value.spectral_variation for value in values], dim=0),
#     )


# # -----------------------------------------------------------------------------
# # Derived inter-class CDS
# # -----------------------------------------------------------------------------


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     """Derive learned + physical inter-class CDS from stored class states."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")

#     learned_identity_by_region = _pairwise_l1_similarity_by_slot(structure.identity)
#     learned_variation_relation = _pairwise_l1_similarity_global(
#         ordered_variation_operator(structure.variation)
#     )

#     # [C,B,3] -> [C,3,B] so the 3 physical components are the slots.
#     physical_identity_by_component = _pairwise_l1_similarity_by_slot(
#         structure.spectral_identity.permute(0, 2, 1)
#     )
#     physical_variation_relation = _pairwise_l1_similarity_global(
#         ordered_variation_operator(structure.spectral_variation)
#     )

#     physical_identity_scalar = physical_identity_by_component.sum(dim=2) / 3.0
#     spectral_confusability = (
#         0.5 * (physical_identity_scalar + physical_variation_relation)
#     ).clamp(0.0, 1.0)

#     c = structure.class_count
#     diagonal = torch.arange(c, device=structure.identity.device)
#     spectral_confusability = spectral_confusability.clone()
#     spectral_confusability[diagonal, diagonal] = 1.0

#     values = (
#         learned_identity_by_region,
#         learned_variation_relation,
#         physical_identity_by_component,
#         physical_variation_relation,
#         spectral_confusability,
#     )
#     if not all(bool(torch.isfinite(value).all()) for value in values):
#         raise RuntimeError("derived HSI-CDS relations contain NaN or Inf")

#     return ClassRelationState(
#         class_ids=structure.class_ids,
#         learned_identity_cosine_by_region=learned_identity_by_region,
#         learned_variation_overlap=learned_variation_relation,
#         spectral_identity_cosine_by_component=physical_identity_by_component,
#         spectral_variation_overlap=physical_variation_relation,
#         spectral_confusability=spectral_confusability,
#     )


# def learned_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     """Learned HSI-CDS confusability from I_Q and V_Q [C,C]."""
#     relations = derive_class_relations(structure)
#     identity_scalar = (
#         relations.learned_identity_cosine_by_region.sum(dim=2)
#         / float(relations.learned_identity_cosine_by_region.size(2))
#     )
#     result = 0.5 * (identity_scalar + relations.learned_variation_overlap)
#     return result.clamp(0.0, 1.0)


# def learned_boundary_overlap_matrix(structure: ClassStructureBatch) -> Tensor:
#     """Legacy alias; this is CDS confusability, not a classifier boundary."""
#     return learned_cds_confusability_matrix(structure)


# def physical_cds_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     """Physical HSI-CDS confusability from I_P and V_P [C,C]."""
#     relations = derive_class_relations(structure)
#     return relations.spectral_confusability


# def physical_region_confusability_matrix(structure: ClassStructureBatch) -> Tensor:
#     """Legacy alias for physical HSI-CDS confusability."""
#     return physical_cds_confusability_matrix(structure)


# def whole_cds_pair_risk(
#     cds: ClassDiscriminativeStructureState,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """One derived HSI risk from learned and physical intra/inter CDS.

#     learned  = 0.5 * (I_Q relation + V_Q relation)
#     physical = 0.5 * (I_P relation + V_P relation)
#     risk     = sqrt(learned * physical)

#     The diagonal is zero because pair risk is only used between different
#     classes.  No class-set normalization or hand-selected threshold is used.
#     """
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")

#     rel = cds.relations
#     learned_identity = (
#         rel.learned_identity_cosine_by_region.sum(dim=2)
#         / float(rel.learned_identity_cosine_by_region.size(2))
#     )
#     learned = 0.5 * (learned_identity + rel.learned_variation_overlap)

#     physical_identity = (
#         rel.spectral_identity_cosine_by_component.sum(dim=2)
#         / float(rel.spectral_identity_cosine_by_component.size(2))
#     )
#     physical = 0.5 * (physical_identity + rel.spectral_variation_overlap)

#     risk = torch.sqrt(
#         learned.clamp_min(0.0) * physical.clamp_min(0.0)
#     ).clamp(0.0, 1.0)

#     c = int(risk.size(0))
#     if c < 2:
#         raise ValueError("pair risk requires at least two classes")
#     diagonal = torch.arange(c, device=risk.device)
#     risk = risk.clone()
#     risk[diagonal, diagonal] = 0.0

#     if not bool(torch.isfinite(risk).all()):
#         raise RuntimeError("whole-CDS pair risk became NaN or Inf")
#     return risk, learned.clamp(0.0, 1.0), physical.clamp(0.0, 1.0)


# def spectral_correspondence_metric(structure: ClassStructureBatch) -> Tensor:
#     """Robust native-band class-balanced V_P reference [3,B,B]."""
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     physical = ordered_variation_operator(structure.spectral_variation)
#     # [C,B,B,3] -> median over class dimension -> [B,B,3] -> [3,B,B]
#     relation = _deterministic_lower_median_dim0(physical)
#     result = relation.permute(2, 0, 1).contiguous()
#     if not bool(torch.isfinite(result).all()):
#         raise RuntimeError("spectral correspondence metric contains NaN or Inf")
#     return result


# # -----------------------------------------------------------------------------
# # Persistent bank
# # -----------------------------------------------------------------------------


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Persistent HSI-CDS bank storing {I_Q,V_Q,I_P,V_P,N}."""

#     def __init__(
#         self,
#         *,
#         spectral_token_count: int,
#         spectral_token_dim: int,
#         spectral_bands: int,
#         spectral_band_positions: Sequence[float] | Tensor | None = None,
#         device: torch.device | str | None = None,
#         dtype: torch.dtype = torch.float32,
#     ) -> None:
#         super().__init__()
#         self.spectral_token_count = _integer(
#             spectral_token_count,
#             "spectral_token_count",
#             minimum=2,
#         )
#         self.spectral_token_dim = _integer(
#             spectral_token_dim,
#             "spectral_token_dim",
#             minimum=1,
#         )
#         self.spectral_bands = _integer(
#             spectral_bands,
#             "spectral_bands",
#             minimum=2,
#         )
#         if dtype not in (torch.float32, torch.float64):
#             raise TypeError("structure dtype must be float32 or float64")

#         positions = _positions(
#             spectral_band_positions,
#             count=self.spectral_bands,
#             device=device,
#             dtype=dtype,
#         )
#         self.register_buffer(
#             "_floating_reference",
#             torch.empty(0, device=device, dtype=dtype),
#             persistent=False,
#         )
#         self.register_buffer("spectral_band_positions", positions, persistent=True)
#         self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long))
#         self.register_buffer("sample_counts", torch.empty(0, device=device, dtype=torch.long))
#         self.register_buffer(
#             "identity_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "spectral_identity_bank",
#             torch.empty(
#                 0,
#                 self.spectral_bands,
#                 3,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.register_buffer(
#             "spectral_variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_bands,
#                 self.spectral_bands,
#                 3,
#                 device=device,
#                 dtype=dtype,
#             ),
#         )
#         self.validate_state()

#     @property
#     def device(self) -> torch.device:
#         return self._floating_reference.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self._floating_reference.dtype

#     def __len__(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(v) for v in self.class_ids.detach().cpu().tolist())

#     @staticmethod
#     def _validate_deformation_semantics(value: Tensor, *, name: str) -> None:
#         if value.ndim != 4 or value.size(1) != value.size(2):
#             raise RuntimeError(f"{name} must be [C,L,L,F]")
#         if value.numel() == 0:
#             return

#         tolerance = 32.0 * torch.finfo(value.dtype).eps
#         if bool((value < -tolerance).any()):
#             raise RuntimeError(f"{name} contains negative variation")
#         if not bool(torch.allclose(value, value.transpose(1, 2), rtol=0.0, atol=tolerance)):
#             raise RuntimeError(f"{name} is not symmetric across ordered spectral pairs")

#         diagonal = value.diagonal(dim1=1, dim2=2)
#         if bool((diagonal.abs() > tolerance).any()):
#             raise RuntimeError(f"{name} diagonal must be zero")

#     def validate_state(self) -> bool:
#         c = len(self)
#         expected = {
#             "class_ids": (c,),
#             "sample_counts": (c,),
#             "identity_bank": (c, self.spectral_token_count, self.spectral_token_dim),
#             "variation_bank": (
#                 c,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#             ),
#             "spectral_identity_bank": (c, self.spectral_bands, 3),
#             "spectral_variation_bank": (
#                 c,
#                 self.spectral_bands,
#                 self.spectral_bands,
#                 3,
#             ),
#         }
#         for name, shape in expected.items():
#             if getattr(self, name).shape != shape:
#                 raise RuntimeError(
#                     f"{name} state is invalid: "
#                     f"{tuple(getattr(self, name).shape)} != {shape}"
#                 )

#         if self.class_ids.dtype != torch.long or self.sample_counts.dtype != torch.long:
#             raise RuntimeError("class IDs/counts must be torch.long")
#         if c:
#             if bool((self.class_ids < 0).any()) or self.class_ids.unique().numel() != c:
#                 raise RuntimeError("class IDs must be unique non-negative values")
#             if bool((self.sample_counts <= 0).any()):
#                 raise RuntimeError("class sample counts must be positive")

#         for name in (
#             "identity_bank",
#             "variation_bank",
#             "spectral_identity_bank",
#             "spectral_variation_bank",
#         ):
#             value = getattr(self, name)
#             if value.device != self.device or value.dtype != self.dtype:
#                 raise RuntimeError(f"{name} has wrong device/dtype")
#             if not bool(torch.isfinite(value).all()):
#                 raise RuntimeError(f"{name} contains NaN or Inf")

#         self._validate_deformation_semantics(self.variation_bank, name="variation_bank")
#         self._validate_deformation_semantics(
#             self.spectral_variation_bank,
#             name="spectral_variation_bank",
#         )
#         return True

#     def _validate_batch(self, batch: ClassStructureBatch) -> None:
#         if not isinstance(batch, ClassStructureBatch) or batch.class_count <= 0:
#             raise TypeError("batch must be a non-empty ClassStructureBatch")
#         c = batch.class_count
#         shapes = (
#             (batch.class_ids, (c,)),
#             (batch.sample_counts, (c,)),
#             (batch.identity, (c, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 batch.variation,
#                 (
#                     c,
#                     self.spectral_token_count,
#                     self.spectral_token_count,
#                     self.spectral_token_dim,
#                 ),
#             ),
#             (batch.spectral_identity, (c, self.spectral_bands, 3)),
#             (
#                 batch.spectral_variation,
#                 (c, self.spectral_bands, self.spectral_bands, 3),
#             ),
#         )
#         if any(value.shape != shape for value, shape in shapes):
#             raise ValueError("class structure batch shape is invalid")
#         if batch.class_ids.dtype != torch.long or batch.sample_counts.dtype != torch.long:
#             raise ValueError("class IDs/counts must be torch.long")

#         tensors = (
#             batch.identity,
#             batch.variation,
#             batch.spectral_identity,
#             batch.spectral_variation,
#         )
#         if any(value.device != self.device or value.dtype != self.dtype for value in tensors):
#             raise ValueError("batch and bank must share device/dtype")
#         if not all(bool(torch.isfinite(value).all()) for value in tensors):
#             raise ValueError("HSI-CDS contains NaN or Inf")
#         if bool((batch.sample_counts <= 0).any()):
#             raise ValueError("class sample counts must be positive")

#         self._validate_deformation_semantics(batch.variation, name="batch.variation")
#         self._validate_deformation_semantics(
#             batch.spectral_variation,
#             name="batch.spectral_variation",
#         )

#     def estimate_physical(
#         self,
#         spectral_evidence: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         p = torch.as_tensor(spectral_evidence)
#         if p.device != self.device or p.dtype != self.dtype:
#             raise ValueError("spectral evidence and bank must share device/dtype")
#         if p.ndim != 3 or p.shape[1:] != (self.spectral_bands, 3):
#             raise ValueError("spectral evidence does not match dataset-native band count")
#         return estimate_physical_structure_batch(
#             p,
#             labels,
#             class_ids=class_ids,
#         )

#     def estimate_classes(
#         self,
#         ordered_features: Tensor,
#         spectral_evidence: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         q = torch.as_tensor(ordered_features)
#         if q.device != self.device or q.dtype != self.dtype:
#             raise ValueError("Q and bank must share device/dtype")

#         if physical_structure is None:
#             if spectral_evidence is None:
#                 raise ValueError("spectral_evidence is required when physical_structure is absent")
#             physical_structure = self.estimate_physical(
#                 spectral_evidence,
#                 labels,
#                 class_ids=class_ids,
#             )
#         else:
#             requested = _class_id_tuple(class_ids, name="class_ids")
#             actual = tuple(
#                 int(v)
#                 for v in physical_structure.class_ids.detach().cpu().tolist()
#             )
#             if actual != requested:
#                 raise ValueError("physical_structure class order does not match class_ids")

#         return combine_learned_and_physical_structure(
#             q,
#             labels,
#             physical_structure,
#         )

#     @torch.no_grad()
#     def append_estimate(self, estimate: ClassStructureBatch) -> None:
#         self._validate_batch(estimate)
#         overlap = set(
#             int(v) for v in estimate.class_ids.detach().cpu().tolist()
#         ).intersection(self.seen_class_ids)
#         if overlap:
#             raise ValueError(f"class structure already exists for {sorted(overlap)}")
#         combined = (
#             estimate
#             if len(self) == 0
#             else merge_class_structure_batches((self.class_structures(), estimate))
#         )
#         self.replace_all(combined)

#     @torch.no_grad()
#     def replace_all(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         self.class_ids = structure.class_ids.detach().clone()
#         self.sample_counts = structure.sample_counts.detach().clone()
#         self.identity_bank = structure.identity.detach().clone()
#         self.variation_bank = sanitize_ordered_variation(structure.variation.detach().clone())
#         self.spectral_identity_bank = structure.spectral_identity.detach().clone()
#         self.spectral_variation_bank = sanitize_ordered_variation(
#             structure.spectral_variation.detach().clone()
#         )
#         self.validate_state()

#     def _row_for_class(self, class_id: int) -> int:
#         cid = _integer(class_id, "class_id", minimum=0)
#         rows = torch.nonzero(self.class_ids.eq(cid), as_tuple=False).flatten()
#         if rows.numel() != 1:
#             raise KeyError(f"class structure is not stored uniquely for class {cid}")
#         return int(rows.item())

#     def class_structure(self, class_id: int) -> ClassStructureView:
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(self.class_ids[row].item()),
#             sample_count=int(self.sample_counts[row].item()),
#             identity=self.identity_bank[row].detach().clone(),
#             variation=self.variation_bank[row].detach().clone(),
#             spectral_identity=self.spectral_identity_bank[row].detach().clone(),
#             spectral_variation=self.spectral_variation_bank[row].detach().clone(),
#         )

#     def class_structures(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassStructureBatch:
#         self.validate_state()
#         if len(self) == 0:
#             raise RuntimeError("class structure bank is empty")
#         requested = (
#             self.seen_class_ids
#             if class_ids is None
#             else _class_id_tuple(class_ids, name="class_ids")
#         )
#         rows = torch.tensor(
#             [self._row_for_class(cid) for cid in requested],
#             device=self.device,
#             dtype=torch.long,
#         )
#         return ClassStructureBatch(
#             class_ids=self.class_ids.index_select(0, rows).detach().clone(),
#             sample_counts=self.sample_counts.index_select(0, rows).detach().clone(),
#             identity=self.identity_bank.index_select(0, rows).detach().clone(),
#             variation=self.variation_bank.index_select(0, rows).detach().clone(),
#             spectral_identity=self.spectral_identity_bank.index_select(0, rows).detach().clone(),
#             spectral_variation=self.spectral_variation_bank.index_select(0, rows).detach().clone(),
#         )

#     def build_cds(self, batch: ClassStructureBatch) -> ClassDiscriminativeStructureState:
#         self._validate_batch(batch)
#         return ClassDiscriminativeStructureState(
#             classes=batch,
#             relations=derive_class_relations(batch),
#             spectral_band_positions=self.spectral_band_positions.detach().clone(),
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
#         dynamic = {
#             "class_ids": torch.long,
#             "sample_counts": torch.long,
#             "identity_bank": self.dtype,
#             "variation_bank": self.dtype,
#             "spectral_identity_bank": self.dtype,
#             "spectral_variation_bank": self.dtype,
#         }
#         for name, dtype in dynamic.items():
#             incoming = state_dict.get(prefix + name)
#             if incoming is not None and getattr(self, name).shape != incoming.shape:
#                 setattr(
#                     self,
#                     name,
#                     torch.empty(incoming.shape, device=self.device, dtype=dtype),
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
#             # Old checkpoints may contain the previous relation semantics.  Shape
#             # compatibility is preserved, but a new run should rebuild CDS from
#             # real TRAIN samples before relying on the state.
#             self.validate_state()


# __all__ = [
#     "PhysicalSpectralStructureBatch",
#     "ClassStructureBatch",
#     "ClassRelationState",
#     "ClassDiscriminativeStructureState",
#     "ClassStructureView",
#     "HSIClassDiscriminativeStructureBank",
#     "merge_class_structure_batches",
#     "estimate_physical_structure_batch",
#     "combine_learned_and_physical_structure",
#     "estimate_structure_batch",
#     "derive_class_relations",
#     "learned_cds_confusability_matrix",
#     "learned_boundary_overlap_matrix",
#     "sanitize_ordered_variation",
#     "project_ordered_variation_to_psd",
#     "minimum_ordered_variation_eigenvalue",
#     "ordered_variation_operator",
#     "spectral_correspondence_metric",
#     "physical_cds_confusability_matrix",
#     "physical_region_confusability_matrix",
#     "whole_cds_pair_risk",
# ]
