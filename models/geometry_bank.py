from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

import torch
import torch.nn as nn

Tensor = torch.Tensor


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


def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
    lower = _deterministic_quantile_dim0(values, 0.25)
    upper = _deterministic_quantile_dim0(values, 0.75)
    return (upper - lower).clamp_min(0.0)


def _validate_ordered_samples(values: Tensor) -> Tensor:
    value = torch.as_tensor(values)
    if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) <= 0:
        raise ValueError("values must be non-empty [N,L,F], L>=2")
    if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError("values must be finite real floating point")
    return value


def _validate_physical_evidence(values: Tensor) -> Tensor:
    value = torch.as_tensor(values)
    if value.ndim == 2:
        value = value.unsqueeze(-1)
    if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) != 1:
        raise ValueError("physical evidence must be direct full-band [N,B,1], B>=2")
    if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
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
    value = torch.as_tensor(variation)
    if value.ndim != 4 or value.size(1) != value.size(2):
        raise ValueError("variation must be [C,L,L,F]")
    if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
        raise ValueError("variation must be finite real floating point")
    clean = value.clamp_min(0.0)
    return (0.5 * (clean + clean.transpose(1, 2))).contiguous()


def ordered_variation_operator(variation: Tensor) -> Tensor:
    return sanitize_ordered_variation(variation)


def ordered_cds_support_components(
    sample: Tensor,
    identity: Tensor,
    variation: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return absolute, ordered-relation and complete CDS support [N,C].

    This is the single canonical sample-to-CDS compatibility rule used by
    preservation, cross-phase separation and evaluation.  V is a robust ordered
    scale field, never covariance or a Gaussian parameter.
    """
    x = torch.as_tensor(sample)
    center = torch.as_tensor(identity, device=x.device, dtype=x.dtype)
    robust_scale = sanitize_ordered_variation(
        torch.as_tensor(variation, device=x.device, dtype=x.dtype)
    )
    if x.ndim != 3 or x.size(0) == 0 or x.size(1) < 2 or x.size(2) <= 0:
        raise ValueError("sample must be non-empty [N,L,F], L>=2")
    if center.ndim != 3 or tuple(center.shape[1:]) != tuple(x.shape[1:]):
        raise ValueError("identity must be [C,L,F] matching sample")
    class_count, length, feature_count = center.shape
    if robust_scale.shape != (class_count, length, length, feature_count):
        raise ValueError("variation must be [C,L,L,F] matching identity")
    if not all(bool(torch.isfinite(v).all()) for v in (x, center, robust_scale)):
        raise ValueError("sample/identity/variation contains NaN or Inf")

    eps = torch.finfo(x.dtype).eps
    diagonal = torch.arange(length, device=x.device)
    absolute_scale = robust_scale[:, diagonal, diagonal, :]
    absolute_deviation = (x[:, None] - center[None]).abs()
    absolute_denominator = absolute_scale[None] + absolute_deviation
    absolute_element = absolute_scale[None] / absolute_denominator.clamp_min(eps)
    absolute_element = torch.where(
        absolute_denominator.le(eps),
        torch.ones_like(absolute_element),
        absolute_element,
    )
    absolute_support = absolute_element.mean(dim=(2, 3)).clamp(0.0, 1.0)

    upper = torch.triu(
        torch.ones(length, length, device=x.device, dtype=torch.bool),
        diagonal=1,
    )
    if int(upper.sum().item()) <= 0:
        raise RuntimeError("ordered CDS support has no off-diagonal relations")
    sample_relation = x[:, :, None, :] - x[:, None, :, :]
    relation_rows: list[Tensor] = []
    for class_index in range(class_count):
        identity_relation = (
            center[class_index, :, None, :] - center[class_index, None, :, :]
        )
        deviation = (sample_relation - identity_relation.unsqueeze(0)).abs()
        class_scale = robust_scale[class_index].unsqueeze(0)
        denominator = class_scale + deviation
        element = class_scale / denominator.clamp_min(eps)
        element = torch.where(denominator.le(eps), torch.ones_like(element), element)
        relation_rows.append(
            element[:, upper, :].mean(dim=(1, 2)).clamp(0.0, 1.0)
        )
    relation_support = torch.stack(relation_rows, dim=1)
    complete_support = torch.sqrt(
        absolute_support.clamp_min(0.0) * relation_support.clamp_min(0.0)
    ).clamp(0.0, 1.0)
    return absolute_support, relation_support, complete_support


def combine_learned_physical_relevance(learned: Tensor, physical: Tensor) -> Tensor:
    """Fuse learned and physical relevance with learned semantics as the gate.

    Physical similarity can strengthen an existing learned-space relation but
    cannot create semantic relevance by itself.  This same rule is used by drift
    correction, feature separation and classifier reconstruction.
    """
    learned_value = torch.as_tensor(learned)
    physical_value = torch.as_tensor(
        physical, device=learned_value.device, dtype=learned_value.dtype
    )
    if learned_value.shape != physical_value.shape:
        raise ValueError("learned and physical relevance must have identical shape")
    if not bool(torch.isfinite(learned_value).all()) or not bool(torch.isfinite(physical_value).all()):
        raise ValueError("learned/physical relevance contains NaN or Inf")
    learned_value = learned_value.clamp(0.0, 1.0)
    physical_value = physical_value.clamp(0.0, 1.0)
    return (
        learned_value
        + (1.0 - learned_value) * learned_value * physical_value
    ).clamp(0.0, 1.0)


def _ordered_identity_and_variation(
    values: Tensor,
    *,
    row_chunk: int = 8,
) -> tuple[Tensor, Tensor]:
    sample = _validate_ordered_samples(values)
    identity = _deterministic_median_dim0(sample)
    _, length, _ = sample.shape
    chunk = _integer(row_chunk, "row_chunk", minimum=1)

    rows: list[Tensor] = []
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
        rows.append(_deterministic_iqr_dim0(differences))
    variation = torch.cat(rows, dim=0)

    diagonal = torch.arange(length, device=sample.device)
    variation[diagonal, diagonal, :] = _deterministic_iqr_dim0(sample)
    variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
    return identity.contiguous(), variation.contiguous()


def _variation_diagonal(variation: Tensor) -> Tensor:
    value = sanitize_ordered_variation(variation)
    length = value.size(1)
    index = torch.arange(length, device=value.device)
    return value[:, index, index, :]


def _robust_scale_proximity(separation: Tensor, combined_scale: Tensor) -> Tensor:
    distance = torch.as_tensor(separation)
    scale = torch.as_tensor(combined_scale, device=distance.device, dtype=distance.dtype)
    if distance.shape != scale.shape:
        raise ValueError("separation and combined_scale must have identical shapes")
    if (
        not bool(torch.isfinite(distance).all())
        or not bool(torch.isfinite(scale).all())
        or bool((distance < 0).any())
        or bool((scale < 0).any())
    ):
        raise ValueError("proximity inputs must be finite non-negative tensors")
    eps = torch.finfo(distance.dtype).eps
    denominator = scale + distance
    result = scale / denominator.clamp_min(eps)
    result = torch.where(denominator.le(eps), torch.ones_like(result), result)
    return result.clamp(0.0, 1.0)


def _absolute_class_proximity_by_slot(identity: Tensor, variation: Tensor) -> Tensor:
    center = torch.as_tensor(identity)
    if center.ndim != 3 or center.size(0) <= 0 or center.size(1) < 2:
        raise ValueError("identity must be non-empty [C,L,F]")
    scale = _variation_diagonal(variation)
    if scale.shape != center.shape:
        raise ValueError("variation diagonal must match identity")
    separation = (center[:, None] - center[None, :]).abs()
    combined_scale = scale[:, None] + scale[None, :]
    proximity = _robust_scale_proximity(separation, combined_scale).mean(dim=3)
    diagonal = torch.arange(center.size(0), device=center.device)
    proximity = proximity.clone()
    proximity[diagonal, diagonal, :] = 1.0
    return proximity.contiguous()


def _relation_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
    center = torch.as_tensor(identity)
    robust_scale = sanitize_ordered_variation(variation)
    if center.ndim != 3:
        raise ValueError("identity must be [C,L,F]")
    class_count, length, feature_count = center.shape
    if robust_scale.shape != (class_count, length, length, feature_count):
        raise ValueError("variation shape does not match identity")

    relation_center = center[:, :, None, :] - center[:, None, :, :]
    separation = (relation_center[:, None] - relation_center[None, :]).abs()
    combined_scale = robust_scale[:, None] + robust_scale[None, :]
    proximity = _robust_scale_proximity(separation, combined_scale)
    upper = torch.triu(
        torch.ones(length, length, device=center.device, dtype=torch.bool),
        diagonal=1,
    )
    result = proximity[:, :, upper, :].mean(dim=(2, 3))
    diagonal = torch.arange(class_count, device=center.device)
    result = result.clone()
    result[diagonal, diagonal] = 1.0
    return result.clamp(0.0, 1.0).contiguous()


@dataclass(frozen=True)
class PhysicalSpectralStructureBatch:
    class_ids: Tensor
    sample_counts: Tensor
    physical_identity: Tensor       # [C,B,1]
    physical_variation: Tensor      # [C,B,B,1]

    @property
    def class_count(self) -> int:
        return int(self.class_ids.numel())


@dataclass(frozen=True)
class ClassStructureBatch:
    class_ids: Tensor
    sample_counts: Tensor
    learned_identity: Tensor        # [C,K,D]
    learned_variation: Tensor       # [C,K,K,D]
    physical_identity: Tensor       # [C,B,1]
    physical_variation: Tensor      # [C,B,B,1]

    @property
    def class_count(self) -> int:
        return int(self.class_ids.numel())


@dataclass(frozen=True)
class ClassRelationState:
    class_ids: Tensor
    learned_identity_proximity_by_region: Tensor   # [C,C,K]
    learned_relation_proximity: Tensor             # [C,C]
    physical_identity_proximity_by_band: Tensor    # [C,C,B]
    physical_relation_proximity: Tensor            # [C,C]

    @property
    def learned_identity_proximity(self) -> Tensor:
        return self.learned_identity_proximity_by_region.mean(dim=2)

    @property
    def physical_identity_proximity(self) -> Tensor:
        return self.physical_identity_proximity_by_band.mean(dim=2)

    @property
    def learned_combined_proximity(self) -> Tensor:
        return torch.sqrt(
            self.learned_identity_proximity.clamp_min(0.0)
            * self.learned_relation_proximity.clamp_min(0.0)
        ).clamp(0.0, 1.0)

    @property
    def physical_combined_proximity(self) -> Tensor:
        return torch.sqrt(
            self.physical_identity_proximity.clamp_min(0.0)
            * self.physical_relation_proximity.clamp_min(0.0)
        ).clamp(0.0, 1.0)


@dataclass(frozen=True)
class ClassDiscriminativeStructureState:
    classes: ClassStructureBatch
    relations: ClassRelationState
    spectral_band_positions: Tensor


@dataclass(frozen=True)
class ClassStructureView:
    class_id: int
    sample_count: int
    learned_identity: Tensor
    learned_variation: Tensor
    physical_identity: Tensor
    physical_variation: Tensor


def estimate_physical_structure_batch(
    physical_evidence: Tensor,
    labels: Tensor,
    *,
    class_ids: Sequence[int] | Tensor,
) -> PhysicalSpectralStructureBatch:
    physical = _validate_physical_evidence(physical_evidence)
    y = torch.as_tensor(labels)
    if y.shape != (physical.size(0),) or y.dtype != torch.long or y.device != physical.device:
        raise ValueError("labels must be torch.long [N] on the physical-evidence device")

    requested = _class_id_tuple(class_ids, name="class_ids")
    observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
    if observed != tuple(sorted(requested)):
        raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

    counts: list[int] = []
    identities: list[Tensor] = []
    variations: list[Tensor] = []
    for class_id in requested:
        mask = y.eq(class_id)
        count = int(mask.sum().item())
        if count <= 0:
            raise RuntimeError(f"class {class_id} has no samples")
        identity, variation = _ordered_identity_and_variation(physical[mask])
        counts.append(count)
        identities.append(identity)
        variations.append(variation)

    return PhysicalSpectralStructureBatch(
        class_ids=torch.tensor(requested, device=physical.device, dtype=torch.long),
        sample_counts=torch.tensor(counts, device=physical.device, dtype=torch.long),
        physical_identity=torch.stack(identities, dim=0),
        physical_variation=torch.stack(variations, dim=0),
    )


def combine_learned_and_physical_structure(
    ordered_features: Tensor,
    labels: Tensor,
    physical_structure: PhysicalSpectralStructureBatch,
) -> ClassStructureBatch:
    learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
    y = torch.as_tensor(labels)
    if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
        raise ValueError("labels must be torch.long [N] on the Q device")
    if not isinstance(physical_structure, PhysicalSpectralStructureBatch) or physical_structure.class_count <= 0:
        raise TypeError("physical_structure must be non-empty")
    if physical_structure.physical_identity.device != learned.device or physical_structure.physical_identity.dtype != learned.dtype:
        raise ValueError("Q and physical CDS must share device/dtype")

    requested = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
    observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
    if observed != tuple(sorted(requested)):
        raise ValueError("Q labels do not match cached physical CDS classes")

    counts: list[int] = []
    identities: list[Tensor] = []
    variations: list[Tensor] = []
    for class_id in requested:
        mask = y.eq(class_id)
        count = int(mask.sum().item())
        if count <= 0:
            raise RuntimeError(f"class {class_id} has no Q samples")
        identity, variation = _ordered_identity_and_variation(learned[mask])
        counts.append(count)
        identities.append(identity)
        variations.append(variation)

    count_tensor = torch.tensor(counts, device=learned.device, dtype=torch.long)
    if not torch.equal(count_tensor, physical_structure.sample_counts):
        raise RuntimeError("learned and physical CDS sample counts disagree")

    return ClassStructureBatch(
        class_ids=physical_structure.class_ids,
        sample_counts=physical_structure.sample_counts,
        learned_identity=torch.stack(identities, dim=0),
        learned_variation=torch.stack(variations, dim=0),
        physical_identity=physical_structure.physical_identity,
        physical_variation=physical_structure.physical_variation,
    )


def estimate_structure_batch(
    ordered_features: Tensor,
    physical_evidence: Tensor,
    labels: Tensor,
    *,
    class_ids: Sequence[int] | Tensor,
) -> ClassStructureBatch:
    physical = estimate_physical_structure_batch(
        physical_evidence,
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

    device = values[0].learned_identity.device
    dtype = values[0].learned_identity.dtype
    learned_shape = tuple(values[0].learned_identity.shape[1:])
    learned_variation_shape = tuple(values[0].learned_variation.shape[1:])
    physical_shape = tuple(values[0].physical_identity.shape[1:])
    physical_variation_shape = tuple(values[0].physical_variation.shape[1:])

    seen: set[int] = set()
    for batch in values:
        ids = [int(value) for value in batch.class_ids.detach().cpu().tolist()]
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"class structure batches overlap at {sorted(overlap)}")
        seen.update(ids)
        tensors = (
            batch.learned_identity,
            batch.learned_variation,
            batch.physical_identity,
            batch.physical_variation,
        )
        if any(tensor.device != device or tensor.dtype != dtype for tensor in tensors):
            raise ValueError("class structure batches must share device/dtype")
        if tuple(batch.learned_identity.shape[1:]) != learned_shape:
            raise ValueError("learned identity shapes disagree")
        if tuple(batch.learned_variation.shape[1:]) != learned_variation_shape:
            raise ValueError("learned variation shapes disagree")
        if tuple(batch.physical_identity.shape[1:]) != physical_shape:
            raise ValueError("physical identity shapes disagree")
        if tuple(batch.physical_variation.shape[1:]) != physical_variation_shape:
            raise ValueError("physical variation shapes disagree")

    return ClassStructureBatch(
        class_ids=torch.cat([batch.class_ids for batch in values], dim=0),
        sample_counts=torch.cat([batch.sample_counts for batch in values], dim=0),
        learned_identity=torch.cat([batch.learned_identity for batch in values], dim=0),
        learned_variation=torch.cat([batch.learned_variation for batch in values], dim=0),
        physical_identity=torch.cat([batch.physical_identity for batch in values], dim=0),
        physical_variation=torch.cat([batch.physical_variation for batch in values], dim=0),
    )


def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
    if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
        raise ValueError("structure must be non-empty")
    learned_identity = _absolute_class_proximity_by_slot(
        structure.learned_identity,
        structure.learned_variation,
    )
    learned_relation = _relation_class_proximity(
        structure.learned_identity,
        structure.learned_variation,
    )
    physical_identity = _absolute_class_proximity_by_slot(
        structure.physical_identity,
        structure.physical_variation,
    )
    physical_relation = _relation_class_proximity(
        structure.physical_identity,
        structure.physical_variation,
    )
    return ClassRelationState(
        class_ids=structure.class_ids,
        learned_identity_proximity_by_region=learned_identity,
        learned_relation_proximity=learned_relation,
        physical_identity_proximity_by_band=physical_identity,
        physical_relation_proximity=physical_relation,
    )


class HSIClassDiscriminativeStructureBank(nn.Module):
    """Persistent exemplar-free class structure bank."""

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
        self.spectral_token_count = _integer(spectral_token_count, "spectral_token_count", minimum=2)
        self.spectral_token_dim = _integer(spectral_token_dim, "spectral_token_dim", minimum=1)
        self.spectral_bands = _integer(spectral_bands, "spectral_bands", minimum=2)
        if dtype not in (torch.float32, torch.float64):
            raise TypeError("CDS bank dtype must be float32 or float64")

        self.register_buffer(
            "spectral_band_positions",
            _band_positions(
                spectral_band_positions,
                count=self.spectral_bands,
                device=device,
                dtype=dtype,
            ),
            persistent=True,
        )
        self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long), persistent=True)
        self.register_buffer(
            "learned_identity_bank",
            torch.empty(0, self.spectral_token_count, self.spectral_token_dim, device=device, dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "learned_variation_bank",
            torch.empty(
                0,
                self.spectral_token_count,
                self.spectral_token_count,
                self.spectral_token_dim,
                device=device,
                dtype=dtype,
            ),
            persistent=True,
        )
        self.register_buffer(
            "physical_identity_bank",
            torch.empty(0, self.spectral_bands, 1, device=device, dtype=dtype),
            persistent=True,
        )
        self.register_buffer(
            "physical_variation_bank",
            torch.empty(0, self.spectral_bands, self.spectral_bands, 1, device=device, dtype=dtype),
            persistent=True,
        )
        self.validate_state()

    def __len__(self) -> int:
        return int(self.class_ids.numel())

    @property
    def device(self) -> torch.device:
        return self.learned_identity_bank.device

    @property
    def dtype(self) -> torch.dtype:
        return self.learned_identity_bank.dtype

    @property
    def seen_class_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.class_ids.detach().cpu().tolist())

    def _validate_variation(self, value: Tensor, *, name: str) -> None:
        if value.ndim != 4 or value.size(1) != value.size(2):
            raise RuntimeError(f"{name} must be [C,L,L,F]")
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise RuntimeError(f"{name} must be finite and non-negative")
        if value.numel() and not torch.allclose(value, value.transpose(1, 2), rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"{name} must be symmetric over ordered pairs")

    def validate_state(self) -> bool:
        class_count = len(self)
        expected_shapes = (
            (self.learned_identity_bank, (class_count, self.spectral_token_count, self.spectral_token_dim)),
            (
                self.learned_variation_bank,
                (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
            ),
            (self.physical_identity_bank, (class_count, self.spectral_bands, 1)),
            (self.physical_variation_bank, (class_count, self.spectral_bands, self.spectral_bands, 1)),
        )
        if self.class_ids.shape != (class_count,):
            raise RuntimeError("class_ids shape is invalid")
        if any(tensor.shape != shape for tensor, shape in expected_shapes):
            raise RuntimeError("CDS bank tensor shape is invalid")
        if self.class_ids.dtype != torch.long:
            raise RuntimeError("class IDs must be torch.long")
        if self.class_ids.unique().numel() != class_count or bool((self.class_ids < 0).any()):
            raise RuntimeError("stored class IDs must be unique and non-negative")
        floating = (
            self.learned_identity_bank,
            self.learned_variation_bank,
            self.physical_identity_bank,
            self.physical_variation_bank,
            self.spectral_band_positions,
        )
        if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in floating):
            raise RuntimeError("CDS bank floating tensors must share device/dtype")
        if not all(bool(torch.isfinite(tensor).all()) for tensor in floating):
            raise RuntimeError("CDS bank contains NaN or Inf")
        self._validate_variation(self.learned_variation_bank, name="learned_variation_bank")
        self._validate_variation(self.physical_variation_bank, name="physical_variation_bank")
        return True

    def _validate_batch(self, structure: ClassStructureBatch) -> None:
        if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
            raise TypeError("structure must be non-empty")
        class_count = structure.class_count
        expected = (
            (structure.class_ids, (class_count,)),
            (structure.sample_counts, (class_count,)),
            (structure.learned_identity, (class_count, self.spectral_token_count, self.spectral_token_dim)),
            (
                structure.learned_variation,
                (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
            ),
            (structure.physical_identity, (class_count, self.spectral_bands, 1)),
            (structure.physical_variation, (class_count, self.spectral_bands, self.spectral_bands, 1)),
        )
        if any(tensor.shape != shape for tensor, shape in expected):
            raise ValueError("class structure shape is invalid")
        tensors = (
            structure.learned_identity,
            structure.learned_variation,
            structure.physical_identity,
            structure.physical_variation,
        )
        if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in tensors):
            raise ValueError("class structure and CDS bank must share device/dtype")
        self._validate_variation(structure.learned_variation, name="learned_variation")
        self._validate_variation(structure.physical_variation, name="physical_variation")

    def estimate_physical(
        self,
        physical_evidence: Tensor,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
    ) -> PhysicalSpectralStructureBatch:
        physical = _validate_physical_evidence(physical_evidence)
        if physical.device != self.device or physical.dtype != self.dtype:
            raise ValueError("physical evidence and CDS bank must share device/dtype")
        if physical.shape[1:] != (self.spectral_bands, 1):
            raise ValueError("physical evidence does not match native band count")
        return estimate_physical_structure_batch(physical, labels, class_ids=class_ids)

    def estimate_classes(
        self,
        ordered_features: Tensor,
        physical_evidence: Tensor | None,
        labels: Tensor,
        *,
        class_ids: Sequence[int] | Tensor,
        physical_structure: PhysicalSpectralStructureBatch | None = None,
    ) -> ClassStructureBatch:
        learned = torch.as_tensor(ordered_features)
        if learned.device != self.device or learned.dtype != self.dtype:
            raise ValueError("Q and CDS bank must share device/dtype")
        if physical_structure is None:
            if physical_evidence is None:
                raise ValueError("physical_evidence is required when physical_structure is absent")
            physical_structure = self.estimate_physical(
                physical_evidence,
                labels,
                class_ids=class_ids,
            )
        else:
            requested = _class_id_tuple(class_ids, name="class_ids")
            actual = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
            if actual != requested:
                raise ValueError("physical_structure class order does not match class_ids")
        return combine_learned_and_physical_structure(learned, labels, physical_structure)

    @torch.no_grad()
    def append_estimate(self, structure: ClassStructureBatch) -> None:
        self._validate_batch(structure)
        overlap = set(int(value) for value in structure.class_ids.detach().cpu().tolist()).intersection(self.seen_class_ids)
        if overlap:
            raise ValueError(f"class structure already exists for {sorted(overlap)}")
        combined = structure if len(self) == 0 else merge_class_structure_batches((self.class_structures(), structure))
        self.replace_all(combined)

    @torch.no_grad()
    def replace_all(self, structure: ClassStructureBatch) -> None:
        self._validate_batch(structure)
        self.class_ids = structure.class_ids.detach().clone()
        self.learned_identity_bank = structure.learned_identity.detach().clone()
        self.learned_variation_bank = sanitize_ordered_variation(structure.learned_variation.detach().clone())
        self.physical_identity_bank = structure.physical_identity.detach().clone()
        self.physical_variation_bank = sanitize_ordered_variation(structure.physical_variation.detach().clone())
        self.validate_state()

    def _row_for_class(self, class_id: int) -> int:
        class_value = _integer(class_id, "class_id", minimum=0)
        rows = torch.nonzero(self.class_ids.eq(class_value), as_tuple=False).flatten()
        if rows.numel() != 1:
            raise KeyError(f"class structure is not stored uniquely for class {class_value}")
        return int(rows.item())

    def class_structure(self, class_id: int) -> ClassStructureView:
        row = self._row_for_class(class_id)
        return ClassStructureView(
            class_id=int(self.class_ids[row].item()),
            sample_count=0,
            learned_identity=self.learned_identity_bank[row].detach().clone(),
            learned_variation=self.learned_variation_bank[row].detach().clone(),
            physical_identity=self.physical_identity_bank[row].detach().clone(),
            physical_variation=self.physical_variation_bank[row].detach().clone(),
        )

    def class_structures(
        self,
        class_ids: Sequence[int] | Tensor | None = None,
    ) -> ClassStructureBatch:
        self.validate_state()
        if len(self) == 0:
            raise RuntimeError("CDS bank is empty")
        requested = self.seen_class_ids if class_ids is None else _class_id_tuple(class_ids, name="class_ids")
        rows = torch.tensor(
            [self._row_for_class(class_id) for class_id in requested],
            device=self.device,
            dtype=torch.long,
        )
        return ClassStructureBatch(
            class_ids=self.class_ids.index_select(0, rows).detach().clone(),
            sample_counts=torch.ones(rows.numel(), device=self.device, dtype=torch.long),
            learned_identity=self.learned_identity_bank.index_select(0, rows).detach().clone(),
            learned_variation=self.learned_variation_bank.index_select(0, rows).detach().clone(),
            physical_identity=self.physical_identity_bank.index_select(0, rows).detach().clone(),
            physical_variation=self.physical_variation_bank.index_select(0, rows).detach().clone(),
        )

    def build_cds(self, structure: ClassStructureBatch) -> ClassDiscriminativeStructureState:
        self._validate_batch(structure)
        return ClassDiscriminativeStructureState(
            classes=structure,
            relations=derive_class_relations(structure),
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
            "learned_identity_bank": self.dtype,
            "learned_variation_bank": self.dtype,
            "physical_identity_bank": self.dtype,
            "physical_variation_bank": self.dtype,
        }
        # sample_counts belonged to an older checkpoint schema. It was metadata,
        # not an algorithmic state variable, so consume it without retaining it.
        state_dict.pop(prefix + "sample_counts", None)
        for name, dtype in dynamic.items():
            incoming = state_dict.get(prefix + name)
            if incoming is not None and getattr(self, name).shape != incoming.shape:
                setattr(self, name, torch.empty(incoming.shape, device=self.device, dtype=dtype))
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




def _differentiable_quantile_dim0(values: Tensor, q: float) -> Tensor:
    """Differentiable counterpart of the persistent-bank linear quantile.

    The mathematical estimator is identical to ``_deterministic_quantile_dim0``;
    unlike the persistent estimator it does not detach or move to CPU, so input
    gradients can reconstruct a transient support set whose CDS matches storage.
    """
    value = torch.as_tensor(values)
    if value.ndim < 1 or value.size(0) <= 0:
        raise ValueError("quantile input must have a non-empty leading dimension")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError("quantile input must be real floating point")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("quantile input contains NaN or Inf")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must be in [0,1]")
    ordered = torch.sort(value, dim=0, stable=True).values
    sample_count = int(ordered.size(0))
    if sample_count == 1:
        return ordered[0]
    position = float(q) * float(sample_count - 1)
    lower = int(position)
    upper = min(lower + 1, sample_count - 1)
    weight = position - float(lower)
    if upper == lower or weight == 0.0:
        return ordered[lower]
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _differentiable_iqr_dim0(values: Tensor) -> Tensor:
    lower = _differentiable_quantile_dim0(values, 0.25)
    upper = _differentiable_quantile_dim0(values, 0.75)
    return (upper - lower).clamp_min(0.0)


def _ordered_identity_and_variation_differentiable(
    values: Tensor,
    *,
    row_chunk: int = 8,
) -> tuple[Tensor, Tensor]:
    sample = _validate_ordered_samples(values)
    identity = _differentiable_quantile_dim0(sample, 0.5)
    _, length, _ = sample.shape
    chunk = _integer(row_chunk, "row_chunk", minimum=1)
    rows: list[Tensor] = []
    for start in range(0, length, chunk):
        stop = min(start + chunk, length)
        differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
        rows.append(_differentiable_iqr_dim0(differences))
    variation = torch.cat(rows, dim=0)
    diagonal = torch.arange(length, device=sample.device)
    variation = variation.clone()
    variation[diagonal, diagonal, :] = _differentiable_iqr_dim0(sample)
    variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
    return identity.contiguous(), variation.contiguous()


def estimate_structure_batch_differentiable(
    ordered_features: Tensor,
    physical_evidence: Tensor,
    labels: Tensor,
    *,
    class_ids: Sequence[int] | Tensor,
) -> ClassStructureBatch:
    """Estimate the exact CDS formula without detaching generation inputs.

    This function exists only for transient support reconstruction. Persistent
    CDS storage continues to use the deterministic CPU estimator above.
    """
    learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
    physical = _validate_physical_evidence(physical_evidence)
    y = torch.as_tensor(labels)
    if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
        raise ValueError("labels must be torch.long [N] on the Q device")
    if physical.size(0) != learned.size(0) or physical.device != learned.device or physical.dtype != learned.dtype:
        raise ValueError("Q and physical evidence must share samples/device/dtype")
    requested = _class_id_tuple(class_ids, name="class_ids")
    observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
    if observed != tuple(sorted(requested)):
        raise ValueError("labels do not match requested classes")

    counts: list[int] = []
    learned_identity: list[Tensor] = []
    learned_variation: list[Tensor] = []
    physical_identity: list[Tensor] = []
    physical_variation: list[Tensor] = []
    for class_id in requested:
        mask = y.eq(class_id)
        count = int(mask.sum().item())
        if count < 2:
            raise ValueError("differentiable CDS reconstruction requires at least two supports per class")
        i_q, v_q = _ordered_identity_and_variation_differentiable(learned[mask])
        i_p, v_p = _ordered_identity_and_variation_differentiable(physical[mask])
        counts.append(count)
        learned_identity.append(i_q)
        learned_variation.append(v_q)
        physical_identity.append(i_p)
        physical_variation.append(v_p)

    return ClassStructureBatch(
        class_ids=torch.tensor(requested, device=learned.device, dtype=torch.long),
        sample_counts=torch.tensor(counts, device=learned.device, dtype=torch.long),
        learned_identity=torch.stack(learned_identity, dim=0),
        learned_variation=torch.stack(learned_variation, dim=0),
        physical_identity=torch.stack(physical_identity, dim=0),
        physical_variation=torch.stack(physical_variation, dim=0),
    )

def whole_cds_pair_risk(
    cds: ClassDiscriminativeStructureState,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return effective, learned and physical pair risk from the complete HSI-CDS.

    Learned risk uses both learned identity and learned ordered-relation structure.
    Physical risk uses both native-spectrum identity and native spectral relations.
    Physical similarity may strengthen an existing learned-space collision, but it
    cannot invent one because the deployed classifier lives in Q space.
    """
    if not isinstance(cds, ClassDiscriminativeStructureState):
        raise TypeError("cds must be ClassDiscriminativeStructureState")
    learned = cds.relations.learned_combined_proximity
    physical = cds.relations.physical_combined_proximity
    if learned.shape != physical.shape or learned.ndim != 2 or learned.size(0) != learned.size(1):
        raise ValueError("CDS pair-risk inputs must be square [C,C] matrices")
    if not bool(torch.isfinite(learned).all()) or not bool(torch.isfinite(physical).all()):
        raise RuntimeError("CDS pair risk contains NaN or Inf")

    learned = learned.clamp(0.0, 1.0)
    physical = physical.clamp(0.0, 1.0)
    effective = combine_learned_physical_relevance(learned, physical)

    diagonal = torch.arange(learned.size(0), device=learned.device)
    learned = learned.clone()
    physical = physical.clone()
    effective = effective.clone()
    learned[diagonal, diagonal] = 0.0
    physical[diagonal, diagonal] = 0.0
    effective[diagonal, diagonal] = 0.0
    return effective.contiguous(), learned.contiguous(), physical.contiguous()

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
    "estimate_structure_batch_differentiable",
    "merge_class_structure_batches",
    "normalize_ordered_decision_features",
    "sanitize_ordered_variation",
    "ordered_variation_operator",
    "ordered_cds_support_components",
    "combine_learned_physical_relevance",
    "derive_class_relations",
    "whole_cds_pair_risk",
]










# from __future__ import annotations
# from dataclasses import dataclass
# from numbers import Integral
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


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


# def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
#     lower = _deterministic_quantile_dim0(values, 0.25)
#     upper = _deterministic_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _validate_ordered_samples(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) <= 0:
#         raise ValueError("values must be non-empty [N,L,F], L>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("values must be finite real floating point")
#     return value


# def _validate_physical_evidence(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim == 2:
#         value = value.unsqueeze(-1)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) != 1:
#         raise ValueError("physical evidence must be direct full-band [N,B,1], B>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
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
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("variation must be finite real floating point")
#     clean = value.clamp_min(0.0)
#     return (0.5 * (clean + clean.transpose(1, 2))).contiguous()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     return sanitize_ordered_variation(variation)


# def ordered_cds_support_components(
#     sample: Tensor,
#     identity: Tensor,
#     variation: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Return absolute, ordered-relation and complete CDS support [N,C].

#     This is the single canonical sample-to-CDS compatibility rule used by
#     preservation, cross-phase separation and evaluation.  V is a robust ordered
#     scale field, never covariance or a Gaussian parameter.
#     """
#     x = torch.as_tensor(sample)
#     center = torch.as_tensor(identity, device=x.device, dtype=x.dtype)
#     robust_scale = sanitize_ordered_variation(
#         torch.as_tensor(variation, device=x.device, dtype=x.dtype)
#     )
#     if x.ndim != 3 or x.size(0) == 0 or x.size(1) < 2 or x.size(2) <= 0:
#         raise ValueError("sample must be non-empty [N,L,F], L>=2")
#     if center.ndim != 3 or tuple(center.shape[1:]) != tuple(x.shape[1:]):
#         raise ValueError("identity must be [C,L,F] matching sample")
#     class_count, length, feature_count = center.shape
#     if robust_scale.shape != (class_count, length, length, feature_count):
#         raise ValueError("variation must be [C,L,L,F] matching identity")
#     if not all(bool(torch.isfinite(v).all()) for v in (x, center, robust_scale)):
#         raise ValueError("sample/identity/variation contains NaN or Inf")

#     eps = torch.finfo(x.dtype).eps
#     diagonal = torch.arange(length, device=x.device)
#     absolute_scale = robust_scale[:, diagonal, diagonal, :]
#     absolute_deviation = (x[:, None] - center[None]).abs()
#     absolute_denominator = absolute_scale[None] + absolute_deviation
#     absolute_element = absolute_scale[None] / absolute_denominator.clamp_min(eps)
#     absolute_element = torch.where(
#         absolute_denominator.le(eps),
#         torch.ones_like(absolute_element),
#         absolute_element,
#     )
#     absolute_support = absolute_element.mean(dim=(2, 3)).clamp(0.0, 1.0)

#     upper = torch.triu(
#         torch.ones(length, length, device=x.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     if int(upper.sum().item()) <= 0:
#         raise RuntimeError("ordered CDS support has no off-diagonal relations")
#     sample_relation = x[:, :, None, :] - x[:, None, :, :]
#     relation_rows: list[Tensor] = []
#     for class_index in range(class_count):
#         identity_relation = (
#             center[class_index, :, None, :] - center[class_index, None, :, :]
#         )
#         deviation = (sample_relation - identity_relation.unsqueeze(0)).abs()
#         class_scale = robust_scale[class_index].unsqueeze(0)
#         denominator = class_scale + deviation
#         element = class_scale / denominator.clamp_min(eps)
#         element = torch.where(denominator.le(eps), torch.ones_like(element), element)
#         relation_rows.append(
#             element[:, upper, :].mean(dim=(1, 2)).clamp(0.0, 1.0)
#         )
#     relation_support = torch.stack(relation_rows, dim=1)
#     complete_support = torch.sqrt(
#         absolute_support.clamp_min(0.0) * relation_support.clamp_min(0.0)
#     ).clamp(0.0, 1.0)
#     return absolute_support, relation_support, complete_support


# def combine_learned_physical_relevance(learned: Tensor, physical: Tensor) -> Tensor:
#     """Fuse learned and physical relevance with learned semantics as the gate.

#     Physical similarity can strengthen an existing learned-space relation but
#     cannot create semantic relevance by itself.  This same rule is used by drift
#     correction, feature separation and classifier reconstruction.
#     """
#     learned_value = torch.as_tensor(learned)
#     physical_value = torch.as_tensor(
#         physical, device=learned_value.device, dtype=learned_value.dtype
#     )
#     if learned_value.shape != physical_value.shape:
#         raise ValueError("learned and physical relevance must have identical shape")
#     if not bool(torch.isfinite(learned_value).all()) or not bool(torch.isfinite(physical_value).all()):
#         raise ValueError("learned/physical relevance contains NaN or Inf")
#     learned_value = learned_value.clamp(0.0, 1.0)
#     physical_value = physical_value.clamp(0.0, 1.0)
#     return (
#         learned_value
#         + (1.0 - learned_value) * learned_value * physical_value
#     ).clamp(0.0, 1.0)


# def _ordered_identity_and_variation(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     sample = _validate_ordered_samples(values)
#     identity = _deterministic_median_dim0(sample)
#     _, length, _ = sample.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)

#     rows: list[Tensor] = []
#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
#         rows.append(_deterministic_iqr_dim0(differences))
#     variation = torch.cat(rows, dim=0)

#     diagonal = torch.arange(length, device=sample.device)
#     variation[diagonal, diagonal, :] = _deterministic_iqr_dim0(sample)
#     variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
#     return identity.contiguous(), variation.contiguous()


# def _variation_diagonal(variation: Tensor) -> Tensor:
#     value = sanitize_ordered_variation(variation)
#     length = value.size(1)
#     index = torch.arange(length, device=value.device)
#     return value[:, index, index, :]


# def _robust_scale_proximity(separation: Tensor, combined_scale: Tensor) -> Tensor:
#     distance = torch.as_tensor(separation)
#     scale = torch.as_tensor(combined_scale, device=distance.device, dtype=distance.dtype)
#     if distance.shape != scale.shape:
#         raise ValueError("separation and combined_scale must have identical shapes")
#     if (
#         not bool(torch.isfinite(distance).all())
#         or not bool(torch.isfinite(scale).all())
#         or bool((distance < 0).any())
#         or bool((scale < 0).any())
#     ):
#         raise ValueError("proximity inputs must be finite non-negative tensors")
#     eps = torch.finfo(distance.dtype).eps
#     denominator = scale + distance
#     result = scale / denominator.clamp_min(eps)
#     result = torch.where(denominator.le(eps), torch.ones_like(result), result)
#     return result.clamp(0.0, 1.0)


# def _absolute_class_proximity_by_slot(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     if center.ndim != 3 or center.size(0) <= 0 or center.size(1) < 2:
#         raise ValueError("identity must be non-empty [C,L,F]")
#     scale = _variation_diagonal(variation)
#     if scale.shape != center.shape:
#         raise ValueError("variation diagonal must match identity")
#     separation = (center[:, None] - center[None, :]).abs()
#     combined_scale = scale[:, None] + scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale).mean(dim=3)
#     diagonal = torch.arange(center.size(0), device=center.device)
#     proximity = proximity.clone()
#     proximity[diagonal, diagonal, :] = 1.0
#     return proximity.contiguous()


# def _relation_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     robust_scale = sanitize_ordered_variation(variation)
#     if center.ndim != 3:
#         raise ValueError("identity must be [C,L,F]")
#     class_count, length, feature_count = center.shape
#     if robust_scale.shape != (class_count, length, length, feature_count):
#         raise ValueError("variation shape does not match identity")

#     relation_center = center[:, :, None, :] - center[:, None, :, :]
#     separation = (relation_center[:, None] - relation_center[None, :]).abs()
#     combined_scale = robust_scale[:, None] + robust_scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale)
#     upper = torch.triu(
#         torch.ones(length, length, device=center.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     result = proximity[:, :, upper, :].mean(dim=(2, 3))
#     diagonal = torch.arange(class_count, device=center.device)
#     result = result.clone()
#     result[diagonal, diagonal] = 1.0
#     return result.clamp(0.0, 1.0).contiguous()


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     learned_identity: Tensor        # [C,K,D]
#     learned_variation: Tensor       # [C,K,K,D]
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_proximity_by_region: Tensor   # [C,C,K]
#     learned_relation_proximity: Tensor             # [C,C]
#     physical_identity_proximity_by_band: Tensor    # [C,C,B]
#     physical_relation_proximity: Tensor            # [C,C]

#     @property
#     def learned_identity_proximity(self) -> Tensor:
#         return self.learned_identity_proximity_by_region.mean(dim=2)

#     @property
#     def physical_identity_proximity(self) -> Tensor:
#         return self.physical_identity_proximity_by_band.mean(dim=2)

#     @property
#     def learned_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.learned_identity_proximity.clamp_min(0.0)
#             * self.learned_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)

#     @property
#     def physical_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.physical_identity_proximity.clamp_min(0.0)
#             * self.physical_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     learned_identity: Tensor
#     learned_variation: Tensor
#     physical_identity: Tensor
#     physical_variation: Tensor


# def estimate_physical_structure_batch(
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> PhysicalSpectralStructureBatch:
#     physical = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels)
#     if y.shape != (physical.size(0),) or y.dtype != torch.long or y.device != physical.device:
#         raise ValueError("labels must be torch.long [N] on the physical-evidence device")

#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no samples")
#         identity, variation = _ordered_identity_and_variation(physical[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     return PhysicalSpectralStructureBatch(
#         class_ids=torch.tensor(requested, device=physical.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=physical.device, dtype=torch.long),
#         physical_identity=torch.stack(identities, dim=0),
#         physical_variation=torch.stack(variations, dim=0),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical_structure: PhysicalSpectralStructureBatch,
# ) -> ClassStructureBatch:
#     learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#     y = torch.as_tensor(labels)
#     if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if not isinstance(physical_structure, PhysicalSpectralStructureBatch) or physical_structure.class_count <= 0:
#         raise TypeError("physical_structure must be non-empty")
#     if physical_structure.physical_identity.device != learned.device or physical_structure.physical_identity.dtype != learned.dtype:
#         raise ValueError("Q and physical CDS must share device/dtype")

#     requested = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("Q labels do not match cached physical CDS classes")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no Q samples")
#         identity, variation = _ordered_identity_and_variation(learned[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     count_tensor = torch.tensor(counts, device=learned.device, dtype=torch.long)
#     if not torch.equal(count_tensor, physical_structure.sample_counts):
#         raise RuntimeError("learned and physical CDS sample counts disagree")

#     return ClassStructureBatch(
#         class_ids=physical_structure.class_ids,
#         sample_counts=physical_structure.sample_counts,
#         learned_identity=torch.stack(identities, dim=0),
#         learned_variation=torch.stack(variations, dim=0),
#         physical_identity=physical_structure.physical_identity,
#         physical_variation=physical_structure.physical_variation,
#     )


# def estimate_structure_batch(
#     ordered_features: Tensor,
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     physical = estimate_physical_structure_batch(
#         physical_evidence,
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

#     device = values[0].learned_identity.device
#     dtype = values[0].learned_identity.dtype
#     learned_shape = tuple(values[0].learned_identity.shape[1:])
#     learned_variation_shape = tuple(values[0].learned_variation.shape[1:])
#     physical_shape = tuple(values[0].physical_identity.shape[1:])
#     physical_variation_shape = tuple(values[0].physical_variation.shape[1:])

#     seen: set[int] = set()
#     for batch in values:
#         ids = [int(value) for value in batch.class_ids.detach().cpu().tolist()]
#         overlap = seen.intersection(ids)
#         if overlap:
#             raise ValueError(f"class structure batches overlap at {sorted(overlap)}")
#         seen.update(ids)
#         tensors = (
#             batch.learned_identity,
#             batch.learned_variation,
#             batch.physical_identity,
#             batch.physical_variation,
#         )
#         if any(tensor.device != device or tensor.dtype != dtype for tensor in tensors):
#             raise ValueError("class structure batches must share device/dtype")
#         if tuple(batch.learned_identity.shape[1:]) != learned_shape:
#             raise ValueError("learned identity shapes disagree")
#         if tuple(batch.learned_variation.shape[1:]) != learned_variation_shape:
#             raise ValueError("learned variation shapes disagree")
#         if tuple(batch.physical_identity.shape[1:]) != physical_shape:
#             raise ValueError("physical identity shapes disagree")
#         if tuple(batch.physical_variation.shape[1:]) != physical_variation_shape:
#             raise ValueError("physical variation shapes disagree")

#     return ClassStructureBatch(
#         class_ids=torch.cat([batch.class_ids for batch in values], dim=0),
#         sample_counts=torch.cat([batch.sample_counts for batch in values], dim=0),
#         learned_identity=torch.cat([batch.learned_identity for batch in values], dim=0),
#         learned_variation=torch.cat([batch.learned_variation for batch in values], dim=0),
#         physical_identity=torch.cat([batch.physical_identity for batch in values], dim=0),
#         physical_variation=torch.cat([batch.physical_variation for batch in values], dim=0),
#     )


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be non-empty")
#     learned_identity = _absolute_class_proximity_by_slot(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     learned_relation = _relation_class_proximity(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     physical_identity = _absolute_class_proximity_by_slot(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     physical_relation = _relation_class_proximity(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     return ClassRelationState(
#         class_ids=structure.class_ids,
#         learned_identity_proximity_by_region=learned_identity,
#         learned_relation_proximity=learned_relation,
#         physical_identity_proximity_by_band=physical_identity,
#         physical_relation_proximity=physical_relation,
#     )


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Persistent exemplar-free class structure bank."""

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
#         self.spectral_token_count = _integer(spectral_token_count, "spectral_token_count", minimum=2)
#         self.spectral_token_dim = _integer(spectral_token_dim, "spectral_token_dim", minimum=1)
#         self.spectral_bands = _integer(spectral_bands, "spectral_bands", minimum=2)
#         if dtype not in (torch.float32, torch.float64):
#             raise TypeError("CDS bank dtype must be float32 or float64")

#         self.register_buffer(
#             "spectral_band_positions",
#             _band_positions(
#                 spectral_band_positions,
#                 count=self.spectral_bands,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long), persistent=True)
#         self.register_buffer(
#             "learned_identity_bank",
#             torch.empty(0, self.spectral_token_count, self.spectral_token_dim, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "learned_variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_identity_bank",
#             torch.empty(0, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_variation_bank",
#             torch.empty(0, self.spectral_bands, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.validate_state()

#     def __len__(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def device(self) -> torch.device:
#         return self.learned_identity_bank.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self.learned_identity_bank.dtype

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(value) for value in self.class_ids.detach().cpu().tolist())

#     def _validate_variation(self, value: Tensor, *, name: str) -> None:
#         if value.ndim != 4 or value.size(1) != value.size(2):
#             raise RuntimeError(f"{name} must be [C,L,L,F]")
#         if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
#             raise RuntimeError(f"{name} must be finite and non-negative")
#         if value.numel() and not torch.allclose(value, value.transpose(1, 2), rtol=1e-6, atol=1e-7):
#             raise RuntimeError(f"{name} must be symmetric over ordered pairs")

#     def validate_state(self) -> bool:
#         class_count = len(self)
#         expected_shapes = (
#             (self.learned_identity_bank, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 self.learned_variation_bank,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (self.physical_identity_bank, (class_count, self.spectral_bands, 1)),
#             (self.physical_variation_bank, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if self.class_ids.shape != (class_count,):
#             raise RuntimeError("class_ids shape is invalid")
#         if any(tensor.shape != shape for tensor, shape in expected_shapes):
#             raise RuntimeError("CDS bank tensor shape is invalid")
#         if self.class_ids.dtype != torch.long:
#             raise RuntimeError("class IDs must be torch.long")
#         if self.class_ids.unique().numel() != class_count or bool((self.class_ids < 0).any()):
#             raise RuntimeError("stored class IDs must be unique and non-negative")
#         floating = (
#             self.learned_identity_bank,
#             self.learned_variation_bank,
#             self.physical_identity_bank,
#             self.physical_variation_bank,
#             self.spectral_band_positions,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in floating):
#             raise RuntimeError("CDS bank floating tensors must share device/dtype")
#         if not all(bool(torch.isfinite(tensor).all()) for tensor in floating):
#             raise RuntimeError("CDS bank contains NaN or Inf")
#         self._validate_variation(self.learned_variation_bank, name="learned_variation_bank")
#         self._validate_variation(self.physical_variation_bank, name="physical_variation_bank")
#         return True

#     def _validate_batch(self, structure: ClassStructureBatch) -> None:
#         if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#             raise TypeError("structure must be non-empty")
#         class_count = structure.class_count
#         expected = (
#             (structure.class_ids, (class_count,)),
#             (structure.sample_counts, (class_count,)),
#             (structure.learned_identity, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 structure.learned_variation,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (structure.physical_identity, (class_count, self.spectral_bands, 1)),
#             (structure.physical_variation, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if any(tensor.shape != shape for tensor, shape in expected):
#             raise ValueError("class structure shape is invalid")
#         tensors = (
#             structure.learned_identity,
#             structure.learned_variation,
#             structure.physical_identity,
#             structure.physical_variation,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in tensors):
#             raise ValueError("class structure and CDS bank must share device/dtype")
#         self._validate_variation(structure.learned_variation, name="learned_variation")
#         self._validate_variation(structure.physical_variation, name="physical_variation")

#     def estimate_physical(
#         self,
#         physical_evidence: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         physical = _validate_physical_evidence(physical_evidence)
#         if physical.device != self.device or physical.dtype != self.dtype:
#             raise ValueError("physical evidence and CDS bank must share device/dtype")
#         if physical.shape[1:] != (self.spectral_bands, 1):
#             raise ValueError("physical evidence does not match native band count")
#         return estimate_physical_structure_batch(physical, labels, class_ids=class_ids)

#     def estimate_classes(
#         self,
#         ordered_features: Tensor,
#         physical_evidence: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         learned = torch.as_tensor(ordered_features)
#         if learned.device != self.device or learned.dtype != self.dtype:
#             raise ValueError("Q and CDS bank must share device/dtype")
#         if physical_structure is None:
#             if physical_evidence is None:
#                 raise ValueError("physical_evidence is required when physical_structure is absent")
#             physical_structure = self.estimate_physical(
#                 physical_evidence,
#                 labels,
#                 class_ids=class_ids,
#             )
#         else:
#             requested = _class_id_tuple(class_ids, name="class_ids")
#             actual = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#             if actual != requested:
#                 raise ValueError("physical_structure class order does not match class_ids")
#         return combine_learned_and_physical_structure(learned, labels, physical_structure)

#     @torch.no_grad()
#     def append_estimate(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         overlap = set(int(value) for value in structure.class_ids.detach().cpu().tolist()).intersection(self.seen_class_ids)
#         if overlap:
#             raise ValueError(f"class structure already exists for {sorted(overlap)}")
#         combined = structure if len(self) == 0 else merge_class_structure_batches((self.class_structures(), structure))
#         self.replace_all(combined)

#     @torch.no_grad()
#     def replace_all(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         self.class_ids = structure.class_ids.detach().clone()
#         self.learned_identity_bank = structure.learned_identity.detach().clone()
#         self.learned_variation_bank = sanitize_ordered_variation(structure.learned_variation.detach().clone())
#         self.physical_identity_bank = structure.physical_identity.detach().clone()
#         self.physical_variation_bank = sanitize_ordered_variation(structure.physical_variation.detach().clone())
#         self.validate_state()

#     def _row_for_class(self, class_id: int) -> int:
#         class_value = _integer(class_id, "class_id", minimum=0)
#         rows = torch.nonzero(self.class_ids.eq(class_value), as_tuple=False).flatten()
#         if rows.numel() != 1:
#             raise KeyError(f"class structure is not stored uniquely for class {class_value}")
#         return int(rows.item())

#     def class_structure(self, class_id: int) -> ClassStructureView:
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(self.class_ids[row].item()),
#             sample_count=0,
#             learned_identity=self.learned_identity_bank[row].detach().clone(),
#             learned_variation=self.learned_variation_bank[row].detach().clone(),
#             physical_identity=self.physical_identity_bank[row].detach().clone(),
#             physical_variation=self.physical_variation_bank[row].detach().clone(),
#         )

#     def class_structures(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassStructureBatch:
#         self.validate_state()
#         if len(self) == 0:
#             raise RuntimeError("CDS bank is empty")
#         requested = self.seen_class_ids if class_ids is None else _class_id_tuple(class_ids, name="class_ids")
#         rows = torch.tensor(
#             [self._row_for_class(class_id) for class_id in requested],
#             device=self.device,
#             dtype=torch.long,
#         )
#         return ClassStructureBatch(
#             class_ids=self.class_ids.index_select(0, rows).detach().clone(),
#             sample_counts=torch.ones(rows.numel(), device=self.device, dtype=torch.long),
#             learned_identity=self.learned_identity_bank.index_select(0, rows).detach().clone(),
#             learned_variation=self.learned_variation_bank.index_select(0, rows).detach().clone(),
#             physical_identity=self.physical_identity_bank.index_select(0, rows).detach().clone(),
#             physical_variation=self.physical_variation_bank.index_select(0, rows).detach().clone(),
#         )

#     def build_cds(self, structure: ClassStructureBatch) -> ClassDiscriminativeStructureState:
#         self._validate_batch(structure)
#         return ClassDiscriminativeStructureState(
#             classes=structure,
#             relations=derive_class_relations(structure),
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
#             "learned_identity_bank": self.dtype,
#             "learned_variation_bank": self.dtype,
#             "physical_identity_bank": self.dtype,
#             "physical_variation_bank": self.dtype,
#         }
#         # sample_counts belonged to an older checkpoint schema. It was metadata,
#         # not an algorithmic state variable, so consume it without retaining it.
#         state_dict.pop(prefix + "sample_counts", None)
#         for name, dtype in dynamic.items():
#             incoming = state_dict.get(prefix + name)
#             if incoming is not None and getattr(self, name).shape != incoming.shape:
#                 setattr(self, name, torch.empty(incoming.shape, device=self.device, dtype=dtype))
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




# def _differentiable_quantile_dim0(values: Tensor, q: float) -> Tensor:
#     """Differentiable counterpart of the persistent-bank linear quantile.

#     The mathematical estimator is identical to ``_deterministic_quantile_dim0``;
#     unlike the persistent estimator it does not detach or move to CPU, so input
#     gradients can reconstruct a transient support set whose CDS matches storage.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("quantile input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("quantile input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("quantile input contains NaN or Inf")
#     if not 0.0 <= float(q) <= 1.0:
#         raise ValueError("q must be in [0,1]")
#     ordered = torch.sort(value, dim=0, stable=True).values
#     sample_count = int(ordered.size(0))
#     if sample_count == 1:
#         return ordered[0]
#     position = float(q) * float(sample_count - 1)
#     lower = int(position)
#     upper = min(lower + 1, sample_count - 1)
#     weight = position - float(lower)
#     if upper == lower or weight == 0.0:
#         return ordered[lower]
#     return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# def _differentiable_iqr_dim0(values: Tensor) -> Tensor:
#     lower = _differentiable_quantile_dim0(values, 0.25)
#     upper = _differentiable_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _ordered_identity_and_variation_differentiable(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     sample = _validate_ordered_samples(values)
#     identity = _differentiable_quantile_dim0(sample, 0.5)
#     _, length, _ = sample.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)
#     rows: list[Tensor] = []
#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
#         rows.append(_differentiable_iqr_dim0(differences))
#     variation = torch.cat(rows, dim=0)
#     diagonal = torch.arange(length, device=sample.device)
#     variation = variation.clone()
#     variation[diagonal, diagonal, :] = _differentiable_iqr_dim0(sample)
#     variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
#     return identity.contiguous(), variation.contiguous()


# def estimate_structure_batch_differentiable(
#     ordered_features: Tensor,
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     """Estimate the exact CDS formula without detaching generation inputs.

#     This function exists only for transient support reconstruction. Persistent
#     CDS storage continues to use the deterministic CPU estimator above.
#     """
#     learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#     physical = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels)
#     if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if physical.size(0) != learned.size(0) or physical.device != learned.device or physical.dtype != learned.dtype:
#         raise ValueError("Q and physical evidence must share samples/device/dtype")
#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("labels do not match requested classes")

#     counts: list[int] = []
#     learned_identity: list[Tensor] = []
#     learned_variation: list[Tensor] = []
#     physical_identity: list[Tensor] = []
#     physical_variation: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count < 2:
#             raise ValueError("differentiable CDS reconstruction requires at least two supports per class")
#         i_q, v_q = _ordered_identity_and_variation_differentiable(learned[mask])
#         i_p, v_p = _ordered_identity_and_variation_differentiable(physical[mask])
#         counts.append(count)
#         learned_identity.append(i_q)
#         learned_variation.append(v_q)
#         physical_identity.append(i_p)
#         physical_variation.append(v_p)

#     return ClassStructureBatch(
#         class_ids=torch.tensor(requested, device=learned.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=learned.device, dtype=torch.long),
#         learned_identity=torch.stack(learned_identity, dim=0),
#         learned_variation=torch.stack(learned_variation, dim=0),
#         physical_identity=torch.stack(physical_identity, dim=0),
#         physical_variation=torch.stack(physical_variation, dim=0),
#     )

# def whole_cds_pair_risk(
#     cds: ClassDiscriminativeStructureState,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Return effective, learned and physical pair risk from the complete HSI-CDS.

#     Learned risk uses both learned identity and learned ordered-relation structure.
#     Physical risk uses both native-spectrum identity and native spectral relations.
#     Physical similarity may strengthen an existing learned-space collision, but it
#     cannot invent one because the deployed classifier lives in Q space.
#     """
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")
#     learned = cds.relations.learned_combined_proximity
#     physical = cds.relations.physical_combined_proximity
#     if learned.shape != physical.shape or learned.ndim != 2 or learned.size(0) != learned.size(1):
#         raise ValueError("CDS pair-risk inputs must be square [C,C] matrices")
#     if not bool(torch.isfinite(learned).all()) or not bool(torch.isfinite(physical).all()):
#         raise RuntimeError("CDS pair risk contains NaN or Inf")

#     learned = learned.clamp(0.0, 1.0)
#     physical = physical.clamp(0.0, 1.0)
#     effective = combine_learned_physical_relevance(learned, physical)

#     diagonal = torch.arange(learned.size(0), device=learned.device)
#     learned = learned.clone()
#     physical = physical.clone()
#     effective = effective.clone()
#     learned[diagonal, diagonal] = 0.0
#     physical[diagonal, diagonal] = 0.0
#     effective[diagonal, diagonal] = 0.0
#     return effective.contiguous(), learned.contiguous(), physical.contiguous()

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
#     "estimate_structure_batch_differentiable",
#     "merge_class_structure_batches",
#     "normalize_ordered_decision_features",
#     "sanitize_ordered_variation",
#     "ordered_variation_operator",
#     "ordered_cds_support_components",
#     "combine_learned_physical_relevance",
#     "derive_class_relations",
#     "whole_cds_pair_risk",
# ]













# from __future__ import annotations
# from dataclasses import dataclass
# from numbers import Integral
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


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


# def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
#     lower = _deterministic_quantile_dim0(values, 0.25)
#     upper = _deterministic_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _validate_ordered_samples(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) <= 0:
#         raise ValueError("values must be non-empty [N,L,F], L>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("values must be finite real floating point")
#     return value


# def _validate_physical_evidence(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim == 2:
#         value = value.unsqueeze(-1)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) != 1:
#         raise ValueError("physical evidence must be direct full-band [N,B,1], B>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
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
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("variation must be finite real floating point")
#     clean = value.clamp_min(0.0)
#     return (0.5 * (clean + clean.transpose(1, 2))).contiguous()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     return sanitize_ordered_variation(variation)


# def _ordered_identity_and_variation(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     sample = _validate_ordered_samples(values)
#     identity = _deterministic_median_dim0(sample)
#     _, length, _ = sample.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)

#     rows: list[Tensor] = []
#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
#         rows.append(_deterministic_iqr_dim0(differences))
#     variation = torch.cat(rows, dim=0)

#     diagonal = torch.arange(length, device=sample.device)
#     variation[diagonal, diagonal, :] = _deterministic_iqr_dim0(sample)
#     variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
#     return identity.contiguous(), variation.contiguous()


# def _variation_diagonal(variation: Tensor) -> Tensor:
#     value = sanitize_ordered_variation(variation)
#     length = value.size(1)
#     index = torch.arange(length, device=value.device)
#     return value[:, index, index, :]


# def _robust_scale_proximity(separation: Tensor, combined_scale: Tensor) -> Tensor:
#     distance = torch.as_tensor(separation)
#     scale = torch.as_tensor(combined_scale, device=distance.device, dtype=distance.dtype)
#     if distance.shape != scale.shape:
#         raise ValueError("separation and combined_scale must have identical shapes")
#     if (
#         not bool(torch.isfinite(distance).all())
#         or not bool(torch.isfinite(scale).all())
#         or bool((distance < 0).any())
#         or bool((scale < 0).any())
#     ):
#         raise ValueError("proximity inputs must be finite non-negative tensors")
#     eps = torch.finfo(distance.dtype).eps
#     denominator = scale + distance
#     result = scale / denominator.clamp_min(eps)
#     result = torch.where(denominator.le(eps), torch.ones_like(result), result)
#     return result.clamp(0.0, 1.0)


# def _absolute_class_proximity_by_slot(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     if center.ndim != 3 or center.size(0) <= 0 or center.size(1) < 2:
#         raise ValueError("identity must be non-empty [C,L,F]")
#     scale = _variation_diagonal(variation)
#     if scale.shape != center.shape:
#         raise ValueError("variation diagonal must match identity")
#     separation = (center[:, None] - center[None, :]).abs()
#     combined_scale = scale[:, None] + scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale).mean(dim=3)
#     diagonal = torch.arange(center.size(0), device=center.device)
#     proximity = proximity.clone()
#     proximity[diagonal, diagonal, :] = 1.0
#     return proximity.contiguous()


# def _relation_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     robust_scale = sanitize_ordered_variation(variation)
#     if center.ndim != 3:
#         raise ValueError("identity must be [C,L,F]")
#     class_count, length, feature_count = center.shape
#     if robust_scale.shape != (class_count, length, length, feature_count):
#         raise ValueError("variation shape does not match identity")

#     relation_center = center[:, :, None, :] - center[:, None, :, :]
#     separation = (relation_center[:, None] - relation_center[None, :]).abs()
#     combined_scale = robust_scale[:, None] + robust_scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale)
#     upper = torch.triu(
#         torch.ones(length, length, device=center.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     result = proximity[:, :, upper, :].mean(dim=(2, 3))
#     diagonal = torch.arange(class_count, device=center.device)
#     result = result.clone()
#     result[diagonal, diagonal] = 1.0
#     return result.clamp(0.0, 1.0).contiguous()


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     learned_identity: Tensor        # [C,K,D]
#     learned_variation: Tensor       # [C,K,K,D]
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_proximity_by_region: Tensor   # [C,C,K]
#     learned_relation_proximity: Tensor             # [C,C]
#     physical_identity_proximity_by_band: Tensor    # [C,C,B]
#     physical_relation_proximity: Tensor            # [C,C]

#     @property
#     def learned_identity_proximity(self) -> Tensor:
#         return self.learned_identity_proximity_by_region.mean(dim=2)

#     @property
#     def physical_identity_proximity(self) -> Tensor:
#         return self.physical_identity_proximity_by_band.mean(dim=2)

#     @property
#     def learned_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.learned_identity_proximity.clamp_min(0.0)
#             * self.learned_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)

#     @property
#     def physical_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.physical_identity_proximity.clamp_min(0.0)
#             * self.physical_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     learned_identity: Tensor
#     learned_variation: Tensor
#     physical_identity: Tensor
#     physical_variation: Tensor


# def estimate_physical_structure_batch(
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> PhysicalSpectralStructureBatch:
#     physical = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels)
#     if y.shape != (physical.size(0),) or y.dtype != torch.long or y.device != physical.device:
#         raise ValueError("labels must be torch.long [N] on the physical-evidence device")

#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no samples")
#         identity, variation = _ordered_identity_and_variation(physical[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     return PhysicalSpectralStructureBatch(
#         class_ids=torch.tensor(requested, device=physical.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=physical.device, dtype=torch.long),
#         physical_identity=torch.stack(identities, dim=0),
#         physical_variation=torch.stack(variations, dim=0),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical_structure: PhysicalSpectralStructureBatch,
# ) -> ClassStructureBatch:
#     learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#     y = torch.as_tensor(labels)
#     if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if not isinstance(physical_structure, PhysicalSpectralStructureBatch) or physical_structure.class_count <= 0:
#         raise TypeError("physical_structure must be non-empty")
#     if physical_structure.physical_identity.device != learned.device or physical_structure.physical_identity.dtype != learned.dtype:
#         raise ValueError("Q and physical CDS must share device/dtype")

#     requested = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("Q labels do not match cached physical CDS classes")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no Q samples")
#         identity, variation = _ordered_identity_and_variation(learned[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     count_tensor = torch.tensor(counts, device=learned.device, dtype=torch.long)
#     if not torch.equal(count_tensor, physical_structure.sample_counts):
#         raise RuntimeError("learned and physical CDS sample counts disagree")

#     return ClassStructureBatch(
#         class_ids=physical_structure.class_ids,
#         sample_counts=physical_structure.sample_counts,
#         learned_identity=torch.stack(identities, dim=0),
#         learned_variation=torch.stack(variations, dim=0),
#         physical_identity=physical_structure.physical_identity,
#         physical_variation=physical_structure.physical_variation,
#     )


# def estimate_structure_batch(
#     ordered_features: Tensor,
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     physical = estimate_physical_structure_batch(
#         physical_evidence,
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

#     device = values[0].learned_identity.device
#     dtype = values[0].learned_identity.dtype
#     learned_shape = tuple(values[0].learned_identity.shape[1:])
#     learned_variation_shape = tuple(values[0].learned_variation.shape[1:])
#     physical_shape = tuple(values[0].physical_identity.shape[1:])
#     physical_variation_shape = tuple(values[0].physical_variation.shape[1:])

#     seen: set[int] = set()
#     for batch in values:
#         ids = [int(value) for value in batch.class_ids.detach().cpu().tolist()]
#         overlap = seen.intersection(ids)
#         if overlap:
#             raise ValueError(f"class structure batches overlap at {sorted(overlap)}")
#         seen.update(ids)
#         tensors = (
#             batch.learned_identity,
#             batch.learned_variation,
#             batch.physical_identity,
#             batch.physical_variation,
#         )
#         if any(tensor.device != device or tensor.dtype != dtype for tensor in tensors):
#             raise ValueError("class structure batches must share device/dtype")
#         if tuple(batch.learned_identity.shape[1:]) != learned_shape:
#             raise ValueError("learned identity shapes disagree")
#         if tuple(batch.learned_variation.shape[1:]) != learned_variation_shape:
#             raise ValueError("learned variation shapes disagree")
#         if tuple(batch.physical_identity.shape[1:]) != physical_shape:
#             raise ValueError("physical identity shapes disagree")
#         if tuple(batch.physical_variation.shape[1:]) != physical_variation_shape:
#             raise ValueError("physical variation shapes disagree")

#     return ClassStructureBatch(
#         class_ids=torch.cat([batch.class_ids for batch in values], dim=0),
#         sample_counts=torch.cat([batch.sample_counts for batch in values], dim=0),
#         learned_identity=torch.cat([batch.learned_identity for batch in values], dim=0),
#         learned_variation=torch.cat([batch.learned_variation for batch in values], dim=0),
#         physical_identity=torch.cat([batch.physical_identity for batch in values], dim=0),
#         physical_variation=torch.cat([batch.physical_variation for batch in values], dim=0),
#     )


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be non-empty")
#     learned_identity = _absolute_class_proximity_by_slot(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     learned_relation = _relation_class_proximity(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     physical_identity = _absolute_class_proximity_by_slot(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     physical_relation = _relation_class_proximity(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     return ClassRelationState(
#         class_ids=structure.class_ids,
#         learned_identity_proximity_by_region=learned_identity,
#         learned_relation_proximity=learned_relation,
#         physical_identity_proximity_by_band=physical_identity,
#         physical_relation_proximity=physical_relation,
#     )


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Persistent exemplar-free class structure bank."""

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
#         self.spectral_token_count = _integer(spectral_token_count, "spectral_token_count", minimum=2)
#         self.spectral_token_dim = _integer(spectral_token_dim, "spectral_token_dim", minimum=1)
#         self.spectral_bands = _integer(spectral_bands, "spectral_bands", minimum=2)
#         if dtype not in (torch.float32, torch.float64):
#             raise TypeError("CDS bank dtype must be float32 or float64")

#         self.register_buffer(
#             "spectral_band_positions",
#             _band_positions(
#                 spectral_band_positions,
#                 count=self.spectral_bands,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long), persistent=True)
#         self.register_buffer("sample_counts", torch.empty(0, device=device, dtype=torch.long), persistent=True)
#         self.register_buffer(
#             "learned_identity_bank",
#             torch.empty(0, self.spectral_token_count, self.spectral_token_dim, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "learned_variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_identity_bank",
#             torch.empty(0, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_variation_bank",
#             torch.empty(0, self.spectral_bands, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.validate_state()

#     def __len__(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def device(self) -> torch.device:
#         return self.learned_identity_bank.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self.learned_identity_bank.dtype

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(value) for value in self.class_ids.detach().cpu().tolist())

#     def _validate_variation(self, value: Tensor, *, name: str) -> None:
#         if value.ndim != 4 or value.size(1) != value.size(2):
#             raise RuntimeError(f"{name} must be [C,L,L,F]")
#         if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
#             raise RuntimeError(f"{name} must be finite and non-negative")
#         if value.numel() and not torch.allclose(value, value.transpose(1, 2), rtol=1e-6, atol=1e-7):
#             raise RuntimeError(f"{name} must be symmetric over ordered pairs")

#     def validate_state(self) -> bool:
#         class_count = len(self)
#         expected_shapes = (
#             (self.sample_counts, (class_count,)),
#             (self.learned_identity_bank, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 self.learned_variation_bank,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (self.physical_identity_bank, (class_count, self.spectral_bands, 1)),
#             (self.physical_variation_bank, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if self.class_ids.shape != (class_count,):
#             raise RuntimeError("class_ids shape is invalid")
#         if any(tensor.shape != shape for tensor, shape in expected_shapes):
#             raise RuntimeError("CDS bank tensor shape is invalid")
#         if self.class_ids.dtype != torch.long or self.sample_counts.dtype != torch.long:
#             raise RuntimeError("class IDs and sample counts must be torch.long")
#         if self.class_ids.unique().numel() != class_count or bool((self.class_ids < 0).any()):
#             raise RuntimeError("stored class IDs must be unique and non-negative")
#         if class_count and bool((self.sample_counts <= 0).any()):
#             raise RuntimeError("stored sample counts must be positive")
#         floating = (
#             self.learned_identity_bank,
#             self.learned_variation_bank,
#             self.physical_identity_bank,
#             self.physical_variation_bank,
#             self.spectral_band_positions,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in floating):
#             raise RuntimeError("CDS bank floating tensors must share device/dtype")
#         if not all(bool(torch.isfinite(tensor).all()) for tensor in floating):
#             raise RuntimeError("CDS bank contains NaN or Inf")
#         self._validate_variation(self.learned_variation_bank, name="learned_variation_bank")
#         self._validate_variation(self.physical_variation_bank, name="physical_variation_bank")
#         return True

#     def _validate_batch(self, structure: ClassStructureBatch) -> None:
#         if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#             raise TypeError("structure must be non-empty")
#         class_count = structure.class_count
#         expected = (
#             (structure.class_ids, (class_count,)),
#             (structure.sample_counts, (class_count,)),
#             (structure.learned_identity, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 structure.learned_variation,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (structure.physical_identity, (class_count, self.spectral_bands, 1)),
#             (structure.physical_variation, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if any(tensor.shape != shape for tensor, shape in expected):
#             raise ValueError("class structure shape is invalid")
#         tensors = (
#             structure.learned_identity,
#             structure.learned_variation,
#             structure.physical_identity,
#             structure.physical_variation,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in tensors):
#             raise ValueError("class structure and CDS bank must share device/dtype")
#         self._validate_variation(structure.learned_variation, name="learned_variation")
#         self._validate_variation(structure.physical_variation, name="physical_variation")

#     def estimate_physical(
#         self,
#         physical_evidence: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         physical = _validate_physical_evidence(physical_evidence)
#         if physical.device != self.device or physical.dtype != self.dtype:
#             raise ValueError("physical evidence and CDS bank must share device/dtype")
#         if physical.shape[1:] != (self.spectral_bands, 1):
#             raise ValueError("physical evidence does not match native band count")
#         return estimate_physical_structure_batch(physical, labels, class_ids=class_ids)

#     def estimate_classes(
#         self,
#         ordered_features: Tensor,
#         physical_evidence: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         learned = torch.as_tensor(ordered_features)
#         if learned.device != self.device or learned.dtype != self.dtype:
#             raise ValueError("Q and CDS bank must share device/dtype")
#         if physical_structure is None:
#             if physical_evidence is None:
#                 raise ValueError("physical_evidence is required when physical_structure is absent")
#             physical_structure = self.estimate_physical(
#                 physical_evidence,
#                 labels,
#                 class_ids=class_ids,
#             )
#         else:
#             requested = _class_id_tuple(class_ids, name="class_ids")
#             actual = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#             if actual != requested:
#                 raise ValueError("physical_structure class order does not match class_ids")
#         return combine_learned_and_physical_structure(learned, labels, physical_structure)

#     @torch.no_grad()
#     def append_estimate(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         overlap = set(int(value) for value in structure.class_ids.detach().cpu().tolist()).intersection(self.seen_class_ids)
#         if overlap:
#             raise ValueError(f"class structure already exists for {sorted(overlap)}")
#         combined = structure if len(self) == 0 else merge_class_structure_batches((self.class_structures(), structure))
#         self.replace_all(combined)

#     @torch.no_grad()
#     def replace_all(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         self.class_ids = structure.class_ids.detach().clone()
#         self.sample_counts = structure.sample_counts.detach().clone()
#         self.learned_identity_bank = structure.learned_identity.detach().clone()
#         self.learned_variation_bank = sanitize_ordered_variation(structure.learned_variation.detach().clone())
#         self.physical_identity_bank = structure.physical_identity.detach().clone()
#         self.physical_variation_bank = sanitize_ordered_variation(structure.physical_variation.detach().clone())
#         self.validate_state()

#     def _row_for_class(self, class_id: int) -> int:
#         class_value = _integer(class_id, "class_id", minimum=0)
#         rows = torch.nonzero(self.class_ids.eq(class_value), as_tuple=False).flatten()
#         if rows.numel() != 1:
#             raise KeyError(f"class structure is not stored uniquely for class {class_value}")
#         return int(rows.item())

#     def class_structure(self, class_id: int) -> ClassStructureView:
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(self.class_ids[row].item()),
#             sample_count=int(self.sample_counts[row].item()),
#             learned_identity=self.learned_identity_bank[row].detach().clone(),
#             learned_variation=self.learned_variation_bank[row].detach().clone(),
#             physical_identity=self.physical_identity_bank[row].detach().clone(),
#             physical_variation=self.physical_variation_bank[row].detach().clone(),
#         )

#     def class_structures(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassStructureBatch:
#         self.validate_state()
#         if len(self) == 0:
#             raise RuntimeError("CDS bank is empty")
#         requested = self.seen_class_ids if class_ids is None else _class_id_tuple(class_ids, name="class_ids")
#         rows = torch.tensor(
#             [self._row_for_class(class_id) for class_id in requested],
#             device=self.device,
#             dtype=torch.long,
#         )
#         return ClassStructureBatch(
#             class_ids=self.class_ids.index_select(0, rows).detach().clone(),
#             sample_counts=self.sample_counts.index_select(0, rows).detach().clone(),
#             learned_identity=self.learned_identity_bank.index_select(0, rows).detach().clone(),
#             learned_variation=self.learned_variation_bank.index_select(0, rows).detach().clone(),
#             physical_identity=self.physical_identity_bank.index_select(0, rows).detach().clone(),
#             physical_variation=self.physical_variation_bank.index_select(0, rows).detach().clone(),
#         )

#     def build_cds(self, structure: ClassStructureBatch) -> ClassDiscriminativeStructureState:
#         self._validate_batch(structure)
#         return ClassDiscriminativeStructureState(
#             classes=structure,
#             relations=derive_class_relations(structure),
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
#             "learned_identity_bank": self.dtype,
#             "learned_variation_bank": self.dtype,
#             "physical_identity_bank": self.dtype,
#             "physical_variation_bank": self.dtype,
#         }
#         for name, dtype in dynamic.items():
#             incoming = state_dict.get(prefix + name)
#             if incoming is not None and getattr(self, name).shape != incoming.shape:
#                 setattr(self, name, torch.empty(incoming.shape, device=self.device, dtype=dtype))
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




# def _differentiable_quantile_dim0(values: Tensor, q: float) -> Tensor:
#     """Differentiable counterpart of the persistent-bank linear quantile.

#     The mathematical estimator is identical to ``_deterministic_quantile_dim0``;
#     unlike the persistent estimator it does not detach or move to CPU, so input
#     gradients can reconstruct a transient support set whose CDS matches storage.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("quantile input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("quantile input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("quantile input contains NaN or Inf")
#     if not 0.0 <= float(q) <= 1.0:
#         raise ValueError("q must be in [0,1]")
#     ordered = torch.sort(value, dim=0, stable=True).values
#     sample_count = int(ordered.size(0))
#     if sample_count == 1:
#         return ordered[0]
#     position = float(q) * float(sample_count - 1)
#     lower = int(position)
#     upper = min(lower + 1, sample_count - 1)
#     weight = position - float(lower)
#     if upper == lower or weight == 0.0:
#         return ordered[lower]
#     return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# def _differentiable_iqr_dim0(values: Tensor) -> Tensor:
#     lower = _differentiable_quantile_dim0(values, 0.25)
#     upper = _differentiable_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _ordered_identity_and_variation_differentiable(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     sample = _validate_ordered_samples(values)
#     identity = _differentiable_quantile_dim0(sample, 0.5)
#     _, length, _ = sample.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)
#     rows: list[Tensor] = []
#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
#         rows.append(_differentiable_iqr_dim0(differences))
#     variation = torch.cat(rows, dim=0)
#     diagonal = torch.arange(length, device=sample.device)
#     variation = variation.clone()
#     variation[diagonal, diagonal, :] = _differentiable_iqr_dim0(sample)
#     variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
#     return identity.contiguous(), variation.contiguous()


# def estimate_structure_batch_differentiable(
#     ordered_features: Tensor,
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     """Estimate the exact CDS formula without detaching generation inputs.

#     This function exists only for transient support reconstruction. Persistent
#     CDS storage continues to use the deterministic CPU estimator above.
#     """
#     learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#     physical = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels)
#     if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if physical.size(0) != learned.size(0) or physical.device != learned.device or physical.dtype != learned.dtype:
#         raise ValueError("Q and physical evidence must share samples/device/dtype")
#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("labels do not match requested classes")

#     counts: list[int] = []
#     learned_identity: list[Tensor] = []
#     learned_variation: list[Tensor] = []
#     physical_identity: list[Tensor] = []
#     physical_variation: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count < 2:
#             raise ValueError("differentiable CDS reconstruction requires at least two supports per class")
#         i_q, v_q = _ordered_identity_and_variation_differentiable(learned[mask])
#         i_p, v_p = _ordered_identity_and_variation_differentiable(physical[mask])
#         counts.append(count)
#         learned_identity.append(i_q)
#         learned_variation.append(v_q)
#         physical_identity.append(i_p)
#         physical_variation.append(v_p)

#     return ClassStructureBatch(
#         class_ids=torch.tensor(requested, device=learned.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=learned.device, dtype=torch.long),
#         learned_identity=torch.stack(learned_identity, dim=0),
#         learned_variation=torch.stack(learned_variation, dim=0),
#         physical_identity=torch.stack(physical_identity, dim=0),
#         physical_variation=torch.stack(physical_variation, dim=0),
#     )

# def whole_cds_pair_risk(
#     cds: ClassDiscriminativeStructureState,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Return effective, learned and physical pair risk from the complete HSI-CDS.

#     Learned risk uses both learned identity and learned ordered-relation structure.
#     Physical risk uses both native-spectrum identity and native spectral relations.
#     Physical similarity may strengthen an existing learned-space collision, but it
#     cannot invent one because the deployed classifier lives in Q space.
#     """
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")
#     learned = cds.relations.learned_combined_proximity
#     physical = cds.relations.physical_combined_proximity
#     if learned.shape != physical.shape or learned.ndim != 2 or learned.size(0) != learned.size(1):
#         raise ValueError("CDS pair-risk inputs must be square [C,C] matrices")
#     if not bool(torch.isfinite(learned).all()) or not bool(torch.isfinite(physical).all()):
#         raise RuntimeError("CDS pair risk contains NaN or Inf")

#     learned = learned.clamp(0.0, 1.0)
#     physical = physical.clamp(0.0, 1.0)
#     effective = (
#         learned
#         + (1.0 - learned) * learned * physical
#     ).clamp(0.0, 1.0)

#     diagonal = torch.arange(learned.size(0), device=learned.device)
#     learned = learned.clone()
#     physical = physical.clone()
#     effective = effective.clone()
#     learned[diagonal, diagonal] = 0.0
#     physical[diagonal, diagonal] = 0.0
#     effective[diagonal, diagonal] = 0.0
#     return effective.contiguous(), learned.contiguous(), physical.contiguous()

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
#     "estimate_structure_batch_differentiable",
#     "merge_class_structure_batches",
#     "normalize_ordered_decision_features",
#     "sanitize_ordered_variation",
#     "ordered_variation_operator",
#     "derive_class_relations",
#     "whole_cds_pair_risk",
# ]







# from __future__ import annotations

# """HSI Class-Discriminative Structure bank.

# Persistent class state is exactly

#     CDS_c = {I_Q, V_Q, I_P, V_P, N}

# I_Q [K,D]
#     Coordinate-wise median of the globally L2-normalized ordered learned
#     representation Q.

# V_Q [K,K,D]
#     Robust learned scale field.  Diagonal entries are coordinate IQRs and
#     off-diagonal entries are IQRs of ordered token differences Q[k]-Q[l].

# I_P [B,1]
#     Coordinate-wise median of the direct robustly normalized native spectrum.

# V_P [B,B,1]
#     Robust physical scale field.  Diagonal entries are per-band IQRs and
#     off-diagonal entries are IQRs of cross-band differences P[b]-P[r].

# N [scalar]
#     Number of real TRAIN samples used to estimate the class state.

# V_Q and V_P are robust scales.  They are never covariances and never hard
# class-region radii.  Inter-class CDS relations are derived on demand and are
# not persistent memory.
# """

# from dataclasses import dataclass
# from numbers import Integral
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


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


# def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
#     lower = _deterministic_quantile_dim0(values, 0.25)
#     upper = _deterministic_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _validate_ordered_samples(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) <= 0:
#         raise ValueError("values must be non-empty [N,L,F], L>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("values must be finite real floating point")
#     return value


# def _validate_physical_evidence(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim == 2:
#         value = value.unsqueeze(-1)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) != 1:
#         raise ValueError("physical evidence must be direct full-band [N,B,1], B>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
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
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("variation must be finite real floating point")
#     clean = value.clamp_min(0.0)
#     return (0.5 * (clean + clean.transpose(1, 2))).contiguous()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     return sanitize_ordered_variation(variation)


# def _ordered_identity_and_variation(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     sample = _validate_ordered_samples(values)
#     identity = _deterministic_median_dim0(sample)
#     _, length, _ = sample.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)

#     rows: list[Tensor] = []
#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
#         rows.append(_deterministic_iqr_dim0(differences))
#     variation = torch.cat(rows, dim=0)

#     diagonal = torch.arange(length, device=sample.device)
#     variation[diagonal, diagonal, :] = _deterministic_iqr_dim0(sample)
#     variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
#     return identity.contiguous(), variation.contiguous()


# def _variation_diagonal(variation: Tensor) -> Tensor:
#     value = sanitize_ordered_variation(variation)
#     length = value.size(1)
#     index = torch.arange(length, device=value.device)
#     return value[:, index, index, :]


# def _robust_scale_proximity(separation: Tensor, combined_scale: Tensor) -> Tensor:
#     distance = torch.as_tensor(separation)
#     scale = torch.as_tensor(combined_scale, device=distance.device, dtype=distance.dtype)
#     if distance.shape != scale.shape:
#         raise ValueError("separation and combined_scale must have identical shapes")
#     if (
#         not bool(torch.isfinite(distance).all())
#         or not bool(torch.isfinite(scale).all())
#         or bool((distance < 0).any())
#         or bool((scale < 0).any())
#     ):
#         raise ValueError("proximity inputs must be finite non-negative tensors")
#     eps = torch.finfo(distance.dtype).eps
#     denominator = scale + distance
#     result = scale / denominator.clamp_min(eps)
#     result = torch.where(denominator.le(eps), torch.ones_like(result), result)
#     return result.clamp(0.0, 1.0)


# def _absolute_class_proximity_by_slot(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     if center.ndim != 3 or center.size(0) <= 0 or center.size(1) < 2:
#         raise ValueError("identity must be non-empty [C,L,F]")
#     scale = _variation_diagonal(variation)
#     if scale.shape != center.shape:
#         raise ValueError("variation diagonal must match identity")
#     separation = (center[:, None] - center[None, :]).abs()
#     combined_scale = scale[:, None] + scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale).mean(dim=3)
#     diagonal = torch.arange(center.size(0), device=center.device)
#     proximity = proximity.clone()
#     proximity[diagonal, diagonal, :] = 1.0
#     return proximity.contiguous()


# def _relation_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     robust_scale = sanitize_ordered_variation(variation)
#     if center.ndim != 3:
#         raise ValueError("identity must be [C,L,F]")
#     class_count, length, feature_count = center.shape
#     if robust_scale.shape != (class_count, length, length, feature_count):
#         raise ValueError("variation shape does not match identity")

#     relation_center = center[:, :, None, :] - center[:, None, :, :]
#     separation = (relation_center[:, None] - relation_center[None, :]).abs()
#     combined_scale = robust_scale[:, None] + robust_scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale)
#     upper = torch.triu(
#         torch.ones(length, length, device=center.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     result = proximity[:, :, upper, :].mean(dim=(2, 3))
#     diagonal = torch.arange(class_count, device=center.device)
#     result = result.clone()
#     result[diagonal, diagonal] = 1.0
#     return result.clamp(0.0, 1.0).contiguous()


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     learned_identity: Tensor        # [C,K,D]
#     learned_variation: Tensor       # [C,K,K,D]
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_proximity_by_region: Tensor   # [C,C,K]
#     learned_relation_proximity: Tensor             # [C,C]
#     physical_identity_proximity_by_band: Tensor    # [C,C,B]
#     physical_relation_proximity: Tensor            # [C,C]

#     @property
#     def learned_identity_proximity(self) -> Tensor:
#         return self.learned_identity_proximity_by_region.mean(dim=2)

#     @property
#     def physical_identity_proximity(self) -> Tensor:
#         return self.physical_identity_proximity_by_band.mean(dim=2)

#     @property
#     def learned_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.learned_identity_proximity.clamp_min(0.0)
#             * self.learned_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)

#     @property
#     def physical_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.physical_identity_proximity.clamp_min(0.0)
#             * self.physical_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     learned_identity: Tensor
#     learned_variation: Tensor
#     physical_identity: Tensor
#     physical_variation: Tensor


# def estimate_physical_structure_batch(
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> PhysicalSpectralStructureBatch:
#     physical = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels)
#     if y.shape != (physical.size(0),) or y.dtype != torch.long or y.device != physical.device:
#         raise ValueError("labels must be torch.long [N] on the physical-evidence device")

#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no samples")
#         identity, variation = _ordered_identity_and_variation(physical[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     return PhysicalSpectralStructureBatch(
#         class_ids=torch.tensor(requested, device=physical.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=physical.device, dtype=torch.long),
#         physical_identity=torch.stack(identities, dim=0),
#         physical_variation=torch.stack(variations, dim=0),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical_structure: PhysicalSpectralStructureBatch,
# ) -> ClassStructureBatch:
#     learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#     y = torch.as_tensor(labels)
#     if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if not isinstance(physical_structure, PhysicalSpectralStructureBatch) or physical_structure.class_count <= 0:
#         raise TypeError("physical_structure must be non-empty")
#     if physical_structure.physical_identity.device != learned.device or physical_structure.physical_identity.dtype != learned.dtype:
#         raise ValueError("Q and physical CDS must share device/dtype")

#     requested = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("Q labels do not match cached physical CDS classes")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no Q samples")
#         identity, variation = _ordered_identity_and_variation(learned[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     count_tensor = torch.tensor(counts, device=learned.device, dtype=torch.long)
#     if not torch.equal(count_tensor, physical_structure.sample_counts):
#         raise RuntimeError("learned and physical CDS sample counts disagree")

#     return ClassStructureBatch(
#         class_ids=physical_structure.class_ids,
#         sample_counts=physical_structure.sample_counts,
#         learned_identity=torch.stack(identities, dim=0),
#         learned_variation=torch.stack(variations, dim=0),
#         physical_identity=physical_structure.physical_identity,
#         physical_variation=physical_structure.physical_variation,
#     )


# def estimate_structure_batch(
#     ordered_features: Tensor,
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     physical = estimate_physical_structure_batch(
#         physical_evidence,
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

#     device = values[0].learned_identity.device
#     dtype = values[0].learned_identity.dtype
#     learned_shape = tuple(values[0].learned_identity.shape[1:])
#     learned_variation_shape = tuple(values[0].learned_variation.shape[1:])
#     physical_shape = tuple(values[0].physical_identity.shape[1:])
#     physical_variation_shape = tuple(values[0].physical_variation.shape[1:])

#     seen: set[int] = set()
#     for batch in values:
#         ids = [int(value) for value in batch.class_ids.detach().cpu().tolist()]
#         overlap = seen.intersection(ids)
#         if overlap:
#             raise ValueError(f"class structure batches overlap at {sorted(overlap)}")
#         seen.update(ids)
#         tensors = (
#             batch.learned_identity,
#             batch.learned_variation,
#             batch.physical_identity,
#             batch.physical_variation,
#         )
#         if any(tensor.device != device or tensor.dtype != dtype for tensor in tensors):
#             raise ValueError("class structure batches must share device/dtype")
#         if tuple(batch.learned_identity.shape[1:]) != learned_shape:
#             raise ValueError("learned identity shapes disagree")
#         if tuple(batch.learned_variation.shape[1:]) != learned_variation_shape:
#             raise ValueError("learned variation shapes disagree")
#         if tuple(batch.physical_identity.shape[1:]) != physical_shape:
#             raise ValueError("physical identity shapes disagree")
#         if tuple(batch.physical_variation.shape[1:]) != physical_variation_shape:
#             raise ValueError("physical variation shapes disagree")

#     return ClassStructureBatch(
#         class_ids=torch.cat([batch.class_ids for batch in values], dim=0),
#         sample_counts=torch.cat([batch.sample_counts for batch in values], dim=0),
#         learned_identity=torch.cat([batch.learned_identity for batch in values], dim=0),
#         learned_variation=torch.cat([batch.learned_variation for batch in values], dim=0),
#         physical_identity=torch.cat([batch.physical_identity for batch in values], dim=0),
#         physical_variation=torch.cat([batch.physical_variation for batch in values], dim=0),
#     )


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be non-empty")
#     learned_identity = _absolute_class_proximity_by_slot(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     learned_relation = _relation_class_proximity(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     physical_identity = _absolute_class_proximity_by_slot(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     physical_relation = _relation_class_proximity(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     return ClassRelationState(
#         class_ids=structure.class_ids,
#         learned_identity_proximity_by_region=learned_identity,
#         learned_relation_proximity=learned_relation,
#         physical_identity_proximity_by_band=physical_identity,
#         physical_relation_proximity=physical_relation,
#     )


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Persistent exemplar-free class structure bank."""

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
#         self.spectral_token_count = _integer(spectral_token_count, "spectral_token_count", minimum=2)
#         self.spectral_token_dim = _integer(spectral_token_dim, "spectral_token_dim", minimum=1)
#         self.spectral_bands = _integer(spectral_bands, "spectral_bands", minimum=2)
#         if dtype not in (torch.float32, torch.float64):
#             raise TypeError("CDS bank dtype must be float32 or float64")

#         self.register_buffer(
#             "spectral_band_positions",
#             _band_positions(
#                 spectral_band_positions,
#                 count=self.spectral_bands,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long), persistent=True)
#         self.register_buffer("sample_counts", torch.empty(0, device=device, dtype=torch.long), persistent=True)
#         self.register_buffer(
#             "learned_identity_bank",
#             torch.empty(0, self.spectral_token_count, self.spectral_token_dim, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "learned_variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_identity_bank",
#             torch.empty(0, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_variation_bank",
#             torch.empty(0, self.spectral_bands, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.validate_state()

#     def __len__(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def device(self) -> torch.device:
#         return self.learned_identity_bank.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self.learned_identity_bank.dtype

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(value) for value in self.class_ids.detach().cpu().tolist())

#     def _validate_variation(self, value: Tensor, *, name: str) -> None:
#         if value.ndim != 4 or value.size(1) != value.size(2):
#             raise RuntimeError(f"{name} must be [C,L,L,F]")
#         if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
#             raise RuntimeError(f"{name} must be finite and non-negative")
#         if value.numel() and not torch.allclose(value, value.transpose(1, 2), rtol=1e-6, atol=1e-7):
#             raise RuntimeError(f"{name} must be symmetric over ordered pairs")

#     def validate_state(self) -> bool:
#         class_count = len(self)
#         expected_shapes = (
#             (self.sample_counts, (class_count,)),
#             (self.learned_identity_bank, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 self.learned_variation_bank,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (self.physical_identity_bank, (class_count, self.spectral_bands, 1)),
#             (self.physical_variation_bank, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if self.class_ids.shape != (class_count,):
#             raise RuntimeError("class_ids shape is invalid")
#         if any(tensor.shape != shape for tensor, shape in expected_shapes):
#             raise RuntimeError("CDS bank tensor shape is invalid")
#         if self.class_ids.dtype != torch.long or self.sample_counts.dtype != torch.long:
#             raise RuntimeError("class IDs and sample counts must be torch.long")
#         if self.class_ids.unique().numel() != class_count or bool((self.class_ids < 0).any()):
#             raise RuntimeError("stored class IDs must be unique and non-negative")
#         if class_count and bool((self.sample_counts <= 0).any()):
#             raise RuntimeError("stored sample counts must be positive")
#         floating = (
#             self.learned_identity_bank,
#             self.learned_variation_bank,
#             self.physical_identity_bank,
#             self.physical_variation_bank,
#             self.spectral_band_positions,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in floating):
#             raise RuntimeError("CDS bank floating tensors must share device/dtype")
#         if not all(bool(torch.isfinite(tensor).all()) for tensor in floating):
#             raise RuntimeError("CDS bank contains NaN or Inf")
#         self._validate_variation(self.learned_variation_bank, name="learned_variation_bank")
#         self._validate_variation(self.physical_variation_bank, name="physical_variation_bank")
#         return True

#     def _validate_batch(self, structure: ClassStructureBatch) -> None:
#         if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#             raise TypeError("structure must be non-empty")
#         class_count = structure.class_count
#         expected = (
#             (structure.class_ids, (class_count,)),
#             (structure.sample_counts, (class_count,)),
#             (structure.learned_identity, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 structure.learned_variation,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (structure.physical_identity, (class_count, self.spectral_bands, 1)),
#             (structure.physical_variation, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if any(tensor.shape != shape for tensor, shape in expected):
#             raise ValueError("class structure shape is invalid")
#         tensors = (
#             structure.learned_identity,
#             structure.learned_variation,
#             structure.physical_identity,
#             structure.physical_variation,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in tensors):
#             raise ValueError("class structure and CDS bank must share device/dtype")
#         self._validate_variation(structure.learned_variation, name="learned_variation")
#         self._validate_variation(structure.physical_variation, name="physical_variation")

#     def estimate_physical(
#         self,
#         physical_evidence: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         physical = _validate_physical_evidence(physical_evidence)
#         if physical.device != self.device or physical.dtype != self.dtype:
#             raise ValueError("physical evidence and CDS bank must share device/dtype")
#         if physical.shape[1:] != (self.spectral_bands, 1):
#             raise ValueError("physical evidence does not match native band count")
#         return estimate_physical_structure_batch(physical, labels, class_ids=class_ids)

#     def estimate_classes(
#         self,
#         ordered_features: Tensor,
#         physical_evidence: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         learned = torch.as_tensor(ordered_features)
#         if learned.device != self.device or learned.dtype != self.dtype:
#             raise ValueError("Q and CDS bank must share device/dtype")
#         if physical_structure is None:
#             if physical_evidence is None:
#                 raise ValueError("physical_evidence is required when physical_structure is absent")
#             physical_structure = self.estimate_physical(
#                 physical_evidence,
#                 labels,
#                 class_ids=class_ids,
#             )
#         else:
#             requested = _class_id_tuple(class_ids, name="class_ids")
#             actual = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#             if actual != requested:
#                 raise ValueError("physical_structure class order does not match class_ids")
#         return combine_learned_and_physical_structure(learned, labels, physical_structure)

#     @torch.no_grad()
#     def append_estimate(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         overlap = set(int(value) for value in structure.class_ids.detach().cpu().tolist()).intersection(self.seen_class_ids)
#         if overlap:
#             raise ValueError(f"class structure already exists for {sorted(overlap)}")
#         combined = structure if len(self) == 0 else merge_class_structure_batches((self.class_structures(), structure))
#         self.replace_all(combined)

#     @torch.no_grad()
#     def replace_all(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         self.class_ids = structure.class_ids.detach().clone()
#         self.sample_counts = structure.sample_counts.detach().clone()
#         self.learned_identity_bank = structure.learned_identity.detach().clone()
#         self.learned_variation_bank = sanitize_ordered_variation(structure.learned_variation.detach().clone())
#         self.physical_identity_bank = structure.physical_identity.detach().clone()
#         self.physical_variation_bank = sanitize_ordered_variation(structure.physical_variation.detach().clone())
#         self.validate_state()

#     def _row_for_class(self, class_id: int) -> int:
#         class_value = _integer(class_id, "class_id", minimum=0)
#         rows = torch.nonzero(self.class_ids.eq(class_value), as_tuple=False).flatten()
#         if rows.numel() != 1:
#             raise KeyError(f"class structure is not stored uniquely for class {class_value}")
#         return int(rows.item())

#     def class_structure(self, class_id: int) -> ClassStructureView:
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(self.class_ids[row].item()),
#             sample_count=int(self.sample_counts[row].item()),
#             learned_identity=self.learned_identity_bank[row].detach().clone(),
#             learned_variation=self.learned_variation_bank[row].detach().clone(),
#             physical_identity=self.physical_identity_bank[row].detach().clone(),
#             physical_variation=self.physical_variation_bank[row].detach().clone(),
#         )

#     def class_structures(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassStructureBatch:
#         self.validate_state()
#         if len(self) == 0:
#             raise RuntimeError("CDS bank is empty")
#         requested = self.seen_class_ids if class_ids is None else _class_id_tuple(class_ids, name="class_ids")
#         rows = torch.tensor(
#             [self._row_for_class(class_id) for class_id in requested],
#             device=self.device,
#             dtype=torch.long,
#         )
#         return ClassStructureBatch(
#             class_ids=self.class_ids.index_select(0, rows).detach().clone(),
#             sample_counts=self.sample_counts.index_select(0, rows).detach().clone(),
#             learned_identity=self.learned_identity_bank.index_select(0, rows).detach().clone(),
#             learned_variation=self.learned_variation_bank.index_select(0, rows).detach().clone(),
#             physical_identity=self.physical_identity_bank.index_select(0, rows).detach().clone(),
#             physical_variation=self.physical_variation_bank.index_select(0, rows).detach().clone(),
#         )

#     def build_cds(self, structure: ClassStructureBatch) -> ClassDiscriminativeStructureState:
#         self._validate_batch(structure)
#         return ClassDiscriminativeStructureState(
#             classes=structure,
#             relations=derive_class_relations(structure),
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
#             "learned_identity_bank": self.dtype,
#             "learned_variation_bank": self.dtype,
#             "physical_identity_bank": self.dtype,
#             "physical_variation_bank": self.dtype,
#         }
#         for name, dtype in dynamic.items():
#             incoming = state_dict.get(prefix + name)
#             if incoming is not None and getattr(self, name).shape != incoming.shape:
#                 setattr(self, name, torch.empty(incoming.shape, device=self.device, dtype=dtype))
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




# def _differentiable_quantile_dim0(values: Tensor, q: float) -> Tensor:
#     """Differentiable counterpart of the persistent-bank linear quantile.

#     The mathematical estimator is identical to ``_deterministic_quantile_dim0``;
#     unlike the persistent estimator it does not detach or move to CPU, so input
#     gradients can reconstruct a transient support set whose CDS matches storage.
#     """
#     value = torch.as_tensor(values)
#     if value.ndim < 1 or value.size(0) <= 0:
#         raise ValueError("quantile input must have a non-empty leading dimension")
#     if not torch.is_floating_point(value) or value.is_complex():
#         raise TypeError("quantile input must be real floating point")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("quantile input contains NaN or Inf")
#     if not 0.0 <= float(q) <= 1.0:
#         raise ValueError("q must be in [0,1]")
#     ordered = torch.sort(value, dim=0, stable=True).values
#     sample_count = int(ordered.size(0))
#     if sample_count == 1:
#         return ordered[0]
#     position = float(q) * float(sample_count - 1)
#     lower = int(position)
#     upper = min(lower + 1, sample_count - 1)
#     weight = position - float(lower)
#     if upper == lower or weight == 0.0:
#         return ordered[lower]
#     return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


# def _differentiable_iqr_dim0(values: Tensor) -> Tensor:
#     lower = _differentiable_quantile_dim0(values, 0.25)
#     upper = _differentiable_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _ordered_identity_and_variation_differentiable(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     sample = _validate_ordered_samples(values)
#     identity = _differentiable_quantile_dim0(sample, 0.5)
#     _, length, _ = sample.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)
#     rows: list[Tensor] = []
#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
#         rows.append(_differentiable_iqr_dim0(differences))
#     variation = torch.cat(rows, dim=0)
#     diagonal = torch.arange(length, device=sample.device)
#     variation = variation.clone()
#     variation[diagonal, diagonal, :] = _differentiable_iqr_dim0(sample)
#     variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
#     return identity.contiguous(), variation.contiguous()


# def estimate_structure_batch_differentiable(
#     ordered_features: Tensor,
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     """Estimate the exact CDS formula without detaching generation inputs.

#     This function exists only for transient support reconstruction. Persistent
#     CDS storage continues to use the deterministic CPU estimator above.
#     """
#     learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#     physical = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels)
#     if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if physical.size(0) != learned.size(0) or physical.device != learned.device or physical.dtype != learned.dtype:
#         raise ValueError("Q and physical evidence must share samples/device/dtype")
#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(v) for v in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("labels do not match requested classes")

#     counts: list[int] = []
#     learned_identity: list[Tensor] = []
#     learned_variation: list[Tensor] = []
#     physical_identity: list[Tensor] = []
#     physical_variation: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count < 2:
#             raise ValueError("differentiable CDS reconstruction requires at least two supports per class")
#         i_q, v_q = _ordered_identity_and_variation_differentiable(learned[mask])
#         i_p, v_p = _ordered_identity_and_variation_differentiable(physical[mask])
#         counts.append(count)
#         learned_identity.append(i_q)
#         learned_variation.append(v_q)
#         physical_identity.append(i_p)
#         physical_variation.append(v_p)

#     return ClassStructureBatch(
#         class_ids=torch.tensor(requested, device=learned.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=learned.device, dtype=torch.long),
#         learned_identity=torch.stack(learned_identity, dim=0),
#         learned_variation=torch.stack(learned_variation, dim=0),
#         physical_identity=torch.stack(physical_identity, dim=0),
#         physical_variation=torch.stack(physical_variation, dim=0),
#     )

# def whole_cds_pair_risk(
#     cds: ClassDiscriminativeStructureState,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Return effective, learned and physical pair risk from the complete HSI-CDS.

#     Learned risk uses both learned identity and learned ordered-relation structure.
#     Physical risk uses both native-spectrum identity and native spectral relations.
#     Physical similarity may strengthen an existing learned-space collision, but it
#     cannot invent one because the deployed classifier lives in Q space.
#     """
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")
#     learned = cds.relations.learned_combined_proximity
#     physical = cds.relations.physical_combined_proximity
#     if learned.shape != physical.shape or learned.ndim != 2 or learned.size(0) != learned.size(1):
#         raise ValueError("CDS pair-risk inputs must be square [C,C] matrices")
#     if not bool(torch.isfinite(learned).all()) or not bool(torch.isfinite(physical).all()):
#         raise RuntimeError("CDS pair risk contains NaN or Inf")

#     learned = learned.clamp(0.0, 1.0)
#     physical = physical.clamp(0.0, 1.0)
#     effective = (
#         learned
#         + (1.0 - learned) * learned * physical
#     ).clamp(0.0, 1.0)

#     diagonal = torch.arange(learned.size(0), device=learned.device)
#     learned = learned.clone()
#     physical = physical.clone()
#     effective = effective.clone()
#     learned[diagonal, diagonal] = 0.0
#     physical[diagonal, diagonal] = 0.0
#     effective[diagonal, diagonal] = 0.0
#     return effective.contiguous(), learned.contiguous(), physical.contiguous()

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
#     "estimate_structure_batch_differentiable",
#     "merge_class_structure_batches",
#     "normalize_ordered_decision_features",
#     "sanitize_ordered_variation",
#     "ordered_variation_operator",
#     "derive_class_relations",
#     "whole_cds_pair_risk",
# ]









# from __future__ import annotations

# """HSI Class-Discriminative Structure bank.

# Persistent class state is exactly

#     CDS_c = {I_Q, V_Q, I_P, V_P, N}

# I_Q [K,D]
#     Coordinate-wise median of the globally L2-normalized ordered learned
#     representation Q.

# V_Q [K,K,D]
#     Robust learned scale field.  Diagonal entries are coordinate IQRs and
#     off-diagonal entries are IQRs of ordered token differences Q[k]-Q[l].

# I_P [B,1]
#     Coordinate-wise median of the direct robustly normalized native spectrum.

# V_P [B,B,1]
#     Robust physical scale field.  Diagonal entries are per-band IQRs and
#     off-diagonal entries are IQRs of cross-band differences P[b]-P[r].

# N [scalar]
#     Number of real TRAIN samples used to estimate the class state.

# V_Q and V_P are robust scales.  They are never covariances and never hard
# class-region radii.  Inter-class CDS relations are derived on demand and are
# not persistent memory.
# """

# from dataclasses import dataclass
# from numbers import Integral
# from typing import Iterable, Sequence

# import torch
# import torch.nn as nn

# Tensor = torch.Tensor


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


# def _deterministic_iqr_dim0(values: Tensor) -> Tensor:
#     lower = _deterministic_quantile_dim0(values, 0.25)
#     upper = _deterministic_quantile_dim0(values, 0.75)
#     return (upper - lower).clamp_min(0.0)


# def _validate_ordered_samples(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) <= 0:
#         raise ValueError("values must be non-empty [N,L,F], L>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("values must be finite real floating point")
#     return value


# def _validate_physical_evidence(values: Tensor) -> Tensor:
#     value = torch.as_tensor(values)
#     if value.ndim == 2:
#         value = value.unsqueeze(-1)
#     if value.ndim != 3 or value.size(0) <= 0 or value.size(1) < 2 or value.size(2) != 1:
#         raise ValueError("physical evidence must be direct full-band [N,B,1], B>=2")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
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
#     value = torch.as_tensor(variation)
#     if value.ndim != 4 or value.size(1) != value.size(2):
#         raise ValueError("variation must be [C,L,L,F]")
#     if not torch.is_floating_point(value) or value.is_complex() or not bool(torch.isfinite(value).all()):
#         raise ValueError("variation must be finite real floating point")
#     clean = value.clamp_min(0.0)
#     return (0.5 * (clean + clean.transpose(1, 2))).contiguous()


# def ordered_variation_operator(variation: Tensor) -> Tensor:
#     return sanitize_ordered_variation(variation)


# def _ordered_identity_and_variation(
#     values: Tensor,
#     *,
#     row_chunk: int = 8,
# ) -> tuple[Tensor, Tensor]:
#     sample = _validate_ordered_samples(values)
#     identity = _deterministic_median_dim0(sample)
#     _, length, _ = sample.shape
#     chunk = _integer(row_chunk, "row_chunk", minimum=1)

#     rows: list[Tensor] = []
#     for start in range(0, length, chunk):
#         stop = min(start + chunk, length)
#         differences = sample[:, start:stop, None, :] - sample[:, None, :, :]
#         rows.append(_deterministic_iqr_dim0(differences))
#     variation = torch.cat(rows, dim=0)

#     diagonal = torch.arange(length, device=sample.device)
#     variation[diagonal, diagonal, :] = _deterministic_iqr_dim0(sample)
#     variation = sanitize_ordered_variation(variation.unsqueeze(0)).squeeze(0)
#     return identity.contiguous(), variation.contiguous()


# def _variation_diagonal(variation: Tensor) -> Tensor:
#     value = sanitize_ordered_variation(variation)
#     length = value.size(1)
#     index = torch.arange(length, device=value.device)
#     return value[:, index, index, :]


# def _robust_scale_proximity(separation: Tensor, combined_scale: Tensor) -> Tensor:
#     distance = torch.as_tensor(separation)
#     scale = torch.as_tensor(combined_scale, device=distance.device, dtype=distance.dtype)
#     if distance.shape != scale.shape:
#         raise ValueError("separation and combined_scale must have identical shapes")
#     if (
#         not bool(torch.isfinite(distance).all())
#         or not bool(torch.isfinite(scale).all())
#         or bool((distance < 0).any())
#         or bool((scale < 0).any())
#     ):
#         raise ValueError("proximity inputs must be finite non-negative tensors")
#     eps = torch.finfo(distance.dtype).eps
#     denominator = scale + distance
#     result = scale / denominator.clamp_min(eps)
#     result = torch.where(denominator.le(eps), torch.ones_like(result), result)
#     return result.clamp(0.0, 1.0)


# def _absolute_class_proximity_by_slot(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     if center.ndim != 3 or center.size(0) <= 0 or center.size(1) < 2:
#         raise ValueError("identity must be non-empty [C,L,F]")
#     scale = _variation_diagonal(variation)
#     if scale.shape != center.shape:
#         raise ValueError("variation diagonal must match identity")
#     separation = (center[:, None] - center[None, :]).abs()
#     combined_scale = scale[:, None] + scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale).mean(dim=3)
#     diagonal = torch.arange(center.size(0), device=center.device)
#     proximity = proximity.clone()
#     proximity[diagonal, diagonal, :] = 1.0
#     return proximity.contiguous()


# def _relation_class_proximity(identity: Tensor, variation: Tensor) -> Tensor:
#     center = torch.as_tensor(identity)
#     robust_scale = sanitize_ordered_variation(variation)
#     if center.ndim != 3:
#         raise ValueError("identity must be [C,L,F]")
#     class_count, length, feature_count = center.shape
#     if robust_scale.shape != (class_count, length, length, feature_count):
#         raise ValueError("variation shape does not match identity")

#     relation_center = center[:, :, None, :] - center[:, None, :, :]
#     separation = (relation_center[:, None] - relation_center[None, :]).abs()
#     combined_scale = robust_scale[:, None] + robust_scale[None, :]
#     proximity = _robust_scale_proximity(separation, combined_scale)
#     upper = torch.triu(
#         torch.ones(length, length, device=center.device, dtype=torch.bool),
#         diagonal=1,
#     )
#     result = proximity[:, :, upper, :].mean(dim=(2, 3))
#     diagonal = torch.arange(class_count, device=center.device)
#     result = result.clone()
#     result[diagonal, diagonal] = 1.0
#     return result.clamp(0.0, 1.0).contiguous()


# @dataclass(frozen=True)
# class PhysicalSpectralStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassStructureBatch:
#     class_ids: Tensor
#     sample_counts: Tensor
#     learned_identity: Tensor        # [C,K,D]
#     learned_variation: Tensor       # [C,K,K,D]
#     physical_identity: Tensor       # [C,B,1]
#     physical_variation: Tensor      # [C,B,B,1]

#     @property
#     def class_count(self) -> int:
#         return int(self.class_ids.numel())


# @dataclass(frozen=True)
# class ClassRelationState:
#     class_ids: Tensor
#     learned_identity_proximity_by_region: Tensor   # [C,C,K]
#     learned_relation_proximity: Tensor             # [C,C]
#     physical_identity_proximity_by_band: Tensor    # [C,C,B]
#     physical_relation_proximity: Tensor            # [C,C]

#     @property
#     def learned_identity_proximity(self) -> Tensor:
#         return self.learned_identity_proximity_by_region.mean(dim=2)

#     @property
#     def physical_identity_proximity(self) -> Tensor:
#         return self.physical_identity_proximity_by_band.mean(dim=2)

#     @property
#     def learned_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.learned_identity_proximity.clamp_min(0.0)
#             * self.learned_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)

#     @property
#     def physical_combined_proximity(self) -> Tensor:
#         return torch.sqrt(
#             self.physical_identity_proximity.clamp_min(0.0)
#             * self.physical_relation_proximity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)


# @dataclass(frozen=True)
# class ClassDiscriminativeStructureState:
#     classes: ClassStructureBatch
#     relations: ClassRelationState
#     spectral_band_positions: Tensor


# @dataclass(frozen=True)
# class ClassStructureView:
#     class_id: int
#     sample_count: int
#     learned_identity: Tensor
#     learned_variation: Tensor
#     physical_identity: Tensor
#     physical_variation: Tensor


# def estimate_physical_structure_batch(
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> PhysicalSpectralStructureBatch:
#     physical = _validate_physical_evidence(physical_evidence)
#     y = torch.as_tensor(labels)
#     if y.shape != (physical.size(0),) or y.dtype != torch.long or y.device != physical.device:
#         raise ValueError("labels must be torch.long [N] on the physical-evidence device")

#     requested = _class_id_tuple(class_ids, name="class_ids")
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError(f"labels contain classes {observed}, expected {tuple(sorted(requested))}")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no samples")
#         identity, variation = _ordered_identity_and_variation(physical[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     return PhysicalSpectralStructureBatch(
#         class_ids=torch.tensor(requested, device=physical.device, dtype=torch.long),
#         sample_counts=torch.tensor(counts, device=physical.device, dtype=torch.long),
#         physical_identity=torch.stack(identities, dim=0),
#         physical_variation=torch.stack(variations, dim=0),
#     )


# def combine_learned_and_physical_structure(
#     ordered_features: Tensor,
#     labels: Tensor,
#     physical_structure: PhysicalSpectralStructureBatch,
# ) -> ClassStructureBatch:
#     learned = normalize_ordered_decision_features(torch.as_tensor(ordered_features))
#     y = torch.as_tensor(labels)
#     if y.shape != (learned.size(0),) or y.dtype != torch.long or y.device != learned.device:
#         raise ValueError("labels must be torch.long [N] on the Q device")
#     if not isinstance(physical_structure, PhysicalSpectralStructureBatch) or physical_structure.class_count <= 0:
#         raise TypeError("physical_structure must be non-empty")
#     if physical_structure.physical_identity.device != learned.device or physical_structure.physical_identity.dtype != learned.dtype:
#         raise ValueError("Q and physical CDS must share device/dtype")

#     requested = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#     observed = tuple(sorted(int(value) for value in y.unique().detach().cpu().tolist()))
#     if observed != tuple(sorted(requested)):
#         raise ValueError("Q labels do not match cached physical CDS classes")

#     counts: list[int] = []
#     identities: list[Tensor] = []
#     variations: list[Tensor] = []
#     for class_id in requested:
#         mask = y.eq(class_id)
#         count = int(mask.sum().item())
#         if count <= 0:
#             raise RuntimeError(f"class {class_id} has no Q samples")
#         identity, variation = _ordered_identity_and_variation(learned[mask])
#         counts.append(count)
#         identities.append(identity)
#         variations.append(variation)

#     count_tensor = torch.tensor(counts, device=learned.device, dtype=torch.long)
#     if not torch.equal(count_tensor, physical_structure.sample_counts):
#         raise RuntimeError("learned and physical CDS sample counts disagree")

#     return ClassStructureBatch(
#         class_ids=physical_structure.class_ids,
#         sample_counts=physical_structure.sample_counts,
#         learned_identity=torch.stack(identities, dim=0),
#         learned_variation=torch.stack(variations, dim=0),
#         physical_identity=physical_structure.physical_identity,
#         physical_variation=physical_structure.physical_variation,
#     )


# def estimate_structure_batch(
#     ordered_features: Tensor,
#     physical_evidence: Tensor,
#     labels: Tensor,
#     *,
#     class_ids: Sequence[int] | Tensor,
# ) -> ClassStructureBatch:
#     physical = estimate_physical_structure_batch(
#         physical_evidence,
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

#     device = values[0].learned_identity.device
#     dtype = values[0].learned_identity.dtype
#     learned_shape = tuple(values[0].learned_identity.shape[1:])
#     learned_variation_shape = tuple(values[0].learned_variation.shape[1:])
#     physical_shape = tuple(values[0].physical_identity.shape[1:])
#     physical_variation_shape = tuple(values[0].physical_variation.shape[1:])

#     seen: set[int] = set()
#     for batch in values:
#         ids = [int(value) for value in batch.class_ids.detach().cpu().tolist()]
#         overlap = seen.intersection(ids)
#         if overlap:
#             raise ValueError(f"class structure batches overlap at {sorted(overlap)}")
#         seen.update(ids)
#         tensors = (
#             batch.learned_identity,
#             batch.learned_variation,
#             batch.physical_identity,
#             batch.physical_variation,
#         )
#         if any(tensor.device != device or tensor.dtype != dtype for tensor in tensors):
#             raise ValueError("class structure batches must share device/dtype")
#         if tuple(batch.learned_identity.shape[1:]) != learned_shape:
#             raise ValueError("learned identity shapes disagree")
#         if tuple(batch.learned_variation.shape[1:]) != learned_variation_shape:
#             raise ValueError("learned variation shapes disagree")
#         if tuple(batch.physical_identity.shape[1:]) != physical_shape:
#             raise ValueError("physical identity shapes disagree")
#         if tuple(batch.physical_variation.shape[1:]) != physical_variation_shape:
#             raise ValueError("physical variation shapes disagree")

#     return ClassStructureBatch(
#         class_ids=torch.cat([batch.class_ids for batch in values], dim=0),
#         sample_counts=torch.cat([batch.sample_counts for batch in values], dim=0),
#         learned_identity=torch.cat([batch.learned_identity for batch in values], dim=0),
#         learned_variation=torch.cat([batch.learned_variation for batch in values], dim=0),
#         physical_identity=torch.cat([batch.physical_identity for batch in values], dim=0),
#         physical_variation=torch.cat([batch.physical_variation for batch in values], dim=0),
#     )


# def derive_class_relations(structure: ClassStructureBatch) -> ClassRelationState:
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be non-empty")
#     learned_identity = _absolute_class_proximity_by_slot(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     learned_relation = _relation_class_proximity(
#         structure.learned_identity,
#         structure.learned_variation,
#     )
#     physical_identity = _absolute_class_proximity_by_slot(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     physical_relation = _relation_class_proximity(
#         structure.physical_identity,
#         structure.physical_variation,
#     )
#     return ClassRelationState(
#         class_ids=structure.class_ids,
#         learned_identity_proximity_by_region=learned_identity,
#         learned_relation_proximity=learned_relation,
#         physical_identity_proximity_by_band=physical_identity,
#         physical_relation_proximity=physical_relation,
#     )


# class HSIClassDiscriminativeStructureBank(nn.Module):
#     """Persistent exemplar-free class structure bank."""

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
#         self.spectral_token_count = _integer(spectral_token_count, "spectral_token_count", minimum=2)
#         self.spectral_token_dim = _integer(spectral_token_dim, "spectral_token_dim", minimum=1)
#         self.spectral_bands = _integer(spectral_bands, "spectral_bands", minimum=2)
#         if dtype not in (torch.float32, torch.float64):
#             raise TypeError("CDS bank dtype must be float32 or float64")

#         self.register_buffer(
#             "spectral_band_positions",
#             _band_positions(
#                 spectral_band_positions,
#                 count=self.spectral_bands,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer("class_ids", torch.empty(0, device=device, dtype=torch.long), persistent=True)
#         self.register_buffer("sample_counts", torch.empty(0, device=device, dtype=torch.long), persistent=True)
#         self.register_buffer(
#             "learned_identity_bank",
#             torch.empty(0, self.spectral_token_count, self.spectral_token_dim, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "learned_variation_bank",
#             torch.empty(
#                 0,
#                 self.spectral_token_count,
#                 self.spectral_token_count,
#                 self.spectral_token_dim,
#                 device=device,
#                 dtype=dtype,
#             ),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_identity_bank",
#             torch.empty(0, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.register_buffer(
#             "physical_variation_bank",
#             torch.empty(0, self.spectral_bands, self.spectral_bands, 1, device=device, dtype=dtype),
#             persistent=True,
#         )
#         self.validate_state()

#     def __len__(self) -> int:
#         return int(self.class_ids.numel())

#     @property
#     def device(self) -> torch.device:
#         return self.learned_identity_bank.device

#     @property
#     def dtype(self) -> torch.dtype:
#         return self.learned_identity_bank.dtype

#     @property
#     def seen_class_ids(self) -> tuple[int, ...]:
#         return tuple(int(value) for value in self.class_ids.detach().cpu().tolist())

#     def _validate_variation(self, value: Tensor, *, name: str) -> None:
#         if value.ndim != 4 or value.size(1) != value.size(2):
#             raise RuntimeError(f"{name} must be [C,L,L,F]")
#         if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
#             raise RuntimeError(f"{name} must be finite and non-negative")
#         if value.numel() and not torch.allclose(value, value.transpose(1, 2), rtol=1e-6, atol=1e-7):
#             raise RuntimeError(f"{name} must be symmetric over ordered pairs")

#     def validate_state(self) -> bool:
#         class_count = len(self)
#         expected_shapes = (
#             (self.sample_counts, (class_count,)),
#             (self.learned_identity_bank, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 self.learned_variation_bank,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (self.physical_identity_bank, (class_count, self.spectral_bands, 1)),
#             (self.physical_variation_bank, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if self.class_ids.shape != (class_count,):
#             raise RuntimeError("class_ids shape is invalid")
#         if any(tensor.shape != shape for tensor, shape in expected_shapes):
#             raise RuntimeError("CDS bank tensor shape is invalid")
#         if self.class_ids.dtype != torch.long or self.sample_counts.dtype != torch.long:
#             raise RuntimeError("class IDs and sample counts must be torch.long")
#         if self.class_ids.unique().numel() != class_count or bool((self.class_ids < 0).any()):
#             raise RuntimeError("stored class IDs must be unique and non-negative")
#         if class_count and bool((self.sample_counts <= 0).any()):
#             raise RuntimeError("stored sample counts must be positive")
#         floating = (
#             self.learned_identity_bank,
#             self.learned_variation_bank,
#             self.physical_identity_bank,
#             self.physical_variation_bank,
#             self.spectral_band_positions,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in floating):
#             raise RuntimeError("CDS bank floating tensors must share device/dtype")
#         if not all(bool(torch.isfinite(tensor).all()) for tensor in floating):
#             raise RuntimeError("CDS bank contains NaN or Inf")
#         self._validate_variation(self.learned_variation_bank, name="learned_variation_bank")
#         self._validate_variation(self.physical_variation_bank, name="physical_variation_bank")
#         return True

#     def _validate_batch(self, structure: ClassStructureBatch) -> None:
#         if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#             raise TypeError("structure must be non-empty")
#         class_count = structure.class_count
#         expected = (
#             (structure.class_ids, (class_count,)),
#             (structure.sample_counts, (class_count,)),
#             (structure.learned_identity, (class_count, self.spectral_token_count, self.spectral_token_dim)),
#             (
#                 structure.learned_variation,
#                 (class_count, self.spectral_token_count, self.spectral_token_count, self.spectral_token_dim),
#             ),
#             (structure.physical_identity, (class_count, self.spectral_bands, 1)),
#             (structure.physical_variation, (class_count, self.spectral_bands, self.spectral_bands, 1)),
#         )
#         if any(tensor.shape != shape for tensor, shape in expected):
#             raise ValueError("class structure shape is invalid")
#         tensors = (
#             structure.learned_identity,
#             structure.learned_variation,
#             structure.physical_identity,
#             structure.physical_variation,
#         )
#         if any(tensor.device != self.device or tensor.dtype != self.dtype for tensor in tensors):
#             raise ValueError("class structure and CDS bank must share device/dtype")
#         self._validate_variation(structure.learned_variation, name="learned_variation")
#         self._validate_variation(structure.physical_variation, name="physical_variation")

#     def estimate_physical(
#         self,
#         physical_evidence: Tensor,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#     ) -> PhysicalSpectralStructureBatch:
#         physical = _validate_physical_evidence(physical_evidence)
#         if physical.device != self.device or physical.dtype != self.dtype:
#             raise ValueError("physical evidence and CDS bank must share device/dtype")
#         if physical.shape[1:] != (self.spectral_bands, 1):
#             raise ValueError("physical evidence does not match native band count")
#         return estimate_physical_structure_batch(physical, labels, class_ids=class_ids)

#     def estimate_classes(
#         self,
#         ordered_features: Tensor,
#         physical_evidence: Tensor | None,
#         labels: Tensor,
#         *,
#         class_ids: Sequence[int] | Tensor,
#         physical_structure: PhysicalSpectralStructureBatch | None = None,
#     ) -> ClassStructureBatch:
#         learned = torch.as_tensor(ordered_features)
#         if learned.device != self.device or learned.dtype != self.dtype:
#             raise ValueError("Q and CDS bank must share device/dtype")
#         if physical_structure is None:
#             if physical_evidence is None:
#                 raise ValueError("physical_evidence is required when physical_structure is absent")
#             physical_structure = self.estimate_physical(
#                 physical_evidence,
#                 labels,
#                 class_ids=class_ids,
#             )
#         else:
#             requested = _class_id_tuple(class_ids, name="class_ids")
#             actual = tuple(int(value) for value in physical_structure.class_ids.detach().cpu().tolist())
#             if actual != requested:
#                 raise ValueError("physical_structure class order does not match class_ids")
#         return combine_learned_and_physical_structure(learned, labels, physical_structure)

#     @torch.no_grad()
#     def append_estimate(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         overlap = set(int(value) for value in structure.class_ids.detach().cpu().tolist()).intersection(self.seen_class_ids)
#         if overlap:
#             raise ValueError(f"class structure already exists for {sorted(overlap)}")
#         combined = structure if len(self) == 0 else merge_class_structure_batches((self.class_structures(), structure))
#         self.replace_all(combined)

#     @torch.no_grad()
#     def replace_all(self, structure: ClassStructureBatch) -> None:
#         self._validate_batch(structure)
#         self.class_ids = structure.class_ids.detach().clone()
#         self.sample_counts = structure.sample_counts.detach().clone()
#         self.learned_identity_bank = structure.learned_identity.detach().clone()
#         self.learned_variation_bank = sanitize_ordered_variation(structure.learned_variation.detach().clone())
#         self.physical_identity_bank = structure.physical_identity.detach().clone()
#         self.physical_variation_bank = sanitize_ordered_variation(structure.physical_variation.detach().clone())
#         self.validate_state()

#     def _row_for_class(self, class_id: int) -> int:
#         class_value = _integer(class_id, "class_id", minimum=0)
#         rows = torch.nonzero(self.class_ids.eq(class_value), as_tuple=False).flatten()
#         if rows.numel() != 1:
#             raise KeyError(f"class structure is not stored uniquely for class {class_value}")
#         return int(rows.item())

#     def class_structure(self, class_id: int) -> ClassStructureView:
#         row = self._row_for_class(class_id)
#         return ClassStructureView(
#             class_id=int(self.class_ids[row].item()),
#             sample_count=int(self.sample_counts[row].item()),
#             learned_identity=self.learned_identity_bank[row].detach().clone(),
#             learned_variation=self.learned_variation_bank[row].detach().clone(),
#             physical_identity=self.physical_identity_bank[row].detach().clone(),
#             physical_variation=self.physical_variation_bank[row].detach().clone(),
#         )

#     def class_structures(
#         self,
#         class_ids: Sequence[int] | Tensor | None = None,
#     ) -> ClassStructureBatch:
#         self.validate_state()
#         if len(self) == 0:
#             raise RuntimeError("CDS bank is empty")
#         requested = self.seen_class_ids if class_ids is None else _class_id_tuple(class_ids, name="class_ids")
#         rows = torch.tensor(
#             [self._row_for_class(class_id) for class_id in requested],
#             device=self.device,
#             dtype=torch.long,
#         )
#         return ClassStructureBatch(
#             class_ids=self.class_ids.index_select(0, rows).detach().clone(),
#             sample_counts=self.sample_counts.index_select(0, rows).detach().clone(),
#             learned_identity=self.learned_identity_bank.index_select(0, rows).detach().clone(),
#             learned_variation=self.learned_variation_bank.index_select(0, rows).detach().clone(),
#             physical_identity=self.physical_identity_bank.index_select(0, rows).detach().clone(),
#             physical_variation=self.physical_variation_bank.index_select(0, rows).detach().clone(),
#         )

#     def build_cds(self, structure: ClassStructureBatch) -> ClassDiscriminativeStructureState:
#         self._validate_batch(structure)
#         return ClassDiscriminativeStructureState(
#             classes=structure,
#             relations=derive_class_relations(structure),
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
#             "learned_identity_bank": self.dtype,
#             "learned_variation_bank": self.dtype,
#             "physical_identity_bank": self.dtype,
#             "physical_variation_bank": self.dtype,
#         }
#         for name, dtype in dynamic.items():
#             incoming = state_dict.get(prefix + name)
#             if incoming is not None and getattr(self, name).shape != incoming.shape:
#                 setattr(self, name, torch.empty(incoming.shape, device=self.device, dtype=dtype))
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
#     "derive_class_relations",
# ]

