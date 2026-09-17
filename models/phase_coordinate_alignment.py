from __future__ import annotations

"""Phase-local transport of the finalized learned HSI CDS.

The ordered spectral-token axis K is never mixed. Paired legal current samples
encoded by the frozen Phase-(t-1) backbone (Q-) and the current backbone (Q+)
estimate one shared scale per learned channel plus a token-specific translation:

    Q+[n,k,d] ~= s[d] * Q-[n,k,d] + t[k,d].

This restriction is deliberate. The finalized V_Q stores pairwise relation
*deformation*, not covariance. Under the map above,

    V_Q+[k,l,d] = |s[d]| * V_Q-[k,l,d],

so the historical CDS can be transported exactly without a covariance rule.

Current-class residuals and CDS affinities are retained only as diagnostics.
They are never transferred into historical classes. Physical {I_P,V_P} is
phase-stable and is never transported. Persistent memory remains exactly
{I_Q,V_Q,I_P,V_P,N}.
"""

from dataclasses import dataclass
import math

import torch

from models.geometry_bank import (
    ClassStructureBatch,
    derive_class_relations,
    merge_class_structure_batches,
    sanitize_ordered_variation,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class PhaseCoordinateTransform:
    task_scale: Tensor        # [K,D], channel-shared over K and repeated for API compatibility
    task_translation: Tensor  # [K,D]


@dataclass(frozen=True)
class CDSAffinityState:
    old_class_ids: Tensor
    new_class_ids: Tensor
    learned_identity_affinity: Tensor
    learned_variation_affinity: Tensor
    spectral_affinity: Tensor
    joint_affinity: Tensor
    transfer_weights: Tensor


@dataclass(frozen=True)
class CurrentClassResidual:
    class_ids: Tensor
    identity: Tensor
    variation: Tensor


@dataclass(frozen=True)
class PhaseAlignmentDiagnostics:
    task_mapping_error: Tensor
    task_mapping_channel_error_mean: Tensor
    task_mapping_channel_error_max: Tensor
    representation_drift_rms: Tensor
    task_translation_rms: Tensor
    task_scale_deviation_rms: Tensor
    task_identity_relative_shift_mean: Tensor
    task_identity_relative_shift_max: Tensor
    task_variation_relative_change_mean: Tensor
    task_variation_relative_change_max: Tensor
    current_identity_residual_relative_mean: Tensor
    current_identity_residual_relative_max: Tensor
    current_variation_residual_relative_mean: Tensor
    current_variation_residual_relative_max: Tensor
    transferred_identity_residual_relative_mean: Tensor
    transferred_identity_residual_relative_max: Tensor
    transferred_variation_residual_relative_mean: Tensor
    transferred_variation_residual_relative_max: Tensor
    calibrated_identity_relation_rmse: Tensor
    calibrated_variation_relation_rmse: Tensor
    spectral_identity_relation_rmse: Tensor
    spectral_variation_relation_rmse: Tensor
    affinity_entropy_mean: Tensor
    affinity_concentration_mean: Tensor
    affinity_zero_row_fraction: Tensor


def _validate_paired(reference_features: Tensor, current_features: Tensor) -> tuple[Tensor, Tensor]:
    reference = torch.as_tensor(reference_features)
    current = torch.as_tensor(current_features)
    if reference.ndim != 3 or reference.size(0) < 2 or reference.size(1) < 2 or reference.size(2) < 1:
        raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
    if current.shape != reference.shape:
        raise ValueError("current_features must match reference_features")
    if reference.device != current.device or reference.dtype != current.dtype:
        raise ValueError("paired Q-/Q+ features must share device/dtype")
    if not torch.is_floating_point(reference) or reference.is_complex():
        raise TypeError("paired Q-/Q+ features must be real floating point")
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
        raise ValueError("paired Q-/Q+ features contain NaN or Inf")
    return reference, current


def _validate_structure_pair(reference: ClassStructureBatch, current: ClassStructureBatch) -> None:
    if not isinstance(reference, ClassStructureBatch) or not isinstance(current, ClassStructureBatch):
        raise TypeError("current reference/current structures must be ClassStructureBatch")
    if reference.class_count <= 0 or current.class_count != reference.class_count:
        raise ValueError("current reference/current structures have incompatible class counts")
    if not torch.equal(reference.class_ids, current.class_ids):
        raise ValueError("current reference/current structures must use identical class order")
    if reference.identity.shape != current.identity.shape or reference.variation.shape != current.variation.shape:
        raise ValueError("current reference/current learned CDS shapes disagree")
    tensors = (
        reference.identity,
        reference.variation,
        reference.spectral_identity,
        reference.spectral_variation,
        current.identity,
        current.variation,
        current.spectral_identity,
        current.spectral_variation,
    )
    if any(v.device != reference.identity.device or v.dtype != reference.identity.dtype for v in tensors):
        raise ValueError("current reference/current CDS must share device/dtype")
    if not all(bool(torch.isfinite(v).all()) for v in tensors):
        raise ValueError("current reference/current CDS contains NaN or Inf")


def _relative_change(after: Tensor, before: Tensor) -> Tensor:
    eps = torch.finfo(before.dtype).eps
    numerator = torch.linalg.vector_norm((after - before).reshape(after.size(0), -1), dim=1)
    denominator = torch.linalg.vector_norm(before.reshape(before.size(0), -1), dim=1).clamp_min(eps)
    return numerator / denominator


def _relative_residual(residual: Tensor, target: Tensor) -> Tensor:
    eps = torch.finfo(target.dtype).eps
    numerator = torch.linalg.vector_norm(residual.reshape(residual.size(0), -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(target.size(0), -1), dim=1).clamp_min(eps)
    return numerator / denominator


def _offdiag_rmse(after: Tensor, before: Tensor) -> Tensor:
    if after.shape != before.shape or after.ndim < 2 or after.size(0) != after.size(1):
        raise ValueError("relation tensors must share [C,C,...] shape")
    c = int(after.size(0))
    if c <= 1:
        return after.new_zeros(())
    mask = ~torch.eye(c, device=after.device, dtype=torch.bool)
    return torch.sqrt((after[mask] - before[mask]).square().mean().clamp_min(0.0))


@torch.no_grad()
def estimate_task_representation_correspondence(
    reference_features: Tensor,
    current_features: Tensor,
) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
    """Fit a CDS-compatible shared Q- -> Q+ affine drift.

    A single scale ``s[d]`` is fitted for each learned channel across all
    spectral tokens, while ``t[k,d]`` remains token-specific. This is the
    simplest affine family that preserves the pairwise-deformation semantics of
    V_Q exactly. The returned scale is repeated to [K,D] only to preserve the
    surrounding repository API.
    """
    reference, current = _validate_paired(reference_features, current_features)
    x = reference.detach().to(device="cpu", dtype=torch.float64)
    y = current.detach().to(device="cpu", dtype=torch.float64)
    _, k, d = x.shape

    # Token-specific intercepts are removed before estimating one channel scale.
    mean_x = x.mean(dim=0)  # [K,D]
    mean_y = y.mean(dim=0)  # [K,D]
    xc = x - mean_x.unsqueeze(0)
    yc = y - mean_y.unsqueeze(0)

    numerator = (xc * yc).sum(dim=(0, 1))      # [D]
    denominator = xc.square().sum(dim=(0, 1)) # [D]
    eps64 = torch.finfo(torch.float64).eps
    tolerance = eps64 * denominator.max()
    identifiable = denominator > tolerance
    channel_scale = torch.ones_like(denominator)
    channel_scale[identifiable] = numerator[identifiable] / denominator[identifiable]

    scale = channel_scale.unsqueeze(0).expand(k, d).clone()
    translation = mean_y - scale * mean_x
    if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
        raise RuntimeError("CDS-compatible phase map contains NaN or Inf")

    prediction = x * scale.unsqueeze(0) + translation.unsqueeze(0)
    residual = y - prediction
    tiny = torch.finfo(torch.float64).tiny
    channel_residual = residual.permute(2, 0, 1)
    channel_target = y.permute(2, 0, 1)
    channel_error = (
        channel_residual.square().sum(dim=(1, 2))
        / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
    )
    mapping_error = residual.square().sum() / y.square().sum().clamp_min(tiny)
    drift_rms = torch.sqrt((y - x).square().mean().clamp_min(0.0))
    translation_rms = torch.sqrt(translation.square().mean().clamp_min(0.0))
    scale_deviation_rms = torch.sqrt(
        (channel_scale - 1.0).square().mean().clamp_min(0.0)
    )

    device, dtype = reference.device, reference.dtype
    transform = PhaseCoordinateTransform(
        task_scale=scale.to(device=device, dtype=dtype).detach(),
        task_translation=translation.to(device=device, dtype=dtype).detach(),
    )
    diagnostics = {
        "task_mapping_error": mapping_error.to(device=device, dtype=dtype),
        "task_mapping_channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
        "task_mapping_channel_error_max": channel_error.max().to(device=device, dtype=dtype),
        "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
        "task_translation_rms": translation_rms.to(device=device, dtype=dtype),
        "task_scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
    }
    return transform, diagnostics


@torch.no_grad()
def apply_task_projection(
    structure: ClassStructureBatch,
    transform: PhaseCoordinateTransform,
) -> ClassStructureBatch:
    """Transport I_Q and pairwise-deformation V_Q into current Q coordinates.

    For the CDS-compatible map Q+ = s[d] Q- + t[k,d]:

        I_Q+ = s[d] I_Q- + t[k,d]
        V_Q+ = |s[d]| V_Q-

    Translation cancels exactly from the pairwise deformation. No covariance,
    PSD transport, eigendecomposition, or residual migration is involved.
    """
    if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
        raise ValueError("structure must be a non-empty ClassStructureBatch")
    _, k, d = structure.identity.shape
    scale = transform.task_scale
    translation = transform.task_translation
    if scale.shape != (k, d) or translation.shape != (k, d):
        raise ValueError("task affine map shape does not match CDS")
    if any(v.device != structure.identity.device or v.dtype != structure.identity.dtype for v in (scale, translation)):
        raise ValueError("task affine map and CDS must share device/dtype")
    if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
        raise ValueError("task affine map contains NaN or Inf")

    # The transport is valid only when scale is shared over ordered token K.
    channel_scale = scale[0]
    expected = channel_scale.unsqueeze(0).expand_as(scale)
    tolerance = 64.0 * torch.finfo(scale.dtype).eps
    if not bool(torch.allclose(scale, expected, rtol=tolerance, atol=tolerance)):
        raise RuntimeError("V_Q deformation transport requires channel-shared scale over K")

    mapped_identity = structure.identity * scale.unsqueeze(0) + translation.unsqueeze(0)
    mapped_variation = sanitize_ordered_variation(
        structure.variation * channel_scale.abs().view(1, 1, 1, d)
    )
    return ClassStructureBatch(
        class_ids=structure.class_ids.detach().clone(),
        sample_counts=structure.sample_counts.detach().clone(),
        identity=mapped_identity.detach().contiguous(),
        variation=mapped_variation.detach().contiguous(),
        spectral_identity=structure.spectral_identity.detach().clone(),
        spectral_variation=structure.spectral_variation.detach().clone(),
    )


@torch.no_grad()
def estimate_current_class_residual(
    *,
    current_reference: ClassStructureBatch,
    current_structure: ClassStructureBatch,
    transform: PhaseCoordinateTransform,
) -> CurrentClassResidual:
    """Observed class-specific migration not explained by shared phase drift."""
    _validate_structure_pair(current_reference, current_structure)
    predicted = apply_task_projection(current_reference, transform)
    if not torch.allclose(current_reference.spectral_identity, current_structure.spectral_identity, rtol=1e-6, atol=1e-7):
        raise RuntimeError("current physical spectral identity changed across neural coordinates")
    if not torch.allclose(current_reference.spectral_variation, current_structure.spectral_variation, rtol=1e-6, atol=1e-7):
        raise RuntimeError("current physical spectral variation changed across neural coordinates")
    identity = current_structure.identity - predicted.identity
    variation = current_structure.variation - predicted.variation
    if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
        raise RuntimeError("current-class residual contains NaN or Inf")
    return CurrentClassResidual(
        class_ids=current_structure.class_ids.detach().clone(),
        identity=identity.detach(),
        variation=variation.detach(),
    )


@torch.no_grad()
def derive_cds_affinity(
    *,
    historical: ClassStructureBatch,
    current_reference: ClassStructureBatch,
) -> CDSAffinityState:
    """Derive phase-start old/current HSI-CDS affinity for diagnostics.

    The geometry bank already returns normalized L1 similarities in [0,1].
    They are used directly; the previous cosine remapping 0.5*(1+r) was invalid
    for the finalized bank. Affinity is not used to move historical CDS.
    """
    if historical.class_count <= 0 or current_reference.class_count <= 0:
        raise ValueError("historical/current reference CDS must be non-empty")
    if historical.identity.device != current_reference.identity.device or historical.identity.dtype != current_reference.identity.dtype:
        raise ValueError("historical/current reference CDS must share device/dtype")
    old_values = set(int(v) for v in historical.class_ids.detach().cpu().tolist())
    new_values = set(int(v) for v in current_reference.class_ids.detach().cpu().tolist())
    if old_values.intersection(new_values):
        raise ValueError("historical/current reference classes overlap")

    merged = merge_class_structure_batches((historical, current_reference))
    relations = derive_class_relations(merged)
    old_count = historical.class_count

    learned_identity = (
        relations.learned_identity_cosine_by_region[:old_count, old_count:].mean(dim=2)
    ).clamp(0.0, 1.0)
    learned_variation = (
        relations.learned_variation_overlap[:old_count, old_count:]
    ).clamp(0.0, 1.0)
    spectral = (
        relations.spectral_confusability[:old_count, old_count:]
    ).clamp(0.0, 1.0)

    learned = 0.5 * (learned_identity + learned_variation)
    joint = torch.sqrt(
        learned.clamp_min(0.0) * spectral.clamp_min(0.0)
    ).clamp(0.0, 1.0)

    tiny = torch.finfo(joint.dtype).tiny
    row_sum = joint.sum(dim=1, keepdim=True)
    active = row_sum.squeeze(1) > tiny
    weights = torch.zeros_like(joint)
    if bool(active.any()):
        weights[active] = joint[active] / row_sum[active]

    return CDSAffinityState(
        old_class_ids=historical.class_ids.detach().clone(),
        new_class_ids=current_reference.class_ids.detach().clone(),
        learned_identity_affinity=learned_identity.detach(),
        learned_variation_affinity=learned_variation.detach(),
        spectral_affinity=spectral.detach(),
        joint_affinity=joint.detach(),
        transfer_weights=weights.detach(),
    )


@torch.no_grad()
def calibrate_historical_structure(
    *,
    historical: ClassStructureBatch,
    task_mapped_historical: ClassStructureBatch,
    residual: CurrentClassResidual,
    affinity: CDSAffinityState,
) -> ClassStructureBatch:
    """Return the shared-drift transported historical CDS unchanged.

    This function name is retained only for repository compatibility. The
    finalized architecture does *not* transfer current-class residuals into
    unavailable historical classes. ``residual`` and ``affinity`` are validated
    and reported as diagnostics only.
    """
    if not torch.equal(historical.class_ids, task_mapped_historical.class_ids):
        raise ValueError("task-mapped historical class IDs changed")
    if not torch.equal(historical.class_ids, affinity.old_class_ids):
        raise ValueError("affinity old class IDs do not match historical CDS")
    if not torch.equal(residual.class_ids, affinity.new_class_ids):
        raise ValueError("residual class IDs do not match current classes")

    return ClassStructureBatch(
        class_ids=task_mapped_historical.class_ids.detach().clone(),
        sample_counts=task_mapped_historical.sample_counts.detach().clone(),
        identity=task_mapped_historical.identity.detach().clone(),
        variation=task_mapped_historical.variation.detach().clone(),
        spectral_identity=task_mapped_historical.spectral_identity.detach().clone(),
        spectral_variation=task_mapped_historical.spectral_variation.detach().clone(),
    )


def _affinity_diagnostics(affinity: CDSAffinityState, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
    weights = affinity.transfer_weights
    active = weights.sum(dim=1) > 0
    zero_fraction = (~active).to(dtype).mean()
    if not bool(active.any()):
        zero = weights.new_zeros(())
        return zero, zero, zero_fraction
    w = weights[active]
    concentration = w.max(dim=1).values.mean()
    if w.size(1) <= 1:
        entropy = weights.new_zeros(())
    else:
        tiny = torch.finfo(weights.dtype).tiny
        entropy = (-(w * w.clamp_min(tiny).log()).sum(dim=1) / math.log(float(w.size(1)))).mean()
    return entropy.to(dtype), concentration.to(dtype), zero_fraction.to(dtype)


@torch.no_grad()
def align_historical_structure(
    *,
    historical: ClassStructureBatch,
    current_reference: ClassStructureBatch,
    current_structure: ClassStructureBatch,
    reference_features: Tensor,
    current_features: Tensor,
    affinity: CDSAffinityState | None = None,
) -> tuple[
    ClassStructureBatch,
    ClassStructureBatch,
    PhaseCoordinateTransform,
    CDSAffinityState,
    CurrentClassResidual,
    PhaseAlignmentDiagnostics,
]:
    """Transport historical CDS with shared observable drift only; no residual transfer."""
    transform, mapping = estimate_task_representation_correspondence(reference_features, current_features)
    task_mapped = apply_task_projection(historical, transform)
    residual = estimate_current_class_residual(
        current_reference=current_reference,
        current_structure=current_structure,
        transform=transform,
    )
    relation = affinity or derive_cds_affinity(
        historical=historical,
        current_reference=current_reference,
    )
    calibrated = calibrate_historical_structure(
        historical=historical,
        task_mapped_historical=task_mapped,
        residual=residual,
        affinity=relation,
    )

    task_identity_shift = _relative_change(task_mapped.identity, historical.identity)
    task_variation_change = _relative_change(task_mapped.variation, historical.variation)
    current_identity_residual = _relative_residual(residual.identity, current_structure.identity)
    current_variation_residual = _relative_residual(residual.variation, current_structure.variation)
    transferred_identity = calibrated.identity - task_mapped.identity
    transferred_variation = calibrated.variation - task_mapped.variation
    transferred_identity_relative = _relative_residual(transferred_identity, task_mapped.identity)
    transferred_variation_relative = _relative_residual(transferred_variation, task_mapped.variation)

    before = derive_class_relations(historical)
    after = derive_class_relations(calibrated)
    identity_rmse = _offdiag_rmse(after.learned_identity_cosine_by_region, before.learned_identity_cosine_by_region)
    variation_rmse = _offdiag_rmse(after.learned_variation_overlap, before.learned_variation_overlap)
    spectral_identity_rmse = _offdiag_rmse(after.spectral_identity_cosine_by_component, before.spectral_identity_cosine_by_component)
    spectral_variation_rmse = _offdiag_rmse(after.spectral_variation_overlap, before.spectral_variation_overlap)
    entropy, concentration, zero_fraction = _affinity_diagnostics(relation, historical.identity.dtype)

    diagnostics = PhaseAlignmentDiagnostics(
        task_mapping_error=mapping["task_mapping_error"],
        task_mapping_channel_error_mean=mapping["task_mapping_channel_error_mean"],
        task_mapping_channel_error_max=mapping["task_mapping_channel_error_max"],
        representation_drift_rms=mapping["representation_drift_rms"],
        task_translation_rms=mapping["task_translation_rms"],
        task_scale_deviation_rms=mapping["task_scale_deviation_rms"],
        task_identity_relative_shift_mean=task_identity_shift.mean(),
        task_identity_relative_shift_max=task_identity_shift.max(),
        task_variation_relative_change_mean=task_variation_change.mean(),
        task_variation_relative_change_max=task_variation_change.max(),
        current_identity_residual_relative_mean=current_identity_residual.mean(),
        current_identity_residual_relative_max=current_identity_residual.max(),
        current_variation_residual_relative_mean=current_variation_residual.mean(),
        current_variation_residual_relative_max=current_variation_residual.max(),
        transferred_identity_residual_relative_mean=transferred_identity_relative.mean(),
        transferred_identity_residual_relative_max=transferred_identity_relative.max(),
        transferred_variation_residual_relative_mean=transferred_variation_relative.mean(),
        transferred_variation_residual_relative_max=transferred_variation_relative.max(),
        calibrated_identity_relation_rmse=identity_rmse,
        calibrated_variation_relation_rmse=variation_rmse,
        spectral_identity_relation_rmse=spectral_identity_rmse,
        spectral_variation_relation_rmse=spectral_variation_rmse,
        affinity_entropy_mean=entropy,
        affinity_concentration_mean=concentration,
        affinity_zero_row_fraction=zero_fraction,
    )
    for field in diagnostics.__dataclass_fields__:
        if not bool(torch.isfinite(getattr(diagnostics, field))):
            raise RuntimeError(f"phase alignment diagnostic {field} is NaN or Inf")
    return task_mapped, calibrated, transform, relation, residual, diagnostics


__all__ = [
    "PhaseCoordinateTransform",
    "CDSAffinityState",
    "CurrentClassResidual",
    "PhaseAlignmentDiagnostics",
    "estimate_task_representation_correspondence",
    "apply_task_projection",
    "estimate_current_class_residual",
    "derive_cds_affinity",
    "calibrate_historical_structure",
    "align_historical_structure",
]












# from __future__ import annotations

# """Phase-local transport of the finalized learned HSI CDS.

# The ordered spectral-token axis K is never mixed. Paired legal current samples
# encoded by the frozen Phase-(t-1) backbone (Q-) and the current backbone (Q+)
# estimate one shared scale per learned channel plus a token-specific translation:

#     Q+[n,k,d] ~= s[d] * Q-[n,k,d] + t[k,d].

# This restriction is deliberate. The finalized V_Q stores pairwise relation
# *deformation*, not covariance. Under the map above,

#     V_Q+[k,l,d] = |s[d]| * V_Q-[k,l,d],

# so the historical CDS can be transported exactly without a covariance rule.

# Current-class residuals and CDS affinities are retained only as diagnostics.
# They are never transferred into historical classes. Physical {I_P,V_P} is
# phase-stable and is never transported. Persistent memory remains exactly
# {I_Q,V_Q,I_P,V_P,N}.
# """

# from dataclasses import dataclass
# import math

# import torch

# from models.geometry_bank import (
#     ClassStructureBatch,
#     derive_class_relations,
#     merge_class_structure_batches,
#     sanitize_ordered_variation,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class PhaseCoordinateTransform:
#     task_scale: Tensor        # [K,D], channel-shared over K and repeated for API compatibility
#     task_translation: Tensor  # [K,D]


# @dataclass(frozen=True)
# class CDSAffinityState:
#     old_class_ids: Tensor
#     new_class_ids: Tensor
#     learned_identity_affinity: Tensor
#     learned_variation_affinity: Tensor
#     spectral_affinity: Tensor
#     joint_affinity: Tensor
#     transfer_weights: Tensor


# @dataclass(frozen=True)
# class CurrentClassResidual:
#     class_ids: Tensor
#     identity: Tensor
#     variation: Tensor


# @dataclass(frozen=True)
# class PhaseAlignmentDiagnostics:
#     task_mapping_error: Tensor
#     task_mapping_channel_error_mean: Tensor
#     task_mapping_channel_error_max: Tensor
#     representation_drift_rms: Tensor
#     task_translation_rms: Tensor
#     task_scale_deviation_rms: Tensor
#     task_identity_relative_shift_mean: Tensor
#     task_identity_relative_shift_max: Tensor
#     task_variation_relative_change_mean: Tensor
#     task_variation_relative_change_max: Tensor
#     current_identity_residual_relative_mean: Tensor
#     current_identity_residual_relative_max: Tensor
#     current_variation_residual_relative_mean: Tensor
#     current_variation_residual_relative_max: Tensor
#     transferred_identity_residual_relative_mean: Tensor
#     transferred_identity_residual_relative_max: Tensor
#     transferred_variation_residual_relative_mean: Tensor
#     transferred_variation_residual_relative_max: Tensor
#     calibrated_identity_relation_rmse: Tensor
#     calibrated_variation_relation_rmse: Tensor
#     spectral_identity_relation_rmse: Tensor
#     spectral_variation_relation_rmse: Tensor
#     affinity_entropy_mean: Tensor
#     affinity_concentration_mean: Tensor
#     affinity_zero_row_fraction: Tensor


# def _validate_paired(reference_features: Tensor, current_features: Tensor) -> tuple[Tensor, Tensor]:
#     reference = torch.as_tensor(reference_features)
#     current = torch.as_tensor(current_features)
#     if reference.ndim != 3 or reference.size(0) < 2 or reference.size(1) < 2 or reference.size(2) < 1:
#         raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if reference.device != current.device or reference.dtype != current.dtype:
#         raise ValueError("paired Q-/Q+ features must share device/dtype")
#     if not torch.is_floating_point(reference) or reference.is_complex():
#         raise TypeError("paired Q-/Q+ features must be real floating point")
#     if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
#         raise ValueError("paired Q-/Q+ features contain NaN or Inf")
#     return reference, current


# def _validate_structure_pair(reference: ClassStructureBatch, current: ClassStructureBatch) -> None:
#     if not isinstance(reference, ClassStructureBatch) or not isinstance(current, ClassStructureBatch):
#         raise TypeError("current reference/current structures must be ClassStructureBatch")
#     if reference.class_count <= 0 or current.class_count != reference.class_count:
#         raise ValueError("current reference/current structures have incompatible class counts")
#     if not torch.equal(reference.class_ids, current.class_ids):
#         raise ValueError("current reference/current structures must use identical class order")
#     if reference.identity.shape != current.identity.shape or reference.variation.shape != current.variation.shape:
#         raise ValueError("current reference/current learned CDS shapes disagree")
#     tensors = (
#         reference.identity,
#         reference.variation,
#         reference.spectral_identity,
#         reference.spectral_variation,
#         current.identity,
#         current.variation,
#         current.spectral_identity,
#         current.spectral_variation,
#     )
#     if any(v.device != reference.identity.device or v.dtype != reference.identity.dtype for v in tensors):
#         raise ValueError("current reference/current CDS must share device/dtype")
#     if not all(bool(torch.isfinite(v).all()) for v in tensors):
#         raise ValueError("current reference/current CDS contains NaN or Inf")


# def _relative_change(after: Tensor, before: Tensor) -> Tensor:
#     eps = torch.finfo(before.dtype).eps
#     numerator = torch.linalg.vector_norm((after - before).reshape(after.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(before.reshape(before.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _relative_residual(residual: Tensor, target: Tensor) -> Tensor:
#     eps = torch.finfo(target.dtype).eps
#     numerator = torch.linalg.vector_norm(residual.reshape(residual.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(target.reshape(target.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _offdiag_rmse(after: Tensor, before: Tensor) -> Tensor:
#     if after.shape != before.shape or after.ndim < 2 or after.size(0) != after.size(1):
#         raise ValueError("relation tensors must share [C,C,...] shape")
#     c = int(after.size(0))
#     if c <= 1:
#         return after.new_zeros(())
#     mask = ~torch.eye(c, device=after.device, dtype=torch.bool)
#     return torch.sqrt((after[mask] - before[mask]).square().mean().clamp_min(0.0))


# @torch.no_grad()
# def estimate_task_representation_correspondence(
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
#     """Fit a CDS-compatible shared Q- -> Q+ affine drift.

#     A single scale ``s[d]`` is fitted for each learned channel across all
#     spectral tokens, while ``t[k,d]`` remains token-specific. This is the
#     simplest affine family that preserves the pairwise-deformation semantics of
#     V_Q exactly. The returned scale is repeated to [K,D] only to preserve the
#     surrounding repository API.
#     """
#     reference, current = _validate_paired(reference_features, current_features)
#     x = reference.detach().to(device="cpu", dtype=torch.float64)
#     y = current.detach().to(device="cpu", dtype=torch.float64)
#     _, k, d = x.shape

#     # Token-specific intercepts are removed before estimating one channel scale.
#     mean_x = x.mean(dim=0)  # [K,D]
#     mean_y = y.mean(dim=0)  # [K,D]
#     xc = x - mean_x.unsqueeze(0)
#     yc = y - mean_y.unsqueeze(0)

#     numerator = (xc * yc).sum(dim=(0, 1))      # [D]
#     denominator = xc.square().sum(dim=(0, 1)) # [D]
#     eps64 = torch.finfo(torch.float64).eps
#     tolerance = eps64 * denominator.max()
#     identifiable = denominator > tolerance
#     channel_scale = torch.ones_like(denominator)
#     channel_scale[identifiable] = numerator[identifiable] / denominator[identifiable]

#     scale = channel_scale.unsqueeze(0).expand(k, d).clone()
#     translation = mean_y - scale * mean_x
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise RuntimeError("CDS-compatible phase map contains NaN or Inf")

#     prediction = x * scale.unsqueeze(0) + translation.unsqueeze(0)
#     residual = y - prediction
#     tiny = torch.finfo(torch.float64).tiny
#     channel_residual = residual.permute(2, 0, 1)
#     channel_target = y.permute(2, 0, 1)
#     channel_error = (
#         channel_residual.square().sum(dim=(1, 2))
#         / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
#     )
#     mapping_error = residual.square().sum() / y.square().sum().clamp_min(tiny)
#     drift_rms = torch.sqrt((y - x).square().mean().clamp_min(0.0))
#     translation_rms = torch.sqrt(translation.square().mean().clamp_min(0.0))
#     scale_deviation_rms = torch.sqrt(
#         (channel_scale - 1.0).square().mean().clamp_min(0.0)
#     )

#     device, dtype = reference.device, reference.dtype
#     transform = PhaseCoordinateTransform(
#         task_scale=scale.to(device=device, dtype=dtype).detach(),
#         task_translation=translation.to(device=device, dtype=dtype).detach(),
#     )
#     diagnostics = {
#         "task_mapping_error": mapping_error.to(device=device, dtype=dtype),
#         "task_mapping_channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
#         "task_mapping_channel_error_max": channel_error.max().to(device=device, dtype=dtype),
#         "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
#         "task_translation_rms": translation_rms.to(device=device, dtype=dtype),
#         "task_scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
#     }
#     return transform, diagnostics


# @torch.no_grad()
# def apply_task_projection(
#     structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> ClassStructureBatch:
#     """Transport I_Q and pairwise-deformation V_Q into current Q coordinates.

#     For the CDS-compatible map Q+ = s[d] Q- + t[k,d]:

#         I_Q+ = s[d] I_Q- + t[k,d]
#         V_Q+ = |s[d]| V_Q-

#     Translation cancels exactly from the pairwise deformation. No covariance,
#     PSD transport, eigendecomposition, or residual migration is involved.
#     """
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     _, k, d = structure.identity.shape
#     scale = transform.task_scale
#     translation = transform.task_translation
#     if scale.shape != (k, d) or translation.shape != (k, d):
#         raise ValueError("task affine map shape does not match CDS")
#     if any(v.device != structure.identity.device or v.dtype != structure.identity.dtype for v in (scale, translation)):
#         raise ValueError("task affine map and CDS must share device/dtype")
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise ValueError("task affine map contains NaN or Inf")

#     # The transport is valid only when scale is shared over ordered token K.
#     channel_scale = scale[0]
#     expected = channel_scale.unsqueeze(0).expand_as(scale)
#     tolerance = 64.0 * torch.finfo(scale.dtype).eps
#     if not bool(torch.allclose(scale, expected, rtol=tolerance, atol=tolerance)):
#         raise RuntimeError("V_Q deformation transport requires channel-shared scale over K")

#     mapped_identity = structure.identity * scale.unsqueeze(0) + translation.unsqueeze(0)
#     mapped_variation = sanitize_ordered_variation(
#         structure.variation * channel_scale.abs().view(1, 1, 1, d)
#     )
#     return ClassStructureBatch(
#         class_ids=structure.class_ids.detach().clone(),
#         sample_counts=structure.sample_counts.detach().clone(),
#         identity=mapped_identity.detach().contiguous(),
#         variation=mapped_variation.detach().contiguous(),
#         spectral_identity=structure.spectral_identity.detach().clone(),
#         spectral_variation=structure.spectral_variation.detach().clone(),
#     )


# @torch.no_grad()
# def estimate_current_class_residual(
#     *,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> CurrentClassResidual:
#     """Observed class-specific migration not explained by shared phase drift."""
#     _validate_structure_pair(current_reference, current_structure)
#     predicted = apply_task_projection(current_reference, transform)
#     if not torch.allclose(current_reference.spectral_identity, current_structure.spectral_identity, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral identity changed across neural coordinates")
#     if not torch.allclose(current_reference.spectral_variation, current_structure.spectral_variation, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral variation changed across neural coordinates")
#     identity = current_structure.identity - predicted.identity
#     variation = current_structure.variation - predicted.variation
#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
#         raise RuntimeError("current-class residual contains NaN or Inf")
#     return CurrentClassResidual(
#         class_ids=current_structure.class_ids.detach().clone(),
#         identity=identity.detach(),
#         variation=variation.detach(),
#     )


# @torch.no_grad()
# def derive_cds_affinity(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
# ) -> CDSAffinityState:
#     """Derive phase-start old/current HSI-CDS affinity for diagnostics.

#     The geometry bank already returns normalized L1 similarities in [0,1].
#     They are used directly; the previous cosine remapping 0.5*(1+r) was invalid
#     for the finalized bank. Affinity is not used to move historical CDS.
#     """
#     if historical.class_count <= 0 or current_reference.class_count <= 0:
#         raise ValueError("historical/current reference CDS must be non-empty")
#     if historical.identity.device != current_reference.identity.device or historical.identity.dtype != current_reference.identity.dtype:
#         raise ValueError("historical/current reference CDS must share device/dtype")
#     old_values = set(int(v) for v in historical.class_ids.detach().cpu().tolist())
#     new_values = set(int(v) for v in current_reference.class_ids.detach().cpu().tolist())
#     if old_values.intersection(new_values):
#         raise ValueError("historical/current reference classes overlap")

#     merged = merge_class_structure_batches((historical, current_reference))
#     relations = derive_class_relations(merged)
#     old_count = historical.class_count

#     learned_identity = (
#         relations.learned_identity_similarity_by_region[:old_count, old_count:].mean(dim=2)
#     ).clamp(0.0, 1.0)
#     learned_variation = (
#         relations.learned_variation_overlap[:old_count, old_count:]
#     ).clamp(0.0, 1.0)
#     spectral = (
#         relations.spectral_confusability[:old_count, old_count:]
#     ).clamp(0.0, 1.0)

#     learned = 0.5 * (learned_identity + learned_variation)
#     joint = torch.sqrt(
#         learned.clamp_min(0.0) * spectral.clamp_min(0.0)
#     ).clamp(0.0, 1.0)

#     tiny = torch.finfo(joint.dtype).tiny
#     row_sum = joint.sum(dim=1, keepdim=True)
#     active = row_sum.squeeze(1) > tiny
#     weights = torch.zeros_like(joint)
#     if bool(active.any()):
#         weights[active] = joint[active] / row_sum[active]

#     return CDSAffinityState(
#         old_class_ids=historical.class_ids.detach().clone(),
#         new_class_ids=current_reference.class_ids.detach().clone(),
#         learned_identity_affinity=learned_identity.detach(),
#         learned_variation_affinity=learned_variation.detach(),
#         spectral_affinity=spectral.detach(),
#         joint_affinity=joint.detach(),
#         transfer_weights=weights.detach(),
#     )


# @torch.no_grad()
# def calibrate_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     task_mapped_historical: ClassStructureBatch,
#     residual: CurrentClassResidual,
#     affinity: CDSAffinityState,
# ) -> ClassStructureBatch:
#     """Return the shared-drift transported historical CDS unchanged.

#     This function name is retained only for repository compatibility. The
#     finalized architecture does *not* transfer current-class residuals into
#     unavailable historical classes. ``residual`` and ``affinity`` are validated
#     and reported as diagnostics only.
#     """
#     if not torch.equal(historical.class_ids, task_mapped_historical.class_ids):
#         raise ValueError("task-mapped historical class IDs changed")
#     if not torch.equal(historical.class_ids, affinity.old_class_ids):
#         raise ValueError("affinity old class IDs do not match historical CDS")
#     if not torch.equal(residual.class_ids, affinity.new_class_ids):
#         raise ValueError("residual class IDs do not match current classes")

#     return ClassStructureBatch(
#         class_ids=task_mapped_historical.class_ids.detach().clone(),
#         sample_counts=task_mapped_historical.sample_counts.detach().clone(),
#         identity=task_mapped_historical.identity.detach().clone(),
#         variation=task_mapped_historical.variation.detach().clone(),
#         spectral_identity=task_mapped_historical.spectral_identity.detach().clone(),
#         spectral_variation=task_mapped_historical.spectral_variation.detach().clone(),
#     )


# def _affinity_diagnostics(affinity: CDSAffinityState, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
#     weights = affinity.transfer_weights
#     active = weights.sum(dim=1) > 0
#     zero_fraction = (~active).to(dtype).mean()
#     if not bool(active.any()):
#         zero = weights.new_zeros(())
#         return zero, zero, zero_fraction
#     w = weights[active]
#     concentration = w.max(dim=1).values.mean()
#     if w.size(1) <= 1:
#         entropy = weights.new_zeros(())
#     else:
#         tiny = torch.finfo(weights.dtype).tiny
#         entropy = (-(w * w.clamp_min(tiny).log()).sum(dim=1) / math.log(float(w.size(1)))).mean()
#     return entropy.to(dtype), concentration.to(dtype), zero_fraction.to(dtype)


# @torch.no_grad()
# def align_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     reference_features: Tensor,
#     current_features: Tensor,
#     affinity: CDSAffinityState | None = None,
# ) -> tuple[
#     ClassStructureBatch,
#     ClassStructureBatch,
#     PhaseCoordinateTransform,
#     CDSAffinityState,
#     CurrentClassResidual,
#     PhaseAlignmentDiagnostics,
# ]:
#     """Transport historical CDS with shared observable drift only; no residual transfer."""
#     transform, mapping = estimate_task_representation_correspondence(reference_features, current_features)
#     task_mapped = apply_task_projection(historical, transform)
#     residual = estimate_current_class_residual(
#         current_reference=current_reference,
#         current_structure=current_structure,
#         transform=transform,
#     )
#     relation = affinity or derive_cds_affinity(
#         historical=historical,
#         current_reference=current_reference,
#     )
#     calibrated = calibrate_historical_structure(
#         historical=historical,
#         task_mapped_historical=task_mapped,
#         residual=residual,
#         affinity=relation,
#     )

#     task_identity_shift = _relative_change(task_mapped.identity, historical.identity)
#     task_variation_change = _relative_change(task_mapped.variation, historical.variation)
#     current_identity_residual = _relative_residual(residual.identity, current_structure.identity)
#     current_variation_residual = _relative_residual(residual.variation, current_structure.variation)
#     transferred_identity = calibrated.identity - task_mapped.identity
#     transferred_variation = calibrated.variation - task_mapped.variation
#     transferred_identity_relative = _relative_residual(transferred_identity, task_mapped.identity)
#     transferred_variation_relative = _relative_residual(transferred_variation, task_mapped.variation)

#     before = derive_class_relations(historical)
#     after = derive_class_relations(calibrated)
#     identity_rmse = _offdiag_rmse(after.learned_identity_similarity_by_region, before.learned_identity_similarity_by_region)
#     variation_rmse = _offdiag_rmse(after.learned_variation_overlap, before.learned_variation_overlap)
#     spectral_identity_rmse = _offdiag_rmse(after.spectral_identity_similarity_by_component, before.spectral_identity_similarity_by_component)
#     spectral_variation_rmse = _offdiag_rmse(after.spectral_variation_overlap, before.spectral_variation_overlap)
#     entropy, concentration, zero_fraction = _affinity_diagnostics(relation, historical.identity.dtype)

#     diagnostics = PhaseAlignmentDiagnostics(
#         task_mapping_error=mapping["task_mapping_error"],
#         task_mapping_channel_error_mean=mapping["task_mapping_channel_error_mean"],
#         task_mapping_channel_error_max=mapping["task_mapping_channel_error_max"],
#         representation_drift_rms=mapping["representation_drift_rms"],
#         task_translation_rms=mapping["task_translation_rms"],
#         task_scale_deviation_rms=mapping["task_scale_deviation_rms"],
#         task_identity_relative_shift_mean=task_identity_shift.mean(),
#         task_identity_relative_shift_max=task_identity_shift.max(),
#         task_variation_relative_change_mean=task_variation_change.mean(),
#         task_variation_relative_change_max=task_variation_change.max(),
#         current_identity_residual_relative_mean=current_identity_residual.mean(),
#         current_identity_residual_relative_max=current_identity_residual.max(),
#         current_variation_residual_relative_mean=current_variation_residual.mean(),
#         current_variation_residual_relative_max=current_variation_residual.max(),
#         transferred_identity_residual_relative_mean=transferred_identity_relative.mean(),
#         transferred_identity_residual_relative_max=transferred_identity_relative.max(),
#         transferred_variation_residual_relative_mean=transferred_variation_relative.mean(),
#         transferred_variation_residual_relative_max=transferred_variation_relative.max(),
#         calibrated_identity_relation_rmse=identity_rmse,
#         calibrated_variation_relation_rmse=variation_rmse,
#         spectral_identity_relation_rmse=spectral_identity_rmse,
#         spectral_variation_relation_rmse=spectral_variation_rmse,
#         affinity_entropy_mean=entropy,
#         affinity_concentration_mean=concentration,
#         affinity_zero_row_fraction=zero_fraction,
#     )
#     for field in diagnostics.__dataclass_fields__:
#         if not bool(torch.isfinite(getattr(diagnostics, field))):
#             raise RuntimeError(f"phase alignment diagnostic {field} is NaN or Inf")
#     return task_mapped, calibrated, transform, relation, residual, diagnostics


# __all__ = [
#     "PhaseCoordinateTransform",
#     "CDSAffinityState",
#     "CurrentClassResidual",
#     "PhaseAlignmentDiagnostics",
#     "estimate_task_representation_correspondence",
#     "apply_task_projection",
#     "estimate_current_class_residual",
#     "derive_cds_affinity",
#     "calibrate_historical_structure",
#     "align_historical_structure",
# ]













# from __future__ import annotations

# """Phase-local transport of the finalized learned HSI CDS.

# The ordered spectral-token axis K is never mixed. Paired legal current samples
# encoded by the frozen Phase-(t-1) backbone (Q-) and the current backbone (Q+)
# estimate one shared scale per learned channel plus a token-specific translation:

#     Q+[n,k,d] ~= s[d] * Q-[n,k,d] + t[k,d].

# This restriction is deliberate. The finalized V_Q stores diagonal absolute IQR
# extents and off-diagonal ordered-relation IQR extents, not covariance. Under the
# map above both transform exactly as

#     V_Q+[k,l,d] = |s[d]| * V_Q-[k,l,d],

# so the historical CDS can be transported without a covariance rule.

# The shared transport is followed by class-specific migration estimated only from
# legal current-phase samples. Current-class identity residuals and multiplicative
# V_Q deformation changes are transferred to historical classes according to the
# complete learned + physical old/current CDS affinity. Physical {I_P,V_P} is
# phase-stable and is never transported. Persistent memory remains exactly
# {I_Q,V_Q,I_P,V_P,N}.
# """

# from dataclasses import dataclass
# import math

# import torch

# from models.geometry_bank import (
#     ClassStructureBatch,
#     derive_class_relations,
#     merge_class_structure_batches,
#     sanitize_ordered_variation,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class PhaseCoordinateTransform:
#     task_scale: Tensor        # [K,D], channel-shared over K and repeated for API compatibility
#     task_translation: Tensor  # [K,D]


# @dataclass(frozen=True)
# class CDSAffinityState:
#     old_class_ids: Tensor
#     new_class_ids: Tensor
#     learned_identity_affinity: Tensor
#     learned_variation_affinity: Tensor
#     spectral_identity_affinity: Tensor
#     spectral_variation_affinity: Tensor
#     joint_affinity: Tensor
#     transfer_weights: Tensor

#     @property
#     def spectral_affinity(self) -> Tensor:
#         """Backward-compatible physical affinity summary."""
#         return torch.sqrt(
#             self.spectral_identity_affinity.clamp_min(0.0)
#             * self.spectral_variation_affinity.clamp_min(0.0)
#         ).clamp(0.0, 1.0)


# @dataclass(frozen=True)
# class CurrentClassResidual:
#     class_ids: Tensor
#     identity: Tensor
#     variation: Tensor
#     variation_log_ratio: Tensor


# @dataclass(frozen=True)
# class PhaseAlignmentDiagnostics:
#     task_mapping_error: Tensor
#     task_mapping_channel_error_mean: Tensor
#     task_mapping_channel_error_max: Tensor
#     representation_drift_rms: Tensor
#     task_translation_rms: Tensor
#     task_scale_deviation_rms: Tensor
#     task_identity_relative_shift_mean: Tensor
#     task_identity_relative_shift_max: Tensor
#     task_variation_relative_change_mean: Tensor
#     task_variation_relative_change_max: Tensor
#     current_identity_residual_relative_mean: Tensor
#     current_identity_residual_relative_max: Tensor
#     current_variation_residual_relative_mean: Tensor
#     current_variation_residual_relative_max: Tensor
#     transferred_identity_residual_relative_mean: Tensor
#     transferred_identity_residual_relative_max: Tensor
#     transferred_variation_residual_relative_mean: Tensor
#     transferred_variation_residual_relative_max: Tensor
#     calibrated_identity_relation_rmse: Tensor
#     calibrated_variation_relation_rmse: Tensor
#     spectral_identity_relation_rmse: Tensor
#     spectral_variation_relation_rmse: Tensor
#     affinity_entropy_mean: Tensor
#     affinity_concentration_mean: Tensor
#     affinity_zero_row_fraction: Tensor


# def _validate_paired(reference_features: Tensor, current_features: Tensor) -> tuple[Tensor, Tensor]:
#     reference = torch.as_tensor(reference_features)
#     current = torch.as_tensor(current_features)
#     if reference.ndim != 3 or reference.size(0) < 2 or reference.size(1) < 2 or reference.size(2) < 1:
#         raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if reference.device != current.device or reference.dtype != current.dtype:
#         raise ValueError("paired Q-/Q+ features must share device/dtype")
#     if not torch.is_floating_point(reference) or reference.is_complex():
#         raise TypeError("paired Q-/Q+ features must be real floating point")
#     if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
#         raise ValueError("paired Q-/Q+ features contain NaN or Inf")
#     return reference, current


# def _validate_structure_pair(reference: ClassStructureBatch, current: ClassStructureBatch) -> None:
#     if not isinstance(reference, ClassStructureBatch) or not isinstance(current, ClassStructureBatch):
#         raise TypeError("current reference/current structures must be ClassStructureBatch")
#     if reference.class_count <= 0 or current.class_count != reference.class_count:
#         raise ValueError("current reference/current structures have incompatible class counts")
#     if not torch.equal(reference.class_ids, current.class_ids):
#         raise ValueError("current reference/current structures must use identical class order")
#     if reference.identity.shape != current.identity.shape or reference.variation.shape != current.variation.shape:
#         raise ValueError("current reference/current learned CDS shapes disagree")
#     tensors = (
#         reference.identity,
#         reference.variation,
#         reference.spectral_identity,
#         reference.spectral_variation,
#         current.identity,
#         current.variation,
#         current.spectral_identity,
#         current.spectral_variation,
#     )
#     if any(v.device != reference.identity.device or v.dtype != reference.identity.dtype for v in tensors):
#         raise ValueError("current reference/current CDS must share device/dtype")
#     if not all(bool(torch.isfinite(v).all()) for v in tensors):
#         raise ValueError("current reference/current CDS contains NaN or Inf")


# def _relative_change(after: Tensor, before: Tensor) -> Tensor:
#     eps = torch.finfo(before.dtype).eps
#     numerator = torch.linalg.vector_norm((after - before).reshape(after.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(before.reshape(before.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _relative_residual(residual: Tensor, target: Tensor) -> Tensor:
#     eps = torch.finfo(target.dtype).eps
#     numerator = torch.linalg.vector_norm(residual.reshape(residual.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(target.reshape(target.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _offdiag_rmse(after: Tensor, before: Tensor) -> Tensor:
#     if after.shape != before.shape or after.ndim < 2 or after.size(0) != after.size(1):
#         raise ValueError("relation tensors must share [C,C,...] shape")
#     c = int(after.size(0))
#     if c <= 1:
#         return after.new_zeros(())
#     mask = ~torch.eye(c, device=after.device, dtype=torch.bool)
#     return torch.sqrt((after[mask] - before[mask]).square().mean().clamp_min(0.0))


# @torch.no_grad()
# def estimate_task_representation_correspondence(
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
#     """Fit a CDS-compatible shared Q- -> Q+ affine drift.

#     A single scale ``s[d]`` is fitted for each learned channel across all
#     spectral tokens, while ``t[k,d]`` remains token-specific. This is the
#     simplest affine family that preserves the complete absolute/relational V_Q
#     extent semantics exactly. The returned scale is repeated to [K,D] only to preserve the
#     surrounding repository API.
#     """
#     reference, current = _validate_paired(reference_features, current_features)
#     x = reference.detach().to(device="cpu", dtype=torch.float64)
#     y = current.detach().to(device="cpu", dtype=torch.float64)
#     _, k, d = x.shape

#     # Token-specific intercepts are removed before estimating one channel scale.
#     mean_x = x.mean(dim=0)  # [K,D]
#     mean_y = y.mean(dim=0)  # [K,D]
#     xc = x - mean_x.unsqueeze(0)
#     yc = y - mean_y.unsqueeze(0)

#     numerator = (xc * yc).sum(dim=(0, 1))      # [D]
#     denominator = xc.square().sum(dim=(0, 1)) # [D]
#     eps64 = torch.finfo(torch.float64).eps
#     tolerance = eps64 * denominator.max()
#     identifiable = denominator > tolerance
#     channel_scale = torch.ones_like(denominator)
#     channel_scale[identifiable] = numerator[identifiable] / denominator[identifiable]

#     scale = channel_scale.unsqueeze(0).expand(k, d).clone()
#     translation = mean_y - scale * mean_x
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise RuntimeError("CDS-compatible phase map contains NaN or Inf")

#     prediction = x * scale.unsqueeze(0) + translation.unsqueeze(0)
#     residual = y - prediction
#     tiny = torch.finfo(torch.float64).tiny
#     channel_residual = residual.permute(2, 0, 1)
#     channel_target = y.permute(2, 0, 1)
#     channel_error = (
#         channel_residual.square().sum(dim=(1, 2))
#         / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
#     )
#     mapping_error = residual.square().sum() / y.square().sum().clamp_min(tiny)
#     drift_rms = torch.sqrt((y - x).square().mean().clamp_min(0.0))
#     translation_rms = torch.sqrt(translation.square().mean().clamp_min(0.0))
#     scale_deviation_rms = torch.sqrt(
#         (channel_scale - 1.0).square().mean().clamp_min(0.0)
#     )

#     device, dtype = reference.device, reference.dtype
#     transform = PhaseCoordinateTransform(
#         task_scale=scale.to(device=device, dtype=dtype).detach(),
#         task_translation=translation.to(device=device, dtype=dtype).detach(),
#     )
#     diagnostics = {
#         "task_mapping_error": mapping_error.to(device=device, dtype=dtype),
#         "task_mapping_channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
#         "task_mapping_channel_error_max": channel_error.max().to(device=device, dtype=dtype),
#         "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
#         "task_translation_rms": translation_rms.to(device=device, dtype=dtype),
#         "task_scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
#     }
#     return transform, diagnostics


# @torch.no_grad()
# def apply_task_projection(
#     structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> ClassStructureBatch:
#     """Transport I_Q and complete absolute/relational V_Q into current Q coordinates.

#     For the CDS-compatible map Q+ = s[d] Q- + t[k,d]:

#         I_Q+ = s[d] I_Q- + t[k,d]
#         V_Q+ = |s[d]| V_Q-

#     Translation cancels from off-diagonal relations; absolute IQR is translation
#     invariant as well. No covariance,
#     PSD transport, eigendecomposition, or residual migration is involved.
#     """
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     _, k, d = structure.identity.shape
#     scale = transform.task_scale
#     translation = transform.task_translation
#     if scale.shape != (k, d) or translation.shape != (k, d):
#         raise ValueError("task affine map shape does not match CDS")
#     if any(v.device != structure.identity.device or v.dtype != structure.identity.dtype for v in (scale, translation)):
#         raise ValueError("task affine map and CDS must share device/dtype")
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise ValueError("task affine map contains NaN or Inf")

#     # The transport is valid only when scale is shared over ordered token K.
#     channel_scale = scale[0]
#     expected = channel_scale.unsqueeze(0).expand_as(scale)
#     tolerance = 64.0 * torch.finfo(scale.dtype).eps
#     if not bool(torch.allclose(scale, expected, rtol=tolerance, atol=tolerance)):
#         raise RuntimeError("V_Q deformation transport requires channel-shared scale over K")

#     mapped_identity = structure.identity * scale.unsqueeze(0) + translation.unsqueeze(0)
#     mapped_variation = sanitize_ordered_variation(
#         structure.variation * channel_scale.abs().view(1, 1, 1, d)
#     )
#     return ClassStructureBatch(
#         class_ids=structure.class_ids.detach().clone(),
#         sample_counts=structure.sample_counts.detach().clone(),
#         identity=mapped_identity.detach().contiguous(),
#         variation=mapped_variation.detach().contiguous(),
#         spectral_identity=structure.spectral_identity.detach().clone(),
#         spectral_variation=structure.spectral_variation.detach().clone(),
#     )


# @torch.no_grad()
# def estimate_current_class_residual(
#     *,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> CurrentClassResidual:
#     """Observed class-specific migration not explained by shared phase drift."""
#     _validate_structure_pair(current_reference, current_structure)
#     predicted = apply_task_projection(current_reference, transform)
#     if not torch.allclose(current_reference.spectral_identity, current_structure.spectral_identity, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral identity changed across neural coordinates")
#     if not torch.allclose(current_reference.spectral_variation, current_structure.spectral_variation, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral variation changed across neural coordinates")
#     identity = current_structure.identity - predicted.identity
#     variation = current_structure.variation - predicted.variation
#     eps = torch.finfo(current_structure.variation.dtype).eps
#     variation_log_ratio = torch.log(
#         (current_structure.variation + eps) / (predicted.variation + eps)
#     )
#     if not all(bool(torch.isfinite(value).all()) for value in (identity, variation, variation_log_ratio)):
#         raise RuntimeError("current-class residual contains NaN or Inf")
#     return CurrentClassResidual(
#         class_ids=current_structure.class_ids.detach().clone(),
#         identity=identity.detach(),
#         variation=variation.detach(),
#         variation_log_ratio=variation_log_ratio.detach(),
#     )


# @torch.no_grad()
# def derive_cds_affinity(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
# ) -> CDSAffinityState:
#     """Derive phase-start old/current affinity from the complete HSI-CDS.

#     Identity and intra-class structure remain separate in both learned and
#     physical spectral spaces.  Their geometric mean is used only here, where a
#     scalar transfer weight is required for class-specific migration.
#     """
#     if historical.class_count <= 0 or current_reference.class_count <= 0:
#         raise ValueError("historical/current reference CDS must be non-empty")
#     if historical.identity.device != current_reference.identity.device or historical.identity.dtype != current_reference.identity.dtype:
#         raise ValueError("historical/current reference CDS must share device/dtype")
#     old_values = set(int(v) for v in historical.class_ids.detach().cpu().tolist())
#     new_values = set(int(v) for v in current_reference.class_ids.detach().cpu().tolist())
#     if old_values.intersection(new_values):
#         raise ValueError("historical/current reference classes overlap")

#     merged = merge_class_structure_batches((historical, current_reference))
#     relations = derive_class_relations(merged)
#     old_count = historical.class_count

#     learned_identity = relations.learned_identity_cosine_by_region[
#         :old_count, old_count:
#     ].mean(dim=2).clamp(0.0, 1.0)
#     learned_variation = relations.learned_variation_overlap[
#         :old_count, old_count:
#     ].clamp(0.0, 1.0)
#     spectral_identity = relations.spectral_identity_cosine_by_component[
#         :old_count, old_count:
#     ].mean(dim=2).clamp(0.0, 1.0)
#     spectral_variation = relations.spectral_variation_overlap[
#         :old_count, old_count:
#     ].clamp(0.0, 1.0)

#     joint = (
#         learned_identity
#         * learned_variation
#         * spectral_identity
#         * spectral_variation
#     ).clamp_min(0.0).pow(0.25).clamp(0.0, 1.0)

#     tiny = torch.finfo(joint.dtype).tiny
#     row_sum = joint.sum(dim=1, keepdim=True)
#     active = row_sum.squeeze(1) > tiny
#     weights = torch.zeros_like(joint)
#     if bool(active.any()):
#         weights[active] = joint[active] / row_sum[active]

#     return CDSAffinityState(
#         old_class_ids=historical.class_ids.detach().clone(),
#         new_class_ids=current_reference.class_ids.detach().clone(),
#         learned_identity_affinity=learned_identity.detach(),
#         learned_variation_affinity=learned_variation.detach(),
#         spectral_identity_affinity=spectral_identity.detach(),
#         spectral_variation_affinity=spectral_variation.detach(),
#         joint_affinity=joint.detach(),
#         transfer_weights=weights.detach(),
#     )


# @torch.no_grad()
# def calibrate_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     task_mapped_historical: ClassStructureBatch,
#     residual: CurrentClassResidual,
#     affinity: CDSAffinityState,
# ) -> ClassStructureBatch:
#     """Apply class-specific whole-CDS migration after shared transport.

#     I_Q receives an affinity-weighted additive residual because it is a location
#     tensor. V_Q receives an affinity-weighted multiplicative change in log space
#     because it is a non-negative absolute/ordered relation-spread tensor.
#     Physical I_P/V_P are copied exactly and never migrate with Q.
#     """
#     if not torch.equal(historical.class_ids, task_mapped_historical.class_ids):
#         raise ValueError("task-mapped historical class IDs changed")
#     if not torch.equal(historical.class_ids, affinity.old_class_ids):
#         raise ValueError("affinity old class IDs do not match historical CDS")
#     if not torch.equal(residual.class_ids, affinity.new_class_ids):
#         raise ValueError("residual class IDs do not match current classes")

#     weights = affinity.transfer_weights.to(
#         device=task_mapped_historical.identity.device,
#         dtype=task_mapped_historical.identity.dtype,
#     )
#     if weights.shape != (historical.class_count, residual.class_ids.numel()):
#         raise ValueError("transfer weight shape disagrees with old/current classes")
#     if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
#         raise ValueError("transfer weights must be finite and non-negative")

#     transferred_identity = torch.einsum("oj,jkd->okd", weights, residual.identity)
#     transferred_log_ratio = torch.einsum(
#         "oj,jkld->okld", weights, residual.variation_log_ratio
#     )
#     multiplier = torch.exp(transferred_log_ratio)
#     eps = torch.finfo(task_mapped_historical.variation.dtype).eps
#     migrated_variation = (task_mapped_historical.variation + eps) * multiplier - eps
#     migrated_variation = sanitize_ordered_variation(migrated_variation.clamp_min(0.0))
#     migrated_identity = task_mapped_historical.identity + transferred_identity

#     if not bool(torch.isfinite(migrated_identity).all()) or not bool(torch.isfinite(migrated_variation).all()):
#         raise RuntimeError("class-specific historical CDS migration produced NaN or Inf")

#     return ClassStructureBatch(
#         class_ids=task_mapped_historical.class_ids.detach().clone(),
#         sample_counts=task_mapped_historical.sample_counts.detach().clone(),
#         identity=migrated_identity.detach().contiguous(),
#         variation=migrated_variation.detach().contiguous(),
#         spectral_identity=historical.spectral_identity.detach().clone(),
#         spectral_variation=historical.spectral_variation.detach().clone(),
#     )


# def _affinity_diagnostics(affinity: CDSAffinityState, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
#     weights = affinity.transfer_weights
#     active = weights.sum(dim=1) > 0
#     zero_fraction = (~active).to(dtype).mean()
#     if not bool(active.any()):
#         zero = weights.new_zeros(())
#         return zero, zero, zero_fraction
#     w = weights[active]
#     concentration = w.max(dim=1).values.mean()
#     if w.size(1) <= 1:
#         entropy = weights.new_zeros(())
#     else:
#         tiny = torch.finfo(weights.dtype).tiny
#         entropy = (-(w * w.clamp_min(tiny).log()).sum(dim=1) / math.log(float(w.size(1)))).mean()
#     return entropy.to(dtype), concentration.to(dtype), zero_fraction.to(dtype)


# @torch.no_grad()
# def align_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     reference_features: Tensor,
#     current_features: Tensor,
#     affinity: CDSAffinityState | None = None,
# ) -> tuple[
#     ClassStructureBatch,
#     ClassStructureBatch,
#     PhaseCoordinateTransform,
#     CDSAffinityState,
#     CurrentClassResidual,
#     PhaseAlignmentDiagnostics,
# ]:
#     """Shared CDS-compatible transport plus whole-CDS class-specific migration."""
#     transform, mapping = estimate_task_representation_correspondence(reference_features, current_features)
#     task_mapped = apply_task_projection(historical, transform)
#     residual = estimate_current_class_residual(
#         current_reference=current_reference,
#         current_structure=current_structure,
#         transform=transform,
#     )
#     relation = affinity or derive_cds_affinity(
#         historical=historical,
#         current_reference=current_reference,
#     )
#     calibrated = calibrate_historical_structure(
#         historical=historical,
#         task_mapped_historical=task_mapped,
#         residual=residual,
#         affinity=relation,
#     )

#     task_identity_shift = _relative_change(task_mapped.identity, historical.identity)
#     task_variation_change = _relative_change(task_mapped.variation, historical.variation)
#     current_identity_residual = _relative_residual(residual.identity, current_structure.identity)
#     current_variation_residual = _relative_residual(residual.variation, current_structure.variation)
#     transferred_identity = calibrated.identity - task_mapped.identity
#     transferred_variation = calibrated.variation - task_mapped.variation
#     transferred_identity_relative = _relative_residual(transferred_identity, task_mapped.identity)
#     transferred_variation_relative = _relative_residual(transferred_variation, task_mapped.variation)

#     before = derive_class_relations(historical)
#     after = derive_class_relations(calibrated)
#     identity_rmse = _offdiag_rmse(after.learned_identity_cosine_by_region, before.learned_identity_cosine_by_region)
#     variation_rmse = _offdiag_rmse(after.learned_variation_overlap, before.learned_variation_overlap)
#     spectral_identity_rmse = _offdiag_rmse(after.spectral_identity_cosine_by_component, before.spectral_identity_cosine_by_component)
#     spectral_variation_rmse = _offdiag_rmse(after.spectral_variation_overlap, before.spectral_variation_overlap)
#     entropy, concentration, zero_fraction = _affinity_diagnostics(relation, historical.identity.dtype)

#     diagnostics = PhaseAlignmentDiagnostics(
#         task_mapping_error=mapping["task_mapping_error"],
#         task_mapping_channel_error_mean=mapping["task_mapping_channel_error_mean"],
#         task_mapping_channel_error_max=mapping["task_mapping_channel_error_max"],
#         representation_drift_rms=mapping["representation_drift_rms"],
#         task_translation_rms=mapping["task_translation_rms"],
#         task_scale_deviation_rms=mapping["task_scale_deviation_rms"],
#         task_identity_relative_shift_mean=task_identity_shift.mean(),
#         task_identity_relative_shift_max=task_identity_shift.max(),
#         task_variation_relative_change_mean=task_variation_change.mean(),
#         task_variation_relative_change_max=task_variation_change.max(),
#         current_identity_residual_relative_mean=current_identity_residual.mean(),
#         current_identity_residual_relative_max=current_identity_residual.max(),
#         current_variation_residual_relative_mean=current_variation_residual.mean(),
#         current_variation_residual_relative_max=current_variation_residual.max(),
#         transferred_identity_residual_relative_mean=transferred_identity_relative.mean(),
#         transferred_identity_residual_relative_max=transferred_identity_relative.max(),
#         transferred_variation_residual_relative_mean=transferred_variation_relative.mean(),
#         transferred_variation_residual_relative_max=transferred_variation_relative.max(),
#         calibrated_identity_relation_rmse=identity_rmse,
#         calibrated_variation_relation_rmse=variation_rmse,
#         spectral_identity_relation_rmse=spectral_identity_rmse,
#         spectral_variation_relation_rmse=spectral_variation_rmse,
#         affinity_entropy_mean=entropy,
#         affinity_concentration_mean=concentration,
#         affinity_zero_row_fraction=zero_fraction,
#     )
#     for field in diagnostics.__dataclass_fields__:
#         if not bool(torch.isfinite(getattr(diagnostics, field))):
#             raise RuntimeError(f"phase alignment diagnostic {field} is NaN or Inf")
#     return task_mapped, calibrated, transform, relation, residual, diagnostics


# __all__ = [
#     "PhaseCoordinateTransform",
#     "CDSAffinityState",
#     "CurrentClassResidual",
#     "PhaseAlignmentDiagnostics",
#     "estimate_task_representation_correspondence",
#     "apply_task_projection",
#     "estimate_current_class_residual",
#     "derive_cds_affinity",
#     "calibrate_historical_structure",
#     "align_historical_structure",
# ]




















# from __future__ import annotations

# """Phase-local transport of the finalized learned HSI CDS.

# The ordered spectral-token axis K is never mixed. Paired legal current samples
# encoded by the frozen Phase-(t-1) backbone (Q-) and the current backbone (Q+)
# estimate one shared scale per learned channel plus a token-specific translation:

#     Q+[n,k,d] ~= s[d] * Q-[n,k,d] + t[k,d].

# This restriction is deliberate. The finalized V_Q stores diagonal absolute IQR
# extents and off-diagonal ordered-relation IQR extents, not covariance. Under the
# map above both transform exactly as

#     V_Q+[k,l,d] = |s[d]| * V_Q-[k,l,d],

# so the historical CDS can be transported without a covariance rule.

# Current-class residuals and CDS affinities are retained only as diagnostics.
# They are never transferred into historical classes. Physical {I_P,V_P} is
# phase-stable and is never transported. Persistent memory remains exactly
# {I_Q,V_Q,I_P,V_P,N}.
# """

# from dataclasses import dataclass
# import math

# import torch

# from models.geometry_bank import (
#     ClassStructureBatch,
#     derive_class_relations,
#     merge_class_structure_batches,
#     sanitize_ordered_variation,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class PhaseCoordinateTransform:
#     task_scale: Tensor        # [K,D], channel-shared over K and repeated for API compatibility
#     task_translation: Tensor  # [K,D]


# @dataclass(frozen=True)
# class CDSAffinityState:
#     old_class_ids: Tensor
#     new_class_ids: Tensor
#     learned_identity_affinity: Tensor
#     learned_variation_affinity: Tensor
#     spectral_affinity: Tensor
#     joint_affinity: Tensor
#     transfer_weights: Tensor


# @dataclass(frozen=True)
# class CurrentClassResidual:
#     class_ids: Tensor
#     identity: Tensor
#     variation: Tensor


# @dataclass(frozen=True)
# class PhaseAlignmentDiagnostics:
#     task_mapping_error: Tensor
#     task_mapping_channel_error_mean: Tensor
#     task_mapping_channel_error_max: Tensor
#     representation_drift_rms: Tensor
#     task_translation_rms: Tensor
#     task_scale_deviation_rms: Tensor
#     task_identity_relative_shift_mean: Tensor
#     task_identity_relative_shift_max: Tensor
#     task_variation_relative_change_mean: Tensor
#     task_variation_relative_change_max: Tensor
#     current_identity_residual_relative_mean: Tensor
#     current_identity_residual_relative_max: Tensor
#     current_variation_residual_relative_mean: Tensor
#     current_variation_residual_relative_max: Tensor
#     transferred_identity_residual_relative_mean: Tensor
#     transferred_identity_residual_relative_max: Tensor
#     transferred_variation_residual_relative_mean: Tensor
#     transferred_variation_residual_relative_max: Tensor
#     calibrated_identity_relation_rmse: Tensor
#     calibrated_variation_relation_rmse: Tensor
#     spectral_identity_relation_rmse: Tensor
#     spectral_variation_relation_rmse: Tensor
#     affinity_entropy_mean: Tensor
#     affinity_concentration_mean: Tensor
#     affinity_zero_row_fraction: Tensor


# def _validate_paired(reference_features: Tensor, current_features: Tensor) -> tuple[Tensor, Tensor]:
#     reference = torch.as_tensor(reference_features)
#     current = torch.as_tensor(current_features)
#     if reference.ndim != 3 or reference.size(0) < 2 or reference.size(1) < 2 or reference.size(2) < 1:
#         raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if reference.device != current.device or reference.dtype != current.dtype:
#         raise ValueError("paired Q-/Q+ features must share device/dtype")
#     if not torch.is_floating_point(reference) or reference.is_complex():
#         raise TypeError("paired Q-/Q+ features must be real floating point")
#     if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
#         raise ValueError("paired Q-/Q+ features contain NaN or Inf")
#     return reference, current


# def _validate_structure_pair(reference: ClassStructureBatch, current: ClassStructureBatch) -> None:
#     if not isinstance(reference, ClassStructureBatch) or not isinstance(current, ClassStructureBatch):
#         raise TypeError("current reference/current structures must be ClassStructureBatch")
#     if reference.class_count <= 0 or current.class_count != reference.class_count:
#         raise ValueError("current reference/current structures have incompatible class counts")
#     if not torch.equal(reference.class_ids, current.class_ids):
#         raise ValueError("current reference/current structures must use identical class order")
#     if reference.identity.shape != current.identity.shape or reference.variation.shape != current.variation.shape:
#         raise ValueError("current reference/current learned CDS shapes disagree")
#     tensors = (
#         reference.identity,
#         reference.variation,
#         reference.spectral_identity,
#         reference.spectral_variation,
#         current.identity,
#         current.variation,
#         current.spectral_identity,
#         current.spectral_variation,
#     )
#     if any(v.device != reference.identity.device or v.dtype != reference.identity.dtype for v in tensors):
#         raise ValueError("current reference/current CDS must share device/dtype")
#     if not all(bool(torch.isfinite(v).all()) for v in tensors):
#         raise ValueError("current reference/current CDS contains NaN or Inf")


# def _relative_change(after: Tensor, before: Tensor) -> Tensor:
#     eps = torch.finfo(before.dtype).eps
#     numerator = torch.linalg.vector_norm((after - before).reshape(after.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(before.reshape(before.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _relative_residual(residual: Tensor, target: Tensor) -> Tensor:
#     eps = torch.finfo(target.dtype).eps
#     numerator = torch.linalg.vector_norm(residual.reshape(residual.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(target.reshape(target.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _offdiag_rmse(after: Tensor, before: Tensor) -> Tensor:
#     if after.shape != before.shape or after.ndim < 2 or after.size(0) != after.size(1):
#         raise ValueError("relation tensors must share [C,C,...] shape")
#     c = int(after.size(0))
#     if c <= 1:
#         return after.new_zeros(())
#     mask = ~torch.eye(c, device=after.device, dtype=torch.bool)
#     return torch.sqrt((after[mask] - before[mask]).square().mean().clamp_min(0.0))


# @torch.no_grad()
# def estimate_task_representation_correspondence(
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
#     """Fit a CDS-compatible shared Q- -> Q+ affine drift.

#     A single scale ``s[d]`` is fitted for each learned channel across all
#     spectral tokens, while ``t[k,d]`` remains token-specific. This is the
#     simplest affine family that preserves the complete absolute/relational V_Q
#     extent semantics exactly. The returned scale is repeated to [K,D] only to preserve the
#     surrounding repository API.
#     """
#     reference, current = _validate_paired(reference_features, current_features)
#     x = reference.detach().to(device="cpu", dtype=torch.float64)
#     y = current.detach().to(device="cpu", dtype=torch.float64)
#     _, k, d = x.shape

#     # Token-specific intercepts are removed before estimating one channel scale.
#     mean_x = x.mean(dim=0)  # [K,D]
#     mean_y = y.mean(dim=0)  # [K,D]
#     xc = x - mean_x.unsqueeze(0)
#     yc = y - mean_y.unsqueeze(0)

#     numerator = (xc * yc).sum(dim=(0, 1))      # [D]
#     denominator = xc.square().sum(dim=(0, 1)) # [D]
#     eps64 = torch.finfo(torch.float64).eps
#     tolerance = eps64 * denominator.max()
#     identifiable = denominator > tolerance
#     channel_scale = torch.ones_like(denominator)
#     channel_scale[identifiable] = numerator[identifiable] / denominator[identifiable]

#     scale = channel_scale.unsqueeze(0).expand(k, d).clone()
#     translation = mean_y - scale * mean_x
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise RuntimeError("CDS-compatible phase map contains NaN or Inf")

#     prediction = x * scale.unsqueeze(0) + translation.unsqueeze(0)
#     residual = y - prediction
#     tiny = torch.finfo(torch.float64).tiny
#     channel_residual = residual.permute(2, 0, 1)
#     channel_target = y.permute(2, 0, 1)
#     channel_error = (
#         channel_residual.square().sum(dim=(1, 2))
#         / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
#     )
#     mapping_error = residual.square().sum() / y.square().sum().clamp_min(tiny)
#     drift_rms = torch.sqrt((y - x).square().mean().clamp_min(0.0))
#     translation_rms = torch.sqrt(translation.square().mean().clamp_min(0.0))
#     scale_deviation_rms = torch.sqrt(
#         (channel_scale - 1.0).square().mean().clamp_min(0.0)
#     )

#     device, dtype = reference.device, reference.dtype
#     transform = PhaseCoordinateTransform(
#         task_scale=scale.to(device=device, dtype=dtype).detach(),
#         task_translation=translation.to(device=device, dtype=dtype).detach(),
#     )
#     diagnostics = {
#         "task_mapping_error": mapping_error.to(device=device, dtype=dtype),
#         "task_mapping_channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
#         "task_mapping_channel_error_max": channel_error.max().to(device=device, dtype=dtype),
#         "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
#         "task_translation_rms": translation_rms.to(device=device, dtype=dtype),
#         "task_scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
#     }
#     return transform, diagnostics


# @torch.no_grad()
# def apply_task_projection(
#     structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> ClassStructureBatch:
#     """Transport I_Q and complete absolute/relational V_Q into current Q coordinates.

#     For the CDS-compatible map Q+ = s[d] Q- + t[k,d]:

#         I_Q+ = s[d] I_Q- + t[k,d]
#         V_Q+ = |s[d]| V_Q-

#     Translation cancels from off-diagonal relations; absolute IQR is translation
#     invariant as well. No covariance,
#     PSD transport, eigendecomposition, or residual migration is involved.
#     """
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     _, k, d = structure.identity.shape
#     scale = transform.task_scale
#     translation = transform.task_translation
#     if scale.shape != (k, d) or translation.shape != (k, d):
#         raise ValueError("task affine map shape does not match CDS")
#     if any(v.device != structure.identity.device or v.dtype != structure.identity.dtype for v in (scale, translation)):
#         raise ValueError("task affine map and CDS must share device/dtype")
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise ValueError("task affine map contains NaN or Inf")

#     # The transport is valid only when scale is shared over ordered token K.
#     channel_scale = scale[0]
#     expected = channel_scale.unsqueeze(0).expand_as(scale)
#     tolerance = 64.0 * torch.finfo(scale.dtype).eps
#     if not bool(torch.allclose(scale, expected, rtol=tolerance, atol=tolerance)):
#         raise RuntimeError("V_Q deformation transport requires channel-shared scale over K")

#     mapped_identity = structure.identity * scale.unsqueeze(0) + translation.unsqueeze(0)
#     mapped_variation = sanitize_ordered_variation(
#         structure.variation * channel_scale.abs().view(1, 1, 1, d)
#     )
#     return ClassStructureBatch(
#         class_ids=structure.class_ids.detach().clone(),
#         sample_counts=structure.sample_counts.detach().clone(),
#         identity=mapped_identity.detach().contiguous(),
#         variation=mapped_variation.detach().contiguous(),
#         spectral_identity=structure.spectral_identity.detach().clone(),
#         spectral_variation=structure.spectral_variation.detach().clone(),
#     )


# @torch.no_grad()
# def estimate_current_class_residual(
#     *,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> CurrentClassResidual:
#     """Observed class-specific migration not explained by shared phase drift."""
#     _validate_structure_pair(current_reference, current_structure)
#     predicted = apply_task_projection(current_reference, transform)
#     if not torch.allclose(current_reference.spectral_identity, current_structure.spectral_identity, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral identity changed across neural coordinates")
#     if not torch.allclose(current_reference.spectral_variation, current_structure.spectral_variation, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral variation changed across neural coordinates")
#     identity = current_structure.identity - predicted.identity
#     variation = current_structure.variation - predicted.variation
#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
#         raise RuntimeError("current-class residual contains NaN or Inf")
#     return CurrentClassResidual(
#         class_ids=current_structure.class_ids.detach().clone(),
#         identity=identity.detach(),
#         variation=variation.detach(),
#     )


# @torch.no_grad()
# def derive_cds_affinity(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
# ) -> CDSAffinityState:
#     """Derive phase-start old/current HSI-CDS affinity for diagnostics.

#     The geometry bank already returns normalized L1 similarities in [0,1].
#     They are used directly; the previous cosine remapping 0.5*(1+r) was invalid
#     for the finalized bank. Affinity is not used to move historical CDS.
#     """
#     if historical.class_count <= 0 or current_reference.class_count <= 0:
#         raise ValueError("historical/current reference CDS must be non-empty")
#     if historical.identity.device != current_reference.identity.device or historical.identity.dtype != current_reference.identity.dtype:
#         raise ValueError("historical/current reference CDS must share device/dtype")
#     old_values = set(int(v) for v in historical.class_ids.detach().cpu().tolist())
#     new_values = set(int(v) for v in current_reference.class_ids.detach().cpu().tolist())
#     if old_values.intersection(new_values):
#         raise ValueError("historical/current reference classes overlap")

#     merged = merge_class_structure_batches((historical, current_reference))
#     relations = derive_class_relations(merged)
#     old_count = historical.class_count

#     learned_identity = (
#         relations.learned_identity_cosine_by_region[:old_count, old_count:].mean(dim=2)
#     ).clamp(0.0, 1.0)
#     learned_variation = (
#         relations.learned_variation_overlap[:old_count, old_count:]
#     ).clamp(0.0, 1.0)
#     spectral = (
#         relations.spectral_confusability[:old_count, old_count:]
#     ).clamp(0.0, 1.0)

#     learned = 0.5 * (learned_identity + learned_variation)
#     joint = torch.sqrt(
#         learned.clamp_min(0.0) * spectral.clamp_min(0.0)
#     ).clamp(0.0, 1.0)

#     tiny = torch.finfo(joint.dtype).tiny
#     row_sum = joint.sum(dim=1, keepdim=True)
#     active = row_sum.squeeze(1) > tiny
#     weights = torch.zeros_like(joint)
#     if bool(active.any()):
#         weights[active] = joint[active] / row_sum[active]

#     return CDSAffinityState(
#         old_class_ids=historical.class_ids.detach().clone(),
#         new_class_ids=current_reference.class_ids.detach().clone(),
#         learned_identity_affinity=learned_identity.detach(),
#         learned_variation_affinity=learned_variation.detach(),
#         spectral_affinity=spectral.detach(),
#         joint_affinity=joint.detach(),
#         transfer_weights=weights.detach(),
#     )


# @torch.no_grad()
# def calibrate_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     task_mapped_historical: ClassStructureBatch,
#     residual: CurrentClassResidual,
#     affinity: CDSAffinityState,
# ) -> ClassStructureBatch:
#     """Return the shared-drift transported historical CDS unchanged.

#     This function name is retained only for repository compatibility. The
#     finalized architecture does *not* transfer current-class residuals into
#     unavailable historical classes. ``residual`` and ``affinity`` are validated
#     and reported as diagnostics only.
#     """
#     if not torch.equal(historical.class_ids, task_mapped_historical.class_ids):
#         raise ValueError("task-mapped historical class IDs changed")
#     if not torch.equal(historical.class_ids, affinity.old_class_ids):
#         raise ValueError("affinity old class IDs do not match historical CDS")
#     if not torch.equal(residual.class_ids, affinity.new_class_ids):
#         raise ValueError("residual class IDs do not match current classes")

#     return ClassStructureBatch(
#         class_ids=task_mapped_historical.class_ids.detach().clone(),
#         sample_counts=task_mapped_historical.sample_counts.detach().clone(),
#         identity=task_mapped_historical.identity.detach().clone(),
#         variation=task_mapped_historical.variation.detach().clone(),
#         spectral_identity=task_mapped_historical.spectral_identity.detach().clone(),
#         spectral_variation=task_mapped_historical.spectral_variation.detach().clone(),
#     )


# def _affinity_diagnostics(affinity: CDSAffinityState, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
#     weights = affinity.transfer_weights
#     active = weights.sum(dim=1) > 0
#     zero_fraction = (~active).to(dtype).mean()
#     if not bool(active.any()):
#         zero = weights.new_zeros(())
#         return zero, zero, zero_fraction
#     w = weights[active]
#     concentration = w.max(dim=1).values.mean()
#     if w.size(1) <= 1:
#         entropy = weights.new_zeros(())
#     else:
#         tiny = torch.finfo(weights.dtype).tiny
#         entropy = (-(w * w.clamp_min(tiny).log()).sum(dim=1) / math.log(float(w.size(1)))).mean()
#     return entropy.to(dtype), concentration.to(dtype), zero_fraction.to(dtype)


# @torch.no_grad()
# def align_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     reference_features: Tensor,
#     current_features: Tensor,
#     affinity: CDSAffinityState | None = None,
# ) -> tuple[
#     ClassStructureBatch,
#     ClassStructureBatch,
#     PhaseCoordinateTransform,
#     CDSAffinityState,
#     CurrentClassResidual,
#     PhaseAlignmentDiagnostics,
# ]:
#     """Transport historical CDS with shared observable drift only; no residual transfer."""
#     transform, mapping = estimate_task_representation_correspondence(reference_features, current_features)
#     task_mapped = apply_task_projection(historical, transform)
#     residual = estimate_current_class_residual(
#         current_reference=current_reference,
#         current_structure=current_structure,
#         transform=transform,
#     )
#     relation = affinity or derive_cds_affinity(
#         historical=historical,
#         current_reference=current_reference,
#     )
#     calibrated = calibrate_historical_structure(
#         historical=historical,
#         task_mapped_historical=task_mapped,
#         residual=residual,
#         affinity=relation,
#     )

#     task_identity_shift = _relative_change(task_mapped.identity, historical.identity)
#     task_variation_change = _relative_change(task_mapped.variation, historical.variation)
#     current_identity_residual = _relative_residual(residual.identity, current_structure.identity)
#     current_variation_residual = _relative_residual(residual.variation, current_structure.variation)
#     transferred_identity = calibrated.identity - task_mapped.identity
#     transferred_variation = calibrated.variation - task_mapped.variation
#     transferred_identity_relative = _relative_residual(transferred_identity, task_mapped.identity)
#     transferred_variation_relative = _relative_residual(transferred_variation, task_mapped.variation)

#     before = derive_class_relations(historical)
#     after = derive_class_relations(calibrated)
#     identity_rmse = _offdiag_rmse(after.learned_identity_cosine_by_region, before.learned_identity_cosine_by_region)
#     variation_rmse = _offdiag_rmse(after.learned_variation_overlap, before.learned_variation_overlap)
#     spectral_identity_rmse = _offdiag_rmse(after.spectral_identity_cosine_by_component, before.spectral_identity_cosine_by_component)
#     spectral_variation_rmse = _offdiag_rmse(after.spectral_variation_overlap, before.spectral_variation_overlap)
#     entropy, concentration, zero_fraction = _affinity_diagnostics(relation, historical.identity.dtype)

#     diagnostics = PhaseAlignmentDiagnostics(
#         task_mapping_error=mapping["task_mapping_error"],
#         task_mapping_channel_error_mean=mapping["task_mapping_channel_error_mean"],
#         task_mapping_channel_error_max=mapping["task_mapping_channel_error_max"],
#         representation_drift_rms=mapping["representation_drift_rms"],
#         task_translation_rms=mapping["task_translation_rms"],
#         task_scale_deviation_rms=mapping["task_scale_deviation_rms"],
#         task_identity_relative_shift_mean=task_identity_shift.mean(),
#         task_identity_relative_shift_max=task_identity_shift.max(),
#         task_variation_relative_change_mean=task_variation_change.mean(),
#         task_variation_relative_change_max=task_variation_change.max(),
#         current_identity_residual_relative_mean=current_identity_residual.mean(),
#         current_identity_residual_relative_max=current_identity_residual.max(),
#         current_variation_residual_relative_mean=current_variation_residual.mean(),
#         current_variation_residual_relative_max=current_variation_residual.max(),
#         transferred_identity_residual_relative_mean=transferred_identity_relative.mean(),
#         transferred_identity_residual_relative_max=transferred_identity_relative.max(),
#         transferred_variation_residual_relative_mean=transferred_variation_relative.mean(),
#         transferred_variation_residual_relative_max=transferred_variation_relative.max(),
#         calibrated_identity_relation_rmse=identity_rmse,
#         calibrated_variation_relation_rmse=variation_rmse,
#         spectral_identity_relation_rmse=spectral_identity_rmse,
#         spectral_variation_relation_rmse=spectral_variation_rmse,
#         affinity_entropy_mean=entropy,
#         affinity_concentration_mean=concentration,
#         affinity_zero_row_fraction=zero_fraction,
#     )
#     for field in diagnostics.__dataclass_fields__:
#         if not bool(torch.isfinite(getattr(diagnostics, field))):
#             raise RuntimeError(f"phase alignment diagnostic {field} is NaN or Inf")
#     return task_mapped, calibrated, transform, relation, residual, diagnostics


# __all__ = [
#     "PhaseCoordinateTransform",
#     "CDSAffinityState",
#     "CurrentClassResidual",
#     "PhaseAlignmentDiagnostics",
#     "estimate_task_representation_correspondence",
#     "apply_task_projection",
#     "estimate_current_class_residual",
#     "derive_cds_affinity",
#     "calibrate_historical_structure",
#     "align_historical_structure",
# ]







# from __future__ import annotations

# """Phase-local transport of the finalized learned HSI CDS.

# The ordered spectral-token axis K is never mixed. Paired legal current samples
# encoded by the frozen Phase-(t-1) backbone (Q-) and the current backbone (Q+)
# estimate one shared scale per learned channel plus a token-specific translation:

#     Q+[n,k,d] ~= s[d] * Q-[n,k,d] + t[k,d].

# This restriction is deliberate. The finalized V_Q stores pairwise relation
# *deformation*, not covariance. Under the map above,

#     V_Q+[k,l,d] = |s[d]| * V_Q-[k,l,d],

# so the historical CDS can be transported exactly without a covariance rule.

# Current-class residuals and CDS affinities are retained only as diagnostics.
# They are never transferred into historical classes. Physical {I_P,V_P} is
# phase-stable and is never transported. Persistent memory remains exactly
# {I_Q,V_Q,I_P,V_P,N}.
# """

# from dataclasses import dataclass
# import math

# import torch

# from models.geometry_bank import (
#     ClassStructureBatch,
#     derive_class_relations,
#     merge_class_structure_batches,
#     sanitize_ordered_variation,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class PhaseCoordinateTransform:
#     task_scale: Tensor        # [K,D], channel-shared over K and repeated for API compatibility
#     task_translation: Tensor  # [K,D]


# @dataclass(frozen=True)
# class CDSAffinityState:
#     old_class_ids: Tensor
#     new_class_ids: Tensor
#     learned_identity_affinity: Tensor
#     learned_variation_affinity: Tensor
#     spectral_affinity: Tensor
#     joint_affinity: Tensor
#     transfer_weights: Tensor


# @dataclass(frozen=True)
# class CurrentClassResidual:
#     class_ids: Tensor
#     identity: Tensor
#     variation: Tensor


# @dataclass(frozen=True)
# class PhaseAlignmentDiagnostics:
#     task_mapping_error: Tensor
#     task_mapping_channel_error_mean: Tensor
#     task_mapping_channel_error_max: Tensor
#     representation_drift_rms: Tensor
#     task_translation_rms: Tensor
#     task_scale_deviation_rms: Tensor
#     task_identity_relative_shift_mean: Tensor
#     task_identity_relative_shift_max: Tensor
#     task_variation_relative_change_mean: Tensor
#     task_variation_relative_change_max: Tensor
#     current_identity_residual_relative_mean: Tensor
#     current_identity_residual_relative_max: Tensor
#     current_variation_residual_relative_mean: Tensor
#     current_variation_residual_relative_max: Tensor
#     transferred_identity_residual_relative_mean: Tensor
#     transferred_identity_residual_relative_max: Tensor
#     transferred_variation_residual_relative_mean: Tensor
#     transferred_variation_residual_relative_max: Tensor
#     calibrated_identity_relation_rmse: Tensor
#     calibrated_variation_relation_rmse: Tensor
#     spectral_identity_relation_rmse: Tensor
#     spectral_variation_relation_rmse: Tensor
#     affinity_entropy_mean: Tensor
#     affinity_concentration_mean: Tensor
#     affinity_zero_row_fraction: Tensor


# def _validate_paired(reference_features: Tensor, current_features: Tensor) -> tuple[Tensor, Tensor]:
#     reference = torch.as_tensor(reference_features)
#     current = torch.as_tensor(current_features)
#     if reference.ndim != 3 or reference.size(0) < 2 or reference.size(1) < 2 or reference.size(2) < 1:
#         raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if reference.device != current.device or reference.dtype != current.dtype:
#         raise ValueError("paired Q-/Q+ features must share device/dtype")
#     if not torch.is_floating_point(reference) or reference.is_complex():
#         raise TypeError("paired Q-/Q+ features must be real floating point")
#     if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(current).all()):
#         raise ValueError("paired Q-/Q+ features contain NaN or Inf")
#     return reference, current


# def _validate_structure_pair(reference: ClassStructureBatch, current: ClassStructureBatch) -> None:
#     if not isinstance(reference, ClassStructureBatch) or not isinstance(current, ClassStructureBatch):
#         raise TypeError("current reference/current structures must be ClassStructureBatch")
#     if reference.class_count <= 0 or current.class_count != reference.class_count:
#         raise ValueError("current reference/current structures have incompatible class counts")
#     if not torch.equal(reference.class_ids, current.class_ids):
#         raise ValueError("current reference/current structures must use identical class order")
#     if reference.identity.shape != current.identity.shape or reference.variation.shape != current.variation.shape:
#         raise ValueError("current reference/current learned CDS shapes disagree")
#     tensors = (
#         reference.identity,
#         reference.variation,
#         reference.spectral_identity,
#         reference.spectral_variation,
#         current.identity,
#         current.variation,
#         current.spectral_identity,
#         current.spectral_variation,
#     )
#     if any(v.device != reference.identity.device or v.dtype != reference.identity.dtype for v in tensors):
#         raise ValueError("current reference/current CDS must share device/dtype")
#     if not all(bool(torch.isfinite(v).all()) for v in tensors):
#         raise ValueError("current reference/current CDS contains NaN or Inf")


# def _relative_change(after: Tensor, before: Tensor) -> Tensor:
#     eps = torch.finfo(before.dtype).eps
#     numerator = torch.linalg.vector_norm((after - before).reshape(after.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(before.reshape(before.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _relative_residual(residual: Tensor, target: Tensor) -> Tensor:
#     eps = torch.finfo(target.dtype).eps
#     numerator = torch.linalg.vector_norm(residual.reshape(residual.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(target.reshape(target.size(0), -1), dim=1).clamp_min(eps)
#     return numerator / denominator


# def _offdiag_rmse(after: Tensor, before: Tensor) -> Tensor:
#     if after.shape != before.shape or after.ndim < 2 or after.size(0) != after.size(1):
#         raise ValueError("relation tensors must share [C,C,...] shape")
#     c = int(after.size(0))
#     if c <= 1:
#         return after.new_zeros(())
#     mask = ~torch.eye(c, device=after.device, dtype=torch.bool)
#     return torch.sqrt((after[mask] - before[mask]).square().mean().clamp_min(0.0))


# @torch.no_grad()
# def estimate_task_representation_correspondence(
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
#     """Fit a CDS-compatible shared Q- -> Q+ affine drift.

#     A single scale ``s[d]`` is fitted for each learned channel across all
#     spectral tokens, while ``t[k,d]`` remains token-specific. This is the
#     simplest affine family that preserves the pairwise-deformation semantics of
#     V_Q exactly. The returned scale is repeated to [K,D] only to preserve the
#     surrounding repository API.
#     """
#     reference, current = _validate_paired(reference_features, current_features)
#     x = reference.detach().to(device="cpu", dtype=torch.float64)
#     y = current.detach().to(device="cpu", dtype=torch.float64)
#     _, k, d = x.shape

#     # Token-specific intercepts are removed before estimating one channel scale.
#     mean_x = x.mean(dim=0)  # [K,D]
#     mean_y = y.mean(dim=0)  # [K,D]
#     xc = x - mean_x.unsqueeze(0)
#     yc = y - mean_y.unsqueeze(0)

#     numerator = (xc * yc).sum(dim=(0, 1))      # [D]
#     denominator = xc.square().sum(dim=(0, 1)) # [D]
#     eps64 = torch.finfo(torch.float64).eps
#     tolerance = eps64 * denominator.max()
#     identifiable = denominator > tolerance
#     channel_scale = torch.ones_like(denominator)
#     channel_scale[identifiable] = numerator[identifiable] / denominator[identifiable]

#     scale = channel_scale.unsqueeze(0).expand(k, d).clone()
#     translation = mean_y - scale * mean_x
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise RuntimeError("CDS-compatible phase map contains NaN or Inf")

#     prediction = x * scale.unsqueeze(0) + translation.unsqueeze(0)
#     residual = y - prediction
#     tiny = torch.finfo(torch.float64).tiny
#     channel_residual = residual.permute(2, 0, 1)
#     channel_target = y.permute(2, 0, 1)
#     channel_error = (
#         channel_residual.square().sum(dim=(1, 2))
#         / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
#     )
#     mapping_error = residual.square().sum() / y.square().sum().clamp_min(tiny)
#     drift_rms = torch.sqrt((y - x).square().mean().clamp_min(0.0))
#     translation_rms = torch.sqrt(translation.square().mean().clamp_min(0.0))
#     scale_deviation_rms = torch.sqrt(
#         (channel_scale - 1.0).square().mean().clamp_min(0.0)
#     )

#     device, dtype = reference.device, reference.dtype
#     transform = PhaseCoordinateTransform(
#         task_scale=scale.to(device=device, dtype=dtype).detach(),
#         task_translation=translation.to(device=device, dtype=dtype).detach(),
#     )
#     diagnostics = {
#         "task_mapping_error": mapping_error.to(device=device, dtype=dtype),
#         "task_mapping_channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
#         "task_mapping_channel_error_max": channel_error.max().to(device=device, dtype=dtype),
#         "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
#         "task_translation_rms": translation_rms.to(device=device, dtype=dtype),
#         "task_scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
#     }
#     return transform, diagnostics


# @torch.no_grad()
# def apply_task_projection(
#     structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> ClassStructureBatch:
#     """Transport I_Q and pairwise-deformation V_Q into current Q coordinates.

#     For the CDS-compatible map Q+ = s[d] Q- + t[k,d]:

#         I_Q+ = s[d] I_Q- + t[k,d]
#         V_Q+ = |s[d]| V_Q-

#     Translation cancels exactly from the pairwise deformation. No covariance,
#     PSD transport, eigendecomposition, or residual migration is involved.
#     """
#     if not isinstance(structure, ClassStructureBatch) or structure.class_count <= 0:
#         raise ValueError("structure must be a non-empty ClassStructureBatch")
#     _, k, d = structure.identity.shape
#     scale = transform.task_scale
#     translation = transform.task_translation
#     if scale.shape != (k, d) or translation.shape != (k, d):
#         raise ValueError("task affine map shape does not match CDS")
#     if any(v.device != structure.identity.device or v.dtype != structure.identity.dtype for v in (scale, translation)):
#         raise ValueError("task affine map and CDS must share device/dtype")
#     if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(translation).all()):
#         raise ValueError("task affine map contains NaN or Inf")

#     # The transport is valid only when scale is shared over ordered token K.
#     channel_scale = scale[0]
#     expected = channel_scale.unsqueeze(0).expand_as(scale)
#     tolerance = 64.0 * torch.finfo(scale.dtype).eps
#     if not bool(torch.allclose(scale, expected, rtol=tolerance, atol=tolerance)):
#         raise RuntimeError("V_Q deformation transport requires channel-shared scale over K")

#     mapped_identity = structure.identity * scale.unsqueeze(0) + translation.unsqueeze(0)
#     mapped_variation = sanitize_ordered_variation(
#         structure.variation * channel_scale.abs().view(1, 1, 1, d)
#     )
#     return ClassStructureBatch(
#         class_ids=structure.class_ids.detach().clone(),
#         sample_counts=structure.sample_counts.detach().clone(),
#         identity=mapped_identity.detach().contiguous(),
#         variation=mapped_variation.detach().contiguous(),
#         spectral_identity=structure.spectral_identity.detach().clone(),
#         spectral_variation=structure.spectral_variation.detach().clone(),
#     )


# @torch.no_grad()
# def estimate_current_class_residual(
#     *,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> CurrentClassResidual:
#     """Observed class-specific migration not explained by shared phase drift."""
#     _validate_structure_pair(current_reference, current_structure)
#     predicted = apply_task_projection(current_reference, transform)
#     if not torch.allclose(current_reference.spectral_identity, current_structure.spectral_identity, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral identity changed across neural coordinates")
#     if not torch.allclose(current_reference.spectral_variation, current_structure.spectral_variation, rtol=1e-6, atol=1e-7):
#         raise RuntimeError("current physical spectral variation changed across neural coordinates")
#     identity = current_structure.identity - predicted.identity
#     variation = current_structure.variation - predicted.variation
#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
#         raise RuntimeError("current-class residual contains NaN or Inf")
#     return CurrentClassResidual(
#         class_ids=current_structure.class_ids.detach().clone(),
#         identity=identity.detach(),
#         variation=variation.detach(),
#     )


# @torch.no_grad()
# def derive_cds_affinity(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
# ) -> CDSAffinityState:
#     """Derive phase-start old/current HSI-CDS affinity for diagnostics.

#     The geometry bank already returns normalized L1 similarities in [0,1].
#     They are used directly; the previous cosine remapping 0.5*(1+r) was invalid
#     for the finalized bank. Affinity is not used to move historical CDS.
#     """
#     if historical.class_count <= 0 or current_reference.class_count <= 0:
#         raise ValueError("historical/current reference CDS must be non-empty")
#     if historical.identity.device != current_reference.identity.device or historical.identity.dtype != current_reference.identity.dtype:
#         raise ValueError("historical/current reference CDS must share device/dtype")
#     old_values = set(int(v) for v in historical.class_ids.detach().cpu().tolist())
#     new_values = set(int(v) for v in current_reference.class_ids.detach().cpu().tolist())
#     if old_values.intersection(new_values):
#         raise ValueError("historical/current reference classes overlap")

#     merged = merge_class_structure_batches((historical, current_reference))
#     relations = derive_class_relations(merged)
#     old_count = historical.class_count

#     learned_identity = (
#         relations.learned_identity_cosine_by_region[:old_count, old_count:].mean(dim=2)
#     ).clamp(0.0, 1.0)
#     learned_variation = (
#         relations.learned_variation_overlap[:old_count, old_count:]
#     ).clamp(0.0, 1.0)
#     spectral = (
#         relations.spectral_confusability[:old_count, old_count:]
#     ).clamp(0.0, 1.0)

#     learned = 0.5 * (learned_identity + learned_variation)
#     joint = torch.sqrt(
#         learned.clamp_min(0.0) * spectral.clamp_min(0.0)
#     ).clamp(0.0, 1.0)

#     tiny = torch.finfo(joint.dtype).tiny
#     row_sum = joint.sum(dim=1, keepdim=True)
#     active = row_sum.squeeze(1) > tiny
#     weights = torch.zeros_like(joint)
#     if bool(active.any()):
#         weights[active] = joint[active] / row_sum[active]

#     return CDSAffinityState(
#         old_class_ids=historical.class_ids.detach().clone(),
#         new_class_ids=current_reference.class_ids.detach().clone(),
#         learned_identity_affinity=learned_identity.detach(),
#         learned_variation_affinity=learned_variation.detach(),
#         spectral_affinity=spectral.detach(),
#         joint_affinity=joint.detach(),
#         transfer_weights=weights.detach(),
#     )


# @torch.no_grad()
# def calibrate_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     task_mapped_historical: ClassStructureBatch,
#     residual: CurrentClassResidual,
#     affinity: CDSAffinityState,
# ) -> ClassStructureBatch:
#     """Return the shared-drift transported historical CDS unchanged.

#     This function name is retained only for repository compatibility. The
#     finalized architecture does *not* transfer current-class residuals into
#     unavailable historical classes. ``residual`` and ``affinity`` are validated
#     and reported as diagnostics only.
#     """
#     if not torch.equal(historical.class_ids, task_mapped_historical.class_ids):
#         raise ValueError("task-mapped historical class IDs changed")
#     if not torch.equal(historical.class_ids, affinity.old_class_ids):
#         raise ValueError("affinity old class IDs do not match historical CDS")
#     if not torch.equal(residual.class_ids, affinity.new_class_ids):
#         raise ValueError("residual class IDs do not match current classes")

#     return ClassStructureBatch(
#         class_ids=task_mapped_historical.class_ids.detach().clone(),
#         sample_counts=task_mapped_historical.sample_counts.detach().clone(),
#         identity=task_mapped_historical.identity.detach().clone(),
#         variation=task_mapped_historical.variation.detach().clone(),
#         spectral_identity=task_mapped_historical.spectral_identity.detach().clone(),
#         spectral_variation=task_mapped_historical.spectral_variation.detach().clone(),
#     )


# def _affinity_diagnostics(affinity: CDSAffinityState, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
#     weights = affinity.transfer_weights
#     active = weights.sum(dim=1) > 0
#     zero_fraction = (~active).to(dtype).mean()
#     if not bool(active.any()):
#         zero = weights.new_zeros(())
#         return zero, zero, zero_fraction
#     w = weights[active]
#     concentration = w.max(dim=1).values.mean()
#     if w.size(1) <= 1:
#         entropy = weights.new_zeros(())
#     else:
#         tiny = torch.finfo(weights.dtype).tiny
#         entropy = (-(w * w.clamp_min(tiny).log()).sum(dim=1) / math.log(float(w.size(1)))).mean()
#     return entropy.to(dtype), concentration.to(dtype), zero_fraction.to(dtype)


# @torch.no_grad()
# def align_historical_structure(
#     *,
#     historical: ClassStructureBatch,
#     current_reference: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     reference_features: Tensor,
#     current_features: Tensor,
#     affinity: CDSAffinityState | None = None,
# ) -> tuple[
#     ClassStructureBatch,
#     ClassStructureBatch,
#     PhaseCoordinateTransform,
#     CDSAffinityState,
#     CurrentClassResidual,
#     PhaseAlignmentDiagnostics,
# ]:
#     """Transport historical CDS with shared observable drift only; no residual transfer."""
#     transform, mapping = estimate_task_representation_correspondence(reference_features, current_features)
#     task_mapped = apply_task_projection(historical, transform)
#     residual = estimate_current_class_residual(
#         current_reference=current_reference,
#         current_structure=current_structure,
#         transform=transform,
#     )
#     relation = affinity or derive_cds_affinity(
#         historical=historical,
#         current_reference=current_reference,
#     )
#     calibrated = calibrate_historical_structure(
#         historical=historical,
#         task_mapped_historical=task_mapped,
#         residual=residual,
#         affinity=relation,
#     )

#     task_identity_shift = _relative_change(task_mapped.identity, historical.identity)
#     task_variation_change = _relative_change(task_mapped.variation, historical.variation)
#     current_identity_residual = _relative_residual(residual.identity, current_structure.identity)
#     current_variation_residual = _relative_residual(residual.variation, current_structure.variation)
#     transferred_identity = calibrated.identity - task_mapped.identity
#     transferred_variation = calibrated.variation - task_mapped.variation
#     transferred_identity_relative = _relative_residual(transferred_identity, task_mapped.identity)
#     transferred_variation_relative = _relative_residual(transferred_variation, task_mapped.variation)

#     before = derive_class_relations(historical)
#     after = derive_class_relations(calibrated)
#     identity_rmse = _offdiag_rmse(after.learned_identity_cosine_by_region, before.learned_identity_cosine_by_region)
#     variation_rmse = _offdiag_rmse(after.learned_variation_overlap, before.learned_variation_overlap)
#     spectral_identity_rmse = _offdiag_rmse(after.spectral_identity_cosine_by_component, before.spectral_identity_cosine_by_component)
#     spectral_variation_rmse = _offdiag_rmse(after.spectral_variation_overlap, before.spectral_variation_overlap)
#     entropy, concentration, zero_fraction = _affinity_diagnostics(relation, historical.identity.dtype)

#     diagnostics = PhaseAlignmentDiagnostics(
#         task_mapping_error=mapping["task_mapping_error"],
#         task_mapping_channel_error_mean=mapping["task_mapping_channel_error_mean"],
#         task_mapping_channel_error_max=mapping["task_mapping_channel_error_max"],
#         representation_drift_rms=mapping["representation_drift_rms"],
#         task_translation_rms=mapping["task_translation_rms"],
#         task_scale_deviation_rms=mapping["task_scale_deviation_rms"],
#         task_identity_relative_shift_mean=task_identity_shift.mean(),
#         task_identity_relative_shift_max=task_identity_shift.max(),
#         task_variation_relative_change_mean=task_variation_change.mean(),
#         task_variation_relative_change_max=task_variation_change.max(),
#         current_identity_residual_relative_mean=current_identity_residual.mean(),
#         current_identity_residual_relative_max=current_identity_residual.max(),
#         current_variation_residual_relative_mean=current_variation_residual.mean(),
#         current_variation_residual_relative_max=current_variation_residual.max(),
#         transferred_identity_residual_relative_mean=transferred_identity_relative.mean(),
#         transferred_identity_residual_relative_max=transferred_identity_relative.max(),
#         transferred_variation_residual_relative_mean=transferred_variation_relative.mean(),
#         transferred_variation_residual_relative_max=transferred_variation_relative.max(),
#         calibrated_identity_relation_rmse=identity_rmse,
#         calibrated_variation_relation_rmse=variation_rmse,
#         spectral_identity_relation_rmse=spectral_identity_rmse,
#         spectral_variation_relation_rmse=spectral_variation_rmse,
#         affinity_entropy_mean=entropy,
#         affinity_concentration_mean=concentration,
#         affinity_zero_row_fraction=zero_fraction,
#     )
#     for field in diagnostics.__dataclass_fields__:
#         if not bool(torch.isfinite(getattr(diagnostics, field))):
#             raise RuntimeError(f"phase alignment diagnostic {field} is NaN or Inf")
#     return task_mapped, calibrated, transform, relation, residual, diagnostics


# __all__ = [
#     "PhaseCoordinateTransform",
#     "CDSAffinityState",
#     "CurrentClassResidual",
#     "PhaseAlignmentDiagnostics",
#     "estimate_task_representation_correspondence",
#     "apply_task_projection",
#     "estimate_current_class_residual",
#     "derive_cds_affinity",
#     "calibrate_historical_structure",
#     "align_historical_structure",
# ]
