from __future__ import annotations
from dataclasses import dataclass

import torch

from models.geometry_bank import (
    ClassStructureBatch,
    combine_learned_physical_relevance,
    derive_class_relations,
    merge_class_structure_batches,
    normalize_ordered_decision_features,
    sanitize_ordered_variation,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class PhaseCoordinateTransform:
    """Single phase-wise map from finalized Q_(t-1) to finalized Q_t.

    ``channel_scale`` is shared over the ordered spectral-token axis so the
    Whole-CDS variation field has an exact compatible transport
    ``V_Q' = |s| V_Q``. ``token_translation`` may differ by ordered token.
    """

    channel_scale: Tensor  # [D]
    token_translation: Tensor  # [K,D]


@dataclass(frozen=True)
class PhaseAlignmentDiagnostics:
    mapping_error: Tensor
    channel_error_mean: Tensor
    channel_error_max: Tensor
    representation_drift_rms: Tensor
    translation_rms: Tensor
    scale_deviation_rms: Tensor
    learned_relevance_mean: Tensor
    physical_relevance_mean: Tensor
    effective_relevance_mean: Tensor
    class_specific_support_mass_mean: Tensor
    class_specific_support_mass_min: Tensor
    class_specific_support_mass_max: Tensor
    identity_correction_relative_mean: Tensor
    identity_correction_relative_max: Tensor
    variation_correction_relative_mean: Tensor
    variation_correction_relative_max: Tensor
    variation_negative_fraction_before_sanitize: Tensor


def _validate_paired_features(
    reference_features: Tensor,
    current_features: Tensor,
) -> tuple[Tensor, Tensor]:
    reference = torch.as_tensor(reference_features)
    current = torch.as_tensor(current_features)
    if (
        reference.ndim != 3
        or reference.size(0) < 2
        or reference.size(1) < 2
        or reference.size(2) <= 0
    ):
        raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
    if current.shape != reference.shape:
        raise ValueError("current_features must match reference_features")
    if reference.device != current.device or reference.dtype != current.dtype:
        raise ValueError("paired Q-/Q+ features must share device/dtype")
    if not torch.is_floating_point(reference) or reference.is_complex():
        raise TypeError("paired Q-/Q+ features must be real floating point")
    if not bool(torch.isfinite(reference).all()) or not bool(
        torch.isfinite(current).all()
    ):
        raise ValueError("paired Q-/Q+ features contain NaN or Inf")
    return reference, current


def _validate_structure_pair(
    reference: ClassStructureBatch, current: ClassStructureBatch
) -> None:
    if not isinstance(reference, ClassStructureBatch) or not isinstance(
        current, ClassStructureBatch
    ):
        raise TypeError("reference/current structures must be ClassStructureBatch")
    if reference.class_count <= 0 or current.class_count != reference.class_count:
        raise ValueError("reference/current structures have incompatible class counts")
    if not torch.equal(reference.class_ids, current.class_ids):
        raise ValueError(
            "reference/current class IDs must be identical and ordered equally"
        )
    if reference.learned_identity.shape != current.learned_identity.shape:
        raise ValueError("reference/current I_Q shapes disagree")
    if reference.learned_variation.shape != current.learned_variation.shape:
        raise ValueError("reference/current V_Q shapes disagree")
    # Physical evidence is a native-spectrum coordinate and must never be moved by
    # a neural feature-space transform.
    if not torch.equal(reference.physical_identity, current.physical_identity):
        raise ValueError("current-class I_P changed across neural coordinates")
    if not torch.equal(reference.physical_variation, current.physical_variation):
        raise ValueError("current-class V_P changed across neural coordinates")


def _relative_batch_norm(delta: Tensor, reference: Tensor) -> Tensor:
    """Per-class relative L2 change with a finite zero-safe denominator."""
    if delta.shape != reference.shape or delta.ndim < 2 or delta.size(0) <= 0:
        raise ValueError("relative-norm tensors must share a non-empty class dimension")
    eps = torch.finfo(reference.dtype).eps
    numerator = torch.linalg.vector_norm(delta.reshape(delta.size(0), -1), dim=1)
    denominator = torch.linalg.vector_norm(
        reference.reshape(reference.size(0), -1), dim=1
    ).clamp_min(eps)
    return numerator / denominator


@torch.no_grad()
def estimate_phase_coordinate_transform(
    reference_features: Tensor,
    current_features: Tensor,
) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
    """Estimate a conservative observable Q_(t-1) -> Q_t map.

    The previous implementation used an unconstrained per-channel least-squares
    slope.  On Indian Pines it reported only ~1.2% paired-Q drift but ~46% scale
    change, which then moved historical CDS far more than the representation had
    actually moved.

    Here each channel solves an identity-centred ridge problem.  The ridge mass is
    *data-derived* from the observed paired drift energy, not a hand-set clamp::

        s_d = (x_d^T y_d + lambda_d) / (x_d^T x_d + lambda_d)
        lambda_d = ||y_d - x_d||^2 + eps

    When a channel is weakly identifiable or noisy, the solution continuously
    shrinks toward ``s_d = 1``.  Well-supported channels remain data-driven.  No
    old-class-specific motion is inferred from unseen old samples.
    """
    reference, current = _validate_paired_features(reference_features, current_features)
    reference = normalize_ordered_decision_features(reference)
    current = normalize_ordered_decision_features(current)

    source = reference.detach().to(device="cpu", dtype=torch.float64)
    target = current.detach().to(device="cpu", dtype=torch.float64)
    _, token_count, channel_count = source.shape

    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean.unsqueeze(0)
    target_centered = target - target_mean.unsqueeze(0)

    cross = (source_centered * target_centered).sum(dim=(0, 1))
    source_energy = source_centered.square().sum(dim=(0, 1))
    drift_energy = (target_centered - source_centered).square().sum(dim=(0, 1))
    eps = torch.finfo(torch.float64).eps
    numerical = eps * source_energy.max().clamp_min(1.0)
    ridge = drift_energy + numerical
    channel_scale = (cross + ridge) / (source_energy + ridge)
    token_translation = target_mean - channel_scale.unsqueeze(0) * source_mean

    prediction = source * channel_scale.view(
        1, 1, channel_count
    ) + token_translation.view(1, token_count, channel_count)
    residual = target - prediction
    tiny = torch.finfo(torch.float64).tiny
    channel_residual = residual.permute(2, 0, 1)
    channel_target = target.permute(2, 0, 1)
    channel_error = channel_residual.square().sum(
        dim=(1, 2)
    ) / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
    mapping_error = residual.square().sum() / target.square().sum().clamp_min(tiny)
    drift_rms = torch.sqrt((target - source).square().mean().clamp_min(0.0))
    translation_rms = torch.sqrt(token_translation.square().mean().clamp_min(0.0))
    scale_deviation_rms = torch.sqrt(
        (channel_scale - 1.0).square().mean().clamp_min(0.0)
    )

    values = (
        channel_scale,
        token_translation,
        mapping_error,
        channel_error,
        drift_rms,
        translation_rms,
        scale_deviation_rms,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("phase coordinate transform contains NaN or Inf")

    device, dtype = reference.device, reference.dtype
    transform = PhaseCoordinateTransform(
        channel_scale=channel_scale.to(device=device, dtype=dtype),
        token_translation=token_translation.to(device=device, dtype=dtype),
    )
    mapping = {
        "mapping_error": mapping_error.to(device=device, dtype=dtype),
        "channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
        "channel_error_max": channel_error.max().to(device=device, dtype=dtype),
        "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
        "translation_rms": translation_rms.to(device=device, dtype=dtype),
        "scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
    }
    return transform, mapping


@torch.no_grad()
def transport_historical_structure(
    historical_structure: ClassStructureBatch,
    transform: PhaseCoordinateTransform,
) -> ClassStructureBatch:
    """Transport learned historical CDS once; physical CDS is copied exactly."""
    if (
        not isinstance(historical_structure, ClassStructureBatch)
        or historical_structure.class_count <= 0
    ):
        raise ValueError("historical_structure must be non-empty")

    learned_identity = historical_structure.learned_identity
    learned_variation = historical_structure.learned_variation
    _, token_count, channel_count = learned_identity.shape

    scale = torch.as_tensor(transform.channel_scale)
    translation = torch.as_tensor(transform.token_translation)
    if scale.shape != (channel_count,):
        raise ValueError("channel_scale must have shape [D]")
    if translation.shape != (token_count, channel_count):
        raise ValueError("token_translation must have shape [K,D]")
    if scale.device != learned_identity.device or scale.dtype != learned_identity.dtype:
        raise ValueError("phase transform and historical CDS must share device/dtype")
    if (
        translation.device != learned_identity.device
        or translation.dtype != learned_identity.dtype
    ):
        raise ValueError("phase transform and historical CDS must share device/dtype")
    if not bool(torch.isfinite(scale).all()) or not bool(
        torch.isfinite(translation).all()
    ):
        raise ValueError("phase transform contains NaN or Inf")

    transported_identity = learned_identity * scale.view(
        1, 1, channel_count
    ) + translation.view(1, token_count, channel_count)
    transported_variation = sanitize_ordered_variation(
        learned_variation * scale.abs().view(1, 1, 1, channel_count)
    )

    return ClassStructureBatch(
        class_ids=historical_structure.class_ids.detach().clone(),
        sample_counts=historical_structure.sample_counts.detach().clone(),
        learned_identity=transported_identity.detach().contiguous(),
        learned_variation=transported_variation.detach().contiguous(),
        physical_identity=historical_structure.physical_identity.detach().clone(),
        physical_variation=historical_structure.physical_variation.detach().clone(),
    )


@torch.no_grad()
def _whole_cds_relevance(
    historical_structure: ClassStructureBatch,
    current_reference_structure: ClassStructureBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Old x current Whole-CDS relevance in the finalized previous Q coordinate.

    The learned and physical CDS jointly decide how much each observed current-
    class residual is relevant to each historical class.  Row mass is bounded by
    the strongest available evidence, so weakly related old classes fall back to
    the conservative global phase transform instead of receiving arbitrary motion.
    """
    if (
        historical_structure.class_count <= 0
        or current_reference_structure.class_count <= 0
    ):
        raise ValueError("historical/current structures must be non-empty")
    old_ids = set(
        int(v) for v in historical_structure.class_ids.detach().cpu().tolist()
    )
    new_ids = set(
        int(v) for v in current_reference_structure.class_ids.detach().cpu().tolist()
    )
    if old_ids.intersection(new_ids):
        raise ValueError("historical/current class sets overlap")

    merged = merge_class_structure_batches(
        (historical_structure, current_reference_structure)
    )
    relations = derive_class_relations(merged)
    old_count = historical_structure.class_count
    learned = relations.learned_combined_proximity[:old_count, old_count:].clamp(
        0.0, 1.0
    )
    physical = relations.physical_combined_proximity[:old_count, old_count:].clamp(
        0.0, 1.0
    )
    effective = combine_learned_physical_relevance(learned, physical)

    eps = torch.finfo(effective.dtype).eps
    row_sum = effective.sum(dim=1, keepdim=True)
    normalized = torch.where(
        row_sum > eps,
        effective / row_sum.clamp_min(eps),
        torch.zeros_like(effective),
    )
    confidence = effective.max(dim=1, keepdim=True).values
    weights = normalized * confidence
    if not all(
        bool(torch.isfinite(v).all()) for v in (learned, physical, effective, weights)
    ):
        raise RuntimeError("CDS relevance became NaN or Inf")
    return learned, physical, effective, weights


@torch.no_grad()
def correct_historical_structure(
    *,
    historical_structure: ClassStructureBatch,
    current_reference_structure: ClassStructureBatch,
    current_structure: ClassStructureBatch,
    reference_features: Tensor,
    current_features: Tensor,
) -> tuple[ClassStructureBatch, PhaseCoordinateTransform, PhaseAlignmentDiagnostics]:
    """Correct historical learned CDS in the finalized current Q coordinate.

    One observable phase map supplies the global transport. The only class-specific
    term is a bounded residual transferred from *observed current classes*. Whole-CDS
    relevance controls the transfer mass, whose row sum is in [0,1]; weak evidence
    therefore falls back continuously to the global map. Physical I_P/V_P are never
    transported or altered.
    """
    if (
        not isinstance(historical_structure, ClassStructureBatch)
        or historical_structure.class_count <= 0
    ):
        raise ValueError("historical_structure must be non-empty")
    _validate_structure_pair(current_reference_structure, current_structure)

    transform, mapping = estimate_phase_coordinate_transform(
        reference_features, current_features
    )
    task_old = transport_historical_structure(historical_structure, transform)
    task_current = transport_historical_structure(
        current_reference_structure, transform
    )

    # Observable current-class residual after the global map.
    identity_residual = (
        current_structure.learned_identity - task_current.learned_identity
    ).detach()

    # V_Q is a non-negative ordered scale field. Transfer relative scale change
    # multiplicatively; zero/weak baselines use the neutral ratio 1.
    task_current_variation = task_current.learned_variation.detach()
    observed_current_variation = current_structure.learned_variation.detach()
    if bool((task_current_variation < 0).any()) or bool(
        (observed_current_variation < 0).any()
    ):
        raise RuntimeError("V_Q must remain non-negative during drift estimation")
    eps = torch.finfo(task_current_variation.dtype).eps
    ratio = torch.ones_like(observed_current_variation)
    identifiable = task_current_variation > eps
    ratio[identifiable] = (
        observed_current_variation[identifiable] / task_current_variation[identifiable]
    )
    tiny = torch.finfo(ratio.dtype).tiny
    variation_log_ratio = ratio.clamp_min(tiny).log().detach()
    if not bool(torch.isfinite(identity_residual).all()) or not bool(
        torch.isfinite(variation_log_ratio).all()
    ):
        raise RuntimeError("observed current-class residual contains NaN or Inf")

    learned, physical, effective, weights = _whole_cds_relevance(
        historical_structure, current_reference_structure
    )
    expected = (historical_structure.class_count, current_structure.class_count)
    if weights.shape != expected:
        raise RuntimeError("Whole-CDS residual-transfer weights have invalid shape")
    support_mass = weights.sum(dim=1)
    tolerance = 64.0 * torch.finfo(weights.dtype).eps
    if bool((support_mass < -tolerance).any()) or bool(
        (support_mass > 1.0 + tolerance).any()
    ):
        raise RuntimeError("Whole-CDS residual-transfer mass escaped [0,1]")

    # Positive weights with row mass <= 1 make this a bounded weighted mixture of
    # observed residuals plus an implicit zero-residual fallback.
    local_identity = torch.einsum("on,nkd->okd", weights, identity_residual)
    local_variation_log_scale = torch.einsum(
        "on,nkld->okld", weights, variation_log_ratio
    )
    variation_scale = local_variation_log_scale.exp()

    corrected_identity = task_old.learned_identity + local_identity
    raw_corrected_variation = task_old.learned_variation * variation_scale
    negative_fraction = (
        raw_corrected_variation.lt(0.0).to(raw_corrected_variation.dtype).mean()
    )
    if bool(negative_fraction > 0):
        raise RuntimeError("multiplicative V_Q residual correction became negative")
    corrected_variation = sanitize_ordered_variation(raw_corrected_variation)

    if not bool(torch.isfinite(corrected_identity).all()) or not bool(
        torch.isfinite(corrected_variation).all()
    ):
        raise RuntimeError("Whole-CDS historical correction became NaN or Inf")

    corrected = ClassStructureBatch(
        class_ids=historical_structure.class_ids.detach().clone(),
        sample_counts=historical_structure.sample_counts.detach().clone(),
        learned_identity=corrected_identity.detach().contiguous(),
        learned_variation=corrected_variation.detach().contiguous(),
        physical_identity=historical_structure.physical_identity.detach().clone(),
        physical_variation=historical_structure.physical_variation.detach().clone(),
    )
    if not torch.equal(
        corrected.physical_identity, historical_structure.physical_identity
    ):
        raise RuntimeError("historical I_P changed during coordinate correction")
    if not torch.equal(
        corrected.physical_variation, historical_structure.physical_variation
    ):
        raise RuntimeError("historical V_P changed during coordinate correction")

    identity_relative = _relative_batch_norm(local_identity, task_old.learned_identity)
    variation_delta = corrected_variation - task_old.learned_variation
    variation_relative = _relative_batch_norm(
        variation_delta, task_old.learned_variation
    )

    diagnostics = PhaseAlignmentDiagnostics(
        mapping_error=mapping["mapping_error"],
        channel_error_mean=mapping["channel_error_mean"],
        channel_error_max=mapping["channel_error_max"],
        representation_drift_rms=mapping["representation_drift_rms"],
        translation_rms=mapping["translation_rms"],
        scale_deviation_rms=mapping["scale_deviation_rms"],
        learned_relevance_mean=learned.mean(),
        physical_relevance_mean=physical.mean(),
        effective_relevance_mean=effective.mean(),
        class_specific_support_mass_mean=support_mass.mean(),
        class_specific_support_mass_min=support_mass.min(),
        class_specific_support_mass_max=support_mass.max(),
        identity_correction_relative_mean=identity_relative.mean(),
        identity_correction_relative_max=identity_relative.max(),
        variation_correction_relative_mean=variation_relative.mean(),
        variation_correction_relative_max=variation_relative.max(),
        variation_negative_fraction_before_sanitize=negative_fraction,
    )
    for name in diagnostics.__dataclass_fields__:
        value = getattr(diagnostics, name)
        if not bool(torch.isfinite(value)):
            raise RuntimeError(f"phase alignment diagnostic {name} is NaN or Inf")
    return corrected, transform, diagnostics


__all__ = [
    "PhaseCoordinateTransform",
    "PhaseAlignmentDiagnostics",
    "estimate_phase_coordinate_transform",
    "transport_historical_structure",
    "correct_historical_structure",
]
