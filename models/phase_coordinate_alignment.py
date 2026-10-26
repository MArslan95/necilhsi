from __future__ import annotations
from dataclasses import dataclass, replace

import torch

from models.geometry_bank import (
    ClassStructureBatch,
    combine_learned_physical_relevance,
    derive_class_relations,
    merge_class_structure_batches,
    normalize_ordered_decision_features,
    sanitize_ordered_variation,
    physical_support_components,
    spectral_conditioned_learned_support,
    normalized_moment_translation_feasible,
)

Tensor = torch.Tensor


@dataclass(frozen=True)
class PhaseCoordinateTransform:
    """An optional held-out-validated low-rank orthogonal map in normalized Q.

    The rotation moves learned class means AND low-rank covariance bases in
    the same coordinates. Legacy fields remain for read-only diagnostics;
    mean-only translation is prohibited.
    """

    channel_scale: Tensor  # [D]
    token_translation: Tensor  # [K,D]
    class_translation: Tensor | None = None  # deprecated legacy diagnostic
    rotation_basis: Tensor | None = None  # [F,R] orthonormal columns
    rotation_matrix: Tensor | None = None  # [R,R] orthogonal, acts on row features


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
    effective_sample_size_mean: Tensor
    effective_sample_size_min: Tensor
    identity_correction_relative_mean: Tensor
    identity_correction_relative_max: Tensor
    variation_correction_relative_mean: Tensor
    variation_correction_relative_max: Tensor
    variation_negative_fraction_before_sanitize: Tensor
    # Observable new-class holdout test, not an old-class transfer certificate.
    holdout_candidate_error: Tensor
    holdout_identity_error: Tensor
    transport_accepted: Tensor
    per_class_holdout_candidate_error: Tensor | None = None
    per_class_holdout_identity_error: Tensor | None = None
    per_class_transport_accepted: Tensor | None = None


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
    for name in ("learned_absolute_deviation",):
        if getattr(reference, name).shape != getattr(current, name).shape:
            raise ValueError(f"reference/current {name} shapes disagree")
    # Physical evidence is a native-spectrum coordinate and must never be moved by
    # a neural feature-space transform.
    if not torch.equal(reference.physical_identity, current.physical_identity):
        raise ValueError("current-class I_P changed across neural coordinates")
    if not torch.equal(reference.physical_variation, current.physical_variation):
        raise ValueError("current-class V_P diagnostics changed")
    for name in ("physical_variance", "physical_delta_variance"):
        if not torch.equal(getattr(reference, name), getattr(current, name)):
            raise ValueError(f"current native spectral statistic {name} changed")
    for name in (
        "learned_covariance_basis",
        "learned_covariance_eigenvalues",
        "learned_covariance_residual",
    ):
        if getattr(reference, name).shape != getattr(current, name).shape:
            raise ValueError(f"learned covariance coordinate mismatch: {name}")


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
    *,
    allow_transport: bool = True,
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

    # MPS supports the device transfer in the input dtype, but not float64.
    # Promote only after both feature tensors are on CPU.
    source = reference.detach().to(device="cpu").to(dtype=torch.float64)
    target = current.detach().to(device="cpu").to(dtype=torch.float64)
    n, token_count, channel_count = source.shape

    # Validate a map on HELD-OUT current-class feature pairs before transporting
    # any old CDS. Fitting-error alone can favor a spurious phase correction.
    # Fixed alternating split is reproducible, never uses historical samples.
    # A failed holdout comparison retains the identity historical coordinate.
    def fit_map(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        xmean, ymean = x.mean(0), y.mean(0)
        xc, yc = x - xmean.unsqueeze(0), y - ymean.unsqueeze(0)
        cross_val = (xc * yc).sum((0, 1))
        energy_val = xc.square().sum((0, 1))
        drift_val = (yc - xc).square().sum((0, 1))
        regularizer = drift_val + torch.finfo(
            torch.float64
        ).eps * energy_val.max().clamp_min(1.0)
        slopes = (cross_val + regularizer) / (energy_val + regularizer)
        offsets = ymean - slopes.unsqueeze(0) * xmean
        return slopes, offsets

    accepted = False
    if n >= 6 and allow_transport:
        fit_index = torch.arange(n) % 2 == 0
        check_index = ~fit_index
        train_scale, train_offset = fit_map(source[fit_index], target[fit_index])
        held_prediction = source[check_index] * train_scale.view(
            1, 1, -1
        ) + train_offset.unsqueeze(0)
        held_candidate = (target[check_index] - held_prediction).square().mean()
        held_identity = (target[check_index] - source[check_index]).square().mean()
        accepted = bool(held_candidate + 1e-12 < held_identity)
    else:
        # Too few paired current samples for a meaningful split; retain the
        # identity map and report the observable identity error with a finite
        # value. The acceptance flag records that transport was not validated.
        held_identity = (target - source).square().mean()
        held_candidate = held_identity.clone()

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
    if not accepted:
        channel_scale = torch.ones_like(channel_scale)
        token_translation = torch.zeros_like(token_translation)

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
        "holdout_candidate_error": held_candidate.to(device=device, dtype=dtype),
        "holdout_identity_error": held_identity.to(device=device, dtype=dtype),
        "transport_accepted": torch.tensor(accepted, device=device, dtype=torch.bool),
    }
    return transform, mapping


@torch.no_grad()
def rotate_historical_flat_rows(
    x: Tensor, transform: PhaseCoordinateTransform
) -> Tensor:
    """Unit-norm preserving right action: x -> x[I+B(A-I)B^T].

    Works for row mean vectors [C,F] and for column basis [C,F,r] after
    transposing the last two axes. Does not require a dense F-by-F matrix.
    """
    basis, rotation = transform.rotation_basis, transform.rotation_matrix
    if not isinstance(x, Tensor) or x.ndim < 1 or not x.is_floating_point() or x.is_complex():
        raise ValueError("rotation rows must be real floating tensors with a feature dimension")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("rotation rows contain nonfinite values")
    if basis is None and rotation is None:
        return x.clone()
    if basis is None or rotation is None:
        raise ValueError("rotation basis and matrix must be supplied together")
    if (not isinstance(basis, Tensor) or not isinstance(rotation, Tensor)
            or not basis.is_floating_point() or not rotation.is_floating_point()
            or basis.is_complex() or rotation.is_complex()
            or basis.ndim != 2 or basis.size(0) != x.shape[-1]
            or not 0 < basis.size(1) <= basis.size(0)
            or rotation.shape != (basis.size(1), basis.size(1))):
        raise ValueError("orthogonal transport basis/rotation shape or dtype mismatch")
    b = basis.to(device=x.device, dtype=x.dtype)
    a = rotation.to(device=x.device, dtype=x.dtype)
    identity = torch.eye(a.size(0), device=x.device, dtype=x.dtype)
    tolerance = max(2e-5, 32 * torch.finfo(x.dtype).eps)
    if (not bool(torch.isfinite(b).all()) or not bool(torch.isfinite(a).all())
            or not torch.allclose(b.T @ b, identity, rtol=tolerance, atol=tolerance)
            or not torch.allclose(a.T @ a, identity, rtol=tolerance, atol=tolerance)):
        raise ValueError("historical transport must have an orthonormal basis and orthogonal rotation")
    return (
        x + ((x @ b) @ (a - identity)) @ b.T
    )


@torch.no_grad()
def evaluate_fixed_phase_coordinate_transform(
    reference_features: Tensor,
    current_features: Tensor,
    transform: PhaseCoordinateTransform,
) -> dict[str, Tensor]:
    """Measure a supplied training frame against final Q without refitting it.

    The held-out errors use the estimator's fixed odd-index current-input split.
    ``transport_accepted`` records whether the supplied frame has a rotation;
    it does not claim that the fixed frame still improves the final features.
    """
    reference, current = _validate_paired_features(reference_features, current_features)
    source = normalize_ordered_decision_features(reference).detach()
    target = normalize_ordered_decision_features(current).detach()
    prediction = rotate_historical_flat_rows(source.flatten(1), transform).reshape_as(source)
    residual = target - prediction
    tiny = torch.finfo(target.dtype).tiny
    channel_error = residual.square().sum((0, 1)) / target.square().sum((0, 1)).clamp_min(tiny)
    check = torch.arange(len(source), device=source.device) % 2 == 1
    zero = target.new_zeros(())
    return {
        "mapping_error": residual.square().sum() / target.square().sum().clamp_min(tiny),
        "channel_error_mean": channel_error.mean(),
        "channel_error_max": channel_error.max(),
        "representation_drift_rms": (target - source).square().mean().sqrt(),
        "translation_rms": zero,
        "scale_deviation_rms": zero.clone(),
        "holdout_candidate_error": residual[check].square().mean(),
        "holdout_identity_error": (target[check] - source[check]).square().mean(),
        "transport_accepted": torch.tensor(transform.rotation_basis is not None, device=source.device, dtype=torch.bool),
    }


@torch.no_grad()
def estimate_orthogonal_phase_transport(
    reference_features: Tensor, current_features: Tensor
):
    """Fit a small orthogonal Q-map using only paired CURRENT training inputs.

    Held-out current pairs gate application. This is NOT a certificate of
    old-class drift. Rejected maps contain no rotation and preserve old stats.
    """
    ref, cur = _validate_paired_features(reference_features, current_features)
    x = normalize_ordered_decision_features(ref).flatten(1).detach()
    y = normalize_ordered_decision_features(cur).flatten(1).detach()
    n, f = x.shape
    initial = (x - y).square().mean()
    candidate = initial.clone()
    accepted = False
    basis = None
    rotation = None
    if n >= 16 and float(initial) > 1e-12:
        fit = torch.arange(n, device=x.device) % 2 == 0
        val = ~fit
        train_x, train_y = x[fit], y[fit]
        # Deterministic paired span: 16 matched inputs supply <=32 axes.
        chosen = torch.linspace(
            0, len(train_x) - 1, steps=min(16, len(train_x)), device=x.device
        ).long()
        stacked = torch.cat((train_x[chosen], train_y[chosen]), dim=0)
        b, r = torch.linalg.qr(stacked.T, mode="reduced")
        # QR of near-dependent columns may include arbitrary complement axes.
        # Keep only numerically supported directions.
        diagonal = torch.abs(torch.diagonal(r))
        threshold = diagonal.max().clamp_min(1.0) * 1e-5
        supported = diagonal > threshold
        b = b[:, supported]
        if b.numel():
            projected_x = train_x @ b
            projected_y = train_y @ b
            covariance = projected_x.T @ projected_y
            u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
            a = u @ vh
            # Proper rotation, not a reflection.
            if float(torch.det(a)) < 0:
                adjust = torch.ones(a.shape[0], device=a.device, dtype=a.dtype)
                adjust[-1] = -1
                a = (u * adjust.unsqueeze(0)) @ vh
            predicted = (
                x[val]
                + (
                    (x[val] @ b)
                    @ (a - torch.eye(len(a), device=a.device, dtype=a.dtype))
                )
                @ b.T
            )
            candidate = (predicted - y[val]).square().mean()
            identity = (x[val] - y[val]).square().mean()
            # Reject overfit or vanishing improvements; bound extrapolation to
            # unseen old means by measured CURRENT paired feature drift.
            relative_gain = (identity - candidate) / identity.clamp_min(1e-12)
            accepted = (
                bool(relative_gain > 0.05)
                and bool(candidate < identity)
                and bool(torch.isfinite(candidate))
            )
            if accepted:
                basis, rotation = b, a
            initial = identity
    tr = PhaseCoordinateTransform(
        channel_scale=torch.ones(ref.shape[-1], device=ref.device, dtype=ref.dtype),
        token_translation=torch.zeros(
            ref.shape[1:], device=ref.device, dtype=ref.dtype
        ),
        rotation_basis=basis,
        rotation_matrix=rotation,
    )
    return tr, {
        "transport_accepted": torch.tensor(accepted, device=x.device, dtype=torch.bool),
        "holdout_candidate_error": candidate,
        "holdout_identity_error": initial,
        "mapping_error": candidate if accepted else initial,
        "translation_rms": torch.zeros((), device=x.device, dtype=x.dtype),
    }


@torch.no_grad()
def transport_historical_structure(
    historical_structure: ClassStructureBatch, transform: PhaseCoordinateTransform
) -> ClassStructureBatch:
    """Rotate old mean and low-rank covariance together; preserve native P.

    Orthogonal maps preserve cosine distances, feature norms, covariance
    eigenvalues and isotropic residuals. Legacy marginal V_Q and A_Q remain
    read-only teacher-coordinate diagnostics (not active classifier evidence).
    """
    if transform.rotation_basis is None and transform.rotation_matrix is None:
        return historical_structure
    mean = rotate_historical_flat_rows(
        historical_structure.learned_identity.flatten(1), transform
    )
    basis = historical_structure.learned_covariance_basis
    new_basis = rotate_historical_flat_rows(basis.transpose(1, 2), transform).transpose(
        1, 2
    )
    return replace(
        historical_structure,
        learned_identity=mean.reshape_as(historical_structure.learned_identity),
        learned_covariance_basis=new_basis,
    )


@torch.no_grad()
def evidence_controlled_historical_translation(
    historical_structure: ClassStructureBatch,
    historical_moments: dict[str, Tensor],
    reference_features: Tensor,
    current_features: Tensor,
    physical_spectral_evidence: Tensor,
    return_errors: bool = False,
):
    """Estimate empirical old-class drift only from supported current inputs.

    Each old class combines its learned conditional-covariance relevance with
    the actual incoming native-band spectral compatibility. Even/odd sample
    partition validates whether its proposed translation predicts unseen
    current pairs better than doing nothing. This test does NOT certify old
    class drift; unsupported proposals are set to zero.
    """
    reference, current = _validate_paired_features(reference_features, current_features)
    reference = normalize_ordered_decision_features(reference)
    current = normalize_ordered_decision_features(current)
    evidence = torch.as_tensor(
        physical_spectral_evidence, device=reference.device, dtype=reference.dtype
    )
    if evidence.shape != (
        reference.size(0),
        historical_structure.physical_identity.size(1),
        1,
    ):
        raise ValueError(
            "current physical evidence must match paired Q and historical native bands"
        )
    if not bool(torch.isfinite(evidence).all()):
        raise ValueError("physical evidence contains nonfinite values")
    historical_q_support = spectral_conditioned_learned_support(
        reference, historical_moments, evidence
    )
    _, _, historical_p_support = physical_support_components(
        evidence,
        historical_structure.physical_identity.to(evidence),
        historical_structure.physical_variation.to(evidence),
    )
    if historical_q_support.shape != historical_p_support.shape:
        raise ValueError("learned and physical class evidence differ")
    weights = (historical_q_support * historical_p_support).detach()
    sample_count = reference.size(0)
    class_count, token_count, feature_dim = historical_structure.learned_identity.shape
    translations = reference.new_zeros(class_count, token_count, feature_dim)
    accepted = torch.zeros(class_count, device=reference.device, dtype=torch.bool)
    support_mass = weights.sum(0)
    effective_samples = support_mass.square() / weights.square().sum(0).clamp_min(1e-12)
    candidate_errors = reference.new_full((class_count,), float("nan"))
    identity_errors = reference.new_full((class_count,), float("nan"))
    if sample_count < 8:
        result = (translations, accepted, support_mass, effective_samples)
        return result + (candidate_errors, identity_errors) if return_errors else result
    fit = torch.arange(sample_count, device=reference.device) % 2 == 0
    check = ~fit
    drift = current - reference
    moments = {
        key: torch.as_tensor(
            value,
            device=reference.device,
            dtype=reference.dtype if torch.is_floating_point(value) else value.dtype,
        )
        for key, value in historical_moments.items()
    }
    for class_index in range(class_count):
        train_weights = weights[fit, class_index]
        check_weights = weights[check, class_index]
        train_mass = train_weights.sum()
        check_mass = check_weights.sum()
        fit_effective = train_mass.square() / train_weights.square().sum().clamp_min(
            1e-12
        )
        if (
            float(train_mass) < 0.5
            or float(check_mass) < 0.5
            or float(fit_effective) < 4
        ):
            continue
        translation = (train_weights[:, None, None] * drift[fit]).sum(
            0
        ) / train_mass.clamp_min(1e-12)
        # These are unit-sphere *mean* embeddings; never extrapolate an
        # unsupported shift that moves an old conditional mean outside its
        # physically feasible unit ball.
        center = historical_structure.learned_identity[class_index].to(translation)
        if float((center + translation).flatten().norm()) > 1.00001:
            continue
        group_mean = moments["mean"][class_index]
        valid_group = moments["counts"][class_index] > 0
        if bool(
            (
                (group_mean[valid_group] + translation.flatten()).norm(dim=1) > 1.00001
            ).any()
        ):
            continue
        identity_error = (
            drift[check].square().mean((1, 2)) * check_weights
        ).sum() / check_mass
        candidate_error = (
            (drift[check] - translation).square().mean((1, 2)) * check_weights
        ).sum() / check_mass
        identity_errors[class_index] = identity_error
        candidate_errors[class_index] = candidate_error
        if not bool(torch.isfinite(candidate_error)) or not bool(
            candidate_error < 0.99 * identity_error
        ):
            continue
        selected = {
            key: value[class_index : class_index + 1]
            for key, value in moments.items()
            if key in ("mean", "basis", "eigenvalues", "residual_variance", "counts")
        }
        if not bool(
            normalized_moment_translation_feasible(
                selected, translation.unsqueeze(0)
            ).all()
        ):
            continue
        translations[class_index] = translation
        accepted[class_index] = True
    result = (translations, accepted, support_mass, effective_samples)
    return result + (candidate_errors, identity_errors) if return_errors else result


@torch.no_grad()
def _whole_cds_relevance(
    historical_structure: ClassStructureBatch,
    current_reference_structure: ClassStructureBatch,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if (
        historical_structure.class_count < 1
        or current_reference_structure.class_count < 1
    ):
        raise ValueError("nonempty structure required")
    merged = merge_class_structure_batches(
        (historical_structure, current_reference_structure)
    )
    relations = derive_class_relations(merged)
    o = historical_structure.class_count
    learned = relations.learned_combined_proximity[:o, o:].clamp(0, 1)
    physical = relations.physical_combined_proximity[:o, o:].clamp(0, 1)
    effective = combine_learned_physical_relevance(learned, physical)
    eps = torch.finfo(effective.dtype).eps
    weights = effective / effective.sum(dim=1, keepdim=True).clamp_min(eps)
    weights = weights * effective.max(dim=1, keepdim=True).values
    return learned, physical, effective, weights


@torch.no_grad()
def correct_historical_structure(
    *,
    historical_structure: ClassStructureBatch,
    current_reference_structure: ClassStructureBatch,
    current_structure: ClassStructureBatch,
    reference_features: Tensor,
    current_features: Tensor,
    use_transport: bool = True,
    historical_moments: dict[str, Tensor] | None = None,
    physical_spectral_evidence: Tensor | None = None,
) -> tuple[ClassStructureBatch, PhaseCoordinateTransform, PhaseAlignmentDiagnostics]:
    """Attempt orthogonal historical transport validated on new paired Q.

    A rejected map is identity. Preserves native spectral statistics exactly;
    never assumes held-out new-class fit proves unavailable old-class drift.
    """
    _validate_structure_pair(current_reference_structure, current_structure)
    # The affine fit is still useful as an observable drift diagnostic, but
    # cannot itself justify transport of normalized old-class covariance.
    observed_transform, mapping = estimate_phase_coordinate_transform(
        reference_features, current_features, allow_transport=False
    )
    transport_candidate_errors = None
    transport_identity_errors = None
    accepted = None
    if use_transport:
        transform, rotation_metrics = estimate_orthogonal_phase_transport(
            reference_features, current_features
        )
        moved = transport_historical_structure(historical_structure, transform)
        if transform.rotation_basis is not None:
            # Paired current-class evidence alone cannot certify old-class
            # motion. Fail closed if extrapolated old mean movement would be
            # more than twice the RMS displacement actually observed on Q_new.
            x = normalize_ordered_decision_features(reference_features).flatten(1)
            y = normalize_ordered_decision_features(current_features).flatten(1)
            observed = (x - y).square().sum(dim=1).mean().sqrt()
            old_motion = (
                (moved.learned_identity - historical_structure.learned_identity)
                .flatten(1)
                .norm(dim=1)
            )
            if bool(old_motion.max() > 2.0 * observed.clamp_min(1e-9)):
                transform = replace(
                    transform, rotation_basis=None, rotation_matrix=None
                )
                moved = historical_structure
                rotation_metrics["transport_accepted"] = torch.tensor(
                    False, device=x.device
                )
        mapping["transport_accepted"] = rotation_metrics["transport_accepted"]
        mapping["holdout_candidate_error"] = rotation_metrics["holdout_candidate_error"]
        mapping["holdout_identity_error"] = rotation_metrics["holdout_identity_error"]
        mapping["mapping_error"] = rotation_metrics["mapping_error"]
        # Native-band evidence remains in unchanged coordinates. The paired
        # teacher/student Q map is global, not class-specific extrapolation.
        alignment_mass = torch.zeros(
            historical_structure.class_count,
            device=historical_structure.learned_identity.device,
            dtype=historical_structure.learned_identity.dtype,
        )
        effective_sample_size = alignment_mass.clone()
    else:
        transform = observed_transform
        moved = historical_structure
        alignment_mass = torch.zeros(
            historical_structure.class_count,
            device=historical_structure.learned_identity.device,
            dtype=historical_structure.learned_identity.dtype,
        )
        effective_sample_size = alignment_mass.clone()
    ref_moved = current_reference_structure
    learned, physical, effective, weights = _whole_cds_relevance(
        historical_structure, current_reference_structure
    )
    support_mass = alignment_mass if use_transport else weights.sum(dim=1)
    # Diagnostics are grounded in measured NEW class transport residuals.
    identity_delta = current_structure.learned_identity - ref_moved.learned_identity
    # Variation diagnostics now measure the ACTUAL learned low-rank spectrum.
    variation_delta = torch.cat(
        (
            current_structure.learned_covariance_eigenvalues
            - ref_moved.learned_covariance_eigenvalues,
            (
                current_structure.learned_covariance_residual
                - ref_moved.learned_covariance_residual
            )[:, None],
        ),
        dim=1,
    )
    variation_reference = torch.cat(
        (
            ref_moved.learned_covariance_eigenvalues,
            ref_moved.learned_covariance_residual[:, None],
        ),
        dim=1,
    )
    identity_relative = _relative_batch_norm(identity_delta, ref_moved.learned_identity)
    variation_relative = _relative_batch_norm(variation_delta, variation_reference)
    z = mapping["mapping_error"] * 0
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
        effective_sample_size_mean=effective_sample_size.mean(),
        effective_sample_size_min=effective_sample_size.min(),
        identity_correction_relative_mean=identity_relative.mean(),
        identity_correction_relative_max=identity_relative.max(),
        variation_correction_relative_mean=variation_relative.mean(),
        variation_correction_relative_max=variation_relative.max(),
        variation_negative_fraction_before_sanitize=z,
        holdout_candidate_error=mapping["holdout_candidate_error"],
        holdout_identity_error=mapping["holdout_identity_error"],
        transport_accepted=mapping["transport_accepted"],
        per_class_holdout_candidate_error=None,
        per_class_holdout_identity_error=None,
        per_class_transport_accepted=None,
    )
    return moved, transform, diagnostics


__all__ = [
    "PhaseCoordinateTransform",
    "PhaseAlignmentDiagnostics",
    "estimate_phase_coordinate_transform",
    "transport_historical_structure",
    "estimate_orthogonal_phase_transport",
    "rotate_historical_flat_rows",
    "evaluate_fixed_phase_coordinate_transform",
    "evidence_controlled_historical_translation",
    "correct_historical_structure",
]



# from __future__ import annotations
# from dataclasses import dataclass, replace

# import torch

# from models.geometry_bank import (
#     ClassStructureBatch,
#     combine_learned_physical_relevance,
#     derive_class_relations,
#     merge_class_structure_batches,
#     normalize_ordered_decision_features,
#     sanitize_ordered_variation,
#     physical_support_components,
#     spectral_conditioned_learned_support,
#     normalized_moment_translation_feasible,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class PhaseCoordinateTransform:
#     """An optional held-out-validated low-rank orthogonal map in normalized Q.

#     The rotation moves learned class means AND low-rank covariance bases in
#     the same coordinates. Legacy fields remain for read-only diagnostics;
#     mean-only translation is prohibited.
#     """

#     channel_scale: Tensor  # [D]
#     token_translation: Tensor  # [K,D]
#     class_translation: Tensor | None = None  # deprecated legacy diagnostic
#     rotation_basis: Tensor | None = None  # [F,R] orthonormal columns
#     rotation_matrix: Tensor | None = None  # [R,R] orthogonal, acts on row features


# @dataclass(frozen=True)
# class PhaseAlignmentDiagnostics:
#     mapping_error: Tensor
#     channel_error_mean: Tensor
#     channel_error_max: Tensor
#     representation_drift_rms: Tensor
#     translation_rms: Tensor
#     scale_deviation_rms: Tensor
#     learned_relevance_mean: Tensor
#     physical_relevance_mean: Tensor
#     effective_relevance_mean: Tensor
#     class_specific_support_mass_mean: Tensor
#     class_specific_support_mass_min: Tensor
#     class_specific_support_mass_max: Tensor
#     effective_sample_size_mean: Tensor
#     effective_sample_size_min: Tensor
#     identity_correction_relative_mean: Tensor
#     identity_correction_relative_max: Tensor
#     variation_correction_relative_mean: Tensor
#     variation_correction_relative_max: Tensor
#     variation_negative_fraction_before_sanitize: Tensor
#     # Observable new-class holdout test, not an old-class transfer certificate.
#     holdout_candidate_error: Tensor
#     holdout_identity_error: Tensor
#     transport_accepted: Tensor
#     per_class_holdout_candidate_error: Tensor | None = None
#     per_class_holdout_identity_error: Tensor | None = None
#     per_class_transport_accepted: Tensor | None = None


# def _validate_paired_features(
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     reference = torch.as_tensor(reference_features)
#     current = torch.as_tensor(current_features)
#     if (
#         reference.ndim != 3
#         or reference.size(0) < 2
#         or reference.size(1) < 2
#         or reference.size(2) <= 0
#     ):
#         raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if reference.device != current.device or reference.dtype != current.dtype:
#         raise ValueError("paired Q-/Q+ features must share device/dtype")
#     if not torch.is_floating_point(reference) or reference.is_complex():
#         raise TypeError("paired Q-/Q+ features must be real floating point")
#     if not bool(torch.isfinite(reference).all()) or not bool(
#         torch.isfinite(current).all()
#     ):
#         raise ValueError("paired Q-/Q+ features contain NaN or Inf")
#     return reference, current


# def _validate_structure_pair(
#     reference: ClassStructureBatch, current: ClassStructureBatch
# ) -> None:
#     if not isinstance(reference, ClassStructureBatch) or not isinstance(
#         current, ClassStructureBatch
#     ):
#         raise TypeError("reference/current structures must be ClassStructureBatch")
#     if reference.class_count <= 0 or current.class_count != reference.class_count:
#         raise ValueError("reference/current structures have incompatible class counts")
#     if not torch.equal(reference.class_ids, current.class_ids):
#         raise ValueError(
#             "reference/current class IDs must be identical and ordered equally"
#         )
#     if reference.learned_identity.shape != current.learned_identity.shape:
#         raise ValueError("reference/current I_Q shapes disagree")
#     if reference.learned_variation.shape != current.learned_variation.shape:
#         raise ValueError("reference/current V_Q shapes disagree")
#     for name in ("learned_absolute_deviation",):
#         if getattr(reference, name).shape != getattr(current, name).shape:
#             raise ValueError(f"reference/current {name} shapes disagree")
#     # Physical evidence is a native-spectrum coordinate and must never be moved by
#     # a neural feature-space transform.
#     if not torch.equal(reference.physical_identity, current.physical_identity):
#         raise ValueError("current-class I_P changed across neural coordinates")
#     if not torch.equal(reference.physical_variation, current.physical_variation):
#         raise ValueError("current-class V_P diagnostics changed")
#     for name in ("physical_variance", "physical_delta_variance"):
#         if not torch.equal(getattr(reference, name), getattr(current, name)):
#             raise ValueError(f"current native spectral statistic {name} changed")
#     for name in (
#         "learned_covariance_basis",
#         "learned_covariance_eigenvalues",
#         "learned_covariance_residual",
#     ):
#         if getattr(reference, name).shape != getattr(current, name).shape:
#             raise ValueError(f"learned covariance coordinate mismatch: {name}")


# def _relative_batch_norm(delta: Tensor, reference: Tensor) -> Tensor:
#     """Per-class relative L2 change with a finite zero-safe denominator."""
#     if delta.shape != reference.shape or delta.ndim < 2 or delta.size(0) <= 0:
#         raise ValueError("relative-norm tensors must share a non-empty class dimension")
#     eps = torch.finfo(reference.dtype).eps
#     numerator = torch.linalg.vector_norm(delta.reshape(delta.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(
#         reference.reshape(reference.size(0), -1), dim=1
#     ).clamp_min(eps)
#     return numerator / denominator


# @torch.no_grad()
# def estimate_phase_coordinate_transform(
#     reference_features: Tensor,
#     current_features: Tensor,
#     *,
#     allow_transport: bool = True,
# ) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
#     """Estimate a conservative observable Q_(t-1) -> Q_t map.

#     The previous implementation used an unconstrained per-channel least-squares
#     slope.  On Indian Pines it reported only ~1.2% paired-Q drift but ~46% scale
#     change, which then moved historical CDS far more than the representation had
#     actually moved.

#     Here each channel solves an identity-centred ridge problem.  The ridge mass is
#     *data-derived* from the observed paired drift energy, not a hand-set clamp::

#         s_d = (x_d^T y_d + lambda_d) / (x_d^T x_d + lambda_d)
#         lambda_d = ||y_d - x_d||^2 + eps

#     When a channel is weakly identifiable or noisy, the solution continuously
#     shrinks toward ``s_d = 1``.  Well-supported channels remain data-driven.  No
#     old-class-specific motion is inferred from unseen old samples.
#     """
#     reference, current = _validate_paired_features(reference_features, current_features)
#     reference = normalize_ordered_decision_features(reference)
#     current = normalize_ordered_decision_features(current)

#     # MPS supports the device transfer in the input dtype, but not float64.
#     # Promote only after both feature tensors are on CPU.
#     source = reference.detach().to(device="cpu").to(dtype=torch.float64)
#     target = current.detach().to(device="cpu").to(dtype=torch.float64)
#     n, token_count, channel_count = source.shape

#     # Validate a map on HELD-OUT current-class feature pairs before transporting
#     # any old CDS. Fitting-error alone can favor a spurious phase correction.
#     # Fixed alternating split is reproducible, never uses historical samples.
#     # A failed holdout comparison retains the identity historical coordinate.
#     def fit_map(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
#         xmean, ymean = x.mean(0), y.mean(0)
#         xc, yc = x - xmean.unsqueeze(0), y - ymean.unsqueeze(0)
#         cross_val = (xc * yc).sum((0, 1))
#         energy_val = xc.square().sum((0, 1))
#         drift_val = (yc - xc).square().sum((0, 1))
#         regularizer = drift_val + torch.finfo(
#             torch.float64
#         ).eps * energy_val.max().clamp_min(1.0)
#         slopes = (cross_val + regularizer) / (energy_val + regularizer)
#         offsets = ymean - slopes.unsqueeze(0) * xmean
#         return slopes, offsets

#     accepted = False
#     if n >= 6 and allow_transport:
#         fit_index = torch.arange(n) % 2 == 0
#         check_index = ~fit_index
#         train_scale, train_offset = fit_map(source[fit_index], target[fit_index])
#         held_prediction = source[check_index] * train_scale.view(
#             1, 1, -1
#         ) + train_offset.unsqueeze(0)
#         held_candidate = (target[check_index] - held_prediction).square().mean()
#         held_identity = (target[check_index] - source[check_index]).square().mean()
#         accepted = bool(held_candidate + 1e-12 < held_identity)
#     else:
#         # Too few paired current samples for a meaningful split; retain the
#         # identity map and report the observable identity error with a finite
#         # value. The acceptance flag records that transport was not validated.
#         held_identity = (target - source).square().mean()
#         held_candidate = held_identity.clone()

#     source_mean = source.mean(dim=0)
#     target_mean = target.mean(dim=0)
#     source_centered = source - source_mean.unsqueeze(0)
#     target_centered = target - target_mean.unsqueeze(0)

#     cross = (source_centered * target_centered).sum(dim=(0, 1))
#     source_energy = source_centered.square().sum(dim=(0, 1))
#     drift_energy = (target_centered - source_centered).square().sum(dim=(0, 1))
#     eps = torch.finfo(torch.float64).eps
#     numerical = eps * source_energy.max().clamp_min(1.0)
#     ridge = drift_energy + numerical
#     channel_scale = (cross + ridge) / (source_energy + ridge)
#     token_translation = target_mean - channel_scale.unsqueeze(0) * source_mean
#     if not accepted:
#         channel_scale = torch.ones_like(channel_scale)
#         token_translation = torch.zeros_like(token_translation)

#     prediction = source * channel_scale.view(
#         1, 1, channel_count
#     ) + token_translation.view(1, token_count, channel_count)
#     residual = target - prediction
#     tiny = torch.finfo(torch.float64).tiny
#     channel_residual = residual.permute(2, 0, 1)
#     channel_target = target.permute(2, 0, 1)
#     channel_error = channel_residual.square().sum(
#         dim=(1, 2)
#     ) / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
#     mapping_error = residual.square().sum() / target.square().sum().clamp_min(tiny)
#     drift_rms = torch.sqrt((target - source).square().mean().clamp_min(0.0))
#     translation_rms = torch.sqrt(token_translation.square().mean().clamp_min(0.0))
#     scale_deviation_rms = torch.sqrt(
#         (channel_scale - 1.0).square().mean().clamp_min(0.0)
#     )

#     values = (
#         channel_scale,
#         token_translation,
#         mapping_error,
#         channel_error,
#         drift_rms,
#         translation_rms,
#         scale_deviation_rms,
#     )
#     if not all(bool(torch.isfinite(value).all()) for value in values):
#         raise RuntimeError("phase coordinate transform contains NaN or Inf")

#     device, dtype = reference.device, reference.dtype
#     transform = PhaseCoordinateTransform(
#         channel_scale=channel_scale.to(device=device, dtype=dtype),
#         token_translation=token_translation.to(device=device, dtype=dtype),
#     )
#     mapping = {
#         "mapping_error": mapping_error.to(device=device, dtype=dtype),
#         "channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
#         "channel_error_max": channel_error.max().to(device=device, dtype=dtype),
#         "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
#         "translation_rms": translation_rms.to(device=device, dtype=dtype),
#         "scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
#         "holdout_candidate_error": held_candidate.to(device=device, dtype=dtype),
#         "holdout_identity_error": held_identity.to(device=device, dtype=dtype),
#         "transport_accepted": torch.tensor(accepted, device=device, dtype=torch.bool),
#     }
#     return transform, mapping


# @torch.no_grad()
# def rotate_historical_flat_rows(
#     x: Tensor, transform: PhaseCoordinateTransform
# ) -> Tensor:
#     """Unit-norm preserving right action: x -> x[I+B(A-I)B^T].

#     Works for row mean vectors [C,F] and for column basis [C,F,r] after
#     transposing the last two axes. Does not require a dense F-by-F matrix.
#     """
#     basis, rotation = transform.rotation_basis, transform.rotation_matrix
#     if basis is None or rotation is None:
#         return x.clone()
#     if x.shape[-1] != basis.shape[0]:
#         raise ValueError("rotation feature dimension mismatch")
#     b = basis.to(device=x.device, dtype=x.dtype)
#     a = rotation.to(device=x.device, dtype=x.dtype)
#     return (
#         x + ((x @ b) @ (a - torch.eye(a.size(0), device=x.device, dtype=x.dtype))) @ b.T
#     )


# @torch.no_grad()
# def estimate_orthogonal_phase_transport(
#     reference_features: Tensor, current_features: Tensor
# ):
#     """Fit a small orthogonal Q-map using only paired CURRENT training inputs.

#     Held-out current pairs gate application. This is NOT a certificate of
#     old-class drift. Rejected maps contain no rotation and preserve old stats.
#     """
#     ref, cur = _validate_paired_features(reference_features, current_features)
#     x = normalize_ordered_decision_features(ref).flatten(1).detach()
#     y = normalize_ordered_decision_features(cur).flatten(1).detach()
#     n, f = x.shape
#     initial = (x - y).square().mean()
#     candidate = initial.clone()
#     accepted = False
#     basis = None
#     rotation = None
#     if n >= 16 and float(initial) > 1e-12:
#         fit = torch.arange(n, device=x.device) % 2 == 0
#         val = ~fit
#         train_x, train_y = x[fit], y[fit]
#         # Deterministic paired span: 16 matched inputs supply <=32 axes.
#         chosen = torch.linspace(
#             0, len(train_x) - 1, steps=min(16, len(train_x)), device=x.device
#         ).long()
#         stacked = torch.cat((train_x[chosen], train_y[chosen]), dim=0)
#         b, r = torch.linalg.qr(stacked.T, mode="reduced")
#         # QR of near-dependent columns may include arbitrary complement axes.
#         # Keep only numerically supported directions.
#         diagonal = torch.abs(torch.diagonal(r))
#         threshold = diagonal.max().clamp_min(1.0) * 1e-5
#         supported = diagonal > threshold
#         b = b[:, supported]
#         if b.numel():
#             projected_x = train_x @ b
#             projected_y = train_y @ b
#             covariance = projected_x.T @ projected_y
#             u, _, vh = torch.linalg.svd(covariance, full_matrices=False)
#             a = u @ vh
#             # Proper rotation, not a reflection.
#             if float(torch.det(a)) < 0:
#                 adjust = torch.ones(a.shape[0], device=a.device, dtype=a.dtype)
#                 adjust[-1] = -1
#                 a = (u * adjust.unsqueeze(0)) @ vh
#             predicted = (
#                 x[val]
#                 + (
#                     (x[val] @ b)
#                     @ (a - torch.eye(len(a), device=a.device, dtype=a.dtype))
#                 )
#                 @ b.T
#             )
#             candidate = (predicted - y[val]).square().mean()
#             identity = (x[val] - y[val]).square().mean()
#             # Reject overfit or vanishing improvements; bound extrapolation to
#             # unseen old means by measured CURRENT paired feature drift.
#             relative_gain = (identity - candidate) / identity.clamp_min(1e-12)
#             accepted = (
#                 bool(relative_gain > 0.05)
#                 and bool(candidate < identity)
#                 and bool(torch.isfinite(candidate))
#             )
#             if accepted:
#                 basis, rotation = b, a
#             initial = identity
#     tr = PhaseCoordinateTransform(
#         channel_scale=torch.ones(ref.shape[-1], device=ref.device, dtype=ref.dtype),
#         token_translation=torch.zeros(
#             ref.shape[1:], device=ref.device, dtype=ref.dtype
#         ),
#         rotation_basis=basis,
#         rotation_matrix=rotation,
#     )
#     return tr, {
#         "transport_accepted": torch.tensor(accepted, device=x.device, dtype=torch.bool),
#         "holdout_candidate_error": candidate,
#         "holdout_identity_error": initial,
#         "mapping_error": candidate if accepted else initial,
#         "translation_rms": torch.zeros((), device=x.device, dtype=x.dtype),
#     }


# @torch.no_grad()
# def transport_historical_structure(
#     historical_structure: ClassStructureBatch, transform: PhaseCoordinateTransform
# ) -> ClassStructureBatch:
#     """Rotate old mean and low-rank covariance together; preserve native P.

#     Orthogonal maps preserve cosine distances, feature norms, covariance
#     eigenvalues and isotropic residuals. Legacy marginal V_Q and A_Q remain
#     read-only teacher-coordinate diagnostics (not active classifier evidence).
#     """
#     if transform.rotation_basis is None:
#         return historical_structure
#     mean = rotate_historical_flat_rows(
#         historical_structure.learned_identity.flatten(1), transform
#     )
#     basis = historical_structure.learned_covariance_basis
#     new_basis = rotate_historical_flat_rows(basis.transpose(1, 2), transform).transpose(
#         1, 2
#     )
#     return replace(
#         historical_structure,
#         learned_identity=mean.reshape_as(historical_structure.learned_identity),
#         learned_covariance_basis=new_basis,
#     )


# @torch.no_grad()
# def evidence_controlled_historical_translation(
#     historical_structure: ClassStructureBatch,
#     historical_moments: dict[str, Tensor],
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
#     return_errors: bool = False,
# ):
#     """Estimate empirical old-class drift only from supported current inputs.

#     Each old class combines its learned conditional-covariance relevance with
#     the actual incoming native-band spectral compatibility. Even/odd sample
#     partition validates whether its proposed translation predicts unseen
#     current pairs better than doing nothing. This test does NOT certify old
#     class drift; unsupported proposals are set to zero.
#     """
#     reference, current = _validate_paired_features(reference_features, current_features)
#     reference = normalize_ordered_decision_features(reference)
#     current = normalize_ordered_decision_features(current)
#     evidence = torch.as_tensor(
#         physical_spectral_evidence, device=reference.device, dtype=reference.dtype
#     )
#     if evidence.shape != (
#         reference.size(0),
#         historical_structure.physical_identity.size(1),
#         1,
#     ):
#         raise ValueError(
#             "current physical evidence must match paired Q and historical native bands"
#         )
#     if not bool(torch.isfinite(evidence).all()):
#         raise ValueError("physical evidence contains nonfinite values")
#     historical_q_support = spectral_conditioned_learned_support(
#         reference, historical_moments, evidence
#     )
#     _, _, historical_p_support = physical_support_components(
#         evidence,
#         historical_structure.physical_identity.to(evidence),
#         historical_structure.physical_variation.to(evidence),
#     )
#     if historical_q_support.shape != historical_p_support.shape:
#         raise ValueError("learned and physical class evidence differ")
#     weights = (historical_q_support * historical_p_support).detach()
#     sample_count = reference.size(0)
#     class_count, token_count, feature_dim = historical_structure.learned_identity.shape
#     translations = reference.new_zeros(class_count, token_count, feature_dim)
#     accepted = torch.zeros(class_count, device=reference.device, dtype=torch.bool)
#     support_mass = weights.sum(0)
#     effective_samples = support_mass.square() / weights.square().sum(0).clamp_min(1e-12)
#     candidate_errors = reference.new_full((class_count,), float("nan"))
#     identity_errors = reference.new_full((class_count,), float("nan"))
#     if sample_count < 8:
#         result = (translations, accepted, support_mass, effective_samples)
#         return result + (candidate_errors, identity_errors) if return_errors else result
#     fit = torch.arange(sample_count, device=reference.device) % 2 == 0
#     check = ~fit
#     drift = current - reference
#     moments = {
#         key: torch.as_tensor(
#             value,
#             device=reference.device,
#             dtype=reference.dtype if torch.is_floating_point(value) else value.dtype,
#         )
#         for key, value in historical_moments.items()
#     }
#     for class_index in range(class_count):
#         train_weights = weights[fit, class_index]
#         check_weights = weights[check, class_index]
#         train_mass = train_weights.sum()
#         check_mass = check_weights.sum()
#         fit_effective = train_mass.square() / train_weights.square().sum().clamp_min(
#             1e-12
#         )
#         if (
#             float(train_mass) < 0.5
#             or float(check_mass) < 0.5
#             or float(fit_effective) < 4
#         ):
#             continue
#         translation = (train_weights[:, None, None] * drift[fit]).sum(
#             0
#         ) / train_mass.clamp_min(1e-12)
#         # These are unit-sphere *mean* embeddings; never extrapolate an
#         # unsupported shift that moves an old conditional mean outside its
#         # physically feasible unit ball.
#         center = historical_structure.learned_identity[class_index].to(translation)
#         if float((center + translation).flatten().norm()) > 1.00001:
#             continue
#         group_mean = moments["mean"][class_index]
#         valid_group = moments["counts"][class_index] > 0
#         if bool(
#             (
#                 (group_mean[valid_group] + translation.flatten()).norm(dim=1) > 1.00001
#             ).any()
#         ):
#             continue
#         identity_error = (
#             drift[check].square().mean((1, 2)) * check_weights
#         ).sum() / check_mass
#         candidate_error = (
#             (drift[check] - translation).square().mean((1, 2)) * check_weights
#         ).sum() / check_mass
#         identity_errors[class_index] = identity_error
#         candidate_errors[class_index] = candidate_error
#         if not bool(torch.isfinite(candidate_error)) or not bool(
#             candidate_error < 0.99 * identity_error
#         ):
#             continue
#         selected = {
#             key: value[class_index : class_index + 1]
#             for key, value in moments.items()
#             if key in ("mean", "basis", "eigenvalues", "residual_variance", "counts")
#         }
#         if not bool(
#             normalized_moment_translation_feasible(
#                 selected, translation.unsqueeze(0)
#             ).all()
#         ):
#             continue
#         translations[class_index] = translation
#         accepted[class_index] = True
#     result = (translations, accepted, support_mass, effective_samples)
#     return result + (candidate_errors, identity_errors) if return_errors else result


# @torch.no_grad()
# def _whole_cds_relevance(
#     historical_structure: ClassStructureBatch,
#     current_reference_structure: ClassStructureBatch,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     if (
#         historical_structure.class_count < 1
#         or current_reference_structure.class_count < 1
#     ):
#         raise ValueError("nonempty structure required")
#     merged = merge_class_structure_batches(
#         (historical_structure, current_reference_structure)
#     )
#     relations = derive_class_relations(merged)
#     o = historical_structure.class_count
#     learned = relations.learned_combined_proximity[:o, o:].clamp(0, 1)
#     physical = relations.physical_combined_proximity[:o, o:].clamp(0, 1)
#     effective = combine_learned_physical_relevance(learned, physical)
#     eps = torch.finfo(effective.dtype).eps
#     weights = effective / effective.sum(dim=1, keepdim=True).clamp_min(eps)
#     weights = weights * effective.max(dim=1, keepdim=True).values
#     return learned, physical, effective, weights


# @torch.no_grad()
# def correct_historical_structure(
#     *,
#     historical_structure: ClassStructureBatch,
#     current_reference_structure: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     reference_features: Tensor,
#     current_features: Tensor,
#     use_transport: bool = True,
#     historical_moments: dict[str, Tensor] | None = None,
#     physical_spectral_evidence: Tensor | None = None,
# ) -> tuple[ClassStructureBatch, PhaseCoordinateTransform, PhaseAlignmentDiagnostics]:
#     """Attempt orthogonal historical transport validated on new paired Q.

#     A rejected map is identity. Preserves native spectral statistics exactly;
#     never assumes held-out new-class fit proves unavailable old-class drift.
#     """
#     _validate_structure_pair(current_reference_structure, current_structure)
#     # The affine fit is still useful as an observable drift diagnostic, but
#     # cannot itself justify transport of normalized old-class covariance.
#     observed_transform, mapping = estimate_phase_coordinate_transform(
#         reference_features, current_features, allow_transport=False
#     )
#     transport_candidate_errors = None
#     transport_identity_errors = None
#     accepted = None
#     if use_transport:
#         transform, rotation_metrics = estimate_orthogonal_phase_transport(
#             reference_features, current_features
#         )
#         moved = transport_historical_structure(historical_structure, transform)
#         if transform.rotation_basis is not None:
#             # Paired current-class evidence alone cannot certify old-class
#             # motion. Fail closed if extrapolated old mean movement would be
#             # more than twice the RMS displacement actually observed on Q_new.
#             x = normalize_ordered_decision_features(reference_features).flatten(1)
#             y = normalize_ordered_decision_features(current_features).flatten(1)
#             observed = (x - y).square().sum(dim=1).mean().sqrt()
#             old_motion = (
#                 (moved.learned_identity - historical_structure.learned_identity)
#                 .flatten(1)
#                 .norm(dim=1)
#             )
#             if bool(old_motion.max() > 2.0 * observed.clamp_min(1e-9)):
#                 transform = replace(
#                     transform, rotation_basis=None, rotation_matrix=None
#                 )
#                 moved = historical_structure
#                 rotation_metrics["transport_accepted"] = torch.tensor(
#                     False, device=x.device
#                 )
#         mapping["transport_accepted"] = rotation_metrics["transport_accepted"]
#         mapping["holdout_candidate_error"] = rotation_metrics["holdout_candidate_error"]
#         mapping["holdout_identity_error"] = rotation_metrics["holdout_identity_error"]
#         mapping["mapping_error"] = rotation_metrics["mapping_error"]
#         # Native-band evidence remains in unchanged coordinates. The paired
#         # teacher/student Q map is global, not class-specific extrapolation.
#         alignment_mass = torch.zeros(
#             historical_structure.class_count,
#             device=historical_structure.learned_identity.device,
#             dtype=historical_structure.learned_identity.dtype,
#         )
#         effective_sample_size = alignment_mass.clone()
#     else:
#         transform = observed_transform
#         moved = historical_structure
#         alignment_mass = torch.zeros(
#             historical_structure.class_count,
#             device=historical_structure.learned_identity.device,
#             dtype=historical_structure.learned_identity.dtype,
#         )
#         effective_sample_size = alignment_mass.clone()
#     ref_moved = current_reference_structure
#     learned, physical, effective, weights = _whole_cds_relevance(
#         historical_structure, current_reference_structure
#     )
#     support_mass = alignment_mass if use_transport else weights.sum(dim=1)
#     # Diagnostics are grounded in measured NEW class transport residuals.
#     identity_delta = current_structure.learned_identity - ref_moved.learned_identity
#     # Variation diagnostics now measure the ACTUAL learned low-rank spectrum.
#     variation_delta = torch.cat(
#         (
#             current_structure.learned_covariance_eigenvalues
#             - ref_moved.learned_covariance_eigenvalues,
#             (
#                 current_structure.learned_covariance_residual
#                 - ref_moved.learned_covariance_residual
#             )[:, None],
#         ),
#         dim=1,
#     )
#     variation_reference = torch.cat(
#         (
#             ref_moved.learned_covariance_eigenvalues,
#             ref_moved.learned_covariance_residual[:, None],
#         ),
#         dim=1,
#     )
#     identity_relative = _relative_batch_norm(identity_delta, ref_moved.learned_identity)
#     variation_relative = _relative_batch_norm(variation_delta, variation_reference)
#     z = mapping["mapping_error"] * 0
#     diagnostics = PhaseAlignmentDiagnostics(
#         mapping_error=mapping["mapping_error"],
#         channel_error_mean=mapping["channel_error_mean"],
#         channel_error_max=mapping["channel_error_max"],
#         representation_drift_rms=mapping["representation_drift_rms"],
#         translation_rms=mapping["translation_rms"],
#         scale_deviation_rms=mapping["scale_deviation_rms"],
#         learned_relevance_mean=learned.mean(),
#         physical_relevance_mean=physical.mean(),
#         effective_relevance_mean=effective.mean(),
#         class_specific_support_mass_mean=support_mass.mean(),
#         class_specific_support_mass_min=support_mass.min(),
#         class_specific_support_mass_max=support_mass.max(),
#         effective_sample_size_mean=effective_sample_size.mean(),
#         effective_sample_size_min=effective_sample_size.min(),
#         identity_correction_relative_mean=identity_relative.mean(),
#         identity_correction_relative_max=identity_relative.max(),
#         variation_correction_relative_mean=variation_relative.mean(),
#         variation_correction_relative_max=variation_relative.max(),
#         variation_negative_fraction_before_sanitize=z,
#         holdout_candidate_error=mapping["holdout_candidate_error"],
#         holdout_identity_error=mapping["holdout_identity_error"],
#         transport_accepted=mapping["transport_accepted"],
#         per_class_holdout_candidate_error=None,
#         per_class_holdout_identity_error=None,
#         per_class_transport_accepted=None,
#     )
#     return moved, transform, diagnostics


# __all__ = [
#     "PhaseCoordinateTransform",
#     "PhaseAlignmentDiagnostics",
#     "estimate_phase_coordinate_transform",
#     "transport_historical_structure",
#     "estimate_orthogonal_phase_transport",
#     "rotate_historical_flat_rows",
#     "evidence_controlled_historical_translation",
#     "correct_historical_structure",
# ]


# from __future__ import annotations
# from dataclasses import dataclass, replace

# import torch

# from models.geometry_bank import (
#     ClassStructureBatch,
#     combine_learned_physical_relevance,
#     derive_class_relations,
#     merge_class_structure_batches,
#     normalize_ordered_decision_features,
#     sanitize_ordered_variation,
#     physical_support_components,
#     spectral_conditioned_learned_support,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class PhaseCoordinateTransform:
#     """An empirically supported per-class displacement in normalized Q.

#     Historical transport uses ``class_translation`` only. The earlier affine
#     scale/offset fields remain for observable paired-feature diagnostics; they
#     do not authorize transporting old covariance in normalized coordinates.
#     """

#     channel_scale: Tensor  # [D]
#     token_translation: Tensor  # [K,D]
#     class_translation: Tensor | None = None  # [C_old,K,D], evidence-gated


# @dataclass(frozen=True)
# class PhaseAlignmentDiagnostics:
#     mapping_error: Tensor
#     channel_error_mean: Tensor
#     channel_error_max: Tensor
#     representation_drift_rms: Tensor
#     translation_rms: Tensor
#     scale_deviation_rms: Tensor
#     learned_relevance_mean: Tensor
#     physical_relevance_mean: Tensor
#     effective_relevance_mean: Tensor
#     class_specific_support_mass_mean: Tensor
#     class_specific_support_mass_min: Tensor
#     class_specific_support_mass_max: Tensor
#     effective_sample_size_mean: Tensor
#     effective_sample_size_min: Tensor
#     identity_correction_relative_mean: Tensor
#     identity_correction_relative_max: Tensor
#     variation_correction_relative_mean: Tensor
#     variation_correction_relative_max: Tensor
#     variation_negative_fraction_before_sanitize: Tensor
#     # Observable new-class holdout test, not an old-class transfer certificate.
#     holdout_candidate_error: Tensor
#     holdout_identity_error: Tensor
#     transport_accepted: Tensor


# def _validate_paired_features(
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     reference = torch.as_tensor(reference_features)
#     current = torch.as_tensor(current_features)
#     if (
#         reference.ndim != 3
#         or reference.size(0) < 2
#         or reference.size(1) < 2
#         or reference.size(2) <= 0
#     ):
#         raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if reference.device != current.device or reference.dtype != current.dtype:
#         raise ValueError("paired Q-/Q+ features must share device/dtype")
#     if not torch.is_floating_point(reference) or reference.is_complex():
#         raise TypeError("paired Q-/Q+ features must be real floating point")
#     if not bool(torch.isfinite(reference).all()) or not bool(
#         torch.isfinite(current).all()
#     ):
#         raise ValueError("paired Q-/Q+ features contain NaN or Inf")
#     return reference, current


# def _validate_structure_pair(
#     reference: ClassStructureBatch, current: ClassStructureBatch
# ) -> None:
#     if not isinstance(reference, ClassStructureBatch) or not isinstance(
#         current, ClassStructureBatch
#     ):
#         raise TypeError("reference/current structures must be ClassStructureBatch")
#     if reference.class_count <= 0 or current.class_count != reference.class_count:
#         raise ValueError("reference/current structures have incompatible class counts")
#     if not torch.equal(reference.class_ids, current.class_ids):
#         raise ValueError(
#             "reference/current class IDs must be identical and ordered equally"
#         )
#     if reference.learned_identity.shape != current.learned_identity.shape:
#         raise ValueError("reference/current I_Q shapes disagree")
#     if reference.learned_variation.shape != current.learned_variation.shape:
#         raise ValueError("reference/current V_Q shapes disagree")
#     for name in ("learned_absolute_deviation",):
#         if getattr(reference, name).shape != getattr(current, name).shape:
#             raise ValueError(f"reference/current {name} shapes disagree")
#     # Physical evidence is a native-spectrum coordinate and must never be moved by
#     # a neural feature-space transform.
#     if not torch.equal(reference.physical_identity, current.physical_identity):
#         raise ValueError("current-class I_P changed across neural coordinates")
#     if not torch.equal(reference.physical_variation, current.physical_variation):
#         raise ValueError("current-class V_P changed across neural coordinates")


# def _relative_batch_norm(delta: Tensor, reference: Tensor) -> Tensor:
#     """Per-class relative L2 change with a finite zero-safe denominator."""
#     if delta.shape != reference.shape or delta.ndim < 2 or delta.size(0) <= 0:
#         raise ValueError("relative-norm tensors must share a non-empty class dimension")
#     eps = torch.finfo(reference.dtype).eps
#     numerator = torch.linalg.vector_norm(delta.reshape(delta.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(
#         reference.reshape(reference.size(0), -1), dim=1
#     ).clamp_min(eps)
#     return numerator / denominator


# @torch.no_grad()
# def estimate_phase_coordinate_transform(
#     reference_features: Tensor,
#     current_features: Tensor,
#     *,
#     allow_transport: bool = True,
# ) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
#     """Estimate a conservative observable Q_(t-1) -> Q_t map.

#     The previous implementation used an unconstrained per-channel least-squares
#     slope.  On Indian Pines it reported only ~1.2% paired-Q drift but ~46% scale
#     change, which then moved historical CDS far more than the representation had
#     actually moved.

#     Here each channel solves an identity-centred ridge problem.  The ridge mass is
#     *data-derived* from the observed paired drift energy, not a hand-set clamp::

#         s_d = (x_d^T y_d + lambda_d) / (x_d^T x_d + lambda_d)
#         lambda_d = ||y_d - x_d||^2 + eps

#     When a channel is weakly identifiable or noisy, the solution continuously
#     shrinks toward ``s_d = 1``.  Well-supported channels remain data-driven.  No
#     old-class-specific motion is inferred from unseen old samples.
#     """
#     reference, current = _validate_paired_features(reference_features, current_features)
#     reference = normalize_ordered_decision_features(reference)
#     current = normalize_ordered_decision_features(current)

#     # MPS supports the device transfer in the input dtype, but not float64.
#     # Promote only after both feature tensors are on CPU.
#     source = reference.detach().to(device="cpu").to(dtype=torch.float64)
#     target = current.detach().to(device="cpu").to(dtype=torch.float64)
#     n, token_count, channel_count = source.shape

#     # Validate a map on HELD-OUT current-class feature pairs before transporting
#     # any old CDS. Fitting-error alone can favor a spurious phase correction.
#     # Fixed alternating split is reproducible, never uses historical samples.
#     # A failed holdout comparison retains the identity historical coordinate.
#     def fit_map(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
#         xmean, ymean = x.mean(0), y.mean(0)
#         xc, yc = x - xmean.unsqueeze(0), y - ymean.unsqueeze(0)
#         cross_val = (xc * yc).sum((0, 1))
#         energy_val = xc.square().sum((0, 1))
#         drift_val = (yc - xc).square().sum((0, 1))
#         regularizer = drift_val + torch.finfo(
#             torch.float64
#         ).eps * energy_val.max().clamp_min(1.0)
#         slopes = (cross_val + regularizer) / (energy_val + regularizer)
#         offsets = ymean - slopes.unsqueeze(0) * xmean
#         return slopes, offsets

#     accepted = False
#     if n >= 6 and allow_transport:
#         fit_index = torch.arange(n) % 2 == 0
#         check_index = ~fit_index
#         train_scale, train_offset = fit_map(source[fit_index], target[fit_index])
#         held_prediction = source[check_index] * train_scale.view(
#             1, 1, -1
#         ) + train_offset.unsqueeze(0)
#         held_candidate = (target[check_index] - held_prediction).square().mean()
#         held_identity = (target[check_index] - source[check_index]).square().mean()
#         accepted = bool(held_candidate + 1e-12 < held_identity)
#     else:
#         # Too few paired current samples for a meaningful split; retain the
#         # identity map and report the observable identity error with a finite
#         # value. The acceptance flag records that transport was not validated.
#         held_identity = (target - source).square().mean()
#         held_candidate = held_identity.clone()

#     source_mean = source.mean(dim=0)
#     target_mean = target.mean(dim=0)
#     source_centered = source - source_mean.unsqueeze(0)
#     target_centered = target - target_mean.unsqueeze(0)

#     cross = (source_centered * target_centered).sum(dim=(0, 1))
#     source_energy = source_centered.square().sum(dim=(0, 1))
#     drift_energy = (target_centered - source_centered).square().sum(dim=(0, 1))
#     eps = torch.finfo(torch.float64).eps
#     numerical = eps * source_energy.max().clamp_min(1.0)
#     ridge = drift_energy + numerical
#     channel_scale = (cross + ridge) / (source_energy + ridge)
#     token_translation = target_mean - channel_scale.unsqueeze(0) * source_mean
#     if not accepted:
#         channel_scale = torch.ones_like(channel_scale)
#         token_translation = torch.zeros_like(token_translation)

#     prediction = source * channel_scale.view(
#         1, 1, channel_count
#     ) + token_translation.view(1, token_count, channel_count)
#     residual = target - prediction
#     tiny = torch.finfo(torch.float64).tiny
#     channel_residual = residual.permute(2, 0, 1)
#     channel_target = target.permute(2, 0, 1)
#     channel_error = channel_residual.square().sum(
#         dim=(1, 2)
#     ) / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
#     mapping_error = residual.square().sum() / target.square().sum().clamp_min(tiny)
#     drift_rms = torch.sqrt((target - source).square().mean().clamp_min(0.0))
#     translation_rms = torch.sqrt(token_translation.square().mean().clamp_min(0.0))
#     scale_deviation_rms = torch.sqrt(
#         (channel_scale - 1.0).square().mean().clamp_min(0.0)
#     )

#     values = (
#         channel_scale,
#         token_translation,
#         mapping_error,
#         channel_error,
#         drift_rms,
#         translation_rms,
#         scale_deviation_rms,
#     )
#     if not all(bool(torch.isfinite(value).all()) for value in values):
#         raise RuntimeError("phase coordinate transform contains NaN or Inf")

#     device, dtype = reference.device, reference.dtype
#     transform = PhaseCoordinateTransform(
#         channel_scale=channel_scale.to(device=device, dtype=dtype),
#         token_translation=token_translation.to(device=device, dtype=dtype),
#     )
#     mapping = {
#         "mapping_error": mapping_error.to(device=device, dtype=dtype),
#         "channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
#         "channel_error_max": channel_error.max().to(device=device, dtype=dtype),
#         "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
#         "translation_rms": translation_rms.to(device=device, dtype=dtype),
#         "scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
#         "holdout_candidate_error": held_candidate.to(device=device, dtype=dtype),
#         "holdout_identity_error": held_identity.to(device=device, dtype=dtype),
#         "transport_accepted": torch.tensor(accepted, device=device, dtype=torch.bool),
#     }
#     return transform, mapping


# @torch.no_grad()
# def transport_historical_structure(
#     historical_structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> ClassStructureBatch:
#     """Transport historical means by class-wise translation in normalized Q.

#     This deliberately does not apply unvalidated affine scale transformations:
#     after sample L2 normalization an arbitrary affine map cannot transport
#     covariance or V_Q exactly. A shared translation for all conditions of a
#     class keeps within-class centered covariance and V_Q unchanged.
#     """
#     translation = transform.class_translation
#     if translation is None:
#         raise ValueError("transport requires validated per-class Q translations")
#     value = torch.as_tensor(
#         translation,
#         device=historical_structure.learned_identity.device,
#         dtype=historical_structure.learned_identity.dtype,
#     )
#     if value.shape != historical_structure.learned_identity.shape:
#         raise ValueError("class translation must match historical [C,K,D] means")
#     if not bool(torch.isfinite(value).all()):
#         raise ValueError("historical translation contains nonfinite values")
#     updated = historical_structure.learned_identity + value
#     if bool((updated.flatten(1).norm(dim=1) > 1.00001).any()):
#         raise ValueError("translated historical means violate normalized Q geometry")
#     return replace(historical_structure, learned_identity=updated)


# @torch.no_grad()
# def evidence_controlled_historical_translation(
#     historical_structure: ClassStructureBatch,
#     historical_moments: dict[str, Tensor],
#     reference_features: Tensor,
#     current_features: Tensor,
#     physical_spectral_evidence: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     """Estimate empirical old-class drift only from supported current inputs.

#     Each old class combines its learned conditional-covariance relevance with
#     the actual incoming native-band spectral compatibility. Even/odd sample
#     partition validates whether its proposed translation predicts unseen
#     current pairs better than doing nothing. This test does NOT certify old
#     class drift; unsupported proposals are set to zero.
#     """
#     reference, current = _validate_paired_features(reference_features, current_features)
#     reference = normalize_ordered_decision_features(reference)
#     current = normalize_ordered_decision_features(current)
#     evidence = torch.as_tensor(
#         physical_spectral_evidence, device=reference.device, dtype=reference.dtype
#     )
#     if evidence.shape != (
#         reference.size(0),
#         historical_structure.physical_identity.size(1),
#         1,
#     ):
#         raise ValueError(
#             "current physical evidence must match paired Q and historical native bands"
#         )
#     if not bool(torch.isfinite(evidence).all()):
#         raise ValueError("physical evidence contains nonfinite values")
#     historical_q_support = spectral_conditioned_learned_support(
#         reference, historical_moments, evidence
#     )
#     _, _, historical_p_support = physical_support_components(
#         evidence,
#         historical_structure.physical_identity.to(evidence),
#         historical_structure.physical_variation.to(evidence),
#     )
#     if historical_q_support.shape != historical_p_support.shape:
#         raise ValueError("learned and physical class evidence differ")
#     weights = (historical_q_support * historical_p_support).detach()
#     sample_count = reference.size(0)
#     class_count, token_count, feature_dim = historical_structure.learned_identity.shape
#     translations = reference.new_zeros(class_count, token_count, feature_dim)
#     accepted = torch.zeros(class_count, device=reference.device, dtype=torch.bool)
#     support_mass = weights.sum(0)
#     effective_samples = support_mass.square() / weights.square().sum(0).clamp_min(1e-12)
#     if sample_count < 8:
#         return translations, accepted, support_mass, effective_samples
#     fit = torch.arange(sample_count, device=reference.device) % 2 == 0
#     check = ~fit
#     drift = current - reference
#     moments = {
#         key: torch.as_tensor(
#             value,
#             device=reference.device,
#             dtype=reference.dtype if torch.is_floating_point(value) else value.dtype,
#         )
#         for key, value in historical_moments.items()
#     }
#     for class_index in range(class_count):
#         train_weights = weights[fit, class_index]
#         check_weights = weights[check, class_index]
#         train_mass = train_weights.sum()
#         check_mass = check_weights.sum()
#         fit_effective = train_mass.square() / train_weights.square().sum().clamp_min(
#             1e-12
#         )
#         if (
#             float(train_mass) < 0.5
#             or float(check_mass) < 0.5
#             or float(fit_effective) < 4
#         ):
#             continue
#         translation = (train_weights[:, None, None] * drift[fit]).sum(
#             0
#         ) / train_mass.clamp_min(1e-12)
#         # These are unit-sphere *mean* embeddings; never extrapolate an
#         # unsupported shift that moves an old conditional mean outside its
#         # physically feasible unit ball.
#         center = historical_structure.learned_identity[class_index].to(translation)
#         if float((center + translation).flatten().norm()) > 1.00001:
#             continue
#         group_mean = moments["mean"][class_index]
#         valid_group = moments["counts"][class_index] > 0
#         if bool(
#             (
#                 (group_mean[valid_group] + translation.flatten()).norm(dim=1) > 1.00001
#             ).any()
#         ):
#             continue
#         identity_error = (
#             drift[check].square().mean((1, 2)) * check_weights
#         ).sum() / check_mass
#         candidate_error = (
#             (drift[check] - translation).square().mean((1, 2)) * check_weights
#         ).sum() / check_mass
#         if not bool(torch.isfinite(candidate_error)) or not bool(
#             candidate_error < 0.99 * identity_error
#         ):
#             continue
#         translations[class_index] = translation
#         accepted[class_index] = True
#     return translations, accepted, support_mass, effective_samples


# @torch.no_grad()
# def _whole_cds_relevance(
#     historical_structure: ClassStructureBatch,
#     current_reference_structure: ClassStructureBatch,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     if (
#         historical_structure.class_count < 1
#         or current_reference_structure.class_count < 1
#     ):
#         raise ValueError("nonempty structure required")
#     merged = merge_class_structure_batches(
#         (historical_structure, current_reference_structure)
#     )
#     relations = derive_class_relations(merged)
#     o = historical_structure.class_count
#     learned = relations.learned_combined_proximity[:o, o:].clamp(0, 1)
#     physical = relations.physical_combined_proximity[:o, o:].clamp(0, 1)
#     effective = combine_learned_physical_relevance(learned, physical)
#     eps = torch.finfo(effective.dtype).eps
#     weights = effective / effective.sum(dim=1, keepdim=True).clamp_min(eps)
#     weights = weights * effective.max(dim=1, keepdim=True).values
#     return learned, physical, effective, weights


# @torch.no_grad()
# def correct_historical_structure(
#     *,
#     historical_structure: ClassStructureBatch,
#     current_reference_structure: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     reference_features: Tensor,
#     current_features: Tensor,
#     use_transport: bool = True,
#     historical_moments: dict[str, Tensor] | None = None,
#     physical_spectral_evidence: Tensor | None = None,
# ) -> tuple[ClassStructureBatch, PhaseCoordinateTransform, PhaseAlignmentDiagnostics]:
#     """Propose native-spectral-gated empirical historical translation.

#     The old covariance and V_Q are not recreated from missing old examples;
#     class-wise translations preserve their centered affine interpretation.
#     This does not certify the movement of unavailable old-class samples.
#     """
#     _validate_structure_pair(current_reference_structure, current_structure)
#     # The affine fit is still useful as an observable drift diagnostic, but
#     # cannot itself justify transport of normalized old-class covariance.
#     observed_transform, mapping = estimate_phase_coordinate_transform(
#         reference_features, current_features, allow_transport=False
#     )
#     if use_transport:
#         if historical_moments is None or physical_spectral_evidence is None:
#             raise ValueError(
#                 "historical transport requires bank moments and paired native P"
#             )
#         translation, accepted, alignment_mass, effective_sample_size = (
#             evidence_controlled_historical_translation(
#                 historical_structure,
#                 historical_moments,
#                 reference_features,
#                 current_features,
#                 physical_spectral_evidence,
#             )
#         )
#         transform = PhaseCoordinateTransform(
#             channel_scale=torch.ones_like(observed_transform.channel_scale),
#             token_translation=torch.zeros_like(observed_transform.token_translation),
#             class_translation=translation,
#         )
#         moved = transport_historical_structure(historical_structure, transform)
#         mapping["transport_accepted"] = accepted.any()
#         mapping["translation_rms"] = translation.square().mean().sqrt()
#     else:
#         transform = observed_transform
#         moved = historical_structure
#         alignment_mass = torch.zeros(
#             historical_structure.class_count,
#             device=historical_structure.learned_identity.device,
#             dtype=historical_structure.learned_identity.dtype,
#         )
#         effective_sample_size = alignment_mass.clone()
#     ref_moved = current_reference_structure
#     learned, physical, effective, weights = _whole_cds_relevance(
#         historical_structure, current_reference_structure
#     )
#     support_mass = alignment_mass if use_transport else weights.sum(dim=1)
#     # Diagnostics are grounded in measured NEW class transport residuals.
#     identity_delta = current_structure.learned_identity - ref_moved.learned_identity
#     variation_delta = current_structure.learned_variation - ref_moved.learned_variation
#     identity_relative = _relative_batch_norm(identity_delta, ref_moved.learned_identity)
#     variation_relative = _relative_batch_norm(
#         variation_delta, ref_moved.learned_variation
#     )
#     z = mapping["mapping_error"] * 0
#     diagnostics = PhaseAlignmentDiagnostics(
#         mapping_error=mapping["mapping_error"],
#         channel_error_mean=mapping["channel_error_mean"],
#         channel_error_max=mapping["channel_error_max"],
#         representation_drift_rms=mapping["representation_drift_rms"],
#         translation_rms=mapping["translation_rms"],
#         scale_deviation_rms=mapping["scale_deviation_rms"],
#         learned_relevance_mean=learned.mean(),
#         physical_relevance_mean=physical.mean(),
#         effective_relevance_mean=effective.mean(),
#         class_specific_support_mass_mean=support_mass.mean(),
#         class_specific_support_mass_min=support_mass.min(),
#         class_specific_support_mass_max=support_mass.max(),
#         effective_sample_size_mean=effective_sample_size.mean(),
#         effective_sample_size_min=effective_sample_size.min(),
#         identity_correction_relative_mean=identity_relative.mean(),
#         identity_correction_relative_max=identity_relative.max(),
#         variation_correction_relative_mean=variation_relative.mean(),
#         variation_correction_relative_max=variation_relative.max(),
#         variation_negative_fraction_before_sanitize=z,
#         holdout_candidate_error=mapping["holdout_candidate_error"],
#         holdout_identity_error=mapping["holdout_identity_error"],
#         transport_accepted=mapping["transport_accepted"],
#     )
#     return moved, transform, diagnostics


# __all__ = [
#     "PhaseCoordinateTransform",
#     "PhaseAlignmentDiagnostics",
#     "estimate_phase_coordinate_transform",
#     "transport_historical_structure",
#     "evidence_controlled_historical_translation",
#     "correct_historical_structure",
# ]


# from __future__ import annotations
# from dataclasses import dataclass

# import torch

# from models.geometry_bank import (
#     ClassStructureBatch,
#     combine_learned_physical_relevance,
#     derive_class_relations,
#     merge_class_structure_batches,
#     normalize_ordered_decision_features,
#     sanitize_ordered_variation,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class PhaseCoordinateTransform:
#     """Single phase-wise map from finalized Q_(t-1) to finalized Q_t.

#     ``channel_scale`` is shared over the ordered spectral-token axis so the
#     Whole-CDS variation field has an exact compatible transport
#     ``V_Q' = |s| V_Q``. ``token_translation`` may differ by ordered token.
#     """

#     channel_scale: Tensor  # [D]
#     token_translation: Tensor  # [K,D]


# @dataclass(frozen=True)
# class PhaseAlignmentDiagnostics:
#     mapping_error: Tensor
#     channel_error_mean: Tensor
#     channel_error_max: Tensor
#     representation_drift_rms: Tensor
#     translation_rms: Tensor
#     scale_deviation_rms: Tensor
#     learned_relevance_mean: Tensor
#     physical_relevance_mean: Tensor
#     effective_relevance_mean: Tensor
#     class_specific_support_mass_mean: Tensor
#     class_specific_support_mass_min: Tensor
#     class_specific_support_mass_max: Tensor
#     identity_correction_relative_mean: Tensor
#     identity_correction_relative_max: Tensor
#     variation_correction_relative_mean: Tensor
#     variation_correction_relative_max: Tensor
#     variation_negative_fraction_before_sanitize: Tensor
#     # Observable new-class holdout test, not an old-class transfer certificate.
#     holdout_candidate_error: Tensor
#     holdout_identity_error: Tensor
#     transport_accepted: Tensor


# def _validate_paired_features(
#     reference_features: Tensor,
#     current_features: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     reference = torch.as_tensor(reference_features)
#     current = torch.as_tensor(current_features)
#     if (
#         reference.ndim != 3
#         or reference.size(0) < 2
#         or reference.size(1) < 2
#         or reference.size(2) <= 0
#     ):
#         raise ValueError("reference_features must be [N,K,D] with N>=2,K>=2,D>=1")
#     if current.shape != reference.shape:
#         raise ValueError("current_features must match reference_features")
#     if reference.device != current.device or reference.dtype != current.dtype:
#         raise ValueError("paired Q-/Q+ features must share device/dtype")
#     if not torch.is_floating_point(reference) or reference.is_complex():
#         raise TypeError("paired Q-/Q+ features must be real floating point")
#     if not bool(torch.isfinite(reference).all()) or not bool(
#         torch.isfinite(current).all()
#     ):
#         raise ValueError("paired Q-/Q+ features contain NaN or Inf")
#     return reference, current


# def _validate_structure_pair(
#     reference: ClassStructureBatch, current: ClassStructureBatch
# ) -> None:
#     if not isinstance(reference, ClassStructureBatch) or not isinstance(
#         current, ClassStructureBatch
#     ):
#         raise TypeError("reference/current structures must be ClassStructureBatch")
#     if reference.class_count <= 0 or current.class_count != reference.class_count:
#         raise ValueError("reference/current structures have incompatible class counts")
#     if not torch.equal(reference.class_ids, current.class_ids):
#         raise ValueError(
#             "reference/current class IDs must be identical and ordered equally"
#         )
#     if reference.learned_identity.shape != current.learned_identity.shape:
#         raise ValueError("reference/current I_Q shapes disagree")
#     if reference.learned_variation.shape != current.learned_variation.shape:
#         raise ValueError("reference/current V_Q shapes disagree")
#     for name in (
#         "learned_coupled_basis",
#         "learned_coupled_scale",
#         "learned_coupled_active_rank",
#         "learned_residual_scale",
#     ):
#         if getattr(reference, name).shape != getattr(current, name).shape:
#             raise ValueError(f"reference/current {name} shapes disagree")
#     # Physical evidence is a native-spectrum coordinate and must never be moved by
#     # a neural feature-space transform.
#     if not torch.equal(reference.physical_identity, current.physical_identity):
#         raise ValueError("current-class I_P changed across neural coordinates")
#     if not torch.equal(reference.physical_variation, current.physical_variation):
#         raise ValueError("current-class V_P changed across neural coordinates")


# def _relative_batch_norm(delta: Tensor, reference: Tensor) -> Tensor:
#     """Per-class relative L2 change with a finite zero-safe denominator."""
#     if delta.shape != reference.shape or delta.ndim < 2 or delta.size(0) <= 0:
#         raise ValueError("relative-norm tensors must share a non-empty class dimension")
#     eps = torch.finfo(reference.dtype).eps
#     numerator = torch.linalg.vector_norm(delta.reshape(delta.size(0), -1), dim=1)
#     denominator = torch.linalg.vector_norm(
#         reference.reshape(reference.size(0), -1), dim=1
#     ).clamp_min(eps)
#     return numerator / denominator


# @torch.no_grad()
# def estimate_phase_coordinate_transform(
#     reference_features: Tensor,
#     current_features: Tensor,
#     *,
#     allow_transport: bool = True,
# ) -> tuple[PhaseCoordinateTransform, dict[str, Tensor]]:
#     """Estimate a conservative observable Q_(t-1) -> Q_t map.

#     The previous implementation used an unconstrained per-channel least-squares
#     slope.  On Indian Pines it reported only ~1.2% paired-Q drift but ~46% scale
#     change, which then moved historical CDS far more than the representation had
#     actually moved.

#     Here each channel solves an identity-centred ridge problem.  The ridge mass is
#     *data-derived* from the observed paired drift energy, not a hand-set clamp::

#         s_d = (x_d^T y_d + lambda_d) / (x_d^T x_d + lambda_d)
#         lambda_d = ||y_d - x_d||^2 + eps

#     When a channel is weakly identifiable or noisy, the solution continuously
#     shrinks toward ``s_d = 1``.  Well-supported channels remain data-driven.  No
#     old-class-specific motion is inferred from unseen old samples.
#     """
#     reference, current = _validate_paired_features(reference_features, current_features)
#     reference = normalize_ordered_decision_features(reference)
#     current = normalize_ordered_decision_features(current)

#     # MPS supports the device transfer in the input dtype, but not float64.
#     # Promote only after both feature tensors are on CPU.
#     source = reference.detach().to(device="cpu").to(dtype=torch.float64)
#     target = current.detach().to(device="cpu").to(dtype=torch.float64)
#     n, token_count, channel_count = source.shape

#     # Validate a map on HELD-OUT current-class feature pairs before transporting
#     # any old CDS. Fitting-error alone can favor a spurious phase correction.
#     # Fixed alternating split is reproducible, never uses historical samples.
#     # A failed holdout comparison retains the identity historical coordinate.
#     def fit_map(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
#         xmean, ymean = x.mean(0), y.mean(0)
#         xc, yc = x - xmean.unsqueeze(0), y - ymean.unsqueeze(0)
#         cross_val = (xc * yc).sum((0, 1))
#         energy_val = xc.square().sum((0, 1))
#         drift_val = (yc - xc).square().sum((0, 1))
#         regularizer = drift_val + torch.finfo(
#             torch.float64
#         ).eps * energy_val.max().clamp_min(1.0)
#         slopes = (cross_val + regularizer) / (energy_val + regularizer)
#         offsets = ymean - slopes.unsqueeze(0) * xmean
#         return slopes, offsets

#     accepted = False
#     if n >= 6 and allow_transport:
#         fit_index = torch.arange(n) % 2 == 0
#         check_index = ~fit_index
#         train_scale, train_offset = fit_map(source[fit_index], target[fit_index])
#         held_prediction = source[check_index] * train_scale.view(
#             1, 1, -1
#         ) + train_offset.unsqueeze(0)
#         held_candidate = (target[check_index] - held_prediction).square().mean()
#         held_identity = (target[check_index] - source[check_index]).square().mean()
#         accepted = bool(held_candidate + 1e-12 < held_identity)
#     else:
#         # Too few paired current samples for a meaningful split; retain the
#         # identity map and report the observable identity error with a finite
#         # value. The acceptance flag records that transport was not validated.
#         held_identity = (target - source).square().mean()
#         held_candidate = held_identity.clone()

#     source_mean = source.mean(dim=0)
#     target_mean = target.mean(dim=0)
#     source_centered = source - source_mean.unsqueeze(0)
#     target_centered = target - target_mean.unsqueeze(0)

#     cross = (source_centered * target_centered).sum(dim=(0, 1))
#     source_energy = source_centered.square().sum(dim=(0, 1))
#     drift_energy = (target_centered - source_centered).square().sum(dim=(0, 1))
#     eps = torch.finfo(torch.float64).eps
#     numerical = eps * source_energy.max().clamp_min(1.0)
#     ridge = drift_energy + numerical
#     channel_scale = (cross + ridge) / (source_energy + ridge)
#     token_translation = target_mean - channel_scale.unsqueeze(0) * source_mean
#     if not accepted:
#         channel_scale = torch.ones_like(channel_scale)
#         token_translation = torch.zeros_like(token_translation)

#     prediction = source * channel_scale.view(
#         1, 1, channel_count
#     ) + token_translation.view(1, token_count, channel_count)
#     residual = target - prediction
#     tiny = torch.finfo(torch.float64).tiny
#     channel_residual = residual.permute(2, 0, 1)
#     channel_target = target.permute(2, 0, 1)
#     channel_error = channel_residual.square().sum(
#         dim=(1, 2)
#     ) / channel_target.square().sum(dim=(1, 2)).clamp_min(tiny)
#     mapping_error = residual.square().sum() / target.square().sum().clamp_min(tiny)
#     drift_rms = torch.sqrt((target - source).square().mean().clamp_min(0.0))
#     translation_rms = torch.sqrt(token_translation.square().mean().clamp_min(0.0))
#     scale_deviation_rms = torch.sqrt(
#         (channel_scale - 1.0).square().mean().clamp_min(0.0)
#     )

#     values = (
#         channel_scale,
#         token_translation,
#         mapping_error,
#         channel_error,
#         drift_rms,
#         translation_rms,
#         scale_deviation_rms,
#     )
#     if not all(bool(torch.isfinite(value).all()) for value in values):
#         raise RuntimeError("phase coordinate transform contains NaN or Inf")

#     device, dtype = reference.device, reference.dtype
#     transform = PhaseCoordinateTransform(
#         channel_scale=channel_scale.to(device=device, dtype=dtype),
#         token_translation=token_translation.to(device=device, dtype=dtype),
#     )
#     mapping = {
#         "mapping_error": mapping_error.to(device=device, dtype=dtype),
#         "channel_error_mean": channel_error.mean().to(device=device, dtype=dtype),
#         "channel_error_max": channel_error.max().to(device=device, dtype=dtype),
#         "representation_drift_rms": drift_rms.to(device=device, dtype=dtype),
#         "translation_rms": translation_rms.to(device=device, dtype=dtype),
#         "scale_deviation_rms": scale_deviation_rms.to(device=device, dtype=dtype),
#         "holdout_candidate_error": held_candidate.to(device=device, dtype=dtype),
#         "holdout_identity_error": held_identity.to(device=device, dtype=dtype),
#         "transport_accepted": torch.tensor(accepted, device=device, dtype=torch.bool),
#     }
#     return transform, mapping


# @torch.no_grad()
# def transport_historical_structure(
#     historical_structure: ClassStructureBatch,
#     transform: PhaseCoordinateTransform,
# ) -> ClassStructureBatch:
#     """Best-effort compact CDS transport through an observed phase map.

#     The affine transform was learned on CURRENT examples; old-class transfer is
#     NOT a certified old-class drift bound. After nonlinear unit normalization
#     coupled-geometry support is an approximate proxy, not a certified enclosure.
#     The native-band structures are copied without modification.
#     """
#     state = historical_structure
#     identity = state.learned_identity
#     c, k, d = identity.shape
#     scale = transform.channel_scale.to(device=identity.device, dtype=identity.dtype)
#     translation = transform.token_translation.to(
#         device=identity.device, dtype=identity.dtype
#     )
#     if scale.shape != (d,) or translation.shape != (k, d):
#         raise ValueError("phase map and historical Q mismatch")
#     raw_center = identity * scale.view(1, 1, d) + translation.view(1, k, d)
#     raw_norm = (
#         raw_center.reshape(c, -1).norm(dim=1).clamp_min(torch.finfo(identity.dtype).eps)
#     )
#     # Preserve the norm of a robust class identity; it is not necessarily unit
#     # length despite every individual decision feature being unit normalized.
#     original_norm = (
#         identity.reshape(c, -1).norm(dim=1).clamp_min(torch.finfo(identity.dtype).eps)
#     )
#     correction = original_norm / raw_norm
#     normalized_center = raw_center * correction.view(c, 1, 1)
#     variation = state.learned_variation
#     if variation.shape != (c, 2, k - 1, d):
#         raise ValueError("expected signed adjacent learned V_Q")
#     lo = variation[:, 0] * scale.view(1, 1, d)
#     hi = variation[:, 1] * scale.view(1, 1, d)
#     transported_v = torch.stack((torch.minimum(lo, hi), torch.maximum(lo, hi)), dim=1)
#     transported_v = transported_v * correction.view(c, 1, 1, 1)

#     # Approximate normalized image of the directional support, with no old
#     # samples invented or persisted. Axis scales are recalculated via SVD.
#     new_basis = torch.zeros_like(state.learned_coupled_basis)
#     new_lam = torch.zeros_like(state.learned_coupled_scale)
#     new_rank = state.learned_coupled_active_rank.detach().clone()
#     new_eps = torch.zeros_like(state.learned_residual_scale)
#     for row in range(c):
#         r = int(new_rank[row].item())
#         amplitude = scale.repeat(k).abs().amax() * correction[row]
#         new_eps[row] = state.learned_residual_scale[row] * amplitude
#         if r:
#             uu = state.learned_coupled_basis[row, :, :r]
#             lam = state.learned_coupled_scale[row, :r]
#             transferred = (scale.repeat(k).unsqueeze(1) * uu) * lam.unsqueeze(0)
#             transferred = transferred * correction[row]
#             left, singular, _ = torch.linalg.svd(transferred, full_matrices=False)
#             new_basis[row, :, :r] = left[:, :r]
#             new_lam[row, :r] = singular[:r]
#     return ClassStructureBatch(
#         class_ids=state.class_ids.detach().clone(),
#         sample_counts=state.sample_counts.detach().clone(),
#         learned_identity=normalized_center.detach().contiguous(),
#         learned_variation=transported_v.detach().contiguous(),
#         learned_coupled_basis=new_basis.detach().contiguous(),
#         learned_coupled_scale=new_lam.detach().contiguous(),
#         learned_coupled_active_rank=new_rank,
#         learned_residual_scale=new_eps.detach().contiguous(),
#         physical_identity=state.physical_identity.detach().clone(),
#         physical_variation=state.physical_variation.detach().clone(),
#     )


# @torch.no_grad()
# def _whole_cds_relevance(
#     historical_structure: ClassStructureBatch,
#     current_reference_structure: ClassStructureBatch,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     if (
#         historical_structure.class_count < 1
#         or current_reference_structure.class_count < 1
#     ):
#         raise ValueError("nonempty structure required")
#     merged = merge_class_structure_batches(
#         (historical_structure, current_reference_structure)
#     )
#     relations = derive_class_relations(merged)
#     o = historical_structure.class_count
#     learned = relations.learned_combined_proximity[:o, o:].clamp(0, 1)
#     physical = relations.physical_combined_proximity[:o, o:].clamp(0, 1)
#     effective = combine_learned_physical_relevance(learned, physical)
#     eps = torch.finfo(effective.dtype).eps
#     weights = effective / effective.sum(dim=1, keepdim=True).clamp_min(eps)
#     weights = weights * effective.max(dim=1, keepdim=True).values
#     return learned, physical, effective, weights


# @torch.no_grad()
# def correct_historical_structure(
#     *,
#     historical_structure: ClassStructureBatch,
#     current_reference_structure: ClassStructureBatch,
#     current_structure: ClassStructureBatch,
#     reference_features: Tensor,
#     current_features: Tensor,
#     use_transport: bool = True,
# ) -> tuple[ClassStructureBatch, PhaseCoordinateTransform, PhaseAlignmentDiagnostics]:
#     """Candidate global transport; no unverifiable class-specific residual graft.

#     Relational support and G_Q move together. This is an empirical alignment,
#     NOT a proof that missing old samples follow the observed current-class map.
#     """
#     _validate_structure_pair(current_reference_structure, current_structure)
#     transform, mapping = estimate_phase_coordinate_transform(
#         reference_features, current_features, allow_transport=use_transport
#     )
#     moved = transport_historical_structure(historical_structure, transform)
#     ref_moved = transport_historical_structure(current_reference_structure, transform)
#     learned, physical, effective, weights = _whole_cds_relevance(
#         historical_structure, current_reference_structure
#     )
#     support_mass = weights.sum(dim=1)
#     # Diagnostics are grounded in measured NEW class transport residuals.
#     identity_delta = current_structure.learned_identity - ref_moved.learned_identity
#     variation_delta = current_structure.learned_variation - ref_moved.learned_variation
#     identity_relative = _relative_batch_norm(identity_delta, ref_moved.learned_identity)
#     variation_relative = _relative_batch_norm(
#         variation_delta, ref_moved.learned_variation
#     )
#     z = mapping["mapping_error"] * 0
#     diagnostics = PhaseAlignmentDiagnostics(
#         mapping_error=mapping["mapping_error"],
#         channel_error_mean=mapping["channel_error_mean"],
#         channel_error_max=mapping["channel_error_max"],
#         representation_drift_rms=mapping["representation_drift_rms"],
#         translation_rms=mapping["translation_rms"],
#         scale_deviation_rms=mapping["scale_deviation_rms"],
#         learned_relevance_mean=learned.mean(),
#         physical_relevance_mean=physical.mean(),
#         effective_relevance_mean=effective.mean(),
#         class_specific_support_mass_mean=support_mass.mean(),
#         class_specific_support_mass_min=support_mass.min(),
#         class_specific_support_mass_max=support_mass.max(),
#         identity_correction_relative_mean=identity_relative.mean(),
#         identity_correction_relative_max=identity_relative.max(),
#         variation_correction_relative_mean=variation_relative.mean(),
#         variation_correction_relative_max=variation_relative.max(),
#         variation_negative_fraction_before_sanitize=z,
#         holdout_candidate_error=mapping["holdout_candidate_error"],
#         holdout_identity_error=mapping["holdout_identity_error"],
#         transport_accepted=mapping["transport_accepted"],
#     )
#     return moved, transform, diagnostics


# __all__ = [
#     "PhaseCoordinateTransform",
#     "PhaseAlignmentDiagnostics",
#     "estimate_phase_coordinate_transform",
#     "transport_historical_structure",
#     "correct_historical_structure",
# ]
