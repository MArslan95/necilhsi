from __future__ import annotations

from dataclasses import dataclass



import torch

import torch.nn as nn

import torch.nn.functional as F



from models.classifier import CosineClassificationHead

from models.geometry_bank import (

    ClassDiscriminativeStructureState,

    combine_learned_physical_relevance,

)



Tensor = torch.Tensor


@dataclass(frozen=True)

class WholeCDSClassifierProtectionObjective:

    """Compatibility view of the finalized all-seen classifier objective."""



    loss: Tensor

    global_reconstruction_loss: Tensor

    old_class_reconstruction_loss: Tensor

    new_class_reconstruction_loss: Tensor

    old_old_retention_loss: Tensor



    historical_identity_margin_mean: Tensor

    historical_identity_margin_min: Tensor

    historical_deformation_exposure_mean: Tensor

    historical_structural_clearance_mean: Tensor

    historical_structural_clearance_min: Tensor



    current_identity_margin_mean: Tensor

    current_identity_margin_min: Tensor

    current_deformation_exposure_mean: Tensor

    current_structural_clearance_mean: Tensor

    current_structural_clearance_min: Tensor



    historical_internal_clearance_mean: Tensor

    historical_internal_clearance_min: Tensor

    historical_internal_retention_mean: Tensor

    historical_internal_retention_min: Tensor

    historical_internal_violation_fraction: Tensor



    joint_structural_clearance_mean: Tensor

    joint_structural_clearance_min: Tensor

    unsafe_pair_fraction: Tensor

    pair_weight_mean: Tensor

    learned_pair_risk_mean: Tensor

    physical_pair_risk_mean: Tensor

    effective_pair_risk_mean: Tensor

    old_new_structural_violation_mean: Tensor

    old_new_violation_fraction: Tensor



    reference_clearance_mean: Tensor

    reference_clearance_min: Tensor

    reference_retention_mean: Tensor

    reference_retention_min: Tensor

    reference_violation_fraction: Tensor



    @property

    def historical_robust_classification_loss(self) -> Tensor:

        return self.old_class_reconstruction_loss



    @property

    def current_robust_structural_loss(self) -> Tensor:

        return self.new_class_reconstruction_loss





@dataclass(frozen=True)

class WholeCDSClassBalancedObjective:

    loss: Tensor

    per_class_loss: Tensor

    clearance_mean: Tensor

    clearance_min: Tensor

    learned_pair_risk_mean: Tensor

    physical_pair_risk_mean: Tensor

    effective_pair_risk_mean: Tensor

    violation_fraction: Tensor

    old_new_clearance_mean: Tensor

    old_new_clearance_min: Tensor

    old_new_violation_fraction: Tensor

    old_old_clearance_mean: Tensor

    old_old_clearance_min: Tensor

    old_old_violation_fraction: Tensor

    new_new_clearance_mean: Tensor

    new_new_clearance_min: Tensor

    new_new_violation_fraction: Tensor






@dataclass(frozen=True)
class IncomingClassifierCalibrationState:
    """State of real-current fitting under historical old->new CDS constraints."""

    real_loss: Tensor
    real_accuracy: Tensor
    real_margin_mean: Tensor
    historical_clearance_mean: Tensor
    historical_clearance_min: Tensor
    historical_violation_fraction: Tensor
    historical_risk_weighted_violation: Tensor
    learned_pair_risk_mean: Tensor
    physical_pair_risk_mean: Tensor
    effective_pair_risk_mean: Tensor


def _validate_cds(

    *,

    classifier: CosineClassificationHead,

    cds: ClassDiscriminativeStructureState,

) -> tuple[Tensor, Tensor, Tensor, int, int, int]:

    if not isinstance(classifier, CosineClassificationHead):

        raise TypeError("classifier must be CosineClassificationHead")

    if not isinstance(cds, ClassDiscriminativeStructureState):

        raise TypeError("cds must be ClassDiscriminativeStructureState")



    classes = cds.classes

    ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()

    count = int(ids.numel())

    if count < 2 or int(torch.unique(ids).numel()) != count:

        raise ValueError("CDS classifier objective requires at least two unique classes")



    identity = classes.learned_identity.to(

        device=classifier.weight.device, dtype=classifier.weight.dtype

    ).detach()

    variation = classes.learned_variation.to(

        device=classifier.weight.device, dtype=classifier.weight.dtype

    ).detach()

    if identity.ndim != 3 or identity.size(0) != count or identity.size(1) < 2:

        raise ValueError("I_Q must be [C,K,D], K>=2")

    c, k, d = identity.shape

    if variation.shape != (c, k, k, d):

        raise ValueError("V_Q must be [C,K,K,D]")

    if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):

        raise ValueError("I_Q/V_Q contains NaN or Inf")

    if bool((variation < 0).any()):

        raise ValueError("V_Q must be non-negative")

    if int(classifier.feature_dim) != k * d:

        raise ValueError("classifier feature dimension does not match Q")

    return ids, identity, variation, count, k, d





def _directed_identity_margin_and_relation_exposure(

    *,

    source_identity: Tensor,

    source_deformation: Tensor,

    boundary: Tensor,

) -> tuple[Tensor, Tensor, Tensor]:

    """Robust directed cosine clearance from complete learned CDS.



    I_Q supplies ordered identity. V_Q supplies diagonal absolute variation and

    off-diagonal ordered-token relation variation. V_Q is a robust scale field,

    never a covariance or Gaussian memory.

    """

    identity = torch.as_tensor(source_identity)

    deformation = torch.as_tensor(source_deformation)

    normal = torch.as_tensor(boundary)

    if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:

        raise ValueError("source_identity must be [S,K,D]")

    s, k, d = identity.shape

    if deformation.shape != (s, k, k, d):

        raise ValueError("source_deformation must be [S,K,K,D]")

    if normal.ndim != 3 or normal.size(0) != s or normal.size(2) != k * d:

        raise ValueError("boundary must be [S,R,K*D]")

    if not all(bool(torch.isfinite(v).all()) for v in (identity, deformation, normal)):

        raise ValueError("boundary-clearance tensors contain NaN or Inf")



    eps = torch.finfo(identity.dtype).eps

    flat_identity = identity.reshape(s, -1)

    identity_norm = flat_identity.norm(dim=1, keepdim=True).clamp_min(eps)

    identity_unit = flat_identity / identity_norm

    margin = (normal * identity_unit[:, None, :]).sum(dim=2)



    tangent = (

        normal - margin[:, :, None] * identity_unit[:, None, :]

    ) / identity_norm[:, None, :]

    tangent_kd = tangent.reshape(s, normal.size(1), k, d)



    index = torch.arange(k, device=identity.device)

    absolute_scale = deformation[:, index, index, :]

    absolute_exposure = (

        tangent_kd.abs() * absolute_scale[:, None, :, :]

    ).sum(dim=(2, 3))



    relation_sensitivity = (

        tangent_kd[:, :, :, None, :] - tangent_kd[:, :, None, :, :]

    ).abs()

    upper = torch.triu(

        torch.ones(k, k, device=identity.device, dtype=torch.bool), diagonal=1

    )

    relational_exposure = (

        relation_sensitivity

        * deformation[:, None, :, :, :]

        * upper.view(1, 1, k, k, 1)

    ).sum(dim=(2, 3, 4)) / float(k)



    exposure = torch.maximum(absolute_exposure, relational_exposure)

    clearance = margin - exposure

    if not all(bool(torch.isfinite(v).all()) for v in (margin, exposure, clearance)):

        raise RuntimeError("Whole-CDS boundary clearance became NaN or Inf")

    return margin, exposure, clearance





def _effective_pair_risk(learned: Tensor, physical: Tensor) -> Tensor:

    learned_value = torch.as_tensor(learned)

    physical_value = torch.as_tensor(

        physical, device=learned_value.device, dtype=learned_value.dtype

    )

    if learned_value.ndim != 2 or learned_value.shape != physical_value.shape:

        raise ValueError("learned/physical risk must share a rank-two shape")

    return combine_learned_physical_relevance(learned_value, physical_value)





def _classwise_robust_loss(

    *,

    clearance: Tensor,

    pair_weight: Tensor,

    valid_mask: Tensor,

    scale: Tensor,

) -> Tensor:

    if clearance.shape != pair_weight.shape or clearance.shape != valid_mask.shape:

        raise ValueError("clearance, pair_weight and valid_mask must share shape")

    if valid_mask.dtype != torch.bool:

        raise TypeError("valid_mask must be boolean")

    eps = torch.finfo(clearance.dtype).eps

    terms = pair_weight.clamp_min(eps).log() - scale * clearance

    terms = terms.masked_fill(~valid_mask, -torch.inf)

    if not bool(valid_mask.any(dim=1).all()):

        raise RuntimeError("every source class must have at least one rival")

    zero = torch.zeros((clearance.size(0), 1), device=clearance.device, dtype=clearance.dtype)

    return torch.logsumexp(torch.cat((zero, terms), dim=1), dim=1)





def _masked_stats(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:

    if values.shape != mask.shape or mask.dtype != torch.bool:

        raise ValueError("values/mask mismatch")

    selected = values[mask]

    zero = values.sum() * 0.0

    if selected.numel() == 0:

        return zero, zero, zero

    return (

        selected.mean(),

        selected.min(),

        selected.lt(0.0).to(values.dtype).mean(),

    )





def _boundary_state(

    *,

    classifier: CosineClassificationHead,

    cds: ClassDiscriminativeStructureState,

    trainable_class_ids: Tensor | None = None,

) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    ids, identity, variation, count, _k, _d = _validate_cds(

        classifier=classifier, cds=cds

    )

    eps = torch.finfo(identity.dtype).eps

    raw_rows = classifier.class_weight(ids)
    if trainable_class_ids is None:
        owned_rows = raw_rows
    else:
        trainable_ids = torch.as_tensor(
            trainable_class_ids, device=ids.device, dtype=torch.long
        ).flatten()
        if trainable_ids.ndim != 1 or trainable_ids.numel() == 0:
            raise ValueError("trainable_class_ids must be a non-empty rank-one tensor")
        if int(torch.unique(trainable_ids).numel()) != int(trainable_ids.numel()):
            raise ValueError("trainable_class_ids must be unique")
        membership = trainable_ids.unsqueeze(1).eq(ids.unsqueeze(0))
        if not bool(membership.sum(dim=1).eq(1).all()):
            raise ValueError("trainable_class_ids must be contained in CDS class IDs")
        trainable_mask = ids.unsqueeze(1).eq(trainable_ids.unsqueeze(0)).any(dim=1)
        owned_rows = torch.where(
            trainable_mask.unsqueeze(1), raw_rows, raw_rows.detach()
        )
    rows = F.normalize(owned_rows, p=2, dim=1, eps=eps)

    boundary = rows[:, None, :] - rows[None, :, :]

    margin, exposure, clearance = _directed_identity_margin_and_relation_exposure(

        source_identity=identity,

        source_deformation=variation,

        boundary=boundary,

    )



    learned = cds.relations.learned_combined_proximity.to(

        device=identity.device, dtype=identity.dtype

    ).detach()

    physical = cds.relations.physical_combined_proximity.to(

        device=identity.device, dtype=identity.dtype

    ).detach()

    if learned.shape != (count, count) or physical.shape != (count, count):

        raise RuntimeError("Whole-CDS relation matrices must be [C,C]")

    effective = _effective_pair_risk(learned, physical)

    pair_weight = 1.0 + effective

    valid = ~torch.eye(count, device=identity.device, dtype=torch.bool)

    return ids, margin, exposure, clearance, learned, physical, effective, pair_weight, valid





def whole_cds_class_balanced_classifier_loss(

    *,

    classifier: CosineClassificationHead,

    cds: ClassDiscriminativeStructureState,

    historical_class_count: int | None = None,

    trainable_class_ids: Tensor | None = None,

) -> WholeCDSClassBalancedObjective:

    """Finalize the one deployed cosine head from the finalized all-seen Whole-CDS.



    I_Q and V_Q define robust directed boundary clearance. Learned and physical

    inter-class CDS relations determine which rival boundaries deserve stronger

    protection. No CDS score is added to inference logits: deployment remains

    cosine(Q,W), while Whole-CDS is a training-time boundary constraint.

    """

    (

        ids, _margin, _exposure, clearance, learned, physical,

        effective, pair_weight, valid,

    ) = _boundary_state(

        classifier=classifier,

        cds=cds,

        trainable_class_ids=trainable_class_ids,

    )

    count = int(ids.numel())

    scale = classifier.logit_scale.to(

        device=clearance.device, dtype=clearance.dtype

    ).detach()

    per_class = _classwise_robust_loss(

        clearance=clearance,

        pair_weight=pair_weight,

        valid_mask=valid,

        scale=scale,

    )

    loss = per_class.mean()



    if historical_class_count is None:

        old_count = 0

    else:

        old_count = int(historical_class_count)

        if old_count <= 0 or old_count >= count:

            raise ValueError("historical_class_count must split old and new seen classes")



    if old_count > 0:

        index = torch.arange(count, device=clearance.device)

        old = index < old_count

        new = ~old

        old_new_mask = (old[:, None] & new[None, :]) | (new[:, None] & old[None, :])

        old_old_mask = old[:, None] & old[None, :] & valid

        new_new_mask = new[:, None] & new[None, :] & valid

    else:

        old_new_mask = torch.zeros_like(valid)

        old_old_mask = torch.zeros_like(valid)

        new_new_mask = valid



    old_new_mean, old_new_min, old_new_violation = _masked_stats(clearance, old_new_mask)

    old_old_mean, old_old_min, old_old_violation = _masked_stats(clearance, old_old_mask)

    new_new_mean, new_new_min, new_new_violation = _masked_stats(clearance, new_new_mask)



    scalars = (

        loss,

        clearance[valid].mean(),

        clearance[valid].min(),

        learned[valid].mean(),

        physical[valid].mean(),

        effective[valid].mean(),

        clearance[valid].lt(0.0).to(clearance.dtype).mean(),

        old_new_mean, old_new_min, old_new_violation,

        old_old_mean, old_old_min, old_old_violation,

        new_new_mean, new_new_min, new_new_violation,

    )

    if not all(bool(torch.isfinite(v)) for v in scalars):

        raise RuntimeError("Whole-CDS classifier finalization became NaN or Inf")



    return WholeCDSClassBalancedObjective(

        loss=loss,

        per_class_loss=per_class,

        clearance_mean=clearance[valid].mean(),

        clearance_min=clearance[valid].min(),

        learned_pair_risk_mean=learned[valid].mean(),

        physical_pair_risk_mean=physical[valid].mean(),

        effective_pair_risk_mean=effective[valid].mean(),

        violation_fraction=clearance[valid].lt(0.0).to(clearance.dtype).mean(),

        old_new_clearance_mean=old_new_mean,

        old_new_clearance_min=old_new_min,

        old_new_violation_fraction=old_new_violation,

        old_old_clearance_mean=old_old_mean,

        old_old_clearance_min=old_old_min,

        old_old_violation_fraction=old_old_violation,

        new_new_clearance_mean=new_new_mean,

        new_new_clearance_min=new_new_min,

        new_new_violation_fraction=new_new_violation,

    )






def _historical_old_to_new_boundary_state(
    *,
    classifier: CosineClassificationHead,
    cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return historical old->new robust clearance and Whole-CDS pair risk.

    Only historical I_Q/V_Q define the robust occupied region. New-class CDS is
    intentionally not used as classifier supervision because real current HSI
    is available for the incoming side.
    """
    ids, identity, variation, count, _k, _d = _validate_cds(
        classifier=classifier, cds=cds
    )
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and new seen classes")

    old_ids = ids[:old_count]
    new_ids = ids[old_count:]
    eps = torch.finfo(identity.dtype).eps

    # Historical rows are fixed anchors. Incoming rows are the only rows that
    # the calibration solver is allowed to rotate.
    old_rows = F.normalize(
        classifier.class_weight(old_ids).detach(), p=2, dim=1, eps=eps
    )
    new_rows = F.normalize(
        classifier.class_weight(new_ids), p=2, dim=1, eps=eps
    )
    boundary = old_rows[:, None, :] - new_rows[None, :, :]

    margin, exposure, clearance = _directed_identity_margin_and_relation_exposure(
        source_identity=identity[:old_count],
        source_deformation=variation[:old_count],
        boundary=boundary,
    )

    learned_all = cds.relations.learned_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    physical_all = cds.relations.physical_combined_proximity.to(
        device=identity.device, dtype=identity.dtype
    ).detach()
    learned = learned_all[:old_count, old_count:]
    physical = physical_all[:old_count, old_count:]
    effective = _effective_pair_risk(learned, physical)

    if not all(
        bool(torch.isfinite(value).all())
        for value in (margin, exposure, clearance, learned, physical, effective)
    ):
        raise RuntimeError("historical old/new CDS boundary state became NaN or Inf")

    return old_ids, new_ids, clearance, learned, physical, effective


def _real_current_classifier_state(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_class_ids: Tensor,
    incoming_class_ids: Tensor,
    sample_weights: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Class-balanced real-current all-seen CE, accuracy and true-vs-rival margin."""
    features = torch.as_tensor(current_features)
    labels = torch.as_tensor(
        current_labels, device=features.device, dtype=torch.long
    ).flatten()
    seen_ids = torch.as_tensor(
        seen_class_ids, device=features.device, dtype=torch.long
    ).flatten()
    incoming_ids = torch.as_tensor(
        incoming_class_ids, device=features.device, dtype=torch.long
    ).flatten()

    if labels.ndim != 1 or labels.numel() != features.size(0):
        raise ValueError("current_labels must match current_features batch dimension")
    if not bool(torch.isin(labels, incoming_ids).all()):
        raise ValueError("classifier calibration may use only current/incoming labels")

    output = classifier(
        features,
        class_ids=seen_ids,
        trainable_class_ids=incoming_ids,
    )
    matches = labels.unsqueeze(1).eq(output.class_ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise RuntimeError("a current label is absent from seen classifier IDs")
    targets = matches.to(torch.long).argmax(dim=1)

    per_sample = F.cross_entropy(output.logits, targets, reduction="none")
    if sample_weights is None:
        weights = torch.ones_like(per_sample)
    else:
        weights = torch.as_tensor(
            sample_weights, device=per_sample.device, dtype=per_sample.dtype
        ).flatten()
        if weights.shape != per_sample.shape:
            raise ValueError("sample_weights must be [N]")
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("sample_weights must be finite and positive")
    loss = (weights * per_sample).sum() / weights.sum().clamp_min(
        torch.finfo(per_sample.dtype).eps
    )

    prediction = output.class_ids.index_select(
        0, output.logits.argmax(dim=1)
    )
    accuracy = prediction.eq(labels).to(output.logits.dtype).mean()

    true_logits = output.logits.gather(1, targets[:, None]).squeeze(1)
    rival_logits = output.logits.masked_fill(
        F.one_hot(targets, num_classes=output.logits.size(1)).to(torch.bool),
        -torch.inf,
    ).max(dim=1).values
    margin = (weights * (true_logits - rival_logits)).sum() / weights.sum().clamp_min(
        torch.finfo(weights.dtype).eps
    )

    if not all(bool(torch.isfinite(v)) for v in (loss, accuracy, margin)):
        raise RuntimeError("real-current classifier state became NaN or Inf")
    return loss, accuracy, margin


def _incoming_calibration_state(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_class_ids: Tensor,
    incoming_class_ids: Tensor,
    sample_weights: Tensor | None,
    cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> IncomingClassifierCalibrationState:
    real_loss, real_accuracy, real_margin = _real_current_classifier_state(
        classifier=classifier,
        current_features=current_features,
        current_labels=current_labels,
        seen_class_ids=seen_class_ids,
        incoming_class_ids=incoming_class_ids,
        sample_weights=sample_weights,
    )
    (
        _old_ids,
        _new_ids,
        clearance,
        learned,
        physical,
        effective,
    ) = _historical_old_to_new_boundary_state(
        classifier=classifier,
        cds=cds,
        historical_class_count=historical_class_count,
    )
    negative = F.relu(-clearance)
    weighted_violation = (
        (1.0 + effective) * negative
    ).sum() / (1.0 + effective).sum().clamp_min(
        torch.finfo(clearance.dtype).eps
    )

    return IncomingClassifierCalibrationState(
        real_loss=real_loss,
        real_accuracy=real_accuracy,
        real_margin_mean=real_margin,
        historical_clearance_mean=clearance.mean(),
        historical_clearance_min=clearance.min(),
        historical_violation_fraction=clearance.lt(0.0).to(clearance.dtype).mean(),
        historical_risk_weighted_violation=weighted_violation,
        learned_pair_risk_mean=learned.mean(),
        physical_pair_risk_mean=physical.mean(),
        effective_pair_risk_mean=effective.mean(),
    )


def calibrate_incoming_classifier_with_whole_cds(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    sample_weights: Tensor | None = None,
) -> tuple[
    IncomingClassifierCalibrationState,
    IncomingClassifierCalibrationState,
    list[dict[str, float]],
]:
    """Calibrate only incoming cosine rows from real current HSI.

    The optimization objective is class-balanced real-current all-seen CE.
    Corrected historical Whole-CDS supplies hard pair-wise old->new feasibility
    constraints.  A historically safe pair may not cross clearance zero; a pair
    already unsafe at the warm start may not become worse.

    This is a deterministic numerical calibration, not a second epoch schedule:
    no optimizer state, learning-rate hyperparameter, replay, pseudo-features,
    prototype classifier or second head is introduced.
    """
    ids, _identity, _variation, count, _k, _d = _validate_cds(
        classifier=classifier, cds=seen_cds
    )
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and new seen classes")
    old_ids = ids[:old_count]
    new_ids = ids[old_count:]

    features = torch.as_tensor(
        current_features, device=classifier.weight.device, dtype=classifier.weight.dtype
    ).detach()
    labels = torch.as_tensor(
        current_labels, device=classifier.weight.device, dtype=torch.long
    ).flatten().detach()
    weights = (
        None
        if sample_weights is None
        else torch.as_tensor(
            sample_weights, device=classifier.weight.device, dtype=classifier.weight.dtype
        ).flatten().detach()
    )
    if features.size(0) != labels.numel():
        raise ValueError("current_features/current_labels size mismatch")
    if not bool(torch.isin(labels, new_ids).all()):
        raise ValueError("calibration labels must belong only to incoming classes")

    parameter = classifier.weight
    if not parameter.requires_grad:
        raise RuntimeError("classifier.weight must require gradients during calibration")

    old_before = classifier.class_weight(old_ids).detach().clone()
    scale_before = classifier.logit_scale.detach().clone()

    (
        _baseline_old,
        _baseline_new,
        baseline_clearance,
        _learned,
        _physical,
        effective_risk,
    ) = _historical_old_to_new_boundary_state(
        classifier=classifier,
        cds=seen_cds,
        historical_class_count=old_count,
    )
    # Geometric feasibility: safe pairs retain non-negative robust clearance.
    # Already-unsafe pairs are permitted to improve, but never to deteriorate.
    threshold = torch.minimum(
        baseline_clearance.detach(), torch.zeros_like(baseline_clearance)
    )

    initial = _incoming_calibration_state(
        classifier=classifier,
        current_features=features,
        current_labels=labels,
        seen_class_ids=ids,
        incoming_class_ids=new_ids,
        sample_weights=weights,
        cds=seen_cds,
        historical_class_count=old_count,
    )
    current = initial
    trace: list[dict[str, float]] = []

    eps = torch.finfo(parameter.dtype).eps
    sqrt_eps = float(eps) ** 0.5
    numeric = 128.0 * float(eps)
    max_iterations = max(32, 8 * int(new_ids.numel()))
    max_backtracks = 40

    for iteration in range(max_iterations):
        real_loss, _real_acc, _real_margin = _real_current_classifier_state(
            classifier=classifier,
            current_features=features,
            current_labels=labels,
            seen_class_ids=ids,
            incoming_class_ids=new_ids,
            sample_weights=weights,
        )
        gradient = torch.autograd.grad(
            real_loss, parameter, retain_graph=False, allow_unused=False
        )[0]
        incoming_gradient = gradient.index_select(0, new_ids)
        if not bool(torch.isfinite(incoming_gradient).all()):
            raise RuntimeError("incoming calibration gradient contains NaN or Inf")

        raw_rows = classifier.class_weight(new_ids).detach()
        row_norm = raw_rows.norm(dim=1, keepdim=True).clamp_min(eps)
        unit_rows = raw_rows / row_norm
        tangent = incoming_gradient - (
            incoming_gradient * unit_rows
        ).sum(dim=1, keepdim=True) * unit_rows
        tangent_norm = torch.linalg.vector_norm(tangent)
        if not bool(torch.isfinite(tangent_norm)):
            raise RuntimeError("incoming calibration tangent gradient is invalid")
        if float(tangent_norm.detach().item()) <= sqrt_eps:
            break

        direction = tangent / tangent_norm.clamp_min(eps)
        baseline_loss = float(current.real_loss.detach().item())
        backup = raw_rows.clone()
        accepted = False
        accepted_step = 0.0
        candidate_state = current
        step = 1.0

        for _ in range(max_backtracks):
            candidate_unit = F.normalize(
                unit_rows - step * direction, p=2, dim=1, eps=eps
            )
            candidate_rows = candidate_unit * row_norm
            with torch.no_grad():
                classifier.weight.index_copy_(0, new_ids, candidate_rows)

            (
                _candidate_old,
                _candidate_new,
                candidate_clearance,
                _candidate_learned,
                _candidate_physical,
                _candidate_effective,
            ) = _historical_old_to_new_boundary_state(
                classifier=classifier,
                cds=seen_cds,
                historical_class_count=old_count,
            )
            feasible = bool((candidate_clearance >= threshold - numeric).all())

            candidate = _incoming_calibration_state(
                classifier=classifier,
                current_features=features,
                current_labels=labels,
                seen_class_ids=ids,
                incoming_class_ids=new_ids,
                sample_weights=weights,
                cds=seen_cds,
                historical_class_count=old_count,
            )
            improves_real = (
                float(candidate.real_loss.detach().item())
                < baseline_loss - numeric
            )

            if feasible and improves_real:
                accepted = True
                accepted_step = step
                candidate_state = candidate
                break

            with torch.no_grad():
                classifier.weight.index_copy_(0, new_ids, backup)
            step *= 0.5
            if step <= sqrt_eps:
                break

        if not accepted:
            with torch.no_grad():
                classifier.weight.index_copy_(0, new_ids, backup)
            break

        improvement = baseline_loss - float(candidate_state.real_loss.detach().item())
        current = candidate_state

        (
            _trace_old,
            _trace_new,
            trace_clearance,
            _trace_learned,
            _trace_physical,
            _trace_effective,
        ) = _historical_old_to_new_boundary_state(
            classifier=classifier,
            cds=seen_cds,
            historical_class_count=old_count,
        )
        slack = trace_clearance - threshold
        # Risk-weighted slack is diagnostic evidence of where spectral/learned
        # confusability consumes historical safety. It does not invent a margin.
        risk_weighted_slack = (
            (1.0 + effective_risk) * slack
        ).sum() / (1.0 + effective_risk).sum().clamp_min(eps)

        trace.append(
            {
                "iteration": float(iteration + 1),
                "step": float(accepted_step),
                "real_loss": float(current.real_loss.detach().item()),
                "real_accuracy": float(current.real_accuracy.detach().item()),
                "real_margin_mean": float(current.real_margin_mean.detach().item()),
                "historical_clearance_mean": float(
                    current.historical_clearance_mean.detach().item()
                ),
                "historical_clearance_min": float(
                    current.historical_clearance_min.detach().item()
                ),
                "historical_violation_fraction": float(
                    current.historical_violation_fraction.detach().item()
                ),
                "historical_risk_weighted_violation": float(
                    current.historical_risk_weighted_violation.detach().item()
                ),
                "risk_weighted_feasibility_slack": float(
                    risk_weighted_slack.detach().item()
                ),
            }
        )

        if improvement <= sqrt_eps * max(1.0, abs(baseline_loss)):
            break

    final = _incoming_calibration_state(
        classifier=classifier,
        current_features=features,
        current_labels=labels,
        seen_class_ids=ids,
        incoming_class_ids=new_ids,
        sample_weights=weights,
        cds=seen_cds,
        historical_class_count=old_count,
    )
    (
        _final_old,
        _final_new,
        final_clearance,
        _final_learned,
        _final_physical,
        _final_effective,
    ) = _historical_old_to_new_boundary_state(
        classifier=classifier,
        cds=seen_cds,
        historical_class_count=old_count,
    )

    if float(final.real_loss.detach().item()) > float(initial.real_loss.detach().item()) + numeric:
        raise RuntimeError("incoming calibration increased real-current all-seen CE")
    if not bool((final_clearance >= threshold - numeric).all()):
        raise RuntimeError("incoming calibration violated historical pair-wise CDS safety")
    if not torch.equal(classifier.class_weight(old_ids).detach(), old_before):
        raise RuntimeError("incoming calibration changed a historical classifier row")
    if not torch.equal(classifier.logit_scale.detach(), scale_before):
        raise RuntimeError("incoming calibration changed cosine logit scale")

    return initial, final, trace


def reconstruct_classifier_from_whole_cds(

    *,

    classifier: CosineClassificationHead,

    cds: ClassDiscriminativeStructureState,

    historical_class_count: int,

) -> tuple[WholeCDSClassBalancedObjective, WholeCDSClassBalancedObjective, list[dict[str, float]]]:

    """Deterministically reconstruct the existing seen-class cosine rows from final Whole-CDS.



    This is a numerical projection, not a second training schedule.  The already-trained

    classifier is the warm start.  The final all-seen CDS supplies the *only* objective:

    I_Q/V_Q define robust directed clearance and learned/physical inter-class relations

    weight the dangerous boundaries.  All seen rows are projected together; future rows

    are never referenced or changed.  Backtracking accepts only monotonic improvements,

    so the operation cannot silently worsen the structural reconstruction objective.



    No exemplar, pseudo-feature, prototype-only carrier, extra network, optimizer state,

    user-facing epoch count, or learning-rate hyperparameter is introduced.

    """

    ids, _identity, _variation, count, _k, _d = _validate_cds(

        classifier=classifier, cds=cds

    )

    old_count = int(historical_class_count)

    if old_count <= 0 or old_count >= count:

        raise ValueError("historical_class_count must split old and new seen classes")



    weight = classifier.weight

    if not weight.requires_grad:

        raise RuntimeError("classifier.weight must require gradients during reconstruction")

    eps = torch.finfo(weight.dtype).eps

    sqrt_eps = float(eps) ** 0.5

    seen_before = classifier.class_weight(ids).detach().clone()

    scale_before = classifier.logit_scale.detach().clone()



    initial = whole_cds_class_balanced_classifier_loss(

        classifier=classifier, cds=cds, historical_class_count=old_count

    )

    current = initial

    trace: list[dict[str, float]] = []



    # These are numerical safety bounds, not method hyperparameters.  Convergence is

    # controlled by monotonic objective decrease and machine precision.

    max_iterations = max(16, 4 * count)

    max_backtracks = 32



    for iteration in range(max_iterations):

        gradient = torch.autograd.grad(

            current.loss, classifier.weight, retain_graph=False, allow_unused=False

        )[0]

        seen_gradient = gradient.index_select(0, ids)

        if not bool(torch.isfinite(seen_gradient).all()):

            raise RuntimeError("Whole-CDS reconstruction gradient contains NaN or Inf")



        raw_rows = classifier.class_weight(ids).detach()

        row_norm = raw_rows.norm(dim=1, keepdim=True).clamp_min(eps)

        unit_rows = raw_rows / row_norm



        # The deployed head normalizes W, so only tangent-space rotation matters.

        tangent = seen_gradient - (seen_gradient * unit_rows).sum(dim=1, keepdim=True) * unit_rows

        tangent_norm = torch.linalg.vector_norm(tangent)

        if not bool(torch.isfinite(tangent_norm)):

            raise RuntimeError("Whole-CDS reconstruction tangent gradient is invalid")

        if float(tangent_norm.detach().item()) <= sqrt_eps:

            break

        direction = tangent / tangent_norm.clamp_min(eps)



        baseline_loss = float(current.loss.detach().item())

        baseline_clearance = float(current.old_new_clearance_mean.detach().item())

        baseline_violation = float(current.old_new_violation_fraction.detach().item())

        baseline_all_violation = float(current.violation_fraction.detach().item())

        baseline_old_old_violation = float(current.old_old_violation_fraction.detach().item())

        baseline_new_new_violation = float(current.new_new_violation_fraction.detach().item())

        backup = raw_rows.clone()

        accepted = False

        accepted_step = 0.0

        candidate_objective = current

        step = 1.0



        for _ in range(max_backtracks):

            candidate_unit = F.normalize(unit_rows - step * direction, p=2, dim=1, eps=eps)

            candidate_rows = candidate_unit * row_norm

            with torch.no_grad():

                classifier.weight.index_copy_(0, ids, candidate_rows)



            candidate = whole_cds_class_balanced_classifier_loss(

                classifier=classifier, cds=cds, historical_class_count=old_count

            )

            candidate_loss = float(candidate.loss.detach().item())

            candidate_clearance = float(candidate.old_new_clearance_mean.detach().item())

            candidate_violation = float(candidate.old_new_violation_fraction.detach().item())

            candidate_all_violation = float(candidate.violation_fraction.detach().item())

            candidate_old_old_violation = float(candidate.old_old_violation_fraction.detach().item())

            candidate_new_new_violation = float(candidate.new_new_violation_fraction.detach().item())



            numeric = 64.0 * float(eps)

            improves_loss = candidate_loss < baseline_loss - numeric

            preserves_cross = candidate_clearance >= baseline_clearance - numeric

            preserves_cross_viol = candidate_violation <= baseline_violation + numeric

            preserves_all_viol = candidate_all_violation <= baseline_all_violation + numeric

            preserves_old_old = candidate_old_old_violation <= baseline_old_old_violation + numeric

            preserves_new_new = candidate_new_new_violation <= baseline_new_new_violation + numeric

            if (

                improves_loss

                and preserves_cross

                and preserves_cross_viol

                and preserves_all_viol

                and preserves_old_old

                and preserves_new_new

            ):

                accepted = True

                accepted_step = step

                candidate_objective = candidate

                break



            with torch.no_grad():

                classifier.weight.index_copy_(0, ids, backup)

            step *= 0.5

            if step <= sqrt_eps:

                break



        if not accepted:

            with torch.no_grad():

                classifier.weight.index_copy_(0, ids, backup)

            break



        improvement = baseline_loss - float(candidate_objective.loss.detach().item())

        current = candidate_objective

        trace.append({

            "iteration": float(iteration + 1),

            "step": float(accepted_step),

            "loss": float(current.loss.detach().item()),

            "clearance_mean": float(current.clearance_mean.detach().item()),

            "clearance_min": float(current.clearance_min.detach().item()),

            "violation_fraction": float(current.violation_fraction.detach().item()),

            "old_new_clearance_mean": float(current.old_new_clearance_mean.detach().item()),

            "old_new_violation_fraction": float(current.old_new_violation_fraction.detach().item()),

            "old_old_violation_fraction": float(current.old_old_violation_fraction.detach().item()),

            "new_new_violation_fraction": float(current.new_new_violation_fraction.detach().item()),

        })



        scale = max(1.0, abs(baseline_loss))

        if improvement <= sqrt_eps * scale:

            break



    final = whole_cds_class_balanced_classifier_loss(

        classifier=classifier, cds=cds, historical_class_count=old_count

    )

    if float(final.loss.detach().item()) > float(initial.loss.detach().item()) + 64.0 * float(eps):

        raise RuntimeError("Whole-CDS reconstruction increased its structural objective")

    if float(final.old_new_violation_fraction.detach().item()) > (

        float(initial.old_new_violation_fraction.detach().item()) + 64.0 * float(eps)

    ):

        raise RuntimeError("Whole-CDS reconstruction increased old/new structural violations")

    if not torch.equal(classifier.logit_scale.detach(), scale_before):

        raise RuntimeError("Whole-CDS reconstruction changed cosine logit scale")



    seen_after = classifier.class_weight(ids).detach()

    if not bool(torch.isfinite(seen_after).all()):

        raise RuntimeError("Whole-CDS reconstruction produced invalid classifier rows")

    # A successful reconstruction may leave the warm start unchanged when it is

    # already structurally stationary; this is intentional and not an error.

    _ = seen_before

    return initial, final, trace





class HSICDSDecisionRegionRegulator(nn.Module):

    """Compatibility wrapper around finalized all-seen Whole-CDS boundary loss.



    Previous classifier directions are intentionally *not* used as a penalty:

    they belong to the previous Q coordinate. Historical retention is supplied by

    the corrected historical CDS in the current coordinate, avoiding the stale-W

    coordinate inconsistency that the finalization stage is meant to repair.

    """



    def __init__(self) -> None:

        super().__init__()



    @property

    def parameter_count(self) -> int:

        return 0



    def forward(

        self,

        *,

        classifier: CosineClassificationHead,

        seen_cds: ClassDiscriminativeStructureState,

        historical_class_count: int,

        reference_classifier: CosineClassificationHead | None = None,

        reference_historical_cds: ClassDiscriminativeStructureState | None = None,

        trainable_class_ids: Tensor | None = None,

    ) -> WholeCDSClassifierProtectionObjective:

        del reference_classifier, reference_historical_cds

        old_count = int(historical_class_count)

        objective = whole_cds_class_balanced_classifier_loss(

            classifier=classifier,

            cds=seen_cds,

            historical_class_count=old_count,

            trainable_class_ids=trainable_class_ids,

        )

        (

            ids, margin, exposure, clearance, learned, physical,

            effective, pair_weight, valid,

        ) = _boundary_state(classifier=classifier, cds=seen_cds)

        count = int(ids.numel())

        new_count = count - old_count

        if old_count <= 0 or new_count <= 0:

            raise ValueError("historical_class_count must split old and new seen classes")



        old_rows = slice(0, old_count)

        new_rows = slice(old_count, count)

        cross_old = clearance[old_rows, new_rows]

        cross_new = clearance[new_rows, old_rows]

        cross_margin_old = margin[old_rows, new_rows]

        cross_margin_new = margin[new_rows, old_rows]

        cross_exposure_old = exposure[old_rows, new_rows]

        cross_exposure_new = exposure[new_rows, old_rows]

        old_old_mask = ~torch.eye(old_count, device=clearance.device, dtype=torch.bool)

        old_internal = clearance[old_rows, old_rows][old_old_mask]

        zero = clearance.sum() * 0.0



        old_loss = objective.per_class_loss[:old_count].mean()

        new_loss = objective.per_class_loss[old_count:].mean()

        cross_weight = pair_weight[old_rows, new_rows]

        cross_violation = torch.cat((cross_old.reshape(-1), cross_new.reshape(-1)))

        old_new_violation_mean = (

            cross_weight * F.relu(-cross_old)

        ).sum() / cross_weight.sum().clamp_min(torch.finfo(clearance.dtype).eps)



        return WholeCDSClassifierProtectionObjective(
            loss=objective.loss,
            global_reconstruction_loss=objective.loss,
            old_class_reconstruction_loss=old_loss,
            new_class_reconstruction_loss=new_loss,
            old_old_retention_loss=zero,
            historical_identity_margin_mean=cross_margin_old.mean(),
            historical_identity_margin_min=cross_margin_old.min(),
            historical_deformation_exposure_mean=cross_exposure_old.mean(),
            historical_structural_clearance_mean=cross_old.mean(),
            historical_structural_clearance_min=cross_old.min(),

            current_identity_margin_mean=cross_margin_new.mean(),
            current_identity_margin_min=cross_margin_new.min(),
            current_deformation_exposure_mean=cross_exposure_new.mean(),
            current_structural_clearance_mean=cross_new.mean(),
            current_structural_clearance_min=cross_new.min(),
            historical_internal_clearance_mean=old_internal.mean(),
            historical_internal_clearance_min=old_internal.min(),
            historical_internal_retention_mean=zero,
            historical_internal_retention_min=zero,
            historical_internal_violation_fraction=objective.old_old_violation_fraction,
            joint_structural_clearance_mean=objective.clearance_mean,
            joint_structural_clearance_min=objective.clearance_min,
            unsafe_pair_fraction=objective.violation_fraction,
            pair_weight_mean=pair_weight[valid].mean(),
            learned_pair_risk_mean=learned[valid].mean(),
            physical_pair_risk_mean=physical[valid].mean(),
            effective_pair_risk_mean=effective[valid].mean(),
            old_new_structural_violation_mean=old_new_violation_mean,
            old_new_violation_fraction=cross_violation.lt(0.0).to(clearance.dtype).mean(),
            reference_clearance_mean=zero,
            reference_clearance_min=zero,
            reference_retention_mean=zero,
            reference_retention_min=zero,
            reference_violation_fraction=zero,

        )





__all__ = [

    "WholeCDSClassifierProtectionObjective",
    "WholeCDSClassBalancedObjective",
    "IncomingClassifierCalibrationState",
    "whole_cds_class_balanced_classifier_loss",
    "calibrate_incoming_classifier_with_whole_cds",
    "reconstruct_classifier_from_whole_cds",
    "HSICDSDecisionRegionRegulator",

]












# from __future__ import annotations
# from dataclasses import dataclass

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     combine_learned_physical_relevance,
# )

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class WholeCDSClassifierProtectionObjective:
#     """Compatibility view of the finalized all-seen classifier objective."""

#     loss: Tensor
#     global_reconstruction_loss: Tensor
#     old_class_reconstruction_loss: Tensor
#     new_class_reconstruction_loss: Tensor
#     old_old_retention_loss: Tensor

#     historical_identity_margin_mean: Tensor
#     historical_identity_margin_min: Tensor
#     historical_deformation_exposure_mean: Tensor
#     historical_structural_clearance_mean: Tensor
#     historical_structural_clearance_min: Tensor

#     current_identity_margin_mean: Tensor
#     current_identity_margin_min: Tensor
#     current_deformation_exposure_mean: Tensor
#     current_structural_clearance_mean: Tensor
#     current_structural_clearance_min: Tensor

#     historical_internal_clearance_mean: Tensor
#     historical_internal_clearance_min: Tensor
#     historical_internal_retention_mean: Tensor
#     historical_internal_retention_min: Tensor
#     historical_internal_violation_fraction: Tensor

#     joint_structural_clearance_mean: Tensor
#     joint_structural_clearance_min: Tensor
#     unsafe_pair_fraction: Tensor
#     pair_weight_mean: Tensor
#     learned_pair_risk_mean: Tensor
#     physical_pair_risk_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     old_new_structural_violation_mean: Tensor
#     old_new_violation_fraction: Tensor

#     reference_clearance_mean: Tensor
#     reference_clearance_min: Tensor
#     reference_retention_mean: Tensor
#     reference_retention_min: Tensor
#     reference_violation_fraction: Tensor

#     @property
#     def historical_robust_classification_loss(self) -> Tensor:
#         return self.old_class_reconstruction_loss

#     @property
#     def current_robust_structural_loss(self) -> Tensor:
#         return self.new_class_reconstruction_loss


# @dataclass(frozen=True)
# class WholeCDSClassBalancedObjective:
#     loss: Tensor
#     per_class_loss: Tensor
#     clearance_mean: Tensor
#     clearance_min: Tensor
#     learned_pair_risk_mean: Tensor
#     physical_pair_risk_mean: Tensor
#     effective_pair_risk_mean: Tensor
#     violation_fraction: Tensor
#     old_new_clearance_mean: Tensor
#     old_new_clearance_min: Tensor
#     old_new_violation_fraction: Tensor
#     old_old_clearance_mean: Tensor
#     old_old_clearance_min: Tensor
#     old_old_violation_fraction: Tensor
#     new_new_clearance_mean: Tensor
#     new_new_clearance_min: Tensor
#     new_new_violation_fraction: Tensor


# def _validate_cds(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
# ) -> tuple[Tensor, Tensor, Tensor, int, int, int]:
#     if not isinstance(classifier, CosineClassificationHead):
#         raise TypeError("classifier must be CosineClassificationHead")
#     if not isinstance(cds, ClassDiscriminativeStructureState):
#         raise TypeError("cds must be ClassDiscriminativeStructureState")

#     classes = cds.classes
#     ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     if count < 2 or int(torch.unique(ids).numel()) != count:
#         raise ValueError("CDS classifier objective requires at least two unique classes")

#     identity = classes.learned_identity.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     variation = classes.learned_variation.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or identity.size(0) != count or identity.size(1) < 2:
#         raise ValueError("I_Q must be [C,K,D], K>=2")
#     c, k, d = identity.shape
#     if variation.shape != (c, k, k, d):
#         raise ValueError("V_Q must be [C,K,K,D]")
#     if not bool(torch.isfinite(identity).all()) or not bool(torch.isfinite(variation).all()):
#         raise ValueError("I_Q/V_Q contains NaN or Inf")
#     if bool((variation < 0).any()):
#         raise ValueError("V_Q must be non-negative")
#     if int(classifier.feature_dim) != k * d:
#         raise ValueError("classifier feature dimension does not match Q")
#     return ids, identity, variation, count, k, d


# def _directed_identity_margin_and_relation_exposure(
#     *,
#     source_identity: Tensor,
#     source_deformation: Tensor,
#     boundary: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Robust directed cosine clearance from complete learned CDS.

#     I_Q supplies ordered identity. V_Q supplies diagonal absolute variation and
#     off-diagonal ordered-token relation variation. V_Q is a robust scale field,
#     never a covariance or Gaussian memory.
#     """
#     identity = torch.as_tensor(source_identity)
#     deformation = torch.as_tensor(source_deformation)
#     normal = torch.as_tensor(boundary)
#     if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:
#         raise ValueError("source_identity must be [S,K,D]")
#     s, k, d = identity.shape
#     if deformation.shape != (s, k, k, d):
#         raise ValueError("source_deformation must be [S,K,K,D]")
#     if normal.ndim != 3 or normal.size(0) != s or normal.size(2) != k * d:
#         raise ValueError("boundary must be [S,R,K*D]")
#     if not all(bool(torch.isfinite(v).all()) for v in (identity, deformation, normal)):
#         raise ValueError("boundary-clearance tensors contain NaN or Inf")

#     eps = torch.finfo(identity.dtype).eps
#     flat_identity = identity.reshape(s, -1)
#     identity_norm = flat_identity.norm(dim=1, keepdim=True).clamp_min(eps)
#     identity_unit = flat_identity / identity_norm
#     margin = (normal * identity_unit[:, None, :]).sum(dim=2)

#     tangent = (
#         normal - margin[:, :, None] * identity_unit[:, None, :]
#     ) / identity_norm[:, None, :]
#     tangent_kd = tangent.reshape(s, normal.size(1), k, d)

#     index = torch.arange(k, device=identity.device)
#     absolute_scale = deformation[:, index, index, :]
#     absolute_exposure = (
#         tangent_kd.abs() * absolute_scale[:, None, :, :]
#     ).sum(dim=(2, 3))

#     relation_sensitivity = (
#         tangent_kd[:, :, :, None, :] - tangent_kd[:, :, None, :, :]
#     ).abs()
#     upper = torch.triu(
#         torch.ones(k, k, device=identity.device, dtype=torch.bool), diagonal=1
#     )
#     relational_exposure = (
#         relation_sensitivity
#         * deformation[:, None, :, :, :]
#         * upper.view(1, 1, k, k, 1)
#     ).sum(dim=(2, 3, 4)) / float(k)

#     exposure = torch.maximum(absolute_exposure, relational_exposure)
#     clearance = margin - exposure
#     if not all(bool(torch.isfinite(v).all()) for v in (margin, exposure, clearance)):
#         raise RuntimeError("Whole-CDS boundary clearance became NaN or Inf")
#     return margin, exposure, clearance


# def _effective_pair_risk(learned: Tensor, physical: Tensor) -> Tensor:
#     learned_value = torch.as_tensor(learned)
#     physical_value = torch.as_tensor(
#         physical, device=learned_value.device, dtype=learned_value.dtype
#     )
#     if learned_value.ndim != 2 or learned_value.shape != physical_value.shape:
#         raise ValueError("learned/physical risk must share a rank-two shape")
#     return combine_learned_physical_relevance(learned_value, physical_value)


# def _classwise_robust_loss(
#     *,
#     clearance: Tensor,
#     pair_weight: Tensor,
#     valid_mask: Tensor,
#     scale: Tensor,
# ) -> Tensor:
#     if clearance.shape != pair_weight.shape or clearance.shape != valid_mask.shape:
#         raise ValueError("clearance, pair_weight and valid_mask must share shape")
#     if valid_mask.dtype != torch.bool:
#         raise TypeError("valid_mask must be boolean")
#     eps = torch.finfo(clearance.dtype).eps
#     terms = pair_weight.clamp_min(eps).log() - scale * clearance
#     terms = terms.masked_fill(~valid_mask, -torch.inf)
#     if not bool(valid_mask.any(dim=1).all()):
#         raise RuntimeError("every source class must have at least one rival")
#     zero = torch.zeros((clearance.size(0), 1), device=clearance.device, dtype=clearance.dtype)
#     return torch.logsumexp(torch.cat((zero, terms), dim=1), dim=1)


# def _masked_stats(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
#     if values.shape != mask.shape or mask.dtype != torch.bool:
#         raise ValueError("values/mask mismatch")
#     selected = values[mask]
#     zero = values.sum() * 0.0
#     if selected.numel() == 0:
#         return zero, zero, zero
#     return (
#         selected.mean(),
#         selected.min(),
#         selected.lt(0.0).to(values.dtype).mean(),
#     )


# def _boundary_state(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
#     ids, identity, variation, count, _k, _d = _validate_cds(
#         classifier=classifier, cds=cds
#     )
#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     margin, exposure, clearance = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity,
#         source_deformation=variation,
#         boundary=boundary,
#     )

#     learned = cds.relations.learned_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     physical = cds.relations.physical_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     if learned.shape != (count, count) or physical.shape != (count, count):
#         raise RuntimeError("Whole-CDS relation matrices must be [C,C]")
#     effective = _effective_pair_risk(learned, physical)
#     pair_weight = 1.0 + effective
#     valid = ~torch.eye(count, device=identity.device, dtype=torch.bool)
#     return ids, margin, exposure, clearance, learned, physical, effective, pair_weight, valid


# def whole_cds_class_balanced_classifier_loss(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
#     historical_class_count: int | None = None,
# ) -> WholeCDSClassBalancedObjective:
#     """Finalize the one deployed cosine head from the finalized all-seen Whole-CDS.

#     I_Q and V_Q define robust directed boundary clearance. Learned and physical
#     inter-class CDS relations determine which rival boundaries deserve stronger
#     protection. No CDS score is added to inference logits: deployment remains
#     cosine(Q,W), while Whole-CDS is a training-time boundary constraint.
#     """
#     (
#         ids, _margin, _exposure, clearance, learned, physical,
#         effective, pair_weight, valid,
#     ) = _boundary_state(classifier=classifier, cds=cds)
#     count = int(ids.numel())
#     scale = classifier.logit_scale.to(
#         device=clearance.device, dtype=clearance.dtype
#     ).detach()
#     per_class = _classwise_robust_loss(
#         clearance=clearance,
#         pair_weight=pair_weight,
#         valid_mask=valid,
#         scale=scale,
#     )
#     loss = per_class.mean()

#     if historical_class_count is None:
#         old_count = 0
#     else:
#         old_count = int(historical_class_count)
#         if old_count <= 0 or old_count >= count:
#             raise ValueError("historical_class_count must split old and new seen classes")

#     if old_count > 0:
#         index = torch.arange(count, device=clearance.device)
#         old = index < old_count
#         new = ~old
#         old_new_mask = (old[:, None] & new[None, :]) | (new[:, None] & old[None, :])
#         old_old_mask = old[:, None] & old[None, :] & valid
#         new_new_mask = new[:, None] & new[None, :] & valid
#     else:
#         old_new_mask = torch.zeros_like(valid)
#         old_old_mask = torch.zeros_like(valid)
#         new_new_mask = valid

#     old_new_mean, old_new_min, old_new_violation = _masked_stats(clearance, old_new_mask)
#     old_old_mean, old_old_min, old_old_violation = _masked_stats(clearance, old_old_mask)
#     new_new_mean, new_new_min, new_new_violation = _masked_stats(clearance, new_new_mask)

#     scalars = (
#         loss,
#         clearance[valid].mean(),
#         clearance[valid].min(),
#         learned[valid].mean(),
#         physical[valid].mean(),
#         effective[valid].mean(),
#         clearance[valid].lt(0.0).to(clearance.dtype).mean(),
#         old_new_mean, old_new_min, old_new_violation,
#         old_old_mean, old_old_min, old_old_violation,
#         new_new_mean, new_new_min, new_new_violation,
#     )
#     if not all(bool(torch.isfinite(v)) for v in scalars):
#         raise RuntimeError("Whole-CDS classifier finalization became NaN or Inf")

#     return WholeCDSClassBalancedObjective(
#         loss=loss,
#         per_class_loss=per_class,
#         clearance_mean=clearance[valid].mean(),
#         clearance_min=clearance[valid].min(),
#         learned_pair_risk_mean=learned[valid].mean(),
#         physical_pair_risk_mean=physical[valid].mean(),
#         effective_pair_risk_mean=effective[valid].mean(),
#         violation_fraction=clearance[valid].lt(0.0).to(clearance.dtype).mean(),
#         old_new_clearance_mean=old_new_mean,
#         old_new_clearance_min=old_new_min,
#         old_new_violation_fraction=old_new_violation,
#         old_old_clearance_mean=old_old_mean,
#         old_old_clearance_min=old_old_min,
#         old_old_violation_fraction=old_old_violation,
#         new_new_clearance_mean=new_new_mean,
#         new_new_clearance_min=new_new_min,
#         new_new_violation_fraction=new_new_violation,
#     )


# class HSICDSDecisionRegionRegulator(nn.Module):
#     """Compatibility wrapper around finalized all-seen Whole-CDS boundary loss.

#     Previous classifier directions are intentionally *not* used as a penalty:
#     they belong to the previous Q coordinate. Historical retention is supplied by
#     the corrected historical CDS in the current coordinate, avoiding the stale-W
#     coordinate inconsistency that the finalization stage is meant to repair.
#     """

#     def __init__(self) -> None:
#         super().__init__()

#     @property
#     def parameter_count(self) -> int:
#         return 0

#     def forward(
#         self,
#         *,
#         classifier: CosineClassificationHead,
#         seen_cds: ClassDiscriminativeStructureState,
#         historical_class_count: int,
#         reference_classifier: CosineClassificationHead | None = None,
#         reference_historical_cds: ClassDiscriminativeStructureState | None = None,
#         trainable_class_ids: Tensor | None = None,
#     ) -> WholeCDSClassifierProtectionObjective:
#         del reference_classifier, reference_historical_cds, trainable_class_ids
#         old_count = int(historical_class_count)
#         objective = whole_cds_class_balanced_classifier_loss(
#             classifier=classifier,
#             cds=seen_cds,
#             historical_class_count=old_count,
#         )
#         (
#             ids, margin, exposure, clearance, learned, physical,
#             effective, pair_weight, valid,
#         ) = _boundary_state(classifier=classifier, cds=seen_cds)
#         count = int(ids.numel())
#         new_count = count - old_count
#         if old_count <= 0 or new_count <= 0:
#             raise ValueError("historical_class_count must split old and new seen classes")

#         old_rows = slice(0, old_count)
#         new_rows = slice(old_count, count)
#         cross_old = clearance[old_rows, new_rows]
#         cross_new = clearance[new_rows, old_rows]
#         cross_margin_old = margin[old_rows, new_rows]
#         cross_margin_new = margin[new_rows, old_rows]
#         cross_exposure_old = exposure[old_rows, new_rows]
#         cross_exposure_new = exposure[new_rows, old_rows]
#         old_old_mask = ~torch.eye(old_count, device=clearance.device, dtype=torch.bool)
#         old_internal = clearance[old_rows, old_rows][old_old_mask]
#         zero = clearance.sum() * 0.0

#         old_loss = objective.per_class_loss[:old_count].mean()
#         new_loss = objective.per_class_loss[old_count:].mean()
#         cross_weight = pair_weight[old_rows, new_rows]
#         cross_violation = torch.cat((cross_old.reshape(-1), cross_new.reshape(-1)))
#         old_new_violation_mean = (
#             cross_weight * F.relu(-cross_old)
#         ).sum() / cross_weight.sum().clamp_min(torch.finfo(clearance.dtype).eps)

#         return WholeCDSClassifierProtectionObjective(
#             loss=objective.loss,
#             global_reconstruction_loss=objective.loss,
#             old_class_reconstruction_loss=old_loss,
#             new_class_reconstruction_loss=new_loss,
#             old_old_retention_loss=zero,
#             historical_identity_margin_mean=cross_margin_old.mean(),
#             historical_identity_margin_min=cross_margin_old.min(),
#             historical_deformation_exposure_mean=cross_exposure_old.mean(),
#             historical_structural_clearance_mean=cross_old.mean(),
#             historical_structural_clearance_min=cross_old.min(),
#             current_identity_margin_mean=cross_margin_new.mean(),
#             current_identity_margin_min=cross_margin_new.min(),
#             current_deformation_exposure_mean=cross_exposure_new.mean(),
#             current_structural_clearance_mean=cross_new.mean(),
#             current_structural_clearance_min=cross_new.min(),
#             historical_internal_clearance_mean=old_internal.mean(),
#             historical_internal_clearance_min=old_internal.min(),
#             historical_internal_retention_mean=zero,
#             historical_internal_retention_min=zero,
#             historical_internal_violation_fraction=objective.old_old_violation_fraction,
#             joint_structural_clearance_mean=objective.clearance_mean,
#             joint_structural_clearance_min=objective.clearance_min,
#             unsafe_pair_fraction=objective.violation_fraction,
#             pair_weight_mean=pair_weight[valid].mean(),
#             learned_pair_risk_mean=learned[valid].mean(),
#             physical_pair_risk_mean=physical[valid].mean(),
#             effective_pair_risk_mean=effective[valid].mean(),
#             old_new_structural_violation_mean=old_new_violation_mean,
#             old_new_violation_fraction=cross_violation.lt(0.0).to(clearance.dtype).mean(),
#             reference_clearance_mean=zero,
#             reference_clearance_min=zero,
#             reference_retention_mean=zero,
#             reference_retention_min=zero,
#             reference_violation_fraction=zero,
#         )


# __all__ = [
#     "WholeCDSClassifierProtectionObjective",
#     "WholeCDSClassBalancedObjective",
#     "whole_cds_class_balanced_classifier_loss",
#     "HSICDSDecisionRegionRegulator",
# ]
