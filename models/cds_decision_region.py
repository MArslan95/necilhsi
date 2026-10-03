from __future__ import annotations


from dataclasses import dataclass


import torch

import torch.nn.functional as F


from models.classifier import CosineClassificationHead

from models.geometry_bank import ClassDiscriminativeStructureState

Tensor = torch.Tensor


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
class BalancedDecisionAdmissionState:

    balanced_loss: Tensor

    historical_positive_loss: Tensor

    current_positive_loss: Tensor

    real_new_loss: Tensor

    real_all_seen_accuracy: Tensor

    new_only_accuracy: Tensor
    new_only_loss: Tensor

    historical_per_class_loss: Tensor

    current_per_class_loss: Tensor

    cds_loss: Tensor

    clearance_mean: Tensor

    clearance_min: Tensor

    violation_fraction: Tensor

    old_new_violation_fraction: Tensor

    old_old_violation_fraction: Tensor

    new_new_violation_fraction: Tensor

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

    ids = classes.class_ids.to(
        device=classifier.weight.device, dtype=torch.long
    ).flatten()

    count = int(ids.numel())

    if count < 2 or int(torch.unique(ids).numel()) != count:

        raise ValueError(
            "Whole-CDS decision objective requires at least two unique classes"
        )

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

    if not bool(torch.isfinite(identity).all()) or not bool(
        torch.isfinite(variation).all()
    ):

        raise ValueError("I_Q/V_Q contains NaN or Inf")

    if bool((variation < 0).any()):

        raise ValueError("V_Q must be non-negative")

    if int(classifier.feature_dim) != k * d:

        raise ValueError("classifier feature dimension does not match ordered Q")

    return ids, identity, variation, count, k, d


def _directed_identity_margin_and_relation_exposure(
    *,
    source_identity: Tensor,
    source_deformation: Tensor,
    boundary: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return identity margin plus both Whole-CDS intra-class exposures.



    ``V_Q`` carries two distinct structures: its diagonal stores absolute ordered

    token/channel variation and its off-diagonal entries store ordered token-pair

    relation variation.  They are not collapsed with a hard max.  Stage 3 requires

    the deployed boundary to satisfy both structural views, so both clearances are

    retained and optimized.

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

    # The differential of normalized I_Q along each cosine boundary direction.

    tangent = (normal - margin[:, :, None] * identity_unit[:, None, :]) / identity_norm[
        :, None, :
    ]

    tangent_kd = tangent.reshape(s, normal.size(1), k, d)

    # Diagonal V_Q: absolute ordered token/channel extent.

    index = torch.arange(k, device=identity.device)

    absolute_scale = deformation[:, index, index, :]

    absolute_exposure = (tangent_kd.abs() * absolute_scale[:, None, :, :]).sum(
        dim=(2, 3)
    )

    # Off-diagonal V_Q: ordered relation extent.  Only one orientation of each

    # pair is accumulated because V_Q stores an ordered pair field symmetrically

    # for this exposure calculation.

    relation_sensitivity = (
        tangent_kd[:, :, :, None, :] - tangent_kd[:, :, None, :, :]
    ).abs()

    upper = torch.triu(
        torch.ones(k, k, device=identity.device, dtype=torch.bool), diagonal=1
    )

    relational_exposure = (
        relation_sensitivity * deformation[:, None, :, :, :] * upper.view(1, 1, k, k, 1)
    ).sum(dim=(2, 3, 4)) / float(k)

    absolute_clearance = margin - absolute_exposure

    relational_clearance = margin - relational_exposure

    values = (
        margin,
        absolute_exposure,
        relational_exposure,
        absolute_clearance,
        relational_clearance,
    )

    if not all(bool(torch.isfinite(v).all()) for v in values):

        raise RuntimeError("Whole-CDS boundary clearance became NaN or Inf")

    return values


def _effective_pair_risk(learned: Tensor, physical: Tensor) -> Tensor:
    """Fuse learned inter-class geometry with physical spectral evidence.

    Learned CDS defines semantic rivalry. Physical I_P/V_P may strengthen an
    already supported learned-space rivalry, but physical similarity alone must
    never create a semantic class relation. This keeps Q as the single learned
    decision space while P remains HSI-specific evidence about which learned
    boundaries deserve stronger protection.

    R_eff = R_Q + (1 - R_Q) * R_Q * R_P.

    Thus R_Q == 0 implies R_eff == 0, while strong physical similarity can
    amplify a non-zero learned rivalry without creating a second semantic system.
    """

    learned_value = torch.as_tensor(learned)
    physical_value = torch.as_tensor(
        physical, device=learned_value.device, dtype=learned_value.dtype
    )
    if learned_value.ndim != 2 or learned_value.shape != physical_value.shape:
        raise ValueError("learned/physical risk must share a rank-two shape")
    if not bool(torch.isfinite(learned_value).all()) or not bool(
        torch.isfinite(physical_value).all()
    ):
        raise ValueError("learned/physical risk contains NaN or Inf")

    learned_value = learned_value.clamp(0.0, 1.0)
    physical_value = physical_value.clamp(0.0, 1.0)
    return (
        learned_value + (1.0 - learned_value) * learned_value * physical_value
    ).clamp(0.0, 1.0)


def _classwise_robust_loss(
    *,
    absolute_clearance: Tensor,
    relational_clearance: Tensor,
    pair_weight: Tensor,
    valid_mask: Tensor,
    scale: Tensor,
) -> Tensor:
    """CE-shaped historical positive loss using both intra-class structures.



    Each class/rival pair receives one total pair mass.  That mass is split

    equally between the absolute and ordered-relation constraints, so retaining

    both branches does not double the neutral loss scale.  Log-sum-exp then lets

    the more violated branch dominate naturally without zeroing the other branch's

    gradient via a hard ``max``.

    """

    if absolute_clearance.shape != relational_clearance.shape:

        raise ValueError("absolute/relational clearance shapes disagree")

    if (
        absolute_clearance.shape != pair_weight.shape
        or pair_weight.shape != valid_mask.shape
    ):

        raise ValueError("clearance, pair_weight and valid_mask must share shape")

    if valid_mask.dtype != torch.bool:

        raise TypeError("valid_mask must be boolean")

    if not bool(valid_mask.any(dim=1).all()):

        raise RuntimeError("every source class must have at least one rival")

    eps = torch.finfo(absolute_clearance.dtype).eps

    # Half the pair mass is assigned to each structural view.  If both views are

    # identical, their summed exponential contribution equals the original

    # single-view pair mass exactly.

    log_branch_weight = (0.5 * pair_weight).clamp_min(eps).log()

    absolute_terms = log_branch_weight - scale * absolute_clearance

    relational_terms = log_branch_weight - scale * relational_clearance

    absolute_terms = absolute_terms.masked_fill(~valid_mask, -torch.inf)

    relational_terms = relational_terms.masked_fill(~valid_mask, -torch.inf)

    zero = torch.zeros(
        (absolute_clearance.size(0), 1),
        device=absolute_clearance.device,
        dtype=absolute_clearance.dtype,
    )

    return torch.logsumexp(
        torch.cat((zero, absolute_terms, relational_terms), dim=1), dim=1
    )


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
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
]:

    ids, identity, variation, count, _k, _d = _validate_cds(
        classifier=classifier, cds=cds
    )

    eps = torch.finfo(identity.dtype).eps

    rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)

    boundary = rows[:, None, :] - rows[None, :, :]

    (
        margin,
        absolute_exposure,
        relational_exposure,
        absolute_clearance,
        relational_clearance,
    ) = _directed_identity_margin_and_relation_exposure(
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

    # Diagnostics use the conservative clearance: both structural views must be

    # valid, so a pair is only as safe as its smaller clearance.  The training

    # objective itself retains the two branches separately.

    conservative_clearance = torch.minimum(absolute_clearance, relational_clearance)

    return (
        ids,
        margin,
        absolute_exposure,
        relational_exposure,
        absolute_clearance,
        relational_clearance,
        conservative_clearance,
        learned,
        physical,
        effective,
        pair_weight,
        valid,
    )


def whole_cds_class_balanced_classifier_loss(
    *,
    classifier: CosineClassificationHead,
    cds: ClassDiscriminativeStructureState,
    historical_class_count: int | None = None,
) -> WholeCDSClassBalancedObjective:

    (
        ids,
        _margin,
        _absolute_exposure,
        _relational_exposure,
        absolute_clearance,
        relational_clearance,
        clearance,
        learned,
        physical,
        effective,
        pair_weight,
        valid,
    ) = _boundary_state(classifier=classifier, cds=cds)

    count = int(ids.numel())

    scale = classifier.logit_scale.to(
        device=clearance.device, dtype=clearance.dtype
    ).detach()

    per_class = _classwise_robust_loss(
        absolute_clearance=absolute_clearance,
        relational_clearance=relational_clearance,
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

            raise ValueError(
                "historical_class_count must split old and new seen classes"
            )

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

    old_new_mean, old_new_min, old_new_violation = _masked_stats(
        clearance, old_new_mask
    )

    old_old_mean, old_old_min, old_old_violation = _masked_stats(
        clearance, old_old_mask
    )

    new_new_mean, new_new_min, new_new_violation = _masked_stats(
        clearance, new_new_mask
    )

    scalars = (
        loss,
        clearance[valid].mean(),
        clearance[valid].min(),
        learned[valid].mean(),
        physical[valid].mean(),
        effective[valid].mean(),
        clearance[valid].lt(0.0).to(clearance.dtype).mean(),
        old_new_mean,
        old_new_min,
        old_new_violation,
        old_old_mean,
        old_old_min,
        old_old_violation,
        new_new_mean,
        new_new_min,
        new_new_violation,
    )

    if not all(bool(torch.isfinite(v)) for v in scalars):

        raise RuntimeError("Whole-CDS classifier objective became NaN or Inf")

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


def _targets_local(labels: Tensor, class_ids: Tensor) -> Tensor:
    matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("labels contain a class outside the requested class IDs")
    return matches.to(torch.long).argmax(dim=1)


def _real_new_diagnostics(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_ids: Tensor,
    new_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Read-only deployed-head diagnostics on the real current observations."""
    with torch.no_grad():
        output = classifier(
            current_features,
            class_ids=seen_ids,
            trainable_class_ids=(),
        )
        targets_seen = _targets_local(current_labels, seen_ids)
        real_new_loss = F.cross_entropy(output.logits, targets_seen)
        accuracy = output.prediction.eq(current_labels).to(output.logits.dtype).mean()

        new_only = classifier(
            current_features,
            class_ids=new_ids,
            trainable_class_ids=(),
        )
        targets_new = _targets_local(current_labels, new_ids)
        new_only_loss = F.cross_entropy(new_only.logits, targets_new)
        new_only_accuracy = (
            new_only.prediction.eq(current_labels).to(output.logits.dtype).mean()
        )
    return real_new_loss, accuracy, new_only_loss, new_only_accuracy


def _historical_cross_phase_per_class_loss(
    *,
    classifier: CosineClassificationHead,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[Tensor, Tensor]:
    """Robust positive evidence for historical classes against current rivals only.

    Old-old boundaries are already solved by the previous finalized classifier and are
    not re-optimized in Stage 3.  Whole-CDS supplies the missing historical positive
    side of each old/new boundary through I_Q/V_Q, while I_P/V_P enter through the
    learned/physical rival risk.
    """
    (
        ids,
        _margin,
        _absolute_exposure,
        _relational_exposure,
        absolute_clearance,
        relational_clearance,
        _clearance,
        _learned,
        _physical,
        _effective,
        pair_weight,
        _valid,
    ) = _boundary_state(classifier=classifier, cds=seen_cds)
    old_count = int(historical_class_count)
    count = int(ids.numel())
    if old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and new seen classes")

    old_to_new_abs = absolute_clearance[:old_count, old_count:]
    old_to_new_rel = relational_clearance[:old_count, old_count:]
    old_to_new_weight = pair_weight[:old_count, old_count:]
    valid = torch.ones_like(old_to_new_weight, dtype=torch.bool)
    scale = classifier.logit_scale.to(
        device=old_to_new_abs.device, dtype=old_to_new_abs.dtype
    ).detach()
    per_class = _classwise_robust_loss(
        absolute_clearance=old_to_new_abs,
        relational_clearance=old_to_new_rel,
        pair_weight=old_to_new_weight,
        valid_mask=valid,
        scale=scale,
    )
    return per_class, pair_weight


def _current_cross_phase_per_class_loss(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_ids: Tensor,
    old_count: int,
    pair_weight: Tensor,
) -> Tensor:
    """Class-balanced real-current CE over the deployed all-seen classifier.

    Current samples are the strongest positive evidence available for incoming
    classes.  They therefore train exactly the decision rule used at inference:
    tau*cosine(Q, W) over *all seen classes*.  The loss is averaged within each
    incoming class before the old/new class-balanced Stage-3 mean, so HSI class
    frequency does not determine a class's vote.

    ``pair_weight`` is retained in the private signature for compatibility with
    the historical branch call site, but current real-HSI evidence is not reweighted
    by CDS; CDS supplies the missing historical positive evidence instead.
    """
    del pair_weight
    count = int(seen_ids.numel())
    old_count = int(old_count)
    if old_count <= 0 or old_count >= count:
        raise ValueError("old_count must split seen_ids")
    new_ids = seen_ids[old_count:]
    if current_labels.ndim != 1 or current_features.size(0) != current_labels.numel():
        raise ValueError("current features/labels are misaligned")
    if not bool(torch.isin(current_labels, new_ids).all()):
        raise ValueError("Stage 3 current evidence must contain current labels only")

    output = classifier(
        current_features,
        class_ids=seen_ids,
        trainable_class_ids=new_ids,
    )
    targets_seen = _targets_local(current_labels, seen_ids)
    sample_loss = F.cross_entropy(output.logits, targets_seen, reduction="none")

    per_class = []
    for class_id in new_ids:
        mask = current_labels.eq(class_id)
        if not bool(mask.any()):
            raise RuntimeError("Stage 3 current evidence misses a current class")
        per_class.append(sample_loss[mask].mean())
    result = torch.stack(per_class)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("real-current all-seen CE became NaN or Inf")
    return result


def _balanced_state(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> BalancedDecisionAdmissionState:
    """Balanced cross-phase decision-admission state.

    The final admission rule uses synchronized historical Whole-CDS as the
    exemplar-free positive evidence for old classes and real current HSI with exact
    all-seen CE for incoming classes.  Old classifier rows stay fixed; only incoming
    directions are admitted into the inherited global cosine decision space.
    """
    ids, _identity, _variation, count, _k, _d = _validate_cds(
        classifier=classifier, cds=seen_cds
    )
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and new seen classes")
    new_ids = ids[old_count:]

    historical_per_class, pair_weight = _historical_cross_phase_per_class_loss(
        classifier=classifier,
        seen_cds=seen_cds,
        historical_class_count=old_count,
    )
    current_per_class = _current_cross_phase_per_class_loss(
        classifier=classifier,
        current_features=current_features,
        current_labels=current_labels,
        seen_ids=ids,
        old_count=old_count,
        pair_weight=pair_weight,
    )

    # One vote per class: no old/new loss coefficient and no dependence on class size.
    balanced = torch.cat((historical_per_class, current_per_class), dim=0).mean()
    historical_positive = historical_per_class.mean()
    current_positive = current_per_class.mean()

    real_new_loss, real_accuracy, new_only_loss, new_only_accuracy = (
        _real_new_diagnostics(
            classifier=classifier,
            current_features=current_features,
            current_labels=current_labels,
            seen_ids=ids,
            new_ids=new_ids,
        )
    )
    cds_objective = whole_cds_class_balanced_classifier_loss(
        classifier=classifier,
        cds=seen_cds,
        historical_class_count=old_count,
    )
    if not all(
        bool(torch.isfinite(v))
        for v in (
            balanced,
            historical_positive,
            current_positive,
            real_new_loss,
            new_only_loss,
        )
    ):
        raise RuntimeError("cross-phase decision admission produced NaN or Inf")

    return BalancedDecisionAdmissionState(
        balanced_loss=balanced,
        historical_positive_loss=historical_positive,
        current_positive_loss=current_positive,
        real_new_loss=real_new_loss,
        real_all_seen_accuracy=real_accuracy,
        new_only_accuracy=new_only_accuracy,
        new_only_loss=new_only_loss,
        historical_per_class_loss=historical_per_class,
        current_per_class_loss=current_per_class,
        cds_loss=cds_objective.loss,
        clearance_mean=cds_objective.clearance_mean,
        clearance_min=cds_objective.clearance_min,
        violation_fraction=cds_objective.violation_fraction,
        old_new_violation_fraction=cds_objective.old_new_violation_fraction,
        old_old_violation_fraction=cds_objective.old_old_violation_fraction,
        new_new_violation_fraction=cds_objective.new_new_violation_fraction,
        learned_pair_risk_mean=cds_objective.learned_pair_risk_mean,
        physical_pair_risk_mean=cds_objective.physical_pair_risk_mean,
        effective_pair_risk_mean=cds_objective.effective_pair_risk_mean,
    )


def _new_tangent_gradient(
    *,
    classifier: CosineClassificationHead,
    new_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    gradient = classifier.weight.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("Stage-3 classifier gradient is missing or non-finite")
    eps = torch.finfo(classifier.weight.dtype).eps
    rows = F.normalize(classifier.class_weight(new_ids).detach(), p=2, dim=1, eps=eps)
    new_gradient = gradient.index_select(0, new_ids)
    tangent = new_gradient - (new_gradient * rows).sum(dim=1, keepdim=True) * rows
    if not bool(torch.isfinite(tangent).all()):
        raise RuntimeError("Stage-3 incoming-row tangent gradient became NaN or Inf")
    return rows, tangent


def _gradient_observability(
    *,
    classifier: CosineClassificationHead,
    state: BalancedDecisionAdmissionState,
    old_ids: Tensor,
    new_ids: Tensor,
) -> dict[str, float]:
    """Audit where historical CDS and real-current evidence apply pressure."""

    def group_norm(loss: Tensor, group: Tensor, *, retain_graph: bool) -> float:
        gradient = torch.autograd.grad(
            loss, classifier.weight, retain_graph=retain_graph, allow_unused=False
        )[0]
        rows = gradient.index_select(0, group)
        return float(torch.linalg.vector_norm(rows).detach().item())

    return {
        "historical_cds_to_old_grad_norm": group_norm(
            state.historical_positive_loss, old_ids, retain_graph=True
        ),
        "historical_cds_to_new_grad_norm": group_norm(
            state.historical_positive_loss, new_ids, retain_graph=True
        ),
        "real_new_to_old_grad_norm": group_norm(
            state.current_positive_loss, old_ids, retain_graph=True
        ),
        "real_new_to_new_grad_norm": group_norm(
            state.current_positive_loss, new_ids, retain_graph=False
        ),
    }


def balanced_positive_decision_admission(
    *,
    classifier: CosineClassificationHead,
    current_features: Tensor,
    current_labels: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[
    BalancedDecisionAdmissionState,
    BalancedDecisionAdmissionState,
    list[dict[str, float]],
]:
    ids, _identity, _variation, count, _k, _d = _validate_cds(
        classifier=classifier, cds=seen_cds
    )
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and new seen classes")

    features = torch.as_tensor(
        current_features,
        device=classifier.weight.device,
        dtype=classifier.weight.dtype,
    ).detach()
    labels = (
        torch.as_tensor(
            current_labels,
            device=classifier.weight.device,
            dtype=torch.long,
        )
        .flatten()
        .detach()
    )
    old_ids = ids[:old_count]
    new_ids = ids[old_count:]
    if features.ndim != 3 or features.size(0) != labels.numel():
        raise ValueError("current_features must be [N,K,D] aligned with labels")
    if not bool(torch.isin(labels, new_ids).all()):
        raise ValueError("Stage 3 accepts real current/new labels only")
    if not classifier.weight.requires_grad:
        raise RuntimeError("classifier.weight must require gradients during Stage 3")

    scale_before = classifier.logit_scale_parameter.detach().clone()
    old_before = classifier.class_weight(old_ids).detach().clone()
    all_ids = torch.arange(
        classifier.total_class_count, device=ids.device, dtype=torch.long
    )
    future_ids = all_ids[~torch.isin(all_ids, ids)]
    future_before = (
        classifier.class_weight(future_ids).detach().clone()
        if future_ids.numel() > 0
        else None
    )
    eps = torch.finfo(classifier.weight.dtype).eps

    initial = _balanced_state(
        classifier=classifier,
        current_features=features,
        current_labels=labels,
        seen_cds=seen_cds,
        historical_class_count=old_count,
    )
    observability = _gradient_observability(
        classifier=classifier,
        state=initial,
        old_ids=old_ids,
        new_ids=new_ids,
    )

    dtype_eps = torch.finfo(classifier.weight.dtype).eps
    gradient_tolerance = max(1.0e-7, 32.0 * dtype_eps)
    relative_tolerance = max(1.0e-8, 64.0 * dtype_eps)
    armijo_constant = 1.0e-4
    backtrack_factor = 0.5
    max_backtracks = 28
    max_iterations = 256

    trace: list[dict[str, float]] = []
    best_value = float(initial.balanced_loss.detach().item())
    best_rows = classifier.class_weight(new_ids).detach().clone()
    step_guess = 1.0

    for iteration in range(max_iterations):
        classifier.weight.grad = None
        classifier.logit_scale_parameter.grad = None
        state = _balanced_state(
            classifier=classifier,
            current_features=features,
            current_labels=labels,
            seen_cds=seen_cds,
            historical_class_count=old_count,
        )
        state.balanced_loss.backward()

        current_rows, tangent_gradient = _new_tangent_gradient(
            classifier=classifier, new_ids=new_ids
        )
        grad_norm = float(torch.linalg.vector_norm(tangent_gradient).detach().item())
        current_value = float(state.balanced_loss.detach().item())
        if grad_norm <= gradient_tolerance:
            break

        grad_sq = float(tangent_gradient.square().sum().detach().item())
        accepted = False
        accepted_step = 0.0
        candidate = state
        candidate_value = current_value
        step_size = step_guess

        for _ in range(max_backtracks):
            with torch.no_grad():
                candidate_rows = F.normalize(
                    current_rows - step_size * tangent_gradient,
                    p=2,
                    dim=1,
                    eps=eps,
                )
                classifier.weight.index_copy_(0, new_ids, candidate_rows)
                candidate = _balanced_state(
                    classifier=classifier,
                    current_features=features,
                    current_labels=labels,
                    seen_cds=seen_cds,
                    historical_class_count=old_count,
                )
                candidate_value = float(candidate.balanced_loss.item())

            required = current_value - armijo_constant * step_size * grad_sq
            if candidate_value <= required:
                accepted = True
                accepted_step = step_size
                break

            with torch.no_grad():
                classifier.weight.index_copy_(0, new_ids, current_rows)
            step_size *= backtrack_factor
            if step_size <= dtype_eps:
                break

        if not accepted:
            with torch.no_grad():
                classifier.weight.index_copy_(0, new_ids, current_rows)
            break

        if candidate_value < best_value:
            best_value = candidate_value
            best_rows = classifier.class_weight(new_ids).detach().clone()

        relative_change = abs(current_value - candidate_value) / max(
            1.0, abs(current_value)
        )
        trace.append(
            {
                "iteration": float(iteration + 1),
                "balanced_loss": candidate_value,
                "historical_positive_loss": float(
                    candidate.historical_positive_loss.detach().item()
                ),
                "current_positive_loss": float(
                    candidate.current_positive_loss.detach().item()
                ),
                "real_new_loss": float(candidate.real_new_loss.detach().item()),
                "real_all_seen_accuracy": float(
                    candidate.real_all_seen_accuracy.detach().item()
                ),
                "new_only_loss": float(candidate.new_only_loss.detach().item()),
                "new_only_accuracy": float(candidate.new_only_accuracy.detach().item()),
                "old_new_violation_fraction": float(
                    candidate.old_new_violation_fraction.detach().item()
                ),
                "tangent_grad_norm": grad_norm,
                "accepted_step": accepted_step,
                "relative_loss_change": relative_change,
                **(observability if iteration == 0 else {}),
            }
        )

        step_guess = min(1.0, accepted_step / backtrack_factor)
        if relative_change <= relative_tolerance:
            break

    with torch.no_grad():
        classifier.weight.index_copy_(0, new_ids, best_rows)

    final = _balanced_state(
        classifier=classifier,
        current_features=features,
        current_labels=labels,
        seen_cds=seen_cds,
        historical_class_count=old_count,
    )
    if (
        float(final.balanced_loss.detach().item())
        > float(initial.balanced_loss.detach().item()) + 64.0 * dtype_eps
    ):
        raise RuntimeError("Stage-3 cross-phase admission objective increased")
    if not torch.equal(classifier.class_weight(old_ids).detach(), old_before):
        raise RuntimeError("Stage 3 changed finalized historical classifier rows")
    if future_before is not None and not torch.equal(
        classifier.class_weight(future_ids).detach(), future_before
    ):
        raise RuntimeError("Stage 3 changed future classifier rows")
    if not torch.equal(classifier.logit_scale_parameter.detach(), scale_before):
        raise RuntimeError("Stage 3 changed the global cosine logit scale")

    with torch.no_grad():
        final_norms = classifier.class_weight(new_ids).norm(dim=1)
        if not torch.allclose(
            final_norms, torch.ones_like(final_norms), rtol=0.0, atol=64.0 * dtype_eps
        ):
            raise RuntimeError("Stage 3 left incoming cosine rows off the unit sphere")

    if not trace:
        trace.append(
            {
                "iteration": 0.0,
                "balanced_loss": float(final.balanced_loss.detach().item()),
                "historical_positive_loss": float(
                    final.historical_positive_loss.detach().item()
                ),
                "current_positive_loss": float(
                    final.current_positive_loss.detach().item()
                ),
                "real_new_loss": float(final.real_new_loss.detach().item()),
                "real_all_seen_accuracy": float(
                    final.real_all_seen_accuracy.detach().item()
                ),
                "new_only_loss": float(final.new_only_loss.detach().item()),
                "new_only_accuracy": float(final.new_only_accuracy.detach().item()),
                "old_new_violation_fraction": float(
                    final.old_new_violation_fraction.detach().item()
                ),
                "tangent_grad_norm": 0.0,
                "accepted_step": 0.0,
                "relative_loss_change": 0.0,
                **observability,
            }
        )

    return initial, final, trace


__all__ = [
    "WholeCDSClassBalancedObjective",
    "BalancedDecisionAdmissionState",
    "whole_cds_class_balanced_classifier_loss",
    "balanced_positive_decision_admission",
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
