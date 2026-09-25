from __future__ import annotations
from dataclasses import dataclass
import math

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


def _validate_seen_cds(
    *,
    classifier: CosineClassificationHead,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> tuple[Tensor, Tensor, Tensor, int, int, int, int]:
    if not isinstance(classifier, CosineClassificationHead):
        raise TypeError("classifier must be CosineClassificationHead")
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

    classes = seen_cds.classes
    ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
    count = int(ids.numel())
    old_count = int(historical_class_count)
    if count < 2 or old_count <= 0 or old_count >= count:
        raise ValueError("historical_class_count must split old and new seen classes")
    if int(torch.unique(ids).numel()) != count:
        raise ValueError("seen class IDs must be unique")

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
    return ids, identity, variation, old_count, count, k, d


def _historical_cds(
    *,
    reference_classifier: CosineClassificationHead,
    reference_historical_cds: ClassDiscriminativeStructureState,
    expected_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(reference_classifier, CosineClassificationHead):
        raise TypeError("reference_classifier must be CosineClassificationHead")
    if not isinstance(reference_historical_cds, ClassDiscriminativeStructureState):
        raise TypeError("reference_historical_cds must be ClassDiscriminativeStructureState")
    ids = reference_historical_cds.classes.class_ids.to(
        device=reference_classifier.weight.device, dtype=torch.long
    ).flatten()
    expected = expected_ids.to(device=ids.device, dtype=torch.long)
    if not torch.equal(ids, expected):
        raise ValueError("reference historical CDS IDs disagree with historical prefix")
    identity = reference_historical_cds.classes.learned_identity.to(
        device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
    ).detach()
    variation = reference_historical_cds.classes.learned_variation.to(
        device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
    ).detach()
    if identity.ndim != 3 or variation.shape != (
        identity.size(0), identity.size(1), identity.size(1), identity.size(2)
    ):
        raise ValueError("reference I_Q/V_Q shape mismatch")
    return ids, identity, variation


def _directed_identity_margin_and_relation_exposure(
    *,
    source_identity: Tensor,
    source_deformation: Tensor,
    boundary: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Robust directed cosine clearance from complete learned CDS.

    ``I_Q`` supplies the ordered identity. ``V_Q`` supplies both diagonal
    token/channel variation and off-diagonal ordered-token relation variation.
    It is never interpreted as covariance or Gaussian memory.
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
        raise RuntimeError("whole-CDS boundary clearance became NaN or Inf")
    return margin, exposure, clearance


@torch.no_grad()
def _previous_historical_clearance(
    *,
    reference_classifier: CosineClassificationHead,
    reference_historical_cds: ClassDiscriminativeStructureState,
    historical_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    ids, identity, variation = _historical_cds(
        reference_classifier=reference_classifier,
        reference_historical_cds=reference_historical_cds,
        expected_ids=historical_ids,
    )
    if ids.numel() < 2:
        raise ValueError("reference historical CDS needs at least two classes")
    eps = torch.finfo(identity.dtype).eps
    rows = F.normalize(reference_classifier.class_weight(ids), p=2, dim=1, eps=eps)
    boundary = rows[:, None, :] - rows[None, :, :]
    _, _, clearance = _directed_identity_margin_and_relation_exposure(
        source_identity=identity,
        source_deformation=variation,
        boundary=boundary,
    )
    diagonal = torch.eye(ids.numel(), device=clearance.device, dtype=torch.bool)
    floor = clearance.masked_fill(diagonal, torch.inf).min(dim=1).values
    if not bool(torch.isfinite(clearance).all()) or not bool(torch.isfinite(floor).all()):
        raise RuntimeError("previous historical clearance became invalid")
    return clearance.detach(), floor.detach()


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


def whole_cds_class_balanced_classifier_loss(
    *,
    classifier: CosineClassificationHead,
    cds: ClassDiscriminativeStructureState,
) -> WholeCDSClassBalancedObjective:
    """Symmetric class-balanced Whole-CDS supervision for one cosine head."""
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
    if identity.ndim != 3 or variation.shape != (
        count, identity.size(1), identity.size(1), identity.size(2)
    ):
        raise ValueError("I_Q/V_Q shape mismatch")

    eps = torch.finfo(identity.dtype).eps
    rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
    boundary = rows[:, None, :] - rows[None, :, :]
    _, _, clearance = _directed_identity_margin_and_relation_exposure(
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
    effective = _effective_pair_risk(learned, physical)
    pair_weight = 1.0 + effective
    valid = ~torch.eye(count, device=identity.device, dtype=torch.bool)
    scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
    per_class = _classwise_robust_loss(
        clearance=clearance,
        pair_weight=pair_weight,
        valid_mask=valid,
        scale=scale,
    )

    # Class-balanced CDS carriers supervise EXACTLY the deployed cosine score.
    # Whole-CDS still controls robust clearance/pair risk through I_Q/V_Q/I_P/V_P,
    # but we never add a CDS-only score to classifier logits because inference is
    # cosine(Q,W) only.  Training and deployment therefore use the same decision
    # function.
    output = classifier(
        identity,
        class_ids=ids,
        trainable_class_ids=ids,
    )
    local_targets = torch.arange(count, device=identity.device, dtype=torch.long)
    whole_cds_ce = F.cross_entropy(output.logits, local_targets)
    loss = per_class.mean() + whole_cds_ce
    scalars = (
        loss, whole_cds_ce,
        clearance[valid].mean(),
        clearance[valid].min(),
        learned[valid].mean(),
        physical[valid].mean(),
        effective[valid].mean(),
    )
    if not all(bool(torch.isfinite(v)) for v in scalars):
        raise RuntimeError("whole-CDS class-balanced objective became NaN or Inf")
    return WholeCDSClassBalancedObjective(
        loss=loss,
        per_class_loss=per_class,
        clearance_mean=clearance[valid].mean(),
        clearance_min=clearance[valid].min(),
        learned_pair_risk_mean=learned[valid].mean(),
        physical_pair_risk_mean=physical[valid].mean(),
        effective_pair_risk_mean=effective[valid].mean(),
    )


class HSICDSDecisionRegionRegulator(nn.Module):
    """Symmetric all-seen classifier protection from the complete Whole-CDS.

    This module has exactly one role: protect the deployed cosine decision
    geometry.  It never initializes rows and never performs a separate optimizer
    stage.  In incremental phases historical classifier rows are immutable anchors;
    only incoming rows are optimized against the complete all-seen Whole-CDS.
    Historical old-old clearance is retained by construction and reported as a
    diagnostic relative to the finalized previous classifier.
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
        ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
            classifier=classifier,
            seen_cds=seen_cds,
            historical_class_count=historical_class_count,
        )
        eps = torch.finfo(identity.dtype).eps

        # Historical classifier rows are finalized knowledge.  Current-phase data
        # has no historical positives, so those rows are anchors, never trainable
        # variables.  Only explicitly owned rows may move.
        raw_rows = classifier.class_weight(ids)
        if trainable_class_ids is None:
            trainable_mask = torch.ones(count, device=ids.device, dtype=torch.bool)
            trainable_ids = ids
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
                raise ValueError("trainable_class_ids must be contained in seen CDS IDs")
            trainable_mask = ids.unsqueeze(1).eq(trainable_ids.unsqueeze(0)).any(dim=1)

        owned_rows = torch.where(
            trainable_mask.unsqueeze(1), raw_rows, raw_rows.detach()
        )
        rows = F.normalize(owned_rows, p=2, dim=1, eps=eps)
        boundary = rows[:, None, :] - rows[None, :, :]
        margin_all, exposure_all, clearance_all = (
            _directed_identity_margin_and_relation_exposure(
                source_identity=identity,
                source_deformation=variation,
                boundary=boundary,
            )
        )

        learned_all = seen_cds.relations.learned_combined_proximity.to(
            device=identity.device, dtype=identity.dtype
        ).detach()
        physical_all = seen_cds.relations.physical_combined_proximity.to(
            device=identity.device, dtype=identity.dtype
        ).detach()
        if learned_all.shape != (count, count) or physical_all.shape != (count, count):
            raise RuntimeError("whole-CDS relation matrices must be [C,C]")
        effective_all = _effective_pair_risk(learned_all, physical_all)
        pair_weight_all = 1.0 + effective_all

        scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
        valid_all = ~torch.eye(count, device=identity.device, dtype=torch.bool)
        # Old-old geometry is guaranteed by frozen historical classifier rows.
        # Do not let constant old-old terms dominate the trainable cross-phase
        # objective; optimize only boundaries containing at least one owned row.
        optimized_pairs = valid_all & (
            trainable_mask.unsqueeze(1) | trainable_mask.unsqueeze(0)
        )
        per_class = _classwise_robust_loss(
            clearance=clearance_all,
            pair_weight=pair_weight_all,
            valid_mask=optimized_pairs,
            scale=scale,
        )
        global_reconstruction_loss = per_class.mean()
        old_reconstruction_loss = per_class[:old_count].mean()
        new_reconstruction_loss = per_class[old_count:].mean()

        old_internal_clearance = clearance_all[:old_count, :old_count]
        old_internal_mask = ~torch.eye(
            old_count, device=identity.device, dtype=torch.bool
        )
        if not bool(old_internal_mask.any()):
            raise ValueError("historical classifier protection needs at least two old classes")

        if reference_classifier is not None or reference_historical_cds is not None:
            if reference_classifier is None or reference_historical_cds is None:
                raise ValueError(
                    "reference classifier and historical CDS must be supplied together"
                )
            reference_internal, reference_floor = _previous_historical_clearance(
                reference_classifier=reference_classifier,
                reference_historical_cds=reference_historical_cds,
                historical_ids=ids[:old_count],
            )
            reference_internal = reference_internal.to(
                device=identity.device, dtype=identity.dtype
            )
            reference_floor = reference_floor.to(
                device=identity.device, dtype=identity.dtype
            )
            internal_retention = old_internal_clearance - reference_internal
        else:
            reference_internal = identity.new_zeros((old_count, old_count))
            reference_floor = identity.new_zeros((old_count,))
            internal_retention = old_internal_clearance

        internal_violation = F.relu(-internal_retention)[old_internal_mask]
        old_old_retention_loss = (
            (
                torch.logsumexp(scale * internal_violation, dim=0)
                - math.log(float(internal_violation.numel()))
            ).clamp_min(0.0)
            if internal_violation.numel()
            else old_internal_clearance.sum() * 0.0
        )
        old_to_new = clearance_all[:old_count, old_count:]
        new_to_old = clearance_all[old_count:, :old_count].transpose(0, 1).contiguous()
        joint_clearance = torch.minimum(old_to_new, new_to_old)
        cross_learned = learned_all[:old_count, old_count:]
        cross_physical = physical_all[:old_count, old_count:]
        cross_effective = effective_all[:old_count, old_count:]
        cross_weight = 1.0 + cross_effective
        old_new_violation = F.relu(-joint_clearance)
        old_new_violation_mean = (
            (cross_weight * old_new_violation).sum()
            / cross_weight.sum().clamp_min(eps)
        )
        old_new_unsafe = joint_clearance.lt(0.0).to(identity.dtype).mean()

        # Class-balanced CDS carriers supervise the SAME cosine logits used at
        # inference.  Whole-CDS influences the carrier geometry, robust variation
        # clearance, and learned+physical pair risk, but never creates a second
        # hidden decision score.  Historical W rows remain detached anchors.
        carrier_output = classifier(
            identity,
            class_ids=ids,
            trainable_class_ids=trainable_ids,
        )
        local_targets = torch.arange(count, device=identity.device, dtype=torch.long)
        whole_cds_ce = F.cross_entropy(carrier_output.logits, local_targets)

        # Old-old classifier geometry needs no optimization because historical W
        # rows are immutable.  The active regulator therefore targets only the
        # boundaries that can actually change: old/new and new/new.
        loss = (
            global_reconstruction_loss
            + old_new_violation_mean
            + whole_cds_ce
        )

        historical_margin = margin_all[:old_count, old_count:]
        historical_exposure = exposure_all[:old_count, old_count:]
        current_margin = margin_all[old_count:, :old_count].transpose(0, 1).contiguous()
        current_exposure = exposure_all[old_count:, :old_count].transpose(0, 1).contiguous()
        current_clearance = clearance_all[old_count:, :old_count].transpose(0, 1).contiguous()

        reference_retention = old_to_new - reference_floor[:, None]
        reference_violation = reference_retention.lt(0.0).to(identity.dtype).mean()
        internal_values = old_internal_clearance[old_internal_mask]
        retention_values = internal_retention[old_internal_mask]
        internal_unsafe = retention_values.lt(0.0).to(identity.dtype).mean()

        scalars = (
            loss, whole_cds_ce,
            global_reconstruction_loss,
            old_reconstruction_loss,
            new_reconstruction_loss,
            old_old_retention_loss,
            historical_margin.mean(), historical_margin.min(),
            historical_exposure.mean(), old_to_new.mean(), old_to_new.min(),
            current_margin.mean(), current_margin.min(), current_exposure.mean(),
            current_clearance.mean(), current_clearance.min(),
            internal_values.mean(), internal_values.min(),
            retention_values.mean(), retention_values.min(), internal_unsafe,
            joint_clearance.mean(), joint_clearance.min(), cross_weight.mean(),
            cross_learned.mean(), cross_physical.mean(), cross_effective.mean(),
            old_new_violation_mean, old_new_unsafe,
            reference_floor.mean(), reference_floor.min(),
            reference_retention.mean(), reference_retention.min(), reference_violation,
        )
        if not all(bool(torch.isfinite(value)) for value in scalars):
            raise RuntimeError("whole-CDS classifier protection became NaN or Inf")

        return WholeCDSClassifierProtectionObjective(
            loss=loss,
            global_reconstruction_loss=global_reconstruction_loss,
            old_class_reconstruction_loss=old_reconstruction_loss,
            new_class_reconstruction_loss=new_reconstruction_loss,
            old_old_retention_loss=old_old_retention_loss,
            historical_identity_margin_mean=historical_margin.mean(),
            historical_identity_margin_min=historical_margin.min(),
            historical_deformation_exposure_mean=historical_exposure.mean(),
            historical_structural_clearance_mean=old_to_new.mean(),
            historical_structural_clearance_min=old_to_new.min(),
            current_identity_margin_mean=current_margin.mean(),
            current_identity_margin_min=current_margin.min(),
            current_deformation_exposure_mean=current_exposure.mean(),
            current_structural_clearance_mean=current_clearance.mean(),
            current_structural_clearance_min=current_clearance.min(),
            historical_internal_clearance_mean=internal_values.mean(),
            historical_internal_clearance_min=internal_values.min(),
            historical_internal_retention_mean=retention_values.mean(),
            historical_internal_retention_min=retention_values.min(),
            historical_internal_violation_fraction=internal_unsafe,
            joint_structural_clearance_mean=joint_clearance.mean(),
            joint_structural_clearance_min=joint_clearance.min(),
            unsafe_pair_fraction=old_new_unsafe,
            pair_weight_mean=cross_weight.mean(),
            learned_pair_risk_mean=cross_learned.mean(),
            physical_pair_risk_mean=cross_physical.mean(),
            effective_pair_risk_mean=cross_effective.mean(),
            old_new_structural_violation_mean=old_new_violation_mean,
            old_new_violation_fraction=old_new_unsafe,
            reference_clearance_mean=reference_floor.mean(),
            reference_clearance_min=reference_floor.min(),
            reference_retention_mean=reference_retention.mean(),
            reference_retention_min=reference_retention.min(),
            reference_violation_fraction=reference_violation,
        )


__all__ = [
    "WholeCDSClassifierProtectionObjective",
    "WholeCDSClassBalancedObjective",
    "whole_cds_class_balanced_classifier_loss",
    "HSICDSDecisionRegionRegulator",
]









# from __future__ import annotations
# from dataclasses import dataclass
# import math

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


# def _validate_seen_cds(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> tuple[Tensor, Tensor, Tensor, int, int, int, int]:
#     if not isinstance(classifier, CosineClassificationHead):
#         raise TypeError("classifier must be CosineClassificationHead")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and new seen classes")
#     if int(torch.unique(ids).numel()) != count:
#         raise ValueError("seen class IDs must be unique")

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
#     return ids, identity, variation, old_count, count, k, d


# def _historical_cds(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     expected_ids: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     if not isinstance(reference_classifier, CosineClassificationHead):
#         raise TypeError("reference_classifier must be CosineClassificationHead")
#     if not isinstance(reference_historical_cds, ClassDiscriminativeStructureState):
#         raise TypeError("reference_historical_cds must be ClassDiscriminativeStructureState")
#     ids = reference_historical_cds.classes.class_ids.to(
#         device=reference_classifier.weight.device, dtype=torch.long
#     ).flatten()
#     expected = expected_ids.to(device=ids.device, dtype=torch.long)
#     if not torch.equal(ids, expected):
#         raise ValueError("reference historical CDS IDs disagree with historical prefix")
#     identity = reference_historical_cds.classes.learned_identity.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     variation = reference_historical_cds.classes.learned_variation.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or variation.shape != (
#         identity.size(0), identity.size(1), identity.size(1), identity.size(2)
#     ):
#         raise ValueError("reference I_Q/V_Q shape mismatch")
#     return ids, identity, variation


# def _directed_identity_margin_and_relation_exposure(
#     *,
#     source_identity: Tensor,
#     source_deformation: Tensor,
#     boundary: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Robust directed cosine clearance from complete learned CDS.

#     ``I_Q`` supplies the ordered identity. ``V_Q`` supplies both diagonal
#     token/channel variation and off-diagonal ordered-token relation variation.
#     It is never interpreted as covariance or Gaussian memory.
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
#         raise RuntimeError("whole-CDS boundary clearance became NaN or Inf")
#     return margin, exposure, clearance


# @torch.no_grad()
# def _previous_historical_clearance(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     historical_ids: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     ids, identity, variation = _historical_cds(
#         reference_classifier=reference_classifier,
#         reference_historical_cds=reference_historical_cds,
#         expected_ids=historical_ids,
#     )
#     if ids.numel() < 2:
#         raise ValueError("reference historical CDS needs at least two classes")
#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(reference_classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     _, _, clearance = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity,
#         source_deformation=variation,
#         boundary=boundary,
#     )
#     diagonal = torch.eye(ids.numel(), device=clearance.device, dtype=torch.bool)
#     floor = clearance.masked_fill(diagonal, torch.inf).min(dim=1).values
#     if not bool(torch.isfinite(clearance).all()) or not bool(torch.isfinite(floor).all()):
#         raise RuntimeError("previous historical clearance became invalid")
#     return clearance.detach(), floor.detach()


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


# def whole_cds_class_balanced_classifier_loss(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
# ) -> WholeCDSClassBalancedObjective:
#     """Symmetric class-balanced Whole-CDS supervision for one cosine head."""
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
#     if identity.ndim != 3 or variation.shape != (
#         count, identity.size(1), identity.size(1), identity.size(2)
#     ):
#         raise ValueError("I_Q/V_Q shape mismatch")

#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     _, _, clearance = _directed_identity_margin_and_relation_exposure(
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
#     effective = _effective_pair_risk(learned, physical)
#     pair_weight = 1.0 + effective
#     valid = ~torch.eye(count, device=identity.device, dtype=torch.bool)
#     scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#     per_class = _classwise_robust_loss(
#         clearance=clearance,
#         pair_weight=pair_weight,
#         valid_mask=valid,
#         scale=scale,
#     )

#     # Direct Whole-CDS classifier supervision.  Native spectral I_P/V_P and
#     # learned I_Q/V_Q are inside the optimized class score rather than only
#     # pair-risk weights.  Inference remains the same cosine head.
#     physical_identity = classes.physical_identity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()
#     output = classifier(
#         identity,
#         class_ids=ids,
#         trainable_class_ids=ids,
#         physical_spectral_evidence=physical_identity,
#         cds=cds,
#     )
#     local_targets = torch.arange(count, device=identity.device, dtype=torch.long)
#     whole_logits = output.logits + output.whole_support.detach().clamp_min(eps).log()
#     whole_cds_ce = F.cross_entropy(whole_logits, local_targets)
#     loss = per_class.mean() + whole_cds_ce
#     scalars = (
#         loss, whole_cds_ce,
#         clearance[valid].mean(),
#         clearance[valid].min(),
#         learned[valid].mean(),
#         physical[valid].mean(),
#         effective[valid].mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in scalars):
#         raise RuntimeError("whole-CDS class-balanced objective became NaN or Inf")
#     return WholeCDSClassBalancedObjective(
#         loss=loss,
#         per_class_loss=per_class,
#         clearance_mean=clearance[valid].mean(),
#         clearance_min=clearance[valid].min(),
#         learned_pair_risk_mean=learned[valid].mean(),
#         physical_pair_risk_mean=physical[valid].mean(),
#         effective_pair_risk_mean=effective[valid].mean(),
#     )


# class HSICDSDecisionRegionRegulator(nn.Module):
#     """Symmetric all-seen classifier protection from the complete Whole-CDS.

#     This module has exactly one role: protect the deployed cosine decision
#     geometry.  It never initializes rows, never performs a separate optimizer
#     stage, and never applies old/new-specific detach rules.  Every seen class is
#     a live source and a live rival.  Historical old-old robust clearance is also
#     retained relative to the finalized previous classifier.
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
#     ) -> WholeCDSClassifierProtectionObjective:
#         ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
#             classifier=classifier,
#             seen_cds=seen_cds,
#             historical_class_count=historical_class_count,
#         )
#         eps = torch.finfo(identity.dtype).eps
#         rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
#         boundary = rows[:, None, :] - rows[None, :, :]
#         margin_all, exposure_all, clearance_all = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity,
#                 source_deformation=variation,
#                 boundary=boundary,
#             )
#         )

#         learned_all = seen_cds.relations.learned_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         physical_all = seen_cds.relations.physical_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         if learned_all.shape != (count, count) or physical_all.shape != (count, count):
#             raise RuntimeError("whole-CDS relation matrices must be [C,C]")
#         effective_all = _effective_pair_risk(learned_all, physical_all)
#         pair_weight_all = 1.0 + effective_all

#         scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#         valid_all = ~torch.eye(count, device=identity.device, dtype=torch.bool)
#         per_class = _classwise_robust_loss(
#             clearance=clearance_all,
#             pair_weight=pair_weight_all,
#             valid_mask=valid_all,
#             scale=scale,
#         )
#         global_reconstruction_loss = per_class.mean()
#         old_reconstruction_loss = per_class[:old_count].mean()
#         new_reconstruction_loss = per_class[old_count:].mean()

#         old_internal_clearance = clearance_all[:old_count, :old_count]
#         old_internal_mask = ~torch.eye(
#             old_count, device=identity.device, dtype=torch.bool
#         )
#         if not bool(old_internal_mask.any()):
#             raise ValueError("historical classifier protection needs at least two old classes")

#         if reference_classifier is not None or reference_historical_cds is not None:
#             if reference_classifier is None or reference_historical_cds is None:
#                 raise ValueError(
#                     "reference classifier and historical CDS must be supplied together"
#                 )
#             reference_internal, reference_floor = _previous_historical_clearance(
#                 reference_classifier=reference_classifier,
#                 reference_historical_cds=reference_historical_cds,
#                 historical_ids=ids[:old_count],
#             )
#             reference_internal = reference_internal.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             reference_floor = reference_floor.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             internal_retention = old_internal_clearance - reference_internal
#         else:
#             reference_internal = identity.new_zeros((old_count, old_count))
#             reference_floor = identity.new_zeros((old_count,))
#             internal_retention = old_internal_clearance

#         internal_violation = F.relu(-internal_retention)[old_internal_mask]
#         old_old_retention_loss = (
#             (
#                 torch.logsumexp(scale * internal_violation, dim=0)
#                 - math.log(float(internal_violation.numel()))
#             ).clamp_min(0.0)
#             if internal_violation.numel()
#             else old_internal_clearance.sum() * 0.0
#         )
#         old_to_new = clearance_all[:old_count, old_count:]
#         new_to_old = clearance_all[old_count:, :old_count].transpose(0, 1).contiguous()
#         joint_clearance = torch.minimum(old_to_new, new_to_old)
#         cross_learned = learned_all[:old_count, old_count:]
#         cross_physical = physical_all[:old_count, old_count:]
#         cross_effective = effective_all[:old_count, old_count:]
#         cross_weight = 1.0 + cross_effective
#         old_new_violation = F.relu(-joint_clearance)
#         old_new_violation_mean = (
#             (cross_weight * old_new_violation).sum()
#             / cross_weight.sum().clamp_min(eps)
#         )
#         old_new_unsafe = joint_clearance.lt(0.0).to(identity.dtype).mean()

#         # Direct Whole-CDS classifier supervision on every seen class.  This
#         # makes I_Q/V_Q and native I_P/V_P part of the actual classifier gradient.
#         physical_identity = seen_cds.classes.physical_identity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         carrier_output = classifier(
#             identity,
#             class_ids=ids,
#             trainable_class_ids=ids,
#             physical_spectral_evidence=physical_identity,
#             cds=seen_cds,
#         )
#         local_targets = torch.arange(count, device=identity.device, dtype=torch.long)
#         whole_logits = carrier_output.logits + carrier_output.whole_support.detach().clamp_min(eps).log()
#         whole_cds_ce = F.cross_entropy(whole_logits, local_targets)

#         # Old/new violation is an optimized term, not only a reported statistic.
#         loss = (
#             global_reconstruction_loss
#             + old_old_retention_loss
#             + old_new_violation_mean
#             + whole_cds_ce
#         )

#         historical_margin = margin_all[:old_count, old_count:]
#         historical_exposure = exposure_all[:old_count, old_count:]
#         current_margin = margin_all[old_count:, :old_count].transpose(0, 1).contiguous()
#         current_exposure = exposure_all[old_count:, :old_count].transpose(0, 1).contiguous()
#         current_clearance = clearance_all[old_count:, :old_count].transpose(0, 1).contiguous()

#         reference_retention = old_to_new - reference_floor[:, None]
#         reference_violation = reference_retention.lt(0.0).to(identity.dtype).mean()
#         internal_values = old_internal_clearance[old_internal_mask]
#         retention_values = internal_retention[old_internal_mask]
#         internal_unsafe = retention_values.lt(0.0).to(identity.dtype).mean()

#         scalars = (
#             loss, whole_cds_ce,
#             global_reconstruction_loss,
#             old_reconstruction_loss,
#             new_reconstruction_loss,
#             old_old_retention_loss,
#             historical_margin.mean(), historical_margin.min(),
#             historical_exposure.mean(), old_to_new.mean(), old_to_new.min(),
#             current_margin.mean(), current_margin.min(), current_exposure.mean(),
#             current_clearance.mean(), current_clearance.min(),
#             internal_values.mean(), internal_values.min(),
#             retention_values.mean(), retention_values.min(), internal_unsafe,
#             joint_clearance.mean(), joint_clearance.min(), cross_weight.mean(),
#             cross_learned.mean(), cross_physical.mean(), cross_effective.mean(),
#             old_new_violation_mean, old_new_unsafe,
#             reference_floor.mean(), reference_floor.min(),
#             reference_retention.mean(), reference_retention.min(), reference_violation,
#         )
#         if not all(bool(torch.isfinite(value)) for value in scalars):
#             raise RuntimeError("whole-CDS classifier protection became NaN or Inf")

#         return WholeCDSClassifierProtectionObjective(
#             loss=loss,
#             global_reconstruction_loss=global_reconstruction_loss,
#             old_class_reconstruction_loss=old_reconstruction_loss,
#             new_class_reconstruction_loss=new_reconstruction_loss,
#             old_old_retention_loss=old_old_retention_loss,
#             historical_identity_margin_mean=historical_margin.mean(),
#             historical_identity_margin_min=historical_margin.min(),
#             historical_deformation_exposure_mean=historical_exposure.mean(),
#             historical_structural_clearance_mean=old_to_new.mean(),
#             historical_structural_clearance_min=old_to_new.min(),
#             current_identity_margin_mean=current_margin.mean(),
#             current_identity_margin_min=current_margin.min(),
#             current_deformation_exposure_mean=current_exposure.mean(),
#             current_structural_clearance_mean=current_clearance.mean(),
#             current_structural_clearance_min=current_clearance.min(),
#             historical_internal_clearance_mean=internal_values.mean(),
#             historical_internal_clearance_min=internal_values.min(),
#             historical_internal_retention_mean=retention_values.mean(),
#             historical_internal_retention_min=retention_values.min(),
#             historical_internal_violation_fraction=internal_unsafe,
#             joint_structural_clearance_mean=joint_clearance.mean(),
#             joint_structural_clearance_min=joint_clearance.min(),
#             unsafe_pair_fraction=old_new_unsafe,
#             pair_weight_mean=cross_weight.mean(),
#             learned_pair_risk_mean=cross_learned.mean(),
#             physical_pair_risk_mean=cross_physical.mean(),
#             effective_pair_risk_mean=cross_effective.mean(),
#             old_new_structural_violation_mean=old_new_violation_mean,
#             old_new_violation_fraction=old_new_unsafe,
#             reference_clearance_mean=reference_floor.mean(),
#             reference_clearance_min=reference_floor.min(),
#             reference_retention_mean=reference_retention.mean(),
#             reference_retention_min=reference_retention.min(),
#             reference_violation_fraction=reference_violation,
#         )


# __all__ = [
#     "WholeCDSClassifierProtectionObjective",
#     "WholeCDSClassBalancedObjective",
#     "whole_cds_class_balanced_classifier_loss",
#     "HSICDSDecisionRegionRegulator",
# ]









 
# from __future__ import annotations
# from dataclasses import dataclass
# import math

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


# def _validate_seen_cds(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> tuple[Tensor, Tensor, Tensor, int, int, int, int]:
#     if not isinstance(classifier, CosineClassificationHead):
#         raise TypeError("classifier must be CosineClassificationHead")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long).flatten()
#     count = int(ids.numel())
#     old_count = int(historical_class_count)
#     if count < 2 or old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and new seen classes")
#     if int(torch.unique(ids).numel()) != count:
#         raise ValueError("seen class IDs must be unique")

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
#     return ids, identity, variation, old_count, count, k, d


# def _historical_cds(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     expected_ids: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     if not isinstance(reference_classifier, CosineClassificationHead):
#         raise TypeError("reference_classifier must be CosineClassificationHead")
#     if not isinstance(reference_historical_cds, ClassDiscriminativeStructureState):
#         raise TypeError("reference_historical_cds must be ClassDiscriminativeStructureState")
#     ids = reference_historical_cds.classes.class_ids.to(
#         device=reference_classifier.weight.device, dtype=torch.long
#     ).flatten()
#     expected = expected_ids.to(device=ids.device, dtype=torch.long)
#     if not torch.equal(ids, expected):
#         raise ValueError("reference historical CDS IDs disagree with historical prefix")
#     identity = reference_historical_cds.classes.learned_identity.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     variation = reference_historical_cds.classes.learned_variation.to(
#         device=reference_classifier.weight.device, dtype=reference_classifier.weight.dtype
#     ).detach()
#     if identity.ndim != 3 or variation.shape != (
#         identity.size(0), identity.size(1), identity.size(1), identity.size(2)
#     ):
#         raise ValueError("reference I_Q/V_Q shape mismatch")
#     return ids, identity, variation


# def _directed_identity_margin_and_relation_exposure(
#     *,
#     source_identity: Tensor,
#     source_deformation: Tensor,
#     boundary: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Robust directed cosine clearance from complete learned CDS.

#     ``I_Q`` supplies the ordered identity. ``V_Q`` supplies both diagonal
#     token/channel variation and off-diagonal ordered-token relation variation.
#     It is never interpreted as covariance or Gaussian memory.
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
#         raise RuntimeError("whole-CDS boundary clearance became NaN or Inf")
#     return margin, exposure, clearance


# @torch.no_grad()
# def _previous_historical_clearance(
#     *,
#     reference_classifier: CosineClassificationHead,
#     reference_historical_cds: ClassDiscriminativeStructureState,
#     historical_ids: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     ids, identity, variation = _historical_cds(
#         reference_classifier=reference_classifier,
#         reference_historical_cds=reference_historical_cds,
#         expected_ids=historical_ids,
#     )
#     if ids.numel() < 2:
#         raise ValueError("reference historical CDS needs at least two classes")
#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(reference_classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     _, _, clearance = _directed_identity_margin_and_relation_exposure(
#         source_identity=identity,
#         source_deformation=variation,
#         boundary=boundary,
#     )
#     diagonal = torch.eye(ids.numel(), device=clearance.device, dtype=torch.bool)
#     floor = clearance.masked_fill(diagonal, torch.inf).min(dim=1).values
#     if not bool(torch.isfinite(clearance).all()) or not bool(torch.isfinite(floor).all()):
#         raise RuntimeError("previous historical clearance became invalid")
#     return clearance.detach(), floor.detach()


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


# def whole_cds_class_balanced_classifier_loss(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
# ) -> WholeCDSClassBalancedObjective:
#     """Symmetric class-balanced Whole-CDS supervision for one cosine head."""
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
#     if identity.ndim != 3 or variation.shape != (
#         count, identity.size(1), identity.size(1), identity.size(2)
#     ):
#         raise ValueError("I_Q/V_Q shape mismatch")

#     eps = torch.finfo(identity.dtype).eps
#     rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
#     boundary = rows[:, None, :] - rows[None, :, :]
#     _, _, clearance = _directed_identity_margin_and_relation_exposure(
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
#     effective = _effective_pair_risk(learned, physical)
#     pair_weight = 1.0 + effective
#     valid = ~torch.eye(count, device=identity.device, dtype=torch.bool)
#     scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#     per_class = _classwise_robust_loss(
#         clearance=clearance,
#         pair_weight=pair_weight,
#         valid_mask=valid,
#         scale=scale,
#     )
#     loss = per_class.mean()
#     scalars = (
#         loss,
#         clearance[valid].mean(),
#         clearance[valid].min(),
#         learned[valid].mean(),
#         physical[valid].mean(),
#         effective[valid].mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in scalars):
#         raise RuntimeError("whole-CDS class-balanced objective became NaN or Inf")
#     return WholeCDSClassBalancedObjective(
#         loss=loss,
#         per_class_loss=per_class,
#         clearance_mean=clearance[valid].mean(),
#         clearance_min=clearance[valid].min(),
#         learned_pair_risk_mean=learned[valid].mean(),
#         physical_pair_risk_mean=physical[valid].mean(),
#         effective_pair_risk_mean=effective[valid].mean(),
#     )


# class HSICDSDecisionRegionRegulator(nn.Module):
#     """Symmetric all-seen classifier protection from the complete Whole-CDS.

#     This module has exactly one role: protect the deployed cosine decision
#     geometry.  It never initializes rows, never performs a separate optimizer
#     stage, and never applies old/new-specific detach rules.  Every seen class is
#     a live source and a live rival.  Historical old-old robust clearance is also
#     retained relative to the finalized previous classifier.
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
#     ) -> WholeCDSClassifierProtectionObjective:
#         ids, identity, variation, old_count, count, _k, _d = _validate_seen_cds(
#             classifier=classifier,
#             seen_cds=seen_cds,
#             historical_class_count=historical_class_count,
#         )
#         eps = torch.finfo(identity.dtype).eps
#         rows = F.normalize(classifier.class_weight(ids), p=2, dim=1, eps=eps)
#         boundary = rows[:, None, :] - rows[None, :, :]
#         margin_all, exposure_all, clearance_all = (
#             _directed_identity_margin_and_relation_exposure(
#                 source_identity=identity,
#                 source_deformation=variation,
#                 boundary=boundary,
#             )
#         )

#         learned_all = seen_cds.relations.learned_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         physical_all = seen_cds.relations.physical_combined_proximity.to(
#             device=identity.device, dtype=identity.dtype
#         ).detach()
#         if learned_all.shape != (count, count) or physical_all.shape != (count, count):
#             raise RuntimeError("whole-CDS relation matrices must be [C,C]")
#         effective_all = _effective_pair_risk(learned_all, physical_all)
#         pair_weight_all = 1.0 + effective_all

#         scale = classifier.logit_scale.to(device=identity.device, dtype=identity.dtype).detach()
#         valid_all = ~torch.eye(count, device=identity.device, dtype=torch.bool)
#         per_class = _classwise_robust_loss(
#             clearance=clearance_all,
#             pair_weight=pair_weight_all,
#             valid_mask=valid_all,
#             scale=scale,
#         )
#         global_reconstruction_loss = per_class.mean()
#         old_reconstruction_loss = per_class[:old_count].mean()
#         new_reconstruction_loss = per_class[old_count:].mean()

#         old_internal_clearance = clearance_all[:old_count, :old_count]
#         old_internal_mask = ~torch.eye(
#             old_count, device=identity.device, dtype=torch.bool
#         )
#         if not bool(old_internal_mask.any()):
#             raise ValueError("historical classifier protection needs at least two old classes")

#         if reference_classifier is not None or reference_historical_cds is not None:
#             if reference_classifier is None or reference_historical_cds is None:
#                 raise ValueError(
#                     "reference classifier and historical CDS must be supplied together"
#                 )
#             reference_internal, reference_floor = _previous_historical_clearance(
#                 reference_classifier=reference_classifier,
#                 reference_historical_cds=reference_historical_cds,
#                 historical_ids=ids[:old_count],
#             )
#             reference_internal = reference_internal.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             reference_floor = reference_floor.to(
#                 device=identity.device, dtype=identity.dtype
#             )
#             internal_retention = old_internal_clearance - reference_internal
#         else:
#             reference_internal = identity.new_zeros((old_count, old_count))
#             reference_floor = identity.new_zeros((old_count,))
#             internal_retention = old_internal_clearance

#         internal_violation = F.relu(-internal_retention)[old_internal_mask]
#         old_old_retention_loss = (
#             (
#                 torch.logsumexp(scale * internal_violation, dim=0)
#                 - math.log(float(internal_violation.numel()))
#             ).clamp_min(0.0)
#             if internal_violation.numel()
#             else old_internal_clearance.sum() * 0.0
#         )
#         loss = global_reconstruction_loss + old_old_retention_loss

#         old_to_new = clearance_all[:old_count, old_count:]
#         new_to_old = clearance_all[old_count:, :old_count].transpose(0, 1).contiguous()
#         joint_clearance = torch.minimum(old_to_new, new_to_old)
#         cross_learned = learned_all[:old_count, old_count:]
#         cross_physical = physical_all[:old_count, old_count:]
#         cross_effective = effective_all[:old_count, old_count:]
#         cross_weight = 1.0 + cross_effective
#         old_new_violation = F.relu(-joint_clearance)
#         old_new_violation_mean = (
#             (cross_weight * old_new_violation).sum()
#             / cross_weight.sum().clamp_min(eps)
#         )
#         old_new_unsafe = joint_clearance.lt(0.0).to(identity.dtype).mean()

#         historical_margin = margin_all[:old_count, old_count:]
#         historical_exposure = exposure_all[:old_count, old_count:]
#         current_margin = margin_all[old_count:, :old_count].transpose(0, 1).contiguous()
#         current_exposure = exposure_all[old_count:, :old_count].transpose(0, 1).contiguous()
#         current_clearance = clearance_all[old_count:, :old_count].transpose(0, 1).contiguous()

#         reference_retention = old_to_new - reference_floor[:, None]
#         reference_violation = reference_retention.lt(0.0).to(identity.dtype).mean()
#         internal_values = old_internal_clearance[old_internal_mask]
#         retention_values = internal_retention[old_internal_mask]
#         internal_unsafe = retention_values.lt(0.0).to(identity.dtype).mean()

#         scalars = (
#             loss,
#             global_reconstruction_loss,
#             old_reconstruction_loss,
#             new_reconstruction_loss,
#             old_old_retention_loss,
#             historical_margin.mean(), historical_margin.min(),
#             historical_exposure.mean(), old_to_new.mean(), old_to_new.min(),
#             current_margin.mean(), current_margin.min(), current_exposure.mean(),
#             current_clearance.mean(), current_clearance.min(),
#             internal_values.mean(), internal_values.min(),
#             retention_values.mean(), retention_values.min(), internal_unsafe,
#             joint_clearance.mean(), joint_clearance.min(), cross_weight.mean(),
#             cross_learned.mean(), cross_physical.mean(), cross_effective.mean(),
#             old_new_violation_mean, old_new_unsafe,
#             reference_floor.mean(), reference_floor.min(),
#             reference_retention.mean(), reference_retention.min(), reference_violation,
#         )
#         if not all(bool(torch.isfinite(value)) for value in scalars):
#             raise RuntimeError("whole-CDS classifier protection became NaN or Inf")

#         return WholeCDSClassifierProtectionObjective(
#             loss=loss,
#             global_reconstruction_loss=global_reconstruction_loss,
#             old_class_reconstruction_loss=old_reconstruction_loss,
#             new_class_reconstruction_loss=new_reconstruction_loss,
#             old_old_retention_loss=old_old_retention_loss,
#             historical_identity_margin_mean=historical_margin.mean(),
#             historical_identity_margin_min=historical_margin.min(),
#             historical_deformation_exposure_mean=historical_exposure.mean(),
#             historical_structural_clearance_mean=old_to_new.mean(),
#             historical_structural_clearance_min=old_to_new.min(),
#             current_identity_margin_mean=current_margin.mean(),
#             current_identity_margin_min=current_margin.min(),
#             current_deformation_exposure_mean=current_exposure.mean(),
#             current_structural_clearance_mean=current_clearance.mean(),
#             current_structural_clearance_min=current_clearance.min(),
#             historical_internal_clearance_mean=internal_values.mean(),
#             historical_internal_clearance_min=internal_values.min(),
#             historical_internal_retention_mean=retention_values.mean(),
#             historical_internal_retention_min=retention_values.min(),
#             historical_internal_violation_fraction=internal_unsafe,
#             joint_structural_clearance_mean=joint_clearance.mean(),
#             joint_structural_clearance_min=joint_clearance.min(),
#             unsafe_pair_fraction=old_new_unsafe,
#             pair_weight_mean=cross_weight.mean(),
#             learned_pair_risk_mean=cross_learned.mean(),
#             physical_pair_risk_mean=cross_physical.mean(),
#             effective_pair_risk_mean=cross_effective.mean(),
#             old_new_structural_violation_mean=old_new_violation_mean,
#             old_new_violation_fraction=old_new_unsafe,
#             reference_clearance_mean=reference_floor.mean(),
#             reference_clearance_min=reference_floor.min(),
#             reference_retention_mean=reference_retention.mean(),
#             reference_retention_min=reference_retention.min(),
#             reference_violation_fraction=reference_violation,
#         )


# __all__ = [
#     "WholeCDSClassifierProtectionObjective",
#     "WholeCDSClassBalancedObjective",
#     "whole_cds_class_balanced_classifier_loss",
#     "HSICDSDecisionRegionRegulator",
# ]
