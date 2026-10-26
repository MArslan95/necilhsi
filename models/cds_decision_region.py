"""G_Q-free read-only classifier diagnostics and legacy trainer compatibility.

Historical training uses HistoricalCDSSupportBank.historical_risk(), not the
read-only mean/classifier checks here. Nothing in this file introduces an
alternative classifier or classifier calibration.
"""

from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from models.geometry_bank import whole_cds_pair_risk


@dataclass(frozen=True)
class WholeCDSClassBalancedObjective:
    loss: torch.Tensor
    per_class_loss: torch.Tensor
    violation_fraction: torch.Tensor
    clearance_min: torch.Tensor
    clearance_mean: torch.Tensor
    old_new_violation_fraction: torch.Tensor
    learned_pair_risk_mean: torch.Tensor
    physical_pair_risk_mean: torch.Tensor
    effective_pair_risk_mean: torch.Tensor


def _head_diagnostic(
    classifier, cds, old_count, current_features=None, current_labels=None
):
    ids = cds.classes.class_ids.to(classifier.weight.device)
    w = F.normalize(classifier.class_weight(ids), dim=1)
    mean = cds.classes.learned_identity.to(w).flatten(1)
    score = classifier.logit_scale.detach() * mean @ w.T
    target = torch.arange(len(ids), device=w.device)
    per_class = F.cross_entropy(score, target, reduction="none")
    historical = per_class[:old_count].mean() if old_count else per_class.sum() * 0
    current = (
        per_class[old_count:].mean() if old_count < len(ids) else per_class.sum() * 0
    )
    all_accuracy = torch.tensor(0.0, device=w.device)
    only_accuracy = torch.tensor(0.0, device=w.device)
    if current_features is not None:
        with torch.no_grad():
            labels = torch.as_tensor(
                current_labels, device=w.device, dtype=torch.long
            ).flatten()
            out = classifier(current_features, class_ids=ids, trainable_class_ids=())
            all_accuracy = out.prediction.eq(labels).float().mean()
            only = classifier(
                current_features, class_ids=ids[old_count:], trainable_class_ids=()
            )
            only_accuracy = only.prediction.eq(labels).float().mean()
            map_target = labels[:, None].eq(ids[None, :]).long().argmax(1)
            actual_ce = F.cross_entropy(out.logits, map_target)
            current = actual_ce
    effective, learned, physical = whole_cds_pair_risk(cds)
    state = SimpleNamespace(
        balanced_loss=0.5 * (historical + current),
        historical_positive_loss=historical,
        current_positive_loss=current,
        real_new_loss=current,
        real_all_seen_accuracy=all_accuracy,
        new_only_accuracy=only_accuracy,
        new_only_loss=current,
        historical_per_class_loss=per_class[:old_count],
        current_per_class_loss=per_class[old_count:],
        learned_pair_risk_mean=learned.mean(),
        physical_pair_risk_mean=physical.mean(),
        effective_pair_risk_mean=effective.mean(),
    )
    return state, score, per_class


def whole_cds_class_balanced_classifier_loss(
    *, classifier, cds, historical_class_count=None
) -> WholeCDSClassBalancedObjective:
    n = len(cds.classes.class_ids)
    old = int(historical_class_count or 0)
    state, scores, per_class = _head_diagnostic(classifier, cds, old)
    target = torch.arange(n, device=scores.device)
    clear = scores.diag()[:, None] - scores
    valid = ~torch.eye(n, device=scores.device, dtype=torch.bool)
    violations = (clear[valid] < 0).float().mean()
    old_new = (target[:, None] < old) != (target[None, :] < old)
    old_new = old_new & valid
    cross = (
        (clear[old_new] < 0).float().mean() if bool(old_new.any()) else violations * 0
    )
    return WholeCDSClassBalancedObjective(
        state.balanced_loss,
        per_class,
        violations,
        clear[valid].min(),
        clear[valid].mean(),
        cross,
        state.learned_pair_risk_mean,
        state.physical_pair_risk_mean,
        state.effective_pair_risk_mean,
    )


def whole_cds_joint_classifier_state(
    *,
    classifier,
    current_features,
    current_labels,
    seen_cds,
    historical_class_count,
    current_sample_weights=None,
):
    return _head_diagnostic(
        classifier, seen_cds, historical_class_count, current_features, current_labels
    )[0]


def historical_whole_cds_per_class_classifier_loss(
    *, classifier, seen_cds, historical_class_count
):
    return _head_diagnostic(classifier, seen_cds, historical_class_count)[2][
        :historical_class_count
    ]


def current_class_feasible_radii(
    *, current_features, current_labels, cds, historical_class_count
):
    """Observed new-Q distance to historical I_Q; NOT a feasible geometric radius.

    Returns each old class's minimum observed feature-to-mean cosine distance
    and the fraction of incoming features that are closer to an old class mean
    than their own new class mean. Read-only cross-phase diagnostic.
    """
    n_old = int(historical_class_count)
    z = F.normalize(current_features.flatten(1), dim=1)
    mu = F.normalize(cds.classes.learned_identity.flatten(1).to(z), dim=1)
    ids = cds.classes.class_ids.to(z.device)
    old_similarity = z @ mu[:n_old].T
    distances = (1 - old_similarity).amin(0)
    labels = torch.as_tensor(current_labels, device=z.device, dtype=torch.long)
    indices = labels[:, None].eq(ids[None, :]).long().argmax(1)
    own = (z * mu[indices]).sum(1)
    intrusion = (old_similarity.max(1).values > own).float().mean()
    return distances.detach(), intrusion.detach()


def feasible_directional_head_objective(
    *,
    classifier,
    current_features,
    current_labels,
    seen_cds,
    historical_class_count,
    feasible_radii,
    intrusion_fraction,
    current_sample_weights=None,
):
    """Compatibility name: read-only empirical new/old margin diagnostics.

    The returned similarity statistics are *not* G_Q feasible-core radii,
    distribution certificates, or optimized training constraints.
    """
    state = whole_cds_joint_classifier_state(
        classifier=classifier,
        current_features=current_features,
        current_labels=current_labels,
        seen_cds=seen_cds,
        historical_class_count=historical_class_count,
        current_sample_weights=current_sample_weights,
    )
    return SimpleNamespace(
        state=state,
        mean_core_radius=feasible_radii.mean(),
        historical_intrusion_fraction=intrusion_fraction,
        effective_core_violation=intrusion_fraction,
        physical_interval_overlap=state.physical_pair_risk_mean,
        ordered_variation_confidence=state.learned_pair_risk_mean,
    )


def resolve_classifier_gradient_conflict(current_gradient, historical_gradient):
    if current_gradient.shape != historical_gradient.shape:
        raise ValueError("gradient shape mismatch")
    dot = (current_gradient * historical_gradient).sum()
    squared = (
        current_gradient.square()
        .sum()
        .clamp_min(torch.finfo(current_gradient.dtype).eps)
    )
    conflict = dot < 0
    result = (
        historical_gradient
        - torch.where(conflict, dot / squared, torch.zeros_like(dot)) * current_gradient
    )
    survival = result.norm() / historical_gradient.norm().clamp_min(
        torch.finfo(current_gradient.dtype).eps
    )
    return result, conflict.detach(), survival.detach()


__all__ = [
    "WholeCDSClassBalancedObjective",
    "whole_cds_class_balanced_classifier_loss",
    "whole_cds_joint_classifier_state",
    "historical_whole_cds_per_class_classifier_loss",
    "current_class_feasible_radii",
    "feasible_directional_head_objective",
    "resolve_classifier_gradient_conflict",
]


# """G_Q-free read-only classifier diagnostics and legacy trainer compatibility.

# Historical training uses HistoricalCDSSupportBank.historical_risk(), not the
# read-only mean/classifier checks here. Nothing in this file introduces an
# alternative classifier or classifier calibration.
# """

# from __future__ import annotations
# from dataclasses import dataclass
# from types import SimpleNamespace
# import torch
# import torch.nn.functional as F
# from models.geometry_bank import whole_cds_pair_risk


# @dataclass(frozen=True)
# class WholeCDSClassBalancedObjective:
#     loss: torch.Tensor
#     per_class_loss: torch.Tensor
#     violation_fraction: torch.Tensor
#     clearance_min: torch.Tensor
#     clearance_mean: torch.Tensor
#     old_new_violation_fraction: torch.Tensor
#     learned_pair_risk_mean: torch.Tensor
#     physical_pair_risk_mean: torch.Tensor
#     effective_pair_risk_mean: torch.Tensor


# def _head_diagnostic(
#     classifier, cds, old_count, current_features=None, current_labels=None
# ):
#     ids = cds.classes.class_ids.to(classifier.weight.device)
#     w = F.normalize(classifier.class_weight(ids), dim=1)
#     mean = cds.classes.learned_identity.to(w).flatten(1)
#     score = classifier.logit_scale.detach() * mean @ w.T
#     target = torch.arange(len(ids), device=w.device)
#     per_class = F.cross_entropy(score, target, reduction="none")
#     historical = per_class[:old_count].mean() if old_count else per_class.sum() * 0
#     current = (
#         per_class[old_count:].mean() if old_count < len(ids) else per_class.sum() * 0
#     )
#     all_accuracy = torch.tensor(0.0, device=w.device)
#     only_accuracy = torch.tensor(0.0, device=w.device)
#     if current_features is not None:
#         with torch.no_grad():
#             labels = torch.as_tensor(
#                 current_labels, device=w.device, dtype=torch.long
#             ).flatten()
#             out = classifier(current_features, class_ids=ids, trainable_class_ids=())
#             all_accuracy = out.prediction.eq(labels).float().mean()
#             only = classifier(
#                 current_features, class_ids=ids[old_count:], trainable_class_ids=()
#             )
#             only_accuracy = only.prediction.eq(labels).float().mean()
#             map_target = labels[:, None].eq(ids[None, :]).long().argmax(1)
#             actual_ce = F.cross_entropy(out.logits, map_target)
#             current = actual_ce
#     effective, learned, physical = whole_cds_pair_risk(cds)
#     state = SimpleNamespace(
#         balanced_loss=0.5 * (historical + current),
#         historical_positive_loss=historical,
#         current_positive_loss=current,
#         real_new_loss=current,
#         real_all_seen_accuracy=all_accuracy,
#         new_only_accuracy=only_accuracy,
#         new_only_loss=current,
#         historical_per_class_loss=per_class[:old_count],
#         current_per_class_loss=per_class[old_count:],
#         learned_pair_risk_mean=learned.mean(),
#         physical_pair_risk_mean=physical.mean(),
#         effective_pair_risk_mean=effective.mean(),
#     )
#     return state, score, per_class


# def whole_cds_class_balanced_classifier_loss(
#     *, classifier, cds, historical_class_count=None
# ) -> WholeCDSClassBalancedObjective:
#     n = len(cds.classes.class_ids)
#     old = int(historical_class_count or 0)
#     state, scores, per_class = _head_diagnostic(classifier, cds, old)
#     target = torch.arange(n, device=scores.device)
#     clear = scores.diag()[:, None] - scores
#     valid = ~torch.eye(n, device=scores.device, dtype=torch.bool)
#     violations = (clear[valid] < 0).float().mean()
#     old_new = (target[:, None] < old) != (target[None, :] < old)
#     old_new = old_new & valid
#     cross = (
#         (clear[old_new] < 0).float().mean() if bool(old_new.any()) else violations * 0
#     )
#     return WholeCDSClassBalancedObjective(
#         state.balanced_loss,
#         per_class,
#         violations,
#         clear[valid].min(),
#         clear[valid].mean(),
#         cross,
#         state.learned_pair_risk_mean,
#         state.physical_pair_risk_mean,
#         state.effective_pair_risk_mean,
#     )


# def whole_cds_joint_classifier_state(
#     *,
#     classifier,
#     current_features,
#     current_labels,
#     seen_cds,
#     historical_class_count,
#     current_sample_weights=None,
# ):
#     return _head_diagnostic(
#         classifier, seen_cds, historical_class_count, current_features, current_labels
#     )[0]


# def historical_whole_cds_per_class_classifier_loss(
#     *, classifier, seen_cds, historical_class_count
# ):
#     return _head_diagnostic(classifier, seen_cds, historical_class_count)[2][
#         :historical_class_count
#     ]


# def current_class_feasible_radii(
#     *, current_features, current_labels, cds, historical_class_count
# ):
#     """Observed new-Q distance to historical I_Q; NOT a feasible geometric radius.

#     Returns each old class's minimum observed feature-to-mean cosine distance
#     and the fraction of incoming features that are closer to an old class mean
#     than their own new class mean. Read-only cross-phase diagnostic.
#     """
#     n_old = int(historical_class_count)
#     z = F.normalize(current_features.flatten(1), dim=1)
#     mu = F.normalize(cds.classes.learned_identity.flatten(1).to(z), dim=1)
#     ids = cds.classes.class_ids.to(z.device)
#     old_similarity = z @ mu[:n_old].T
#     distances = (1 - old_similarity).amin(0)
#     labels = torch.as_tensor(current_labels, device=z.device, dtype=torch.long)
#     indices = labels[:, None].eq(ids[None, :]).long().argmax(1)
#     own = (z * mu[indices]).sum(1)
#     intrusion = (old_similarity.max(1).values > own).float().mean()
#     return distances.detach(), intrusion.detach()


# def feasible_directional_head_objective(
#     *,
#     classifier,
#     current_features,
#     current_labels,
#     seen_cds,
#     historical_class_count,
#     feasible_radii,
#     intrusion_fraction,
#     current_sample_weights=None,
# ):
#     """Compatibility name: read-only empirical new/old margin diagnostics.

#     The returned similarity statistics are *not* G_Q feasible-core radii,
#     distribution certificates, or optimized training constraints.
#     """
#     state = whole_cds_joint_classifier_state(
#         classifier=classifier,
#         current_features=current_features,
#         current_labels=current_labels,
#         seen_cds=seen_cds,
#         historical_class_count=historical_class_count,
#         current_sample_weights=current_sample_weights,
#     )
#     return SimpleNamespace(
#         state=state,
#         mean_core_radius=feasible_radii.mean(),
#         historical_intrusion_fraction=intrusion_fraction,
#         effective_core_violation=intrusion_fraction,
#         physical_interval_overlap=state.physical_pair_risk_mean,
#         ordered_variation_confidence=state.learned_pair_risk_mean,
#     )


# def resolve_classifier_gradient_conflict(current_gradient, historical_gradient):
#     if current_gradient.shape != historical_gradient.shape:
#         raise ValueError("gradient shape mismatch")
#     dot = (current_gradient * historical_gradient).sum()
#     squared = (
#         current_gradient.square()
#         .sum()
#         .clamp_min(torch.finfo(current_gradient.dtype).eps)
#     )
#     conflict = dot < 0
#     result = (
#         historical_gradient
#         - torch.where(conflict, dot / squared, torch.zeros_like(dot)) * current_gradient
#     )
#     survival = result.norm() / historical_gradient.norm().clamp_min(
#         torch.finfo(current_gradient.dtype).eps
#     )
#     return result, conflict.detach(), survival.detach()


# __all__ = [
#     "WholeCDSClassBalancedObjective",
#     "whole_cds_class_balanced_classifier_loss",
#     "whole_cds_joint_classifier_state",
#     "historical_whole_cds_per_class_classifier_loss",
#     "current_class_feasible_radii",
#     "feasible_directional_head_objective",
#     "resolve_classifier_gradient_conflict",
# ]


# from __future__ import annotations
# from dataclasses import dataclass, replace
# import torch
# import torch.nn.functional as F

# from models.classifier import CosineClassificationHead
# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     coupled_geometry_exposure,
#     coupled_geometry_gauge,
# )

# Tensor = torch.Tensor


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


# @dataclass(frozen=True)
# class HistoricalRegionProtectionObjective:
#     """Old-class decision-region protection against incoming classifier rows."""

#     loss: Tensor
#     absolute_clearance_mean: Tensor
#     relational_clearance_mean: Tensor
#     conservative_clearance_mean: Tensor
#     conservative_clearance_min: Tensor
#     violation_fraction: Tensor
#     learned_pair_risk_mean: Tensor
#     physical_pair_risk_mean: Tensor
#     effective_pair_risk_mean: Tensor


# @dataclass(frozen=True)
# class BalancedDecisionAdmissionState:

#     balanced_loss: Tensor
#     historical_positive_loss: Tensor
#     current_positive_loss: Tensor
#     real_new_loss: Tensor
#     real_all_seen_accuracy: Tensor
#     new_only_accuracy: Tensor
#     new_only_loss: Tensor
#     historical_per_class_loss: Tensor
#     current_per_class_loss: Tensor
#     cds_loss: Tensor
#     clearance_mean: Tensor
#     clearance_min: Tensor
#     violation_fraction: Tensor
#     old_new_violation_fraction: Tensor
#     old_old_violation_fraction: Tensor
#     new_new_violation_fraction: Tensor
#     learned_pair_risk_mean: Tensor
#     physical_pair_risk_mean: Tensor
#     effective_pair_risk_mean: Tensor


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

#     ids = classes.class_ids.to(
#         device=classifier.weight.device, dtype=torch.long
#     ).flatten()

#     count = int(ids.numel())

#     if count < 2 or int(torch.unique(ids).numel()) != count:

#         raise ValueError(
#             "Whole-CDS decision objective requires at least two unique classes"
#         )

#     identity = classes.learned_identity.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()

#     variation = classes.learned_variation.to(
#         device=classifier.weight.device, dtype=classifier.weight.dtype
#     ).detach()

#     if identity.ndim != 3 or identity.size(0) != count or identity.size(1) < 2:

#         raise ValueError("I_Q must be [C,K,D], K>=2")

#     c, k, d = identity.shape

#     if variation.shape != (c, 2, k - 1, d):

#         raise ValueError("V_Q must be signed relational bounds [C,2,K-1,D]")

#     if not bool(torch.isfinite(identity).all()) or not bool(
#         torch.isfinite(variation).all()
#     ):

#         raise ValueError("I_Q/V_Q contains NaN or Inf")

#     if bool((variation[:, 0] > variation[:, 1]).any()):

#         raise ValueError("V_Q lower bound exceeds upper bound")

#     if int(classifier.feature_dim) != k * d:

#         raise ValueError("classifier feature dimension does not match ordered Q")

#     return ids, identity, variation, count, k, d


# def _directed_structural_clearance(
#     *,
#     source_identity: Tensor,
#     source_deformation: Tensor,
#     boundary: Tensor,
#     source_basis: Tensor,
#     source_scale: Tensor,
#     source_rank: Tensor,
#     source_residual: Tensor,
# ) -> tuple[Tensor, Tensor]:
#     """Exact lower bound on deployed normalized-cosine margins over G_Q.

#     The ellipsoid plus orthogonal residual ball is calibrated on normalized
#     training features. The bound is exact for that stored enclosing set.
#     Signed V_Q intervals are descriptive and do NOT certify containment.
#     The two return fields preserve legacy metric names but are the SAME
#     exact coupled-geometry bound, not two independent structural views.
#     """
#     if boundary.shape[:1] != source_identity.shape[:1]:
#         raise ValueError("class count mismatch")
#     if boundary.size(-1) != source_identity[0].numel():
#         raise ValueError("cosine normal must match learned dimension")
#     center = source_identity.reshape(source_identity.size(0), -1).detach()
#     margin = (boundary * center[:, None, :]).sum(dim=-1)
#     exposure = coupled_geometry_exposure(
#         boundary,
#         source_basis.detach(),
#         source_scale.detach(),
#         source_rank.detach(),
#         source_residual.detach(),
#     )
#     lower = margin - exposure
#     if not bool(torch.isfinite(lower).all()):
#         raise RuntimeError("coupled-geometry clearance is nonfinite")
#     return lower, lower


# def _effective_pair_risk(learned: Tensor, physical: Tensor) -> Tensor:
#     """Fuse learned inter-class geometry with physical spectral evidence.

#     Learned CDS defines semantic rivalry. Physical I_P/V_P may strengthen an
#     already supported learned-space rivalry, but physical similarity alone must
#     never create a semantic class relation. This keeps Q as the single learned
#     decision space while P remains HSI-specific evidence about which learned
#     boundaries deserve stronger protection.

#     R_eff = R_Q + (1 - R_Q) * R_Q * R_P.

#     Thus R_Q == 0 implies R_eff == 0, while strong physical similarity can
#     amplify a non-zero learned rivalry without creating a second semantic system.
#     """

#     learned_value = torch.as_tensor(learned)
#     physical_value = torch.as_tensor(
#         physical, device=learned_value.device, dtype=learned_value.dtype
#     )
#     if learned_value.ndim != 2 or learned_value.shape != physical_value.shape:
#         raise ValueError("learned/physical risk must share a rank-two shape")
#     if not bool(torch.isfinite(learned_value).all()) or not bool(
#         torch.isfinite(physical_value).all()
#     ):
#         raise ValueError("learned/physical risk contains NaN or Inf")

#     learned_value = learned_value.clamp(0.0, 1.0)
#     physical_value = physical_value.clamp(0.0, 1.0)
#     return (
#         learned_value + (1.0 - learned_value) * learned_value * physical_value
#     ).clamp(0.0, 1.0)


# def _classwise_robust_loss(
#     *,
#     absolute_clearance: Tensor,
#     relational_clearance: Tensor,
#     pair_weight: Tensor,
#     valid_mask: Tensor,
#     scale: Tensor,
# ) -> Tensor:
#     """CE-shaped historical positive loss using both intra-class structures.


#     Each class/rival pair receives one total pair mass.  That mass is split

#     equally between the absolute and ordered-relation constraints, so retaining

#     both branches does not double the neutral loss scale.  Log-sum-exp then lets

#     the more violated branch dominate naturally without zeroing the other branch's

#     gradient via a hard ``max``.

#     """

#     if absolute_clearance.shape != relational_clearance.shape:

#         raise ValueError("absolute/relational clearance shapes disagree")

#     if (
#         absolute_clearance.shape != pair_weight.shape
#         or pair_weight.shape != valid_mask.shape
#     ):

#         raise ValueError("clearance, pair_weight and valid_mask must share shape")

#     if valid_mask.dtype != torch.bool:

#         raise TypeError("valid_mask must be boolean")

#     if not bool(valid_mask.any(dim=1).all()):

#         raise RuntimeError("every source class must have at least one rival")

#     eps = torch.finfo(absolute_clearance.dtype).eps

#     # Half the pair mass is assigned to each structural view.  If both views are

#     # identical, their summed exponential contribution equals the original

#     # single-view pair mass exactly.

#     log_branch_weight = (0.5 * pair_weight).clamp_min(eps).log()

#     absolute_terms = log_branch_weight - scale * absolute_clearance

#     relational_terms = log_branch_weight - scale * relational_clearance

#     absolute_terms = absolute_terms.masked_fill(~valid_mask, -torch.inf)

#     relational_terms = relational_terms.masked_fill(~valid_mask, -torch.inf)

#     zero = torch.zeros(
#         (absolute_clearance.size(0), 1),
#         device=absolute_clearance.device,
#         dtype=absolute_clearance.dtype,
#     )

#     return torch.logsumexp(
#         torch.cat((zero, absolute_terms, relational_terms), dim=1), dim=1
#     )


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

#     absolute_clearance, relational_clearance = _directed_structural_clearance(
#         source_identity=identity,
#         source_deformation=variation,
#         boundary=boundary,
#         source_basis=cds.classes.learned_coupled_basis,
#         source_scale=cds.classes.learned_coupled_scale,
#         source_rank=cds.classes.learned_coupled_active_rank,
#         source_residual=cds.classes.learned_residual_scale,
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

#     # Diagnostics use the conservative clearance: both structural views must be

#     # valid, so a pair is only as safe as its smaller clearance.  The training

#     # objective itself retains the two branches separately.

#     conservative_clearance = torch.minimum(absolute_clearance, relational_clearance)

#     return (
#         ids,
#         absolute_clearance,
#         relational_clearance,
#         conservative_clearance,
#         learned,
#         physical,
#         effective,
#         pair_weight,
#         valid,
#     )


# def historical_cds_region_protection_loss(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> HistoricalRegionProtectionObjective:
#     ids, identity, variation, count, _k, _d = _validate_cds(
#         classifier=classifier, cds=cds
#     )
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and current classes")

#     eps = torch.finfo(identity.dtype).eps
#     old_ids = ids[:old_count]
#     new_ids = ids[old_count:]

#     old_rows = F.normalize(
#         classifier.class_weight(old_ids).detach(), p=2, dim=1, eps=eps
#     )
#     new_rows = F.normalize(classifier.class_weight(new_ids), p=2, dim=1, eps=eps)

#     # Boundary orientation is old-positive: a positive clearance means the stored
#     # old-class region lies on W_old's side of the old/new cosine hyperplane.
#     boundary = old_rows[:, None, :] - new_rows[None, :, :]

#     old_identity = identity[:old_count].detach()
#     old_variation = variation[:old_count].detach()
#     absolute_clearance, relational_clearance = _directed_structural_clearance(
#         source_identity=old_identity,
#         source_deformation=old_variation,
#         boundary=boundary,
#         source_basis=cds.classes.learned_coupled_basis[:old_count],
#         source_scale=cds.classes.learned_coupled_scale[:old_count],
#         source_rank=cds.classes.learned_coupled_active_rank[:old_count],
#         source_residual=cds.classes.learned_residual_scale[:old_count],
#     )
#     conservative = torch.minimum(absolute_clearance, relational_clearance)

#     learned = cds.relations.learned_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()[:old_count, old_count:]
#     physical = cds.relations.physical_combined_proximity.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()[:old_count, old_count:]
#     if learned.shape != conservative.shape or physical.shape != conservative.shape:
#         raise RuntimeError(
#             "old/new CDS relation matrices disagree with classifier split"
#         )

#     effective = _effective_pair_risk(learned, physical)
#     pair_weight = 1.0 + effective
#     scale = classifier.logit_scale.to(
#         device=identity.device, dtype=identity.dtype
#     ).detach()

#     # Each historical class must be safe against its most dangerous incoming row.
#     # Keep absolute and ordered-relation envelopes as separate structural views;
#     # log-sum-exp makes a violated view/rival dominate without discarding the other.
#     log_branch_weight = (0.5 * pair_weight).clamp_min(eps).log()
#     abs_terms = log_branch_weight - scale * absolute_clearance
#     rel_terms = log_branch_weight - scale * relational_clearance
#     zero = torch.zeros((old_count, 1), device=identity.device, dtype=identity.dtype)
#     per_old_class = torch.logsumexp(
#         torch.cat((zero, abs_terms, rel_terms), dim=1), dim=1
#     )
#     loss = per_old_class.mean()

#     values = (
#         loss,
#         absolute_clearance.mean(),
#         relational_clearance.mean(),
#         conservative.mean(),
#         conservative.min(),
#         conservative.lt(0.0).to(identity.dtype).mean(),
#         learned.mean(),
#         physical.mean(),
#         effective.mean(),
#     )
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("historical CDS region protection became NaN or Inf")

#     return HistoricalRegionProtectionObjective(
#         loss=loss,
#         absolute_clearance_mean=absolute_clearance.mean(),
#         relational_clearance_mean=relational_clearance.mean(),
#         conservative_clearance_mean=conservative.mean(),
#         conservative_clearance_min=conservative.min(),
#         violation_fraction=conservative.lt(0.0).to(identity.dtype).mean(),
#         learned_pair_risk_mean=learned.mean(),
#         physical_pair_risk_mean=physical.mean(),
#         effective_pair_risk_mean=effective.mean(),
#     )


# def whole_cds_class_balanced_classifier_loss(
#     *,
#     classifier: CosineClassificationHead,
#     cds: ClassDiscriminativeStructureState,
#     historical_class_count: int | None = None,
# ) -> WholeCDSClassBalancedObjective:

#     (
#         ids,
#         absolute_clearance,
#         relational_clearance,
#         clearance,
#         learned,
#         physical,
#         effective,
#         pair_weight,
#         valid,
#     ) = _boundary_state(classifier=classifier, cds=cds)

#     count = int(ids.numel())

#     scale = classifier.logit_scale.to(
#         device=clearance.device, dtype=clearance.dtype
#     ).detach()

#     per_class = _classwise_robust_loss(
#         absolute_clearance=absolute_clearance,
#         relational_clearance=relational_clearance,
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

#             raise ValueError(
#                 "historical_class_count must split old and new seen classes"
#             )

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

#     old_new_mean, old_new_min, old_new_violation = _masked_stats(
#         clearance, old_new_mask
#     )

#     old_old_mean, old_old_min, old_old_violation = _masked_stats(
#         clearance, old_old_mask
#     )

#     new_new_mean, new_new_min, new_new_violation = _masked_stats(
#         clearance, new_new_mask
#     )

#     scalars = (
#         loss,
#         clearance[valid].mean(),
#         clearance[valid].min(),
#         learned[valid].mean(),
#         physical[valid].mean(),
#         effective[valid].mean(),
#         clearance[valid].lt(0.0).to(clearance.dtype).mean(),
#         old_new_mean,
#         old_new_min,
#         old_new_violation,
#         old_old_mean,
#         old_old_min,
#         old_old_violation,
#         new_new_mean,
#         new_new_min,
#         new_new_violation,
#     )

#     if not all(bool(torch.isfinite(v)) for v in scalars):

#         raise RuntimeError("Whole-CDS classifier objective became NaN or Inf")

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


# def _targets_local(labels: Tensor, class_ids: Tensor) -> Tensor:
#     matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("labels contain a class outside the requested class IDs")
#     return matches.to(torch.long).argmax(dim=1)


# def _real_new_diagnostics(
#     *,
#     classifier: CosineClassificationHead,
#     current_features: Tensor,
#     current_labels: Tensor,
#     seen_ids: Tensor,
#     new_ids: Tensor,
# ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
#     """Read-only deployed-head diagnostics on the real current observations."""
#     with torch.no_grad():
#         output = classifier(
#             current_features,
#             class_ids=seen_ids,
#             trainable_class_ids=(),
#         )
#         targets_seen = _targets_local(current_labels, seen_ids)
#         real_new_loss = F.cross_entropy(output.logits, targets_seen)
#         accuracy = output.prediction.eq(current_labels).to(output.logits.dtype).mean()

#         new_only = classifier(
#             current_features,
#             class_ids=new_ids,
#             trainable_class_ids=(),
#         )
#         targets_new = _targets_local(current_labels, new_ids)
#         new_only_loss = F.cross_entropy(new_only.logits, targets_new)
#         new_only_accuracy = (
#             new_only.prediction.eq(current_labels).to(output.logits.dtype).mean()
#         )
#     return real_new_loss, accuracy, new_only_loss, new_only_accuracy


# def _historical_all_seen_per_class_loss(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> tuple[Tensor, Tensor]:
#     (
#         ids,
#         absolute_clearance,
#         relational_clearance,
#         _clearance,
#         _learned,
#         _physical,
#         _effective,
#         pair_weight,
#         valid,
#     ) = _boundary_state(classifier=classifier, cds=seen_cds)
#     old_count = int(historical_class_count)
#     count = int(ids.numel())
#     if old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and new seen classes")

#     # Historical Whole-CDS supervises the same deployed cosine head.
#     # Keep the shared decision scale differentiable here so tau is calibrated
#     # by balanced historical + current evidence rather than new data alone.
#     scale = classifier.logit_scale.to(
#         device=absolute_clearance.device, dtype=absolute_clearance.dtype
#     )
#     per_class = _classwise_robust_loss(
#         absolute_clearance=absolute_clearance[:old_count],
#         relational_clearance=relational_clearance[:old_count],
#         pair_weight=pair_weight[:old_count],
#         valid_mask=valid[:old_count],
#         scale=scale,
#     )
#     return per_class, pair_weight


# def _current_cross_phase_per_class_loss_source_owned(
#     *,
#     classifier: CosineClassificationHead,
#     current_features: Tensor,
#     current_labels: Tensor,
#     seen_ids: Tensor,
#     historical_class_count: int,
#     sample_weights: Tensor | None = None,
# ) -> Tensor:
#     old_count = int(historical_class_count)
#     count = int(seen_ids.numel())
#     if old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split seen classes")
#     new_ids = seen_ids[old_count:]
#     if current_labels.ndim != 1 or current_features.size(0) != current_labels.numel():
#         raise ValueError("current features/labels are misaligned")
#     if not bool(torch.isin(current_labels, new_ids).all()):
#         raise ValueError("joint classifier objective accepts current/new labels only")

#     output = classifier(
#         current_features,
#         class_ids=seen_ids,
#         trainable_class_ids=new_ids,
#     )
#     targets_seen = _targets_local(current_labels, seen_ids)
#     sample_loss = F.cross_entropy(output.logits, targets_seen, reduction="none")
#     weights = (
#         torch.ones_like(sample_loss)
#         if sample_weights is None
#         else torch.as_tensor(
#             sample_weights, device=sample_loss.device, dtype=sample_loss.dtype
#         )
#     )
#     if (
#         weights.shape != sample_loss.shape
#         or not bool(torch.isfinite(weights).all())
#         or bool((weights < 0).any())
#     ):
#         raise ValueError("current sample weights must be finite non-negative [N]")
#     per_class = []
#     for class_id in new_ids:
#         mask = current_labels.eq(class_id)
#         if bool(mask.any()):
#             class_weights = weights[mask]
#             per_class.append(
#                 (sample_loss[mask] * class_weights).sum()
#                 / class_weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
#             )
#     if not per_class:
#         raise RuntimeError("joint classifier batch contains no current class")
#     result = torch.stack(per_class)
#     if not bool(torch.isfinite(result).all()):
#         raise RuntimeError("joint real-current all-seen CE became NaN or Inf")
#     return result


# def _balanced_state_all_rows(
#     *,
#     classifier: CosineClassificationHead,
#     current_features: Tensor,
#     current_labels: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     current_sample_weights: Tensor | None = None,
# ) -> BalancedDecisionAdmissionState:
#     ids, _identity, _variation, count, _k, _d = _validate_cds(
#         classifier=classifier, cds=seen_cds
#     )
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= count:
#         raise ValueError("historical_class_count must split old and new seen classes")
#     new_ids = ids[old_count:]

#     historical_per_class, _pair_weight = _historical_all_seen_per_class_loss(
#         classifier=classifier,
#         seen_cds=seen_cds,
#         historical_class_count=old_count,
#     )
#     current_per_class = _current_cross_phase_per_class_loss_source_owned(
#         classifier=classifier,
#         current_features=current_features,
#         current_labels=current_labels,
#         seen_ids=ids,
#         historical_class_count=old_count,
#         sample_weights=current_sample_weights,
#     )

#     # Both task origins receive equal mass on EVERY minibatch.  The number of
#     # current classes represented in a batch can be smaller than the task's
#     # full class count (especially on Indian Pines with Oats and Grass-mowed).
#     # A concatenated mean would silently give old classes 6/(6 + n_present)
#     # of the classifier loss, increasing old bias whenever a new class is absent.
#     historical_positive = historical_per_class.mean()
#     current_positive = current_per_class.mean()
#     balanced = 0.5 * (historical_positive + current_positive)
#     real_new_loss, real_accuracy, new_only_loss, new_only_accuracy = (
#         _real_new_diagnostics(
#             classifier=classifier,
#             current_features=current_features,
#             current_labels=current_labels,
#             seen_ids=ids,
#             new_ids=new_ids,
#         )
#     )
#     cds_objective = whole_cds_class_balanced_classifier_loss(
#         classifier=classifier,
#         cds=seen_cds,
#         historical_class_count=old_count,
#     )
#     return BalancedDecisionAdmissionState(
#         balanced_loss=balanced,
#         historical_positive_loss=historical_positive,
#         current_positive_loss=current_positive,
#         real_new_loss=real_new_loss,
#         real_all_seen_accuracy=real_accuracy,
#         new_only_accuracy=new_only_accuracy,
#         new_only_loss=new_only_loss,
#         historical_per_class_loss=historical_per_class,
#         current_per_class_loss=current_per_class,
#         cds_loss=cds_objective.loss,
#         clearance_mean=cds_objective.clearance_mean,
#         clearance_min=cds_objective.clearance_min,
#         violation_fraction=cds_objective.violation_fraction,
#         old_new_violation_fraction=cds_objective.old_new_violation_fraction,
#         old_old_violation_fraction=cds_objective.old_old_violation_fraction,
#         new_new_violation_fraction=cds_objective.new_new_violation_fraction,
#         learned_pair_risk_mean=cds_objective.learned_pair_risk_mean,
#         physical_pair_risk_mean=cds_objective.physical_pair_risk_mean,
#         effective_pair_risk_mean=cds_objective.effective_pair_risk_mean,
#     )


# def whole_cds_joint_classifier_state(
#     *,
#     classifier: CosineClassificationHead,
#     current_features: Tensor,
#     current_labels: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     current_sample_weights: Tensor | None = None,
# ) -> BalancedDecisionAdmissionState:

#     return _balanced_state_all_rows(
#         classifier=classifier,
#         current_features=current_features,
#         current_labels=current_labels,
#         seen_cds=seen_cds,
#         historical_class_count=historical_class_count,
#         current_sample_weights=current_sample_weights,
#     )


# def historical_whole_cds_per_class_classifier_loss(
#     *,
#     classifier: CosineClassificationHead,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> Tensor:
#     per_class, _ = _historical_all_seen_per_class_loss(
#         classifier=classifier,
#         seen_cds=seen_cds,
#         historical_class_count=historical_class_count,
#     )
#     return per_class


# def resolve_classifier_gradient_conflict(
#     current_gradient: Tensor, historical_gradient: Tensor
# ) -> tuple[Tensor, Tensor, Tensor]:
#     """Project ONLY adverse historical directions away from real current CE.

#     The returned historical direction has nonnegative Euclidean dot product
#     with the current CE direction (within floating-point error). The next
#     AdamW step is not itself guaranteed to be a Pareto step.
#     """
#     if current_gradient.shape != historical_gradient.shape:
#         raise ValueError("head gradient tensors must match")
#     if not all(
#         bool(torch.isfinite(t).all()) for t in (current_gradient, historical_gradient)
#     ):
#         raise ValueError("head gradients must be finite")
#     dot = (current_gradient * historical_gradient).sum()
#     energy = current_gradient.square().sum()
#     conflict = (dot < 0) & (energy > 0)
#     safe_energy = energy.clamp_min(torch.finfo(current_gradient.dtype).eps)
#     corrected = (
#         historical_gradient
#         - torch.where(conflict, dot / safe_energy, torch.zeros_like(dot))
#         * current_gradient
#     )
#     original_norm = torch.linalg.vector_norm(historical_gradient)
#     surviving = (
#         torch.linalg.vector_norm(corrected)
#         / original_norm.clamp_min(torch.finfo(current_gradient.dtype).eps)
#     ).clamp(0, 1)
#     return corrected, conflict.detach(), surviving.detach()


# @dataclass(frozen=True)
# class FeasibleDirectionalHeadObjective:
#     """Current real CE + G_Q feasible historical core, no historical replay.

#     "core" is an INNER subset of the calibrated G_Q region, selected using
#     currently observed negative examples. It is not a guarantee for unseen
#     old examples. The full G_Q remains stored unchanged and is reported.
#     """

#     state: BalancedDecisionAdmissionState
#     effective_core_violation: Tensor
#     core_clearance_mean: Tensor
#     historical_intrusion_fraction: Tensor
#     mean_core_radius: Tensor
#     physical_interval_overlap: Tensor
#     ordered_variation_confidence: Tensor


# @torch.no_grad()
# def current_class_feasible_radii(
#     *,
#     current_features: Tensor,
#     current_labels: Tensor,
#     cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> tuple[Tensor, Tensor]:
#     """Compute nearest current negative-class gauge for each old/new pair.

#     With r=min(1,min_x rho_{G_old}(x)) the r-scaled historical core does
#     not STRICTLY contain any observed new-class feature (boundary equality
#     is possible). No features or per-sample gauges are saved across phases.
#     """
#     old_count = int(historical_class_count)
#     classes = cds.classes
#     ids = classes.class_ids
#     if not 0 < old_count < int(ids.numel()):
#         raise ValueError("old_count must split old/new classes")
#     new_ids = ids[old_count:].to(current_labels.device)
#     if not bool(torch.isin(current_labels, new_ids).all()):
#         raise ValueError("feasibility samples must be current classes only")
#     gauge = coupled_geometry_gauge(
#         current_features.detach(),
#         classes.learned_identity[:old_count],
#         classes.learned_coupled_basis[:old_count],
#         classes.learned_coupled_scale[:old_count],
#         classes.learned_coupled_active_rank[:old_count],
#         classes.learned_residual_scale[:old_count],
#     )
#     radii = []
#     intrusion = []
#     for j in new_ids:
#         sub = gauge[current_labels == j]
#         if sub.numel() == 0:
#             raise ValueError("every incoming class must have observed training data")
#         nearest = sub.amin(dim=0)
#         radii.append(nearest.clamp(min=0.0, max=1.0))
#         intrusion.append((sub <= 1.0).to(sub.dtype).mean(dim=0))
#     return torch.stack(radii, dim=1).detach(), torch.stack(intrusion, dim=1).detach()


# def _spectral_interval_overlap(
#     cds: ClassDiscriminativeStructureState, old_count: int
# ) -> Tensor:
#     """Band-wise actual interval overlap from existing I_P and V_P.

#     Signed V_P first B entries are residual envelopes of native ordered
#     bands; the remaining entries describe adjacent-band differences.
#     """
#     cls = cds.classes
#     ip = cls.physical_identity.squeeze(-1).detach()
#     vp = cls.physical_variation.detach()
#     bands = ip.size(1)
#     lo = ip + vp[:, 0, :bands, 0]
#     hi = ip + vp[:, 1, :bands, 0]
#     low = torch.maximum(lo[:old_count, None], lo[None, old_count:])
#     high = torch.minimum(hi[:old_count, None], hi[None, old_count:])
#     intersect = (high - low).clamp_min(0)
#     union = (
#         torch.maximum(hi[:old_count, None], hi[None, old_count:])
#         - torch.minimum(lo[:old_count, None], lo[None, old_count:])
#     ).clamp_min(0)
#     eps = torch.finfo(ip.dtype).eps
#     overlap = torch.where(
#         union > eps,
#         intersect / union.clamp_min(eps),
#         (lo[:old_count, None] == lo[None, old_count:]).to(ip.dtype),
#     )
#     return overlap.mean(dim=-1).clamp(0, 1)


# def feasible_directional_head_objective(
#     *,
#     classifier: CosineClassificationHead,
#     current_features: Tensor,
#     current_labels: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     feasible_radii: Tensor,
#     intrusion_fraction: Tensor,
#     current_sample_weights: Tensor | None = None,
# ) -> FeasibleDirectionalHeadObjective:
#     """Classifier learning with source-owned G_Q constraints and real new labels.

#     Historical W is reference-fixed. For each old/new pair, the exposed
#     G_Q support is scaled by the empirically feasible radius; signed V_Q
#     provides a *descriptive reliability* (not an extra certified margin);
#     R_Q and I_P/V_P interval overlap prioritize confusable pairs.
#     Historical pressure ends once the robust core margin is nonnegative.
#     """
#     ids, center, vq, count, k, d = _validate_cds(classifier=classifier, cds=seen_cds)
#     old = int(historical_class_count)
#     nnew = count - old
#     expected = (old, nnew)
#     if old <= 0 or nnew <= 0:
#         raise ValueError("old/new split invalid")
#     if feasible_radii.shape != expected or intrusion_fraction.shape != expected:
#         raise ValueError("feasibility matrices must be [old,new]")
#     radius = feasible_radii.to(center).detach().clamp(0, 1)
#     intrusion = intrusion_fraction.to(center).detach().clamp(0, 1)
#     if not all(bool(torch.isfinite(v).all()) for v in (radius, intrusion)):
#         raise ValueError("nonfinite feasibility geometry")

#     wold = F.normalize(classifier.class_weight(ids[:old]).detach(), dim=1)
#     wnew = F.normalize(classifier.class_weight(ids[old:]), dim=1)
#     b = wold[:, None] - wnew[None, :]
#     cflat = center[:old].reshape(old, -1).detach()
#     exposure = coupled_geometry_exposure(
#         b,
#         seen_cds.classes.learned_coupled_basis[:old],
#         seen_cds.classes.learned_coupled_scale[:old],
#         seen_cds.classes.learned_coupled_active_rank[:old],
#         seen_cds.classes.learned_residual_scale[:old],
#     )
#     core_margin = (b * cflat[:, None]).sum(-1) - radius * exposure

#     # Signed ordered V_Q affects reliability according to actual classifier
#     # direction differences along the ordered token axis. No fake V_Q
#     # containment certificate or duplicated absolute/relational branch.
#     bw = b.reshape(old, nnew, k, d)
#     db = bw[:, :, 1:] - bw[:, :, :-1]
#     interval_width = (vq[:old, 1] - vq[:old, 0]).clamp_min(0)
#     eps = torch.finfo(center.dtype).eps
#     var_uncertainty = (db.abs() * interval_width[:, None]).sum((-1, -2)) / db.abs().sum(
#         (-1, -2)
#     ).clamp_min(eps)
#     variation_confidence = (1 / (1 + var_uncertainty)).detach()

#     # Physically measured intervals decide spectral rivalry. No auxiliary
#     # spectral inference head is introduced.
#     p_overlap = _spectral_interval_overlap(seen_cds, old)
#     r_q = (
#         seen_cds.relations.learned_combined_proximity[:old, old:]
#         .to(center)
#         .detach()
#         .clamp(0, 1)
#     )
#     effective_risk = _effective_pair_risk(r_q, p_overlap)
#     relevance = (1.0 + effective_risk) * variation_confidence
#     # Near an actual new example's historical G_Q center, its old/new region
#     # protection is not sufficiently identifiable; smoothly reduce mass.
#     mass = relevance * radius
#     scale = classifier.logit_scale.detach()
#     raw_violation = F.relu(-scale * core_margin)
#     per_old = (mass * raw_violation).sum(dim=1) / float(nnew)
#     old_loss = per_old.mean()

#     current_per_class = _current_cross_phase_per_class_loss_source_owned(
#         classifier=classifier,
#         current_features=current_features,
#         current_labels=current_labels,
#         seen_ids=ids,
#         historical_class_count=old,
#         sample_weights=current_sample_weights,
#     )
#     new_loss = current_per_class.mean()
#     balanced = 0.5 * (old_loss + new_loss)
#     new_ce, real_acc, new_only_loss, new_only_acc = _real_new_diagnostics(
#         classifier=classifier,
#         current_features=current_features,
#         current_labels=current_labels,
#         seen_ids=ids,
#         new_ids=ids[old:],
#     )
#     # The legacy full-region objective is read-only, for apples-to-apples
#     # measurement against previous runs (including its strict violations).
#     with torch.no_grad():
#         base = whole_cds_class_balanced_classifier_loss(
#             classifier=classifier, cds=seen_cds, historical_class_count=old
#         )
#     state = BalancedDecisionAdmissionState(
#         balanced_loss=balanced,
#         historical_positive_loss=old_loss,
#         current_positive_loss=new_loss,
#         real_new_loss=new_ce,
#         real_all_seen_accuracy=real_acc,
#         new_only_accuracy=new_only_acc,
#         new_only_loss=new_only_loss,
#         historical_per_class_loss=per_old,
#         current_per_class_loss=current_per_class,
#         cds_loss=base.loss.detach(),
#         clearance_mean=base.clearance_mean.detach(),
#         clearance_min=base.clearance_min.detach(),
#         violation_fraction=base.violation_fraction.detach(),
#         old_new_violation_fraction=base.old_new_violation_fraction.detach(),
#         old_old_violation_fraction=base.old_old_violation_fraction.detach(),
#         new_new_violation_fraction=base.new_new_violation_fraction.detach(),
#         learned_pair_risk_mean=r_q.mean().detach(),
#         physical_pair_risk_mean=p_overlap.mean().detach(),
#         effective_pair_risk_mean=effective_risk.mean().detach(),
#     )
#     return FeasibleDirectionalHeadObjective(
#         state=state,
#         effective_core_violation=core_margin.lt(0)
#         .to(core_margin.dtype)
#         .mean()
#         .detach(),
#         core_clearance_mean=core_margin.mean().detach(),
#         historical_intrusion_fraction=intrusion.mean().detach(),
#         mean_core_radius=radius.mean().detach(),
#         physical_interval_overlap=p_overlap.mean().detach(),
#         ordered_variation_confidence=variation_confidence.mean().detach(),
#     )


# __all__ = [
#     "FeasibleDirectionalHeadObjective",
#     "resolve_classifier_gradient_conflict",
#     "current_class_feasible_radii",
#     "feasible_directional_head_objective",
#     "WholeCDSClassBalancedObjective",
#     "HistoricalRegionProtectionObjective",
#     "BalancedDecisionAdmissionState",
#     "whole_cds_class_balanced_classifier_loss",
#     "historical_cds_region_protection_loss",
#     "whole_cds_joint_classifier_state",
#     "historical_whole_cds_per_class_classifier_loss",
# ]
