from __future__ import annotations

"""Losses for the full-CDS historical-supervision NECIL-HSI architecture.

The deployed decision rule remains one trainable bias-free cosine classifier.
Persistent class state remains exactly {I_Q,V_Q,I_P,V_P,N}; inter-class CDS is
re-derived from the stored state every refresh.

Training roles:
    direct-Q contrast -> same real-sample contrastive separation in base and incremental phases;

Incremental roles:
    current CE       -> current-class plasticity with all seen rows as rivals;
    HDC              -> preserve the previous historical decision relation;
    CDS separation   -> keep real current HSI away from risky historical regions;
    historical CDS   -> give old classes direct positive classifier supervision
                        and require every all-seen boundary to clear the transported
                        I_Q/V_Q class region, prioritized by physical HSI-CDS risk.

No pseudo-feature generation, inversion, covariance model, second classifier,
post-hoc Stage-B solve, or decision-region regulator is used.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.classifier import SpectralCDSClassifier
from models.geometry_bank import ClassDiscriminativeStructureState, whole_cds_pair_risk

Tensor = torch.Tensor


@dataclass(frozen=True)
class ClassificationObjective:
    loss: Tensor
    accuracy: Tensor
    targets_local: Tensor


@dataclass(frozen=True)
class SupervisedContrastiveObjective:
    loss: Tensor
    valid_anchor_fraction: Tensor
    positive_similarity_mean: Tensor
    negative_similarity_mean: Tensor
    similarity_gap: Tensor


@dataclass(frozen=True)
class HistoricalDecisionConsistencyObjective:
    loss: Tensor
    relation_rmse: Tensor
    teacher_relation_norm_mean: Tensor
    student_relation_norm_mean: Tensor
    pair_weight_mean: Tensor
    learned_region_risk_mean: Tensor
    spectral_region_risk_mean: Tensor


@dataclass(frozen=True)
class HSICDSSeparationObjective:
    loss: Tensor
    margin_mean: Tensor
    margin_min: Tensor
    pair_weight_mean: Tensor
    learned_region_risk_mean: Tensor
    spectral_region_risk_mean: Tensor


@dataclass(frozen=True)
class HistoricalCDSSupervisionObjective:
    loss: Tensor
    positive_ce: Tensor
    decision_anchor_ce: Tensor
    relation_loss: Tensor
    identity_accuracy: Tensor
    decision_anchor_accuracy: Tensor
    identity_margin_mean: Tensor
    identity_margin_min: Tensor
    deformation_exposure_mean: Tensor
    structural_clearance_mean: Tensor
    structural_clearance_min: Tensor
    pair_weight_mean: Tensor
    learned_region_risk_mean: Tensor
    spectral_region_risk_mean: Tensor


def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
    labels = torch.as_tensor(labels_global)
    ids = torch.as_tensor(class_ids)
    if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
        raise ValueError("labels_global must be non-empty torch.long [N]")
    if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
        raise ValueError("class_ids must be torch.long [C], C>=2")
    if labels.device != ids.device:
        raise ValueError("labels_global and class_ids must share device")
    matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("a target label is absent from class_ids")
    return matches.to(torch.long).argmax(dim=1)


def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
    if sample_weights is None:
        return per_sample.mean()
    weights = torch.as_tensor(sample_weights)
    if weights.shape != per_sample.shape or weights.device != per_sample.device or weights.dtype != per_sample.dtype:
        raise ValueError("sample_weights must match per-sample loss")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
        raise ValueError("sample_weights must be finite, non-negative and have positive mass")
    return (per_sample * weights).sum() / weights.sum()


def classification_objective(
    *,
    logits: Tensor,
    class_ids: Tensor,
    labels_global: Tensor,
    sample_weights: Tensor | None = None,
) -> ClassificationObjective:
    scores = torch.as_tensor(logits)
    ids = torch.as_tensor(class_ids)
    if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
        raise ValueError("logits must be non-empty [N,C], C>=2")
    if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
        raise ValueError("class_ids must match logits")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("logits contain NaN or Inf")
    targets = _targets_local(labels_global, ids)
    per_sample = F.cross_entropy(scores, targets, reduction="none")
    loss = _weighted_mean(per_sample, sample_weights)
    accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("classification loss became NaN or Inf")
    return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


def supervised_contrastive_q_loss(
    *,
    ordered_features: Tensor,
    labels_global: Tensor,
    logit_scale: Tensor,
    sample_weights: Tensor | None = None,
) -> SupervisedContrastiveObjective:
    """Supervised contrast directly on the deployed ordered Q representation.

    The same real-sample contrast is used in Phase 0 and every incremental phase.
    It introduces no projection head, prototype, replay state, CDS anchor, or
    classifier parameter. Each sample is contrasted only against other real
    samples in the current minibatch. Flattened Q is L2-normalized so this
    objective uses the same angular geometry as the deployed cosine classifier.
    The classifier's shared positive scale is detached: contrastive learning
    updates Q only and cannot calibrate or bias classifier rows.

    Anchors without a same-class peer in the minibatch are ignored.
    """
    q = torch.as_tensor(ordered_features)
    labels = torch.as_tensor(labels_global)
    if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
        raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
    if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
        raise ValueError("ordered_features must be finite real floating point")
    if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
        raise ValueError("labels_global must be torch.long [N] on the Q device")

    scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
    if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
        raise ValueError("logit_scale must be a finite positive scalar")

    eps = torch.finfo(q.dtype).eps
    unit = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
    similarity = (unit @ unit.t()).clamp(-1.0, 1.0) * scale

    n = int(q.size(0))
    eye = torch.eye(n, device=q.device, dtype=torch.bool)
    same = labels[:, None].eq(labels[None, :])
    positive_mask = same & ~eye
    negative_mask = (~same) & ~eye
    candidate_mask = ~eye

    masked_similarity = similarity.masked_fill(~candidate_mask, -torch.inf)
    log_denom = torch.logsumexp(masked_similarity, dim=1)
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0

    if bool(valid_anchor.any()):
        log_probability = similarity - log_denom.unsqueeze(1)
        per_anchor = -(
            log_probability * positive_mask.to(dtype=q.dtype)
        ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
        valid_loss = per_anchor[valid_anchor]
        if sample_weights is None:
            loss = valid_loss.mean()
        else:
            weights = torch.as_tensor(sample_weights)
            if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
                raise ValueError("sample_weights must be [N] and match Q device/dtype")
            loss = _weighted_mean(valid_loss, weights[valid_anchor])
    else:
        loss = q.sum() * 0.0

    positive_values = similarity[positive_mask]
    negative_values = similarity[negative_mask]
    zero = q.new_zeros(())
    positive_mean = positive_values.mean() if positive_values.numel() else zero
    negative_mean = negative_values.mean() if negative_values.numel() else zero
    gap = positive_mean - negative_mean
    valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

    values = (loss, valid_fraction, positive_mean, negative_mean, gap)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise RuntimeError("supervised Q contrastive objective became NaN or Inf")

    return SupervisedContrastiveObjective(
        loss=loss,
        valid_anchor_fraction=valid_fraction,
        positive_similarity_mean=positive_mean,
        negative_similarity_mean=negative_mean,
        similarity_gap=gap,
    )


def historical_decision_consistency_loss(
    *,
    reference_features: Tensor,
    current_features: Tensor,
    historical_classifier_rows: Tensor,
    labels_global: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
) -> HistoricalDecisionConsistencyObjective:
    """Preserve phase-start old-class relative decision geometry on current HSI."""
    ref = torch.as_tensor(reference_features)
    cur = torch.as_tensor(current_features)
    rows = torch.as_tensor(historical_classifier_rows)
    labels = torch.as_tensor(labels_global)
    old_count = int(historical_class_count)
    if ref.ndim != 3 or ref.size(0) == 0 or cur.shape != ref.shape:
        raise ValueError("reference/current features must share non-empty [N,K,D]")
    if old_count < 2 or rows.shape != (old_count, ref.size(1) * ref.size(2)):
        raise ValueError("historical_classifier_rows must be [O,K*D], O>=2")
    if rows.device != ref.device or rows.dtype != ref.dtype or cur.device != ref.device or cur.dtype != ref.dtype:
        raise ValueError("HDC tensors must share device/dtype")
    if labels.shape != (ref.size(0),) or labels.dtype != torch.long or labels.device != ref.device:
        raise ValueError("labels_global must be torch.long [N]")
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

    eps = torch.finfo(ref.dtype).eps
    old_unit = F.normalize(rows.detach(), p=2, dim=1, eps=eps)
    ref_unit = F.normalize(ref.reshape(ref.size(0), -1), p=2, dim=1, eps=eps)
    cur_unit = F.normalize(cur.reshape(cur.size(0), -1), p=2, dim=1, eps=eps)
    teacher = F.linear(ref_unit, old_unit).detach()
    student = F.linear(cur_unit, old_unit)
    teacher_centered = teacher - teacher.mean(dim=1, keepdim=True)
    student_centered = student - student.mean(dim=1, keepdim=True)

    seen_ids = seen_cds.classes.class_ids.to(device=labels.device, dtype=torch.long)
    matches = labels.unsqueeze(1).eq(seen_ids.unsqueeze(0))
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise RuntimeError("current label absent from all-seen CDS")
    current_rows = matches.to(torch.long).argmax(dim=1)
    risk, learned, spectral = whole_cds_pair_risk(seen_cds)
    weight = risk.index_select(0, current_rows)[:, :old_count].to(dtype=ref.dtype)
    mass = weight.sum().clamp_min(eps)

    teacher_scale = torch.sqrt((weight * teacher_centered.square()).sum() / mass).detach().clamp_min(eps)
    teacher_relation = (teacher_centered / teacher_scale).detach()
    student_relation = student_centered / teacher_scale
    difference = student_relation - teacher_relation
    loss = (weight * difference.square()).sum() / mass
    rmse = difference.square().mean().sqrt()
    selected_learned = learned.index_select(0, current_rows)[:, :old_count]
    selected_spectral = spectral.index_select(0, current_rows)[:, :old_count]
    values = (
        loss,
        rmse,
        teacher_relation.norm(dim=1).mean(),
        student_relation.norm(dim=1).mean(),
        weight.mean(),
        selected_learned.mean(),
        selected_spectral.mean(),
    )
    if not all(bool(torch.isfinite(v)) for v in values):
        raise RuntimeError("historical decision consistency became NaN or Inf")
    return HistoricalDecisionConsistencyObjective(
        loss=loss,
        relation_rmse=rmse,
        teacher_relation_norm_mean=teacher_relation.norm(dim=1).mean(),
        student_relation_norm_mean=student_relation.norm(dim=1).mean(),
        pair_weight_mean=weight.mean(),
        learned_region_risk_mean=selected_learned.mean(),
        spectral_region_risk_mean=selected_spectral.mean(),
    )


def hsi_cds_separation_loss(
    *,
    logits: Tensor,
    class_ids: Tensor,
    labels_global: Tensor,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    sample_weights: Tensor | None = None,
) -> HSICDSSeparationObjective:
    """Risk-weighted new->old separation using the complete all-seen HSI CDS."""
    scores = torch.as_tensor(logits)
    ids = torch.as_tensor(class_ids)
    labels = torch.as_tensor(labels_global)
    old_count = int(historical_class_count)
    if scores.ndim != 2 or scores.size(0) == 0:
        raise ValueError("logits must be non-empty [N,C]")
    if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
        raise ValueError("class_ids must match logits")
    if old_count <= 0 or old_count >= int(ids.numel()):
        raise ValueError("historical_class_count must split all-seen classes")
    if labels.shape != (scores.size(0),) or labels.dtype != torch.long or labels.device != scores.device:
        raise ValueError("labels_global must be torch.long [N]")
    targets = _targets_local(labels, ids)
    if bool((targets < old_count).any()):
        raise ValueError("HSI-CDS separation expects current-phase labels only")

    true_logits = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    old_logits = scores[:, :old_count]
    margin = true_logits.unsqueeze(1) - old_logits
    risk, learned, spectral = whole_cds_pair_risk(seen_cds)
    selected = risk.index_select(0, targets)[:, :old_count].to(dtype=scores.dtype)
    eps = torch.finfo(scores.dtype).eps
    per_sample = (selected * F.softplus(-margin)).sum(dim=1) / selected.sum(dim=1).clamp_min(eps)
    loss = _weighted_mean(per_sample, sample_weights)
    learned_selected = learned.index_select(0, targets)[:, :old_count]
    spectral_selected = spectral.index_select(0, targets)[:, :old_count]
    values = (loss, margin.mean(), margin.min(), selected.mean(), learned_selected.mean(), spectral_selected.mean())
    if not all(bool(torch.isfinite(v)) for v in values):
        raise RuntimeError("HSI-CDS separation became NaN or Inf")
    return HSICDSSeparationObjective(
        loss=loss,
        margin_mean=margin.mean(),
        margin_min=margin.min(),
        pair_weight_mean=selected.mean(),
        learned_region_risk_mean=learned_selected.mean(),
        spectral_region_risk_mean=spectral_selected.mean(),
    )


def _historical_region_clearance(
    *,
    historical_identity: Tensor,
    historical_variation: Tensor,
    classifier_rows: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Measure every historical CDS region against every seen cosine boundary.

    For historical class c and seen rival r, the deployed normalized cosine
    boundary is b_cr = w_c - w_r.  I_Q supplies the signed identity margin while
    V_Q supplies the maximum first-order exposure of the stored ordered
    spectral-relation deformation to that same boundary:

        M_cr = <b_cr, I_Q^c> / ||I_Q^c||
        E_cr = sum_{k<l,d} |b_cr[k,d]-b_cr[l,d]| V_Q^c[k,l,d]
               / (K ||I_Q^c||)
        C_cr = M_cr - E_cr

    This is not a second classifier and it does not synthesize historical
    features.  It is direct supervision of the one deployed classifier from the
    complete transported historical class region.
    """
    identity = torch.as_tensor(historical_identity)
    variation = torch.as_tensor(historical_variation)
    rows = torch.as_tensor(classifier_rows)
    if identity.ndim != 3 or identity.size(0) <= 0 or identity.size(1) < 2:
        raise ValueError("historical_identity must be non-empty [O,K,D], K>=2")
    old_count, token_count, token_dim = identity.shape
    if variation.shape != (old_count, token_count, token_count, token_dim):
        raise ValueError("historical_variation must be [O,K,K,D]")
    if rows.ndim != 2 or rows.size(0) < old_count or rows.size(1) != token_count * token_dim:
        raise ValueError("classifier_rows must be [C,K*D] with C>=O")
    if any(value.device != identity.device or value.dtype != identity.dtype for value in (variation, rows)):
        raise ValueError("historical CDS and classifier rows must share device/dtype")
    if not all(bool(torch.isfinite(value).all()) for value in (identity, variation, rows)):
        raise ValueError("historical CDS classifier tensors contain NaN or Inf")
    if bool((variation < 0).any()):
        raise ValueError("V_Q must be non-negative")

    eps = torch.finfo(identity.dtype).eps
    unit_rows = F.normalize(rows, p=2, dim=1, eps=eps)
    boundary = unit_rows[:old_count, None, :] - unit_rows[None, :, :]

    flat_identity = identity.reshape(old_count, -1)
    identity_norm = flat_identity.norm(dim=1).clamp_min(eps)
    margin = (boundary * flat_identity[:, None, :]).sum(dim=2) / identity_norm[:, None]

    boundary_kd = boundary.reshape(old_count, rows.size(0), token_count, token_dim)
    relation_sensitivity = (
        boundary_kd[:, :, :, None, :] - boundary_kd[:, :, None, :, :]
    ).abs()
    upper = torch.triu(
        torch.ones(token_count, token_count, device=identity.device, dtype=torch.bool),
        diagonal=1,
    )
    exposure = (
        relation_sensitivity
        * variation[:, None, :, :, :]
        * upper.view(1, 1, token_count, token_count, 1)
    ).sum(dim=(2, 3, 4)) / (float(token_count) * identity_norm[:, None])
    clearance = margin - exposure
    if not all(bool(torch.isfinite(value).all()) for value in (margin, exposure, clearance)):
        raise RuntimeError("historical CDS region clearance became NaN or Inf")
    return margin, exposure, clearance


def historical_full_cds_supervision_loss(
    *,
    classifier: SpectralCDSClassifier,
    seen_cds: ClassDiscriminativeStructureState,
    historical_class_count: int,
    historical_classifier_rows: Tensor | None = None,
) -> HistoricalCDSSupervisionObjective:
    """Train the one all-seen classifier from real-new and full historical CDS.

    HyperSC restores historical positive supervision by mixing reconstructed old
    HSI with real current HSI before training the same all-seen classifier.  This
    architecture keeps the same principle without inversion: transported
    historical CDS is the old-class supervision source.

    I_Q supplies an old positive CE target, while V_Q *directly* defines the
    historical class region that every seen classifier boundary must clear.
    I_P/V_P remain phase-stable and, together with learned inter-CDS, prioritize
    spectrally dangerous class pairs through whole_cds_pair_risk.  Thus all four
    stored structural tensors affect actual classifier gradients.
    """
    if not isinstance(classifier, SpectralCDSClassifier):
        raise TypeError("classifier must be SpectralCDSClassifier")
    if not isinstance(seen_cds, ClassDiscriminativeStructureState):
        raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

    classes = seen_cds.classes
    ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long)
    old_count = int(historical_class_count)
    if old_count <= 0 or old_count >= int(ids.numel()):
        raise ValueError("historical_class_count must split all-seen classes")

    identity = classes.identity.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
    variation = classes.variation.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
    historical_identity = identity[:old_count].detach()
    historical_variation = variation[:old_count].detach()

    # Historical positive supervision has TWO complementary sources in the same
    # classifier space.  Transported I_Q is the structural class identity, while
    # the frozen Phase-(t-1) classifier rows are the decision directions that were
    # actually validated on real historical HSI.  Using both avoids declaring the
    # classifier safe merely because one structural centre is correct while a new
    # row still invades the previous decision direction.
    output = classifier(
        historical_identity,
        class_ids=ids,
        trainable_class_ids=ids,
    )
    logits = output.logits
    target_local = torch.arange(old_count, device=logits.device, dtype=torch.long)
    identity_ce = F.cross_entropy(logits, target_local)
    identity_accuracy = logits.argmax(dim=1).eq(target_local).to(logits.dtype).mean()

    anchor_rows = (
        classifier.class_weight(ids[:old_count]).detach()
        if historical_classifier_rows is None
        else torch.as_tensor(
            historical_classifier_rows,
            device=classifier.weight.device,
            dtype=classifier.weight.dtype,
        ).detach()
    )
    if anchor_rows.shape != (old_count, classifier.feature_dim):
        raise ValueError("historical_classifier_rows must be [O,K*D]")
    if not bool(torch.isfinite(anchor_rows).all()):
        raise ValueError("historical_classifier_rows contain NaN or Inf")
    anchor_q = anchor_rows.reshape(
        old_count, classifier.spectral_token_count, classifier.spectral_token_dim
    )
    anchor_output = classifier(
        anchor_q,
        class_ids=ids,
        trainable_class_ids=ids,
    )
    decision_anchor_ce = F.cross_entropy(anchor_output.logits, target_local)
    decision_anchor_accuracy = (
        anchor_output.logits.argmax(dim=1).eq(target_local).to(logits.dtype).mean()
    )

    # Equal averaging keeps structural identity and previously validated decision
    # geometry on the same scale without introducing a new dataset-specific weight.
    positive_ce = 0.5 * (identity_ce + decision_anchor_ce)

    # Complete historical region supervision.  Unlike the previous implementation,
    # V_Q is not reduced to a pair-weight diagnostic: it changes the actual
    # old-vs-seen boundary clearance and therefore the classifier gradient.
    rows = classifier.class_weight(ids)
    identity_margin, deformation_exposure, structural_clearance = _historical_region_clearance(
        historical_identity=historical_identity,
        historical_variation=historical_variation,
        classifier_rows=rows,
    )

    risk, learned, physical = whole_cds_pair_risk(seen_cds)
    risk_old = risk[:old_count].to(device=logits.device, dtype=logits.dtype)
    learned_old = learned[:old_count].to(device=logits.device, dtype=logits.dtype)
    physical_old = physical[:old_count].to(device=logits.device, dtype=logits.dtype)

    # Every seen rival stays active, as in an all-seen CE classifier.  Whole-CDS
    # risk only increases priority for HSI-confusable pairs; it never silently
    # removes a low-risk boundary.  Each old class is normalized independently
    # before the final mean so large/high-risk classes cannot dominate history.
    pair_weight = 1.0 + risk_old
    self_mask = torch.ones_like(pair_weight, dtype=torch.bool)
    self_mask[target_local, target_local] = False
    pair_weight = pair_weight * self_mask.to(dtype=pair_weight.dtype)
    pair_penalty = F.softplus(-structural_clearance)
    eps = torch.finfo(logits.dtype).eps
    per_old_mass = pair_weight.sum(dim=1).clamp_min(eps)
    per_old_region_loss = (pair_weight * pair_penalty).sum(dim=1) / per_old_mass
    relation_loss = per_old_region_loss.mean()

    # Current real CE and this historical objective are both normalized
    # classification-scale losses.  No dataset-specific coefficient is added.
    loss = 0.5 * (positive_ce + relation_loss)

    active = self_mask
    active_margin = identity_margin[active]
    active_exposure = deformation_exposure[active]
    active_clearance = structural_clearance[active]
    active_weight = pair_weight[active]
    active_learned = learned_old[active]
    active_physical = physical_old[active]

    values = (
        loss,
        positive_ce,
        decision_anchor_ce,
        relation_loss,
        identity_accuracy,
        decision_anchor_accuracy,
        active_margin.mean(),
        active_margin.min(),
        active_exposure.mean(),
        active_clearance.mean(),
        active_clearance.min(),
        active_weight.mean(),
        active_learned.mean(),
        active_physical.mean(),
    )
    if not all(bool(torch.isfinite(value)) for value in values):
        raise RuntimeError("historical full-CDS supervision became NaN or Inf")

    return HistoricalCDSSupervisionObjective(
        loss=loss,
        positive_ce=positive_ce,
        decision_anchor_ce=decision_anchor_ce,
        relation_loss=relation_loss,
        identity_accuracy=identity_accuracy,
        decision_anchor_accuracy=decision_anchor_accuracy,
        identity_margin_mean=active_margin.mean(),
        identity_margin_min=active_margin.min(),
        deformation_exposure_mean=active_exposure.mean(),
        structural_clearance_mean=active_clearance.mean(),
        structural_clearance_min=active_clearance.min(),
        pair_weight_mean=active_weight.mean(),
        learned_region_risk_mean=active_learned.mean(),
        spectral_region_risk_mean=active_physical.mean(),
    )


__all__ = [
    "ClassificationObjective",
    "SupervisedContrastiveObjective",
    "HistoricalDecisionConsistencyObjective",
    "HSICDSSeparationObjective",
    "HistoricalCDSSupervisionObjective",
    "classification_objective",
    "supervised_contrastive_q_loss",
    "historical_decision_consistency_loss",
    "hsi_cds_separation_loss",
    "historical_full_cds_supervision_loss",
]









# from __future__ import annotations

# """Losses for the locked NECIL-HSI architecture.

# Q[K,D] is the only learned sample representation. Phase-0/current real samples
# use class-balanced cosine CE plus direct-Q supervised contrast. Incremental
# stability uses HDC and a feature-only new-vs-old separation term. Historical
# classifier protection is intentionally NOT implemented here: it is owned by
# models.cds_decision_region and acts on the same cosine classifier using the
# complete all-seen CDS {I_Q,V_Q,I_P,V_P,N}.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import (
#     ClassDiscriminativeStructureState,
#     whole_cds_pair_risk,
# )
# from models.classifier import ordered_complete_cds_support

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class SupervisedContrastiveObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     negative_similarity_mean: Tensor
#     similarity_gap: Tensor


# @dataclass(frozen=True)
# class HistoricalDecisionConsistencyObjective:
#     loss: Tensor
#     relation_rmse: Tensor
#     teacher_relation_norm_mean: Tensor
#     student_relation_norm_mean: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor


# @dataclass(frozen=True)
# class HSICDSSeparationObjective:
#     loss: Tensor
#     margin_mean: Tensor
#     margin_min: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor




# def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
#     labels = torch.as_tensor(labels_global)
#     ids = torch.as_tensor(class_ids)
#     if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
#         raise ValueError("labels_global must be non-empty torch.long [N]")
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C], C>=2")
#     if labels.device != ids.device:
#         raise ValueError("labels_global and class_ids must share device")
#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a target label is absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
#     if sample_weights is None:
#         return per_sample.mean()
#     weights = torch.as_tensor(sample_weights)
#     if weights.shape != per_sample.shape or weights.device != per_sample.device or weights.dtype != per_sample.dtype:
#         raise ValueError("sample_weights must match per-sample loss")
#     if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
#         raise ValueError("sample_weights must be finite, non-negative and have positive mass")
#     return (per_sample * weights).sum() / weights.sum()


# def classification_objective(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def supervised_contrastive_q_loss(
#     *,
#     ordered_features: Tensor,
#     labels_global: Tensor,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> SupervisedContrastiveObjective:
#     """Supervised contrast directly on the deployed ordered Q representation.

#     This is the narrow feature-overlap correction.  It introduces no projection
#     head, prototype, CDS anchor, replay state, or classifier parameter.  Each
#     sample is contrasted only against other *real samples in the current batch*.
#     The flattened Q vectors are L2-normalized so the contrastive geometry matches
#     the cosine classifier geometry.  The classifier's single positive logit scale
#     is reused only as a detached similarity scale, so this objective cannot update
#     classifier rows or logit calibration.

#     Anchors that have no same-class peer in the current batch are ignored rather
#     than producing an invalid or biased loss.
#     """
#     q = torch.as_tensor(ordered_features)
#     labels = torch.as_tensor(labels_global)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
#         raise ValueError("labels_global must be torch.long [N] on the Q device")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")

#     eps = torch.finfo(q.dtype).eps
#     unit = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     similarity = (unit @ unit.t()).clamp(-1.0, 1.0) * scale

#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     candidate_mask = ~eye

#     masked_similarity = similarity.masked_fill(~candidate_mask, -torch.inf)
#     log_denom = torch.logsumexp(masked_similarity, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0

#     if bool(valid_anchor.any()):
#         log_probability = similarity - log_denom.unsqueeze(1)
#         per_anchor = -(
#             log_probability * positive_mask.to(dtype=q.dtype)
#         ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         valid_loss = per_anchor[valid_anchor]
#         if sample_weights is None:
#             loss = valid_loss.mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(valid_loss, weights[valid_anchor])
#     else:
#         # Keep a valid zero gradient path to Q for batches with no positive pair.
#         loss = q.sum() * 0.0

#     positive_values = similarity[positive_mask]
#     negative_values = similarity[negative_mask]
#     zero = q.new_zeros(())
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     negative_mean = negative_values.mean() if negative_values.numel() else zero
#     gap = positive_mean - negative_mean
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

#     values = (loss, valid_fraction, positive_mean, negative_mean, gap)
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("supervised Q contrastive objective became NaN or Inf")

#     return SupervisedContrastiveObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         negative_similarity_mean=negative_mean,
#         similarity_gap=gap,
#     )


# def historical_decision_consistency_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     historical_classifier_rows: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> HistoricalDecisionConsistencyObjective:
#     """Preserve phase-start old-class relative decision geometry on current HSI."""
#     ref = torch.as_tensor(reference_features)
#     cur = torch.as_tensor(current_features)
#     rows = torch.as_tensor(historical_classifier_rows)
#     labels = torch.as_tensor(labels_global)
#     old_count = int(historical_class_count)
#     if ref.ndim != 3 or ref.size(0) == 0 or cur.shape != ref.shape:
#         raise ValueError("reference/current features must share non-empty [N,K,D]")
#     if old_count < 2 or rows.shape != (old_count, ref.size(1) * ref.size(2)):
#         raise ValueError("historical_classifier_rows must be [O,K*D], O>=2")
#     if rows.device != ref.device or rows.dtype != ref.dtype or cur.device != ref.device or cur.dtype != ref.dtype:
#         raise ValueError("HDC tensors must share device/dtype")
#     if labels.shape != (ref.size(0),) or labels.dtype != torch.long or labels.device != ref.device:
#         raise ValueError("labels_global must be torch.long [N]")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     eps = torch.finfo(ref.dtype).eps
#     old_unit = F.normalize(rows.detach(), p=2, dim=1, eps=eps)
#     ref_unit = F.normalize(ref.reshape(ref.size(0), -1), p=2, dim=1, eps=eps)
#     cur_unit = F.normalize(cur.reshape(cur.size(0), -1), p=2, dim=1, eps=eps)
#     teacher = F.linear(ref_unit, old_unit).detach()
#     student = F.linear(cur_unit, old_unit)
#     teacher_centered = teacher - teacher.mean(dim=1, keepdim=True)
#     student_centered = student - student.mean(dim=1, keepdim=True)

#     seen_ids = seen_cds.classes.class_ids.to(device=labels.device, dtype=torch.long)
#     matches = labels.unsqueeze(1).eq(seen_ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise RuntimeError("current label absent from all-seen CDS")
#     current_rows = matches.to(torch.long).argmax(dim=1)
#     risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     weight = 1.0 + risk.index_select(0, current_rows)[:, :old_count].to(dtype=ref.dtype)
#     mass = weight.sum().clamp_min(eps)

#     teacher_scale = torch.sqrt((weight * teacher_centered.square()).sum() / mass).detach().clamp_min(eps)
#     teacher_relation = (teacher_centered / teacher_scale).detach()
#     student_relation = student_centered / teacher_scale
#     difference = student_relation - teacher_relation
#     loss = (weight * difference.square()).sum() / mass
#     rmse = difference.square().mean().sqrt()
#     selected_learned = learned.index_select(0, current_rows)[:, :old_count]
#     selected_spectral = spectral.index_select(0, current_rows)[:, :old_count]
#     values = (
#         loss,
#         rmse,
#         teacher_relation.norm(dim=1).mean(),
#         student_relation.norm(dim=1).mean(),
#         weight.mean(),
#         selected_learned.mean(),
#         selected_spectral.mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("historical decision consistency became NaN or Inf")
#     return HistoricalDecisionConsistencyObjective(
#         loss=loss,
#         relation_rmse=rmse,
#         teacher_relation_norm_mean=teacher_relation.norm(dim=1).mean(),
#         student_relation_norm_mean=student_relation.norm(dim=1).mean(),
#         pair_weight_mean=weight.mean(),
#         learned_region_risk_mean=selected_learned.mean(),
#         spectral_region_risk_mean=selected_spectral.mean(),
#     )


# def _historical_complete_learned_support(
#     *,
#     ordered_features: Tensor,
#     historical_identity: Tensor,
#     historical_variation: Tensor,
# ) -> Tensor:
#     """Return direct Q-vs-historical complete learned CDS support [N,O].

#     I_Q contributes absolute learned identity compatibility and V_Q contributes
#     ordered-relation compatibility.  This deliberately uses the same support
#     definition as evaluation so training and reported CDS overlap cannot drift
#     into different semantics.
#     """
#     return ordered_complete_cds_support(
#         ordered_features,
#         historical_identity,
#         historical_variation,
#     )


# def hsi_cds_separation_loss(
#     *,
#     ordered_features: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     sample_weights: Tensor | None = None,
# ) -> HSICDSSeparationObjective:
#     """Repel real current Q directly from transported historical CDS regions.

#     This objective contains no classifier logits. For every current sample, I_Q
#     supplies absolute learned identity compatibility and V_Q supplies ordered
#     relation compatibility against each transported historical class.
#     I_P/V_P from the old and current class enter through whole_cds_pair_risk,
#     which prioritizes spectrally confusable class pairs while the positive
#     baseline keeps every historical class active.
#     """
#     q = torch.as_tensor(ordered_features)
#     labels = torch.as_tensor(labels_global)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
#         raise ValueError("labels_global must be torch.long [N] on the Q device")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=q.device, dtype=torch.long)
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= int(ids.numel()):
#         raise ValueError("historical_class_count must split all-seen classes")
#     targets = _targets_local(labels, ids)
#     if bool((targets < old_count).any()):
#         raise ValueError("HSI-CDS separation expects current-phase labels only")

#     historical_identity = classes.identity[:old_count].to(device=q.device, dtype=q.dtype).detach()
#     historical_variation = classes.variation[:old_count].to(device=q.device, dtype=q.dtype).detach()
#     support = _historical_complete_learned_support(
#         ordered_features=q,
#         historical_identity=historical_identity,
#         historical_variation=historical_variation,
#     )

#     risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     risk = risk.to(device=q.device, dtype=q.dtype)
#     learned = learned.to(device=q.device, dtype=q.dtype)
#     spectral = spectral.to(device=q.device, dtype=q.dtype)
#     pair_weight = 1.0 + risk.index_select(0, targets)[:, :old_count]
#     eps = torch.finfo(q.dtype).eps

#     # Support=1 means the current sample lies on the historical relation-CDS;
#     # support=0 means it is structurally outside. -log(1-support) gives a smooth
#     # barrier with the natural endpoints and introduces no hand-set margin.
#     pair_penalty = -torch.log((1.0 - support).clamp_min(eps))
#     # Unit baseline keeps every historical class active.  The unnormalized
#     # 1+R_CDS factor lets absolute learned+physical confusability change the
#     # real feature-exclusion gradient instead of cancelling in a weighted mean.
#     per_sample = (pair_weight * pair_penalty).mean(dim=1)
#     loss = _weighted_mean(per_sample, sample_weights)

#     # Natural exclusion clearance: 1-support. Positive and larger is safer.
#     clearance = 1.0 - support
#     learned_selected = learned.index_select(0, targets)[:, :old_count]
#     spectral_selected = spectral.index_select(0, targets)[:, :old_count]
#     values = (
#         loss, clearance.mean(), clearance.min(), pair_weight.mean(),
#         learned_selected.mean(), spectral_selected.mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("HSI-CDS structural separation became NaN or Inf")
#     return HSICDSSeparationObjective(
#         loss=loss,
#         margin_mean=clearance.mean(),
#         margin_min=clearance.min(),
#         pair_weight_mean=pair_weight.mean(),
#         learned_region_risk_mean=learned_selected.mean(),
#         spectral_region_risk_mean=spectral_selected.mean(),
#     )

# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "HistoricalDecisionConsistencyObjective",
#     "HSICDSSeparationObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "historical_decision_consistency_loss",
#     "hsi_cds_separation_loss",
# ]












# from __future__ import annotations

# """Losses for the locked NECIL-HSI architecture.

# Q[K,D] is the only learned sample representation. Phase-0/current real samples
# use class-balanced cosine CE plus direct-Q supervised contrast. Incremental
# stability uses HDC and a feature-only new-vs-old separation term. Historical
# classifier protection is intentionally NOT implemented here: it is owned by
# models.cds_decision_region and acts on the same cosine classifier using the
# complete all-seen CDS {I_Q,V_Q,I_P,V_P,N}.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.geometry_bank import ClassDiscriminativeStructureState, whole_cds_pair_risk

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class SupervisedContrastiveObjective:
#     loss: Tensor
#     valid_anchor_fraction: Tensor
#     positive_similarity_mean: Tensor
#     negative_similarity_mean: Tensor
#     similarity_gap: Tensor


# @dataclass(frozen=True)
# class HistoricalDecisionConsistencyObjective:
#     loss: Tensor
#     relation_rmse: Tensor
#     teacher_relation_norm_mean: Tensor
#     student_relation_norm_mean: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor


# @dataclass(frozen=True)
# class HSICDSSeparationObjective:
#     loss: Tensor
#     margin_mean: Tensor
#     margin_min: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor




# def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
#     labels = torch.as_tensor(labels_global)
#     ids = torch.as_tensor(class_ids)
#     if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
#         raise ValueError("labels_global must be non-empty torch.long [N]")
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C], C>=2")
#     if labels.device != ids.device:
#         raise ValueError("labels_global and class_ids must share device")
#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a target label is absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
#     if sample_weights is None:
#         return per_sample.mean()
#     weights = torch.as_tensor(sample_weights)
#     if weights.shape != per_sample.shape or weights.device != per_sample.device or weights.dtype != per_sample.dtype:
#         raise ValueError("sample_weights must match per-sample loss")
#     if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
#         raise ValueError("sample_weights must be finite, non-negative and have positive mass")
#     return (per_sample * weights).sum() / weights.sum()


# def classification_objective(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def supervised_contrastive_q_loss(
#     *,
#     ordered_features: Tensor,
#     labels_global: Tensor,
#     logit_scale: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> SupervisedContrastiveObjective:
#     """Supervised contrast directly on the deployed ordered Q representation.

#     This is the narrow feature-overlap correction.  It introduces no projection
#     head, prototype, CDS anchor, replay state, or classifier parameter.  Each
#     sample is contrasted only against other *real samples in the current batch*.
#     The flattened Q vectors are L2-normalized so the contrastive geometry matches
#     the cosine classifier geometry.  The classifier's single positive logit scale
#     is reused only as a detached similarity scale, so this objective cannot update
#     classifier rows or logit calibration.

#     Anchors that have no same-class peer in the current batch are ignored rather
#     than producing an invalid or biased loss.
#     """
#     q = torch.as_tensor(ordered_features)
#     labels = torch.as_tensor(labels_global)
#     if q.ndim != 3 or q.size(0) == 0 or q.size(1) < 2 or q.size(2) <= 0:
#         raise ValueError("ordered_features must be non-empty [N,K,D], K>=2")
#     if not torch.is_floating_point(q) or q.is_complex() or not bool(torch.isfinite(q).all()):
#         raise ValueError("ordered_features must be finite real floating point")
#     if labels.shape != (q.size(0),) or labels.dtype != torch.long or labels.device != q.device:
#         raise ValueError("labels_global must be torch.long [N] on the Q device")

#     scale = torch.as_tensor(logit_scale, device=q.device, dtype=q.dtype).detach()
#     if scale.ndim != 0 or not bool(torch.isfinite(scale)) or not bool(scale > 0):
#         raise ValueError("logit_scale must be a finite positive scalar")

#     eps = torch.finfo(q.dtype).eps
#     unit = F.normalize(q.reshape(q.size(0), -1), p=2, dim=1, eps=eps)
#     similarity = (unit @ unit.t()).clamp(-1.0, 1.0) * scale

#     n = int(q.size(0))
#     eye = torch.eye(n, device=q.device, dtype=torch.bool)
#     same = labels[:, None].eq(labels[None, :])
#     positive_mask = same & ~eye
#     negative_mask = (~same) & ~eye
#     candidate_mask = ~eye

#     masked_similarity = similarity.masked_fill(~candidate_mask, -torch.inf)
#     log_denom = torch.logsumexp(masked_similarity, dim=1)
#     positive_count = positive_mask.sum(dim=1)
#     valid_anchor = positive_count > 0

#     if bool(valid_anchor.any()):
#         log_probability = similarity - log_denom.unsqueeze(1)
#         per_anchor = -(
#             log_probability * positive_mask.to(dtype=q.dtype)
#         ).sum(dim=1) / positive_count.clamp_min(1).to(dtype=q.dtype)
#         valid_loss = per_anchor[valid_anchor]
#         if sample_weights is None:
#             loss = valid_loss.mean()
#         else:
#             weights = torch.as_tensor(sample_weights)
#             if weights.shape != (n,) or weights.device != q.device or weights.dtype != q.dtype:
#                 raise ValueError("sample_weights must be [N] and match Q device/dtype")
#             loss = _weighted_mean(valid_loss, weights[valid_anchor])
#     else:
#         # Keep a valid zero gradient path to Q for batches with no positive pair.
#         loss = q.sum() * 0.0

#     positive_values = similarity[positive_mask]
#     negative_values = similarity[negative_mask]
#     zero = q.new_zeros(())
#     positive_mean = positive_values.mean() if positive_values.numel() else zero
#     negative_mean = negative_values.mean() if negative_values.numel() else zero
#     gap = positive_mean - negative_mean
#     valid_fraction = valid_anchor.to(dtype=q.dtype).mean()

#     values = (loss, valid_fraction, positive_mean, negative_mean, gap)
#     if not all(bool(torch.isfinite(value)) for value in values):
#         raise RuntimeError("supervised Q contrastive objective became NaN or Inf")

#     return SupervisedContrastiveObjective(
#         loss=loss,
#         valid_anchor_fraction=valid_fraction,
#         positive_similarity_mean=positive_mean,
#         negative_similarity_mean=negative_mean,
#         similarity_gap=gap,
#     )


# def historical_decision_consistency_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     historical_classifier_rows: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> HistoricalDecisionConsistencyObjective:
#     """Preserve phase-start old-class relative decision geometry on current HSI."""
#     ref = torch.as_tensor(reference_features)
#     cur = torch.as_tensor(current_features)
#     rows = torch.as_tensor(historical_classifier_rows)
#     labels = torch.as_tensor(labels_global)
#     old_count = int(historical_class_count)
#     if ref.ndim != 3 or ref.size(0) == 0 or cur.shape != ref.shape:
#         raise ValueError("reference/current features must share non-empty [N,K,D]")
#     if old_count < 2 or rows.shape != (old_count, ref.size(1) * ref.size(2)):
#         raise ValueError("historical_classifier_rows must be [O,K*D], O>=2")
#     if rows.device != ref.device or rows.dtype != ref.dtype or cur.device != ref.device or cur.dtype != ref.dtype:
#         raise ValueError("HDC tensors must share device/dtype")
#     if labels.shape != (ref.size(0),) or labels.dtype != torch.long or labels.device != ref.device:
#         raise ValueError("labels_global must be torch.long [N]")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     eps = torch.finfo(ref.dtype).eps
#     old_unit = F.normalize(rows.detach(), p=2, dim=1, eps=eps)
#     ref_unit = F.normalize(ref.reshape(ref.size(0), -1), p=2, dim=1, eps=eps)
#     cur_unit = F.normalize(cur.reshape(cur.size(0), -1), p=2, dim=1, eps=eps)
#     teacher = F.linear(ref_unit, old_unit).detach()
#     student = F.linear(cur_unit, old_unit)
#     teacher_centered = teacher - teacher.mean(dim=1, keepdim=True)
#     student_centered = student - student.mean(dim=1, keepdim=True)

#     seen_ids = seen_cds.classes.class_ids.to(device=labels.device, dtype=torch.long)
#     matches = labels.unsqueeze(1).eq(seen_ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise RuntimeError("current label absent from all-seen CDS")
#     current_rows = matches.to(torch.long).argmax(dim=1)
#     risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     weight = 1.0 + risk.index_select(0, current_rows)[:, :old_count].to(dtype=ref.dtype)
#     mass = weight.sum().clamp_min(eps)

#     teacher_scale = torch.sqrt((weight * teacher_centered.square()).sum() / mass).detach().clamp_min(eps)
#     teacher_relation = (teacher_centered / teacher_scale).detach()
#     student_relation = student_centered / teacher_scale
#     difference = student_relation - teacher_relation
#     loss = (weight * difference.square()).sum() / mass
#     rmse = difference.square().mean().sqrt()
#     selected_learned = learned.index_select(0, current_rows)[:, :old_count]
#     selected_spectral = spectral.index_select(0, current_rows)[:, :old_count]
#     values = (
#         loss,
#         rmse,
#         teacher_relation.norm(dim=1).mean(),
#         student_relation.norm(dim=1).mean(),
#         weight.mean(),
#         selected_learned.mean(),
#         selected_spectral.mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("historical decision consistency became NaN or Inf")
#     return HistoricalDecisionConsistencyObjective(
#         loss=loss,
#         relation_rmse=rmse,
#         teacher_relation_norm_mean=teacher_relation.norm(dim=1).mean(),
#         student_relation_norm_mean=student_relation.norm(dim=1).mean(),
#         pair_weight_mean=weight.mean(),
#         learned_region_risk_mean=selected_learned.mean(),
#         spectral_region_risk_mean=selected_spectral.mean(),
#     )


# def hsi_cds_separation_loss(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     sample_weights: Tensor | None = None,
# ) -> HSICDSSeparationObjective:
#     """Feature-only new->old separation with complete-CDS pair prioritization.

# Classifier rows are expected to be detached by the caller. Every historical
# class remains an active rival through the positive baseline ``1 + risk``; the
# complete learned/physical CDS only prioritizes confusable pairs.
# """
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     labels = torch.as_tensor(labels_global)
#     old_count = int(historical_class_count)
#     if scores.ndim != 2 or scores.size(0) == 0:
#         raise ValueError("logits must be non-empty [N,C]")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if old_count <= 0 or old_count >= int(ids.numel()):
#         raise ValueError("historical_class_count must split all-seen classes")
#     if labels.shape != (scores.size(0),) or labels.dtype != torch.long or labels.device != scores.device:
#         raise ValueError("labels_global must be torch.long [N]")
#     targets = _targets_local(labels, ids)
#     if bool((targets < old_count).any()):
#         raise ValueError("HSI-CDS separation expects current-phase labels only")

#     true_logits = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
#     old_logits = scores[:, :old_count]
#     margin = true_logits.unsqueeze(1) - old_logits
#     risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     selected = 1.0 + risk.index_select(0, targets)[:, :old_count].to(dtype=scores.dtype)
#     eps = torch.finfo(scores.dtype).eps
#     per_sample = (selected * F.softplus(-margin)).sum(dim=1) / selected.sum(dim=1).clamp_min(eps)
#     loss = _weighted_mean(per_sample, sample_weights)
#     learned_selected = learned.index_select(0, targets)[:, :old_count]
#     spectral_selected = spectral.index_select(0, targets)[:, :old_count]
#     values = (loss, margin.mean(), margin.min(), selected.mean(), learned_selected.mean(), spectral_selected.mean())
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("HSI-CDS separation became NaN or Inf")
#     return HSICDSSeparationObjective(
#         loss=loss,
#         margin_mean=margin.mean(),
#         margin_min=margin.min(),
#         pair_weight_mean=selected.mean(),
#         learned_region_risk_mean=learned_selected.mean(),
#         spectral_region_risk_mean=spectral_selected.mean(),
#     )

# __all__ = [
#     "ClassificationObjective",
#     "SupervisedContrastiveObjective",
#     "HistoricalDecisionConsistencyObjective",
#     "HSICDSSeparationObjective",
#     "classification_objective",
#     "supervised_contrastive_q_loss",
#     "historical_decision_consistency_loss",
#     "hsi_cds_separation_loss",
# ]



# from __future__ import annotations

# """Losses for the full-CDS historical-supervision NECIL-HSI architecture.

# The deployed decision rule remains one trainable bias-free cosine classifier.
# Persistent class state remains exactly {I_Q,V_Q,I_P,V_P,N}; inter-class CDS is
# re-derived from the stored state every refresh.

# Incremental roles:
#     current CE       -> current-class plasticity with all seen rows as rivals;
#     HDC              -> preserve the previous historical decision relation;
#     CDS separation   -> keep real current HSI away from risky historical regions;
#     historical CDS   -> give old classes direct positive classifier supervision,
#                         with pairwise competition weighted by complete learned +
#                         physical intra/inter HSI-CDS risk.

# No pseudo-feature generation, inversion, covariance model, second classifier,
# post-hoc Stage-B solve, or decision-region regulator is used.
# """

# from dataclasses import dataclass

# import torch
# import torch.nn.functional as F

# from models.classifier import SpectralCDSClassifier
# from models.geometry_bank import ClassDiscriminativeStructureState, whole_cds_pair_risk

# Tensor = torch.Tensor


# @dataclass(frozen=True)
# class ClassificationObjective:
#     loss: Tensor
#     accuracy: Tensor
#     targets_local: Tensor


# @dataclass(frozen=True)
# class HistoricalDecisionConsistencyObjective:
#     loss: Tensor
#     relation_rmse: Tensor
#     teacher_relation_norm_mean: Tensor
#     student_relation_norm_mean: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor


# @dataclass(frozen=True)
# class HSICDSSeparationObjective:
#     loss: Tensor
#     margin_mean: Tensor
#     margin_min: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor


# @dataclass(frozen=True)
# class HistoricalCDSSupervisionObjective:
#     loss: Tensor
#     positive_ce: Tensor
#     relation_loss: Tensor
#     identity_accuracy: Tensor
#     identity_margin_mean: Tensor
#     identity_margin_min: Tensor
#     pair_weight_mean: Tensor
#     learned_region_risk_mean: Tensor
#     spectral_region_risk_mean: Tensor


# def _targets_local(labels_global: Tensor, class_ids: Tensor) -> Tensor:
#     labels = torch.as_tensor(labels_global)
#     ids = torch.as_tensor(class_ids)
#     if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
#         raise ValueError("labels_global must be non-empty torch.long [N]")
#     if ids.ndim != 1 or ids.numel() < 2 or ids.dtype != torch.long:
#         raise ValueError("class_ids must be torch.long [C], C>=2")
#     if labels.device != ids.device:
#         raise ValueError("labels_global and class_ids must share device")
#     matches = labels.unsqueeze(1).eq(ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise ValueError("a target label is absent from class_ids")
#     return matches.to(torch.long).argmax(dim=1)


# def _weighted_mean(per_sample: Tensor, sample_weights: Tensor | None) -> Tensor:
#     if sample_weights is None:
#         return per_sample.mean()
#     weights = torch.as_tensor(sample_weights)
#     if weights.shape != per_sample.shape or weights.device != per_sample.device or weights.dtype != per_sample.dtype:
#         raise ValueError("sample_weights must match per-sample loss")
#     if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()) or not bool(weights.sum() > 0):
#         raise ValueError("sample_weights must be finite, non-negative and have positive mass")
#     return (per_sample * weights).sum() / weights.sum()


# def classification_objective(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     sample_weights: Tensor | None = None,
# ) -> ClassificationObjective:
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) < 2:
#         raise ValueError("logits must be non-empty [N,C], C>=2")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if not bool(torch.isfinite(scores).all()):
#         raise ValueError("logits contain NaN or Inf")
#     targets = _targets_local(labels_global, ids)
#     per_sample = F.cross_entropy(scores, targets, reduction="none")
#     loss = _weighted_mean(per_sample, sample_weights)
#     accuracy = scores.argmax(dim=1).eq(targets).to(scores.dtype).mean()
#     if not bool(torch.isfinite(loss)):
#         raise RuntimeError("classification loss became NaN or Inf")
#     return ClassificationObjective(loss=loss, accuracy=accuracy, targets_local=targets)


# def historical_decision_consistency_loss(
#     *,
#     reference_features: Tensor,
#     current_features: Tensor,
#     historical_classifier_rows: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> HistoricalDecisionConsistencyObjective:
#     """Preserve phase-start old-class relative decision geometry on current HSI."""
#     ref = torch.as_tensor(reference_features)
#     cur = torch.as_tensor(current_features)
#     rows = torch.as_tensor(historical_classifier_rows)
#     labels = torch.as_tensor(labels_global)
#     old_count = int(historical_class_count)
#     if ref.ndim != 3 or ref.size(0) == 0 or cur.shape != ref.shape:
#         raise ValueError("reference/current features must share non-empty [N,K,D]")
#     if old_count < 2 or rows.shape != (old_count, ref.size(1) * ref.size(2)):
#         raise ValueError("historical_classifier_rows must be [O,K*D], O>=2")
#     if rows.device != ref.device or rows.dtype != ref.dtype or cur.device != ref.device or cur.dtype != ref.dtype:
#         raise ValueError("HDC tensors must share device/dtype")
#     if labels.shape != (ref.size(0),) or labels.dtype != torch.long or labels.device != ref.device:
#         raise ValueError("labels_global must be torch.long [N]")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     eps = torch.finfo(ref.dtype).eps
#     old_unit = F.normalize(rows.detach(), p=2, dim=1, eps=eps)
#     ref_unit = F.normalize(ref.reshape(ref.size(0), -1), p=2, dim=1, eps=eps)
#     cur_unit = F.normalize(cur.reshape(cur.size(0), -1), p=2, dim=1, eps=eps)
#     teacher = F.linear(ref_unit, old_unit).detach()
#     student = F.linear(cur_unit, old_unit)
#     teacher_centered = teacher - teacher.mean(dim=1, keepdim=True)
#     student_centered = student - student.mean(dim=1, keepdim=True)

#     seen_ids = seen_cds.classes.class_ids.to(device=labels.device, dtype=torch.long)
#     matches = labels.unsqueeze(1).eq(seen_ids.unsqueeze(0))
#     if not bool(matches.sum(dim=1).eq(1).all()):
#         raise RuntimeError("current label absent from all-seen CDS")
#     current_rows = matches.to(torch.long).argmax(dim=1)
#     risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     weight = risk.index_select(0, current_rows)[:, :old_count].to(dtype=ref.dtype)
#     mass = weight.sum().clamp_min(eps)

#     teacher_scale = torch.sqrt((weight * teacher_centered.square()).sum() / mass).detach().clamp_min(eps)
#     teacher_relation = (teacher_centered / teacher_scale).detach()
#     student_relation = student_centered / teacher_scale
#     difference = student_relation - teacher_relation
#     loss = (weight * difference.square()).sum() / mass
#     rmse = difference.square().mean().sqrt()
#     selected_learned = learned.index_select(0, current_rows)[:, :old_count]
#     selected_spectral = spectral.index_select(0, current_rows)[:, :old_count]
#     values = (
#         loss,
#         rmse,
#         teacher_relation.norm(dim=1).mean(),
#         student_relation.norm(dim=1).mean(),
#         weight.mean(),
#         selected_learned.mean(),
#         selected_spectral.mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("historical decision consistency became NaN or Inf")
#     return HistoricalDecisionConsistencyObjective(
#         loss=loss,
#         relation_rmse=rmse,
#         teacher_relation_norm_mean=teacher_relation.norm(dim=1).mean(),
#         student_relation_norm_mean=student_relation.norm(dim=1).mean(),
#         pair_weight_mean=weight.mean(),
#         learned_region_risk_mean=selected_learned.mean(),
#         spectral_region_risk_mean=selected_spectral.mean(),
#     )


# def hsi_cds_separation_loss(
#     *,
#     logits: Tensor,
#     class_ids: Tensor,
#     labels_global: Tensor,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
#     sample_weights: Tensor | None = None,
# ) -> HSICDSSeparationObjective:
#     """Risk-weighted new->old separation using the complete all-seen HSI CDS."""
#     scores = torch.as_tensor(logits)
#     ids = torch.as_tensor(class_ids)
#     labels = torch.as_tensor(labels_global)
#     old_count = int(historical_class_count)
#     if scores.ndim != 2 or scores.size(0) == 0:
#         raise ValueError("logits must be non-empty [N,C]")
#     if ids.shape != (scores.size(1),) or ids.dtype != torch.long or ids.device != scores.device:
#         raise ValueError("class_ids must match logits")
#     if old_count <= 0 or old_count >= int(ids.numel()):
#         raise ValueError("historical_class_count must split all-seen classes")
#     if labels.shape != (scores.size(0),) or labels.dtype != torch.long or labels.device != scores.device:
#         raise ValueError("labels_global must be torch.long [N]")
#     targets = _targets_local(labels, ids)
#     if bool((targets < old_count).any()):
#         raise ValueError("HSI-CDS separation expects current-phase labels only")

#     true_logits = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
#     old_logits = scores[:, :old_count]
#     margin = true_logits.unsqueeze(1) - old_logits
#     risk, learned, spectral = whole_cds_pair_risk(seen_cds)
#     selected = risk.index_select(0, targets)[:, :old_count].to(dtype=scores.dtype)
#     eps = torch.finfo(scores.dtype).eps
#     per_sample = (selected * F.softplus(-margin)).sum(dim=1) / selected.sum(dim=1).clamp_min(eps)
#     loss = _weighted_mean(per_sample, sample_weights)
#     learned_selected = learned.index_select(0, targets)[:, :old_count]
#     spectral_selected = spectral.index_select(0, targets)[:, :old_count]
#     values = (loss, margin.mean(), margin.min(), selected.mean(), learned_selected.mean(), spectral_selected.mean())
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("HSI-CDS separation became NaN or Inf")
#     return HSICDSSeparationObjective(
#         loss=loss,
#         margin_mean=margin.mean(),
#         margin_min=margin.min(),
#         pair_weight_mean=selected.mean(),
#         learned_region_risk_mean=learned_selected.mean(),
#         spectral_region_risk_mean=spectral_selected.mean(),
#     )


# def historical_full_cds_supervision_loss(
#     *,
#     classifier: SpectralCDSClassifier,
#     seen_cds: ClassDiscriminativeStructureState,
#     historical_class_count: int,
# ) -> HistoricalCDSSupervisionObjective:
#     """Direct old-class supervision from the complete transported HSI CDS.

#     The aligned historical I_Q tensors are the legal classifier-shaped anchors.
#     They are not treated as the whole class structure. The complete intra/inter
#     HSI-CDS controls how strongly every historical class competes with every seen
#     rival through ``whole_cds_pair_risk``:

#         learned  = 0.5 * (inter(I_Q) + inter(V_Q))
#         physical = 0.5 * (inter(I_P) + inter(V_P))
#         risk     = sqrt(learned * physical)

#     Thus V_Q, I_P and V_P affect the actual classifier gradients rather than
#     serving only as diagnostics. No synthetic sample or pseudo-feature is built.
#     """
#     if not isinstance(classifier, SpectralCDSClassifier):
#         raise TypeError("classifier must be SpectralCDSClassifier")
#     if not isinstance(seen_cds, ClassDiscriminativeStructureState):
#         raise TypeError("seen_cds must be ClassDiscriminativeStructureState")

#     classes = seen_cds.classes
#     ids = classes.class_ids.to(device=classifier.weight.device, dtype=torch.long)
#     old_count = int(historical_class_count)
#     if old_count <= 0 or old_count >= int(ids.numel()):
#         raise ValueError("historical_class_count must split all-seen classes")

#     identity = classes.identity.to(device=classifier.weight.device, dtype=classifier.weight.dtype)
#     historical_identity = identity[:old_count].detach()
#     old_ids = ids[:old_count]

#     # All seen classifier rows are live for this objective. Old rows receive
#     # legitimate old-class positive supervision; current rows receive only the
#     # rival-side gradients induced by historical CDS competition.
#     output = classifier(
#         historical_identity,
#         class_ids=ids,
#         trainable_class_ids=ids,
#     )
#     logits = output.logits
#     target_local = torch.arange(old_count, device=logits.device, dtype=torch.long)
#     positive_ce = F.cross_entropy(logits, target_local)
#     identity_accuracy = logits.argmax(dim=1).eq(target_local).to(logits.dtype).mean()

#     true_logits = logits.gather(1, target_local.unsqueeze(1)).squeeze(1)
#     margins = true_logits.unsqueeze(1) - logits

#     risk, learned, physical = whole_cds_pair_risk(seen_cds)
#     risk_old = risk[:old_count].to(device=logits.device, dtype=logits.dtype)
#     learned_old = learned[:old_count].to(device=logits.device, dtype=logits.dtype)
#     physical_old = physical[:old_count].to(device=logits.device, dtype=logits.dtype)

#     # Diagonal is already zero by definition. Risk therefore contains only
#     # historical-vs-rival inter-class relations and is derived from all four
#     # stored structural tensors I_Q,V_Q,I_P,V_P.
#     pair_penalty = F.softplus(-margins)
#     eps = torch.finfo(logits.dtype).eps
#     mass = risk_old.sum()
#     relation_loss = torch.where(
#         mass > eps,
#         (risk_old * pair_penalty).sum() / mass.clamp_min(eps),
#         logits.new_zeros(()),
#     )

#     # Both terms are normalized scalar classification objectives. Averaging keeps
#     # the historical objective on the same order as one CE term without adding a
#     # tunable method weight.
#     loss = 0.5 * (positive_ce + relation_loss)

#     rival_mask = torch.ones_like(risk_old, dtype=torch.bool)
#     rival_mask[target_local, target_local] = False
#     rival_margin = margins[rival_mask]
#     weighted_mask = risk_old > 0
#     if bool(weighted_mask.any()):
#         weighted_margin = margins[weighted_mask]
#         margin_mean = weighted_margin.mean()
#         margin_min = weighted_margin.min()
#     else:
#         margin_mean = rival_margin.mean()
#         margin_min = rival_margin.min()

#     values = (
#         loss,
#         positive_ce,
#         relation_loss,
#         identity_accuracy,
#         margin_mean,
#         margin_min,
#         risk_old.mean(),
#         learned_old.mean(),
#         physical_old.mean(),
#     )
#     if not all(bool(torch.isfinite(v)) for v in values):
#         raise RuntimeError("historical full-CDS supervision became NaN or Inf")

#     return HistoricalCDSSupervisionObjective(
#         loss=loss,
#         positive_ce=positive_ce,
#         relation_loss=relation_loss,
#         identity_accuracy=identity_accuracy,
#         identity_margin_mean=margin_mean,
#         identity_margin_min=margin_min,
#         pair_weight_mean=risk_old.mean(),
#         learned_region_risk_mean=learned_old.mean(),
#         spectral_region_risk_mean=physical_old.mean(),
#     )


# __all__ = [
#     "ClassificationObjective",
#     "HistoricalDecisionConsistencyObjective",
#     "HSICDSSeparationObjective",
#     "HistoricalCDSSupervisionObjective",
#     "classification_objective",
#     "historical_decision_consistency_loss",
#     "hsi_cds_separation_loss",
#     "historical_full_cds_supervision_loss",
# ]
